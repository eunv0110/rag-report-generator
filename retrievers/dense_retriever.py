#!/usr/bin/env python3
"""Dense Retriever - LangChain Qdrant 기반 벡터 검색"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_qdrant import QdrantVectorStore
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from config.settings import QDRANT_COLLECTION, get_qdrant_path
from models.embeddings.factory import get_embedder
from langchain_core.documents import Document
from typing import List
import os

# 싱글톤 클라이언트 캐시 (Qdrant lock 방지)
_qdrant_client_cache = {}


def get_langchain_embeddings(embedder):
    """기존 embedder를 LangChain Embeddings로 래핑"""
    from langchain_core.embeddings import Embeddings
    from typing import List

    # embedder가 이미 Embeddings 인터페이스를 구현하고 있으면 그대로 반환
    if isinstance(embedder, Embeddings):
        return embedder

    # 그렇지 않으면 wrapper 생성
    class CustomEmbeddings(Embeddings):
        def __init__(self, embedder):
            self.embedder = embedder

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """문서 임베딩"""
            if hasattr(self.embedder, 'embed_documents'):
                return self.embedder.embed_documents(texts)
            elif hasattr(self.embedder, 'embed_texts'):
                return self.embedder.embed_texts(texts)
            else:
                raise AttributeError("embedder에 embed_documents 또는 embed_texts 메서드가 없습니다")

        def embed_query(self, text: str) -> List[float]:
            """쿼리 임베딩"""
            if hasattr(self.embedder, 'embed_query'):
                return self.embedder.embed_query(text)
            elif hasattr(self.embedder, 'embed_texts'):
                return self.embedder.embed_texts([text])[0]
            else:
                raise AttributeError("embedder에 embed_query 또는 embed_texts 메서드가 없습니다")

    return CustomEmbeddings(embedder)


def get_dense_retriever(
    k: int = 5,
    use_singleton: bool = True,
    date_filter: tuple = None
) -> BaseRetriever:
    """
    Dense Retriever 생성 (Qdrant 벡터 검색)

    Args:
        k: 반환할 문서 수
        use_singleton: True면 기존 client를 재사용 (Qdrant lock 방지) - 기본값 True
        date_filter: (start_date, end_date) 튜플 (ISO 형식)

    Returns:
        Qdrant VectorStore Retriever 인스턴스
    """
    # 임베더 로드
    base_embedder = get_embedder()
    langchain_embeddings = get_langchain_embeddings(base_embedder)

    # Qdrant 경로 동적 계산 (현재 MODEL_PRESET 기반)
    qdrant_path = get_qdrant_path()
    model_preset = os.getenv("MODEL_PRESET", "default")

    # Qdrant client 생성 (싱글톤 사용 시 캐시에서 재사용)
    # 캐시 키에 embedding preset 포함 (각 preset마다 별도 클라이언트)
    cache_key = f"{model_preset}_{qdrant_path}_{QDRANT_COLLECTION}"

    if use_singleton and cache_key in _qdrant_client_cache:
        client = _qdrant_client_cache[cache_key]
    else:
        client = QdrantClient(path=qdrant_path)
        if use_singleton:
            _qdrant_client_cache[cache_key] = client

    # Qdrant vectorstore 로드
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=langchain_embeddings,
    )

    # 날짜 필터가 있는 경우 Python 레벨 사후 필터링 적용
    if date_filter:
        start_date, end_date = date_filter
        print(f"✅ Dense Retriever 날짜 필터 적용: {start_date[:10]} ~ {end_date[:10]}")

        # 사후 필터링을 위한 커스텀 Retriever 래퍼
        class DateFilteredRetriever(BaseRetriever):
            base_retriever: object
            start_date: str
            end_date: str
            target_k: int

            class Config:
                arbitrary_types_allowed = True

            def __init__(self, base_retriever, start_date, end_date, target_k):
                super().__init__(
                    base_retriever=base_retriever,
                    start_date=start_date,
                    end_date=end_date,
                    target_k=target_k
                )

            def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                # 더 많이 가져와서 필터링 후 k개 확보
                # search_kwargs를 딕셔너리로 전달
                all_docs = self.base_retriever.invoke(query)

                # 날짜 필터링
                filtered_docs = []
                for doc in all_docs:
                    props = doc.metadata.get('properties', {})
                    date_start = props.get('날짜_start', '')

                    if date_start and self.start_date <= date_start <= self.end_date:
                        filtered_docs.append(doc)
                        if len(filtered_docs) >= self.target_k:
                            break

                return filtered_docs[:self.target_k]

            async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                return self._get_relevant_documents(query, **kwargs)

        # 필터링 후에도 k개를 확보하기 위해 더 많이 검색
        search_kwargs = {"k": k * 3}
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        return DateFilteredRetriever(base_retriever, start_date, end_date, k)

    else:
        # 날짜 필터 없으면 일반 retriever
        search_kwargs = {"k": k}
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        return retriever


if __name__ == "__main__":
    # 테스트
    print("🔍 Dense Retriever 테스트")
    print("=" * 60)

    # Retriever 생성
    retriever = get_dense_retriever(k=3)

    # 테스트 쿼리
    test_queries = [
        "RAG 시스템은 어떻게 동작하나요?",
        "임베딩 모델에 대해 알려주세요",
        "벡터 데이터베이스란 무엇인가요?"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)

        results = retriever.invoke(query)

        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.metadata.get('page_title', 'Unknown')}")
            print(f"    Section: {doc.metadata.get('section_title', 'N/A')}")
            print(f"    Content: {doc.page_content[:200]}...")

    print("\n" + "=" * 60)
    print("✅ Dense Retriever 테스트 완료")
