#!/usr/bin/env python3
"""Dense Retriever - LangChain Qdrant 기반 벡터 검색"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_qdrant import QdrantVectorStore
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from config.settings import QDRANT_PATH, QDRANT_COLLECTION
from models.embeddings.factory import get_embedder


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


def get_dense_retriever(k: int = 5, use_singleton: bool = False) -> BaseRetriever:
    """
    Dense Retriever 생성 (Qdrant 벡터 검색)

    Args:
        k: 반환할 문서 수
        use_singleton: True면 기존 client를 재사용 (Qdrant lock 방지)

    Returns:
        Qdrant VectorStore Retriever 인스턴스
    """
    # 임베더 로드
    base_embedder = get_embedder()
    langchain_embeddings = get_langchain_embeddings(base_embedder)

    # Qdrant client 생성
    client = QdrantClient(path=QDRANT_PATH)

    # Qdrant vectorstore 로드
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=langchain_embeddings,
    )

    # Retriever로 변환
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
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
