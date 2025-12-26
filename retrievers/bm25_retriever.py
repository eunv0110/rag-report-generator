#!/usr/bin/env python3
"""BM25 Retriever - LangChain 기반 키워드 검색"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from config.settings import QDRANT_PATH, QDRANT_COLLECTION


def load_documents_from_qdrant() -> List[Document]:
    """Qdrant에서 모든 문서를 로드하여 LangChain Document로 변환"""
    client = QdrantClient(path=QDRANT_PATH)

    try:
        # 모든 포인트 가져오기
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=True,
            with_vectors=False  # 벡터는 필요 없음
        )

        documents = []
        for point in scroll_result[0]:
            # payload 구조: {"page_content": "...", "metadata": {...}}
            page_content = point.payload.get("page_content", "")
            metadata_dict = point.payload.get("metadata", {})

            # metadata 추출
            metadata = {
                "page_id": metadata_dict.get("page_id", ""),
                "page_title": metadata_dict.get("page_title", ""),
                "section_title": metadata_dict.get("section_title", ""),
                "section_path": metadata_dict.get("section_path", ""),
                "chunk_id": metadata_dict.get("chunk_id", ""),
                "has_image": metadata_dict.get("has_image", False),
                "image_paths": metadata_dict.get("image_paths", []),
                "image_descriptions": metadata_dict.get("image_descriptions", []),
            }

            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            documents.append(doc)

        return documents
    finally:
        # client 명시적으로 닫기
        client.close()


def get_bm25_retriever(k: int = 5) -> BM25Retriever:
    """
    BM25 Retriever 생성

    Args:
        k: 반환할 문서 수

    Returns:
        BM25Retriever 인스턴스
    """
    # Qdrant에서 문서 로드
    documents = load_documents_from_qdrant()

    # BM25 Retriever 생성
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k

    return retriever


if __name__ == "__main__":
    # 테스트
    print("🔍 BM25 Retriever 테스트")
    print("=" * 60)

    # Retriever 생성
    retriever = get_bm25_retriever(k=3)

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
    print("✅ BM25 Retriever 테스트 완료")
