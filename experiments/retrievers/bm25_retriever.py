#!/usr/bin/env python3
"""BM25 Retriever - LangChain 기반 키워드 검색"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from config.settings import (
    QDRANT_COLLECTION,
    QDRANT_URL,
    QDRANT_USE_SERVER,
    get_qdrant_path,
    get_collection_name
)

# 싱글톤 패턴: 문서 캐싱 (Qdrant lock 방지)
_documents_cache = {}


def load_documents_from_qdrant(date_filter: tuple = None, preset: str = None) -> List[Document]:
    """
    Qdrant에서 모든 문서를 로드하여 LangChain Document로 변환

    Args:
        date_filter: (start_date, end_date) 튜플 (ISO 형식)
        preset: 임베딩 프리셋 (None이면 환경변수 사용)
    """
    import os

    # 컬렉션 이름 결정
    collection_name = get_collection_name(preset)

    # Qdrant 클라이언트 생성 (서버 모드 우선)
    if QDRANT_USE_SERVER:
        # 캐시 키에 컬렉션 이름 포함
        cache_key = (QDRANT_URL, collection_name, date_filter)
        client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    else:
        # 레거시: 로컬 파일 모드
        qdrant_path = get_qdrant_path()
        cache_key = (qdrant_path, QDRANT_COLLECTION, date_filter)
        client = QdrantClient(path=qdrant_path)
        collection_name = QDRANT_COLLECTION

    # 캐시 확인
    if cache_key in _documents_cache:
        return _documents_cache[cache_key]

    try:
        # scroll 파라미터 설정
        scroll_params = {
            "collection_name": collection_name,
            "limit": 10000,
            "with_payload": True,
            "with_vectors": False  # 벡터는 필요 없음
        }

        # 모든 포인트 가져오기 (날짜 필터링은 Python에서 처리)
        scroll_result = client.scroll(**scroll_params)

        documents = []
        for point in scroll_result[0]:
            # payload 구조 감지 (두 가지 형식 지원)
            # 형식 1: metadata 필드에 중첩 (upstage 등)
            # 형식 2: payload 최상위에 직접 저장 (openai-large 등)

            # page_content 추출 (여러 필드명 시도)
            page_content = (
                point.payload.get("page_content") or
                point.payload.get("combined_text") or
                point.payload.get("text") or
                ""
            )

            # 빈 문서는 건너뛰기
            if not page_content or not page_content.strip():
                continue

            # metadata 추출 (두 가지 구조 지원)
            if "metadata" in point.payload:
                # 형식 1: metadata 필드에 중첩
                metadata_dict = point.payload["metadata"]
            else:
                # 형식 2: payload 최상위에 직접 저장
                metadata_dict = point.payload

            # 날짜 필터링 (Python 레벨에서 처리)
            if date_filter:
                start_date, end_date = date_filter

                # properties에서 날짜 추출
                properties = metadata_dict.get("properties", {})

                # 새로운 날짜_start 필드 우선 사용 (vectordb 재구축 후)
                date_start = properties.get("날짜_start", "")

                # 날짜_start 필드가 없으면 기존 방식 사용 (하위 호환성)
                if not date_start:
                    last_edited = properties.get("최종 편집 일시", "")
                    created = properties.get("생성 일시", "")

                    # 날짜 범위 체크
                    in_range = False
                    for date_str in [last_edited, created]:
                        if date_str and start_date <= date_str <= end_date:
                            in_range = True
                            break
                else:
                    # 날짜_start 필드로 필터링 (권장)
                    in_range = date_start and start_date <= date_start <= end_date

                # 범위 밖이면 건너뛰기
                if not in_range:
                    continue

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

        # 캐시에 저장
        _documents_cache[cache_key] = documents

        return documents
    finally:
        # client 명시적으로 닫기
        client.close()


def get_bm25_retriever(k: int = 5, date_filter: tuple = None, preset: str = None) -> BM25Retriever:
    """
    BM25 Retriever 생성

    Args:
        k: 반환할 문서 수
        date_filter: (start_date, end_date) 튜플 (ISO 형식)
        preset: 임베딩 프리셋 (None이면 환경변수 사용)

    Returns:
        BM25Retriever 인스턴스
    """
    # Qdrant에서 문서 로드
    documents = load_documents_from_qdrant(date_filter=date_filter, preset=preset)

    # 문서가 없는 경우 처리
    if not documents:
        print(f"⚠️ 날짜 필터({date_filter})에 해당하는 문서가 없습니다.")
        # 빈 retriever 반환 (최소 1개의 더미 문서 필요)
        from langchain_core.documents import Document
        documents = [Document(page_content="", metadata={})]

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
