#!/usr/bin/env python3
"""RAPTOR 리트리버 테스트 스크립트"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrievers.raptor_retriever import RaptorRetriever
from retrievers.factory import RetrieverFactory
from config.settings import QDRANT_PATH
from qdrant_client import QdrantClient


def test_raptor_retriever():
    """RAPTOR 리트리버 기본 테스트"""
    print("=" * 60)
    print("🧪 RAPTOR Retriever 테스트")
    print("=" * 60)

    # Qdrant 클라이언트 초기화
    print("\n1. Qdrant 클라이언트 초기화...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    # RAPTOR 컬렉션 확인
    collections = [col.name for col in qdrant_client.get_collections().collections]
    print(f"   사용 가능한 컬렉션: {collections}")

    if "notion_raptor" not in collections:
        print("\n⚠️  'notion_raptor' 컬렉션이 없습니다!")
        print("먼저 다음 명령으로 RAPTOR vectordb를 구축하세요:")
        print("   python vectorstore/build_raptor_vectordb.py --limit 5")
        return

    # RAPTOR 리트리버 생성 (Factory 사용)
    print("\n2. RAPTOR 리트리버 생성 (Factory)...")
    try:
        retriever = RetrieverFactory.create(
            "raptor",
            qdrant_client,
            collection_name="notion_raptor"
        )
        print(f"   ✓ {retriever.name} 리트리버 생성 완료")
        print(f"   정보: {retriever.get_info()}")
    except Exception as e:
        print(f"   ❌ 리트리버 생성 실패: {e}")
        return

    # 컬렉션 정보 확인
    print("\n3. RAPTOR 컬렉션 정보...")
    collection_info = qdrant_client.get_collection("notion_raptor")
    print(f"   총 노드 수: {collection_info.points_count}")
    print(f"   벡터 차원: {collection_info.config.params.vectors.size}")

    # 샘플 검색 테스트
    test_queries = [
        "프로젝트 관리 방법",
        "팀 협업 도구",
        "일정 관리"
    ]

    print("\n4. 검색 테스트...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n   [{i}] 쿼리: '{query}'")

        try:
            results = retriever.search(query, top_k=3)
            print(f"       검색 결과: {len(results)}개")

            for j, result in enumerate(results, 1):
                level = result.properties.get("level", 0)
                print(f"\n       {j}. [Level {level}] {result.page_title}")
                print(f"          점수: {result.score:.4f}")
                print(f"          텍스트: {result.text[:100]}...")

        except Exception as e:
            print(f"       ❌ 검색 실패: {e}")

    # 레벨별 검색 테스트
    print("\n5. 레벨별 검색 테스트...")
    query = test_queries[0]

    for level in [0, 1, 2]:
        print(f"\n   Level {level} 검색: '{query}'")
        try:
            results = retriever.search_by_level(query, level=level, top_k=3)
            print(f"   결과: {len(results)}개")

            for j, result in enumerate(results, 1):
                print(f"      {j}. {result.page_title} (점수: {result.score:.4f})")

        except Exception as e:
            print(f"      검색 결과 없음 또는 오류: {e}")

    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_raptor_retriever()
