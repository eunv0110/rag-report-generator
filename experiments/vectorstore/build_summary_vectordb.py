#!/usr/bin/env python3
"""S11: Summary-Level Vector DB 구축 스크립트 (요약 기반 검색)

Mixed 컬렉션에서 요약본을 복사하여 Summary 전용 컬렉션 생성
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import *
from utils.langfuse import get_langfuse_client, trace_operation
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid


def init_summary_collection(
    client: QdrantClient,
    collection_name: str,
    dimension: int,
    recreate: bool = False
):
    """
    Summary Vector DB용 Qdrant 컬렉션 초기화

    Args:
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
        dimension: 벡터 차원
        recreate: 기존 컬렉션 삭제 후 재생성 여부
    """
    if recreate:
        try:
            client.delete_collection(collection_name)
            print(f"  ✓ 기존 컬렉션 삭제: {collection_name}")
        except:
            pass

    # 컬렉션 존재 여부 확인
    collections = [col.name for col in client.get_collections().collections]

    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
        print(f"  ✓ 컬렉션 생성: {collection_name}")
    else:
        print(f"  ✓ 기존 컬렉션 사용: {collection_name}")


def main(
    force_recreate: bool = False,
    limit: int = None,
    collection_name: str = "notion_summary",
    source_collection: str = "notion_mixed",
    summary_length: int = 200
):
    """
    S11: Summary-Level Vector DB 구축 메인 함수

    Mixed 컬렉션에서 요약본을 가져와서 Summary 전용 컬렉션 생성

    Args:
        force_recreate: True면 전체 재생성
        limit: 처리할 페이지 수 제한 (None이면 전체)
        collection_name: 저장할 컬렉션 이름
        source_collection: 요약본을 가져올 Mixed 컬렉션 이름
        summary_length: 요약 최대 길이 (자) - Mixed에서 가져오므로 사용 안 함
    """
    print("=" * 60)
    print("📝 S11: Summary-Level Vector DB 구축 시작")
    print("   (요약 기반 검색 - 속도 개선)")
    print(f"   ✨ Mixed 컬렉션의 요약본 재사용!")
    if limit:
        print(f"📊 제한: {limit}개 페이지만 처리")
    print(f"📦 타겟 컬렉션: {collection_name}")
    print(f"📥 소스 컬렉션: {source_collection}")
    print("=" * 60)

    # Langfuse 초기화
    get_langfuse_client()

    # 전체 프로세스를 Langfuse로 트레이싱
    with trace_operation(
        name="summary_vectordb_build",
        metadata={
            "force_recreate": force_recreate,
            "limit": limit,
            "collection_name": collection_name,
            "source_collection": source_collection
        }
    ) as trace:

        # 1. Qdrant 클라이언트 초기화
        print("\n📦 Qdrant 클라이언트 초기화...")
        qdrant_client = QdrantClient(path=QDRANT_PATH)

        # 2. Mixed 컬렉션 확인
        print(f"\n📥 Mixed 컬렉션 확인 중: {source_collection}")
        try:
            source_info = qdrant_client.get_collection(source_collection)
            print(f"  ✓ Mixed 컬렉션 발견: {source_info.points_count}개 포인트")
        except Exception as e:
            print(f"❌ Mixed 컬렉션이 없습니다: {source_collection}")
            print("먼저 build_mixed_vectordb.py를 실행하세요!")
            return

        # 3. Mixed 컬렉션에서 요약본만 가져오기
        print(f"\n📝 요약본 추출 중...")
        if trace:
            extraction_span = trace.span(name="summary_extraction")

        # 요약본만 스크롤로 모두 가져오기
        summary_points = []
        offset = None
        batch_size = 100

        while True:
            results = qdrant_client.scroll(
                collection_name=source_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="properties.content_type",
                            match=MatchValue(value="summary")
                        )
                    ]
                ),
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            points, offset = results

            if not points:
                break

            summary_points.extend(points)

            if offset is None:
                break

            if limit and len(summary_points) >= limit:
                summary_points = summary_points[:limit]
                break

        if trace:
            extraction_span.end(metadata={"total_summaries": len(summary_points)})

        print(f"  ✓ {len(summary_points)}개 요약본 추출 완료")

        if not summary_points:
            print("❌ Mixed 컬렉션에 요약본이 없습니다!")
            return

        # 통계
        avg_len = sum(len(p.payload.get("text", "")) for p in summary_points) / len(summary_points)
        print(f"  ✓ 평균 요약 길이: {avg_len:.1f}자")

        # 4. Summary 컬렉션 생성 및 저장
        print(f"\n💾 Summary 컬렉션 저장 중: {collection_name}")
        if trace:
            storage_span = trace.span(name="qdrant_storage")

        # 임베딩 차원 확인 (첫 번째 포인트에서)
        embedding_dimension = len(summary_points[0].vector)

        # 컬렉션 초기화
        init_summary_collection(
            qdrant_client,
            collection_name,
            embedding_dimension,
            recreate=force_recreate
        )

        # 포인트 복사 (새 UUID로)
        new_points = []
        for point in summary_points:
            new_point = PointStruct(
                id=str(uuid.uuid4()),  # 새 ID 생성
                vector=point.vector,
                payload=point.payload
            )
            new_points.append(new_point)

        # 배치로 저장
        batch_size = 100
        for i in range(0, len(new_points), batch_size):
            batch = new_points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )

        print(f"  ✓ {len(new_points)}개 요약본 저장 완료")

        if trace:
            storage_span.end()

        # 5. 검증
        print("\n🔍 저장 검증...")
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"  ✓ 저장된 포인트 수: {collection_info.points_count}")

        print("\n" + "=" * 60)
        print("🎉 S11: Summary-Level Vector DB 구축 완료!")
        print(f"📦 컬렉션: {collection_name}")
        print(f"📥 소스: {source_collection} (요약본만)")
        print(f"📊 총 청크 수: {len(summary_points)}")
        print(f"📝 평균 요약 길이: {avg_len:.1f}자")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S11: Summary-Level Vector DB 구축 (Mixed 요약본 재사용)")
    parser.add_argument("--force", action="store_true", help="전체 재생성")
    parser.add_argument("--limit", type=int, default=None, help="처리할 페이지 수 제한")
    parser.add_argument("--collection", type=str, default="notion_summary", help="타겟 컬렉션 이름")
    parser.add_argument("--source", type=str, default="notion_mixed", help="소스 Mixed 컬렉션 이름")

    args = parser.parse_args()

    main(
        force_recreate=args.force,
        limit=args.limit,
        collection_name=args.collection,
        source_collection=args.source
    )
