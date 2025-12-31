#!/usr/bin/env python3
"""RAPTOR Vector DB 구축 스크립트"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import *
from models.embeddings.factory import get_embedder
from models.vision.factory import get_vision_model
from core.data_collector import NotionDataSourceCollector
from core.chunker import process_page_data
from retrievers.raptor_tree import RaptorTreeBuilder
from utils.file_utils import save_json, load_json
from utils.langfuse_utils import get_langfuse_client, trace_operation
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from tqdm import tqdm


def init_raptor_collection(
    client: QdrantClient,
    collection_name: str,
    dimension: int,
    recreate: bool = False
):
    """
    RAPTOR용 Qdrant 컬렉션 초기화

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


def store_raptor_nodes_to_qdrant(
    nodes,
    client: QdrantClient,
    collection_name: str
):
    """
    RAPTOR 노드들을 Qdrant에 저장

    Args:
        nodes: RaptorNode 리스트
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
    """
    points = []

    for node in nodes:
        # 메타데이터 준비
        payload = {
            "chunk_id": node.node_id,
            "page_id": node.metadata.get("page_id", ""),
            "text": node.text,
            "combined_text": node.text,
            "page_title": node.metadata.get("page_title", ""),
            "section_title": node.metadata.get("section_title"),
            "section_path": node.metadata.get("section_path"),
            "has_image": node.metadata.get("has_image", False),
            "image_descriptions": node.metadata.get("image_descriptions", []),
            "level": node.level,
            "children_ids": node.children_ids,
            "parent_id": node.parent_id,
            "properties": {
                "level": node.level,
                "children_ids": node.children_ids,
                "parent_id": node.parent_id,
                **node.metadata
            }
        }

        # Point 생성
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=node.embedding.tolist(),
            payload=payload
        )
        points.append(point)

    # 배치로 저장
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )

    print(f"  ✓ {len(points)}개 노드 저장 완료")


def main(
    force_recreate: bool = False,
    limit: int = None,
    collection_name: str = "notion_raptor",
    max_clusters: int = 10,
    max_tree_depth: int = 3,
    summarizer_type: str = "map_reduce"
):
    """
    RAPTOR Vector DB 구축 메인 함수

    Args:
        force_recreate: True면 전체 재생성
        limit: 처리할 페이지 수 제한 (None이면 전체)
        collection_name: 저장할 컬렉션 이름
        max_clusters: 클러스터링 시 최대 클러스터 개수
        max_tree_depth: RAPTOR 트리 최대 깊이
        summarizer_type: 요약 방식 ("map_reduce" 또는 "refine")
    """
    print("=" * 60)
    print("🌳 RAPTOR Vector DB 구축 시작")
    if limit:
        print(f"📊 제한: {limit}개 페이지만 처리")
    print(f"📦 컬렉션: {collection_name}")
    print(f"📝 요약 방식: {summarizer_type}")
    print("=" * 60)

    # Langfuse 초기화
    get_langfuse_client()

    # 전체 프로세스를 Langfuse로 트레이싱
    with trace_operation(
        name="raptor_vectordb_build",
        metadata={
            "force_recreate": force_recreate,
            "limit": limit,
            "collection_name": collection_name,
            "max_clusters": max_clusters,
            "max_tree_depth": max_tree_depth
        }
    ) as trace:

        data_file = DATA_DIR / "notion_data.json"

        # 1. 모델 초기화
        print("\n📦 모델 로딩...")
        if trace:
            model_span = trace.span(name="model_initialization")

        embedder = get_embedder()
        vision_model = get_vision_model()
        qdrant_client = QdrantClient(path=QDRANT_PATH)

        # RAPTOR Tree Builder 초기화
        tree_builder = RaptorTreeBuilder(
            embedder=embedder,
            max_clusters=max_clusters,
            max_tree_depth=max_tree_depth,
            summarizer_type=summarizer_type
        )

        if trace:
            model_span.end()

        # 2. 데이터 로드
        print("\n📂 데이터 로딩...")
        if not data_file.exists():
            print("❌ notion_data.json 파일이 없습니다!")
            print("먼저 build_vectordb.py를 실행하여 데이터를 수집하세요.")
            return

        all_data = load_json(str(data_file))
        pages_to_process = all_data[:limit] if limit else all_data
        print(f"  ✓ {len(pages_to_process)}개 페이지 로드")

        # 3. 페이지별로 청크 생성
        print("\n✂️ 청킹 및 트리 구축 준비...")
        if trace:
            chunking_span = trace.span(name="chunking")

        all_chunks = []
        chunk_metadata_list = []

        for page in tqdm(pages_to_process, desc="청킹"):
            chunks = process_page_data(page, embedder, vision_model)

            for chunk in chunks:
                all_chunks.append(chunk.combined_text)
                chunk_metadata_list.append({
                    "page_id": chunk.page_id,
                    "page_title": chunk.page_title,
                    "section_title": chunk.section_title,
                    "section_path": chunk.section_path,
                    "has_image": chunk.has_image,
                    "image_descriptions": chunk.image_descriptions
                })

        if trace:
            chunking_span.end(metadata={"total_chunks": len(all_chunks)})

        print(f"  ✓ 총 {len(all_chunks)}개 청크 생성")

        if not all_chunks:
            print("❌ 청크가 생성되지 않았습니다!")
            return

        # 4. RAPTOR 트리 구축
        print("\n🌳 RAPTOR 트리 구축 중...")
        if trace:
            tree_span = trace.span(name="raptor_tree_build")

        raptor_nodes = tree_builder.build_tree(
            texts=all_chunks,
            metadatas=chunk_metadata_list
        )

        if trace:
            tree_span.end(metadata={
                "total_nodes": len(raptor_nodes),
                "levels": max([node.level for node in raptor_nodes]) + 1
            })

        # 레벨별 통계
        level_counts = {}
        for node in raptor_nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1

        print("\n📊 레벨별 노드 수:")
        for level in sorted(level_counts.keys()):
            print(f"  레벨 {level}: {level_counts[level]}개")

        # 5. Qdrant에 저장
        print(f"\n💾 Qdrant 저장 중 (컬렉션: {collection_name})...")
        if trace:
            storage_span = trace.span(name="qdrant_storage")

        # 임베딩 차원 확인
        embedding_dimension = len(raptor_nodes[0].embedding)

        # 컬렉션 초기화
        init_raptor_collection(
            qdrant_client,
            collection_name,
            embedding_dimension,
            recreate=force_recreate
        )

        # 노드 저장
        store_raptor_nodes_to_qdrant(
            raptor_nodes,
            qdrant_client,
            collection_name
        )

        if trace:
            storage_span.end()

        # 6. 검증
        print("\n🔍 저장 검증...")
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"  ✓ 저장된 포인트 수: {collection_info.points_count}")

        print("\n" + "=" * 60)
        print("🎉 RAPTOR Vector DB 구축 완료!")
        print(f"📦 컬렉션: {collection_name}")
        print(f"📊 총 노드 수: {len(raptor_nodes)}")
        print(f"🌲 트리 레벨: {len(level_counts)}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAPTOR Vector DB 구축")
    parser.add_argument("--force", action="store_true", help="전체 재생성")
    parser.add_argument("--limit", type=int, default=None, help="처리할 페이지 수 제한")
    parser.add_argument("--collection", type=str, default="notion_raptor", help="컬렉션 이름")
    parser.add_argument("--max-clusters", type=int, default=10, help="최대 클러스터 개수")
    parser.add_argument("--max-depth", type=int, default=3, help="최대 트리 깊이")
    parser.add_argument("--summarizer", type=str, default="map_reduce", choices=["map_reduce", "refine"], help="요약 방식 (기본값: map_reduce)")

    args = parser.parse_args()

    main(
        force_recreate=args.force,
        limit=args.limit,
        collection_name=args.collection,
        max_clusters=args.max_clusters,
        max_tree_depth=args.max_depth,
        summarizer_type=args.summarizer
    )
