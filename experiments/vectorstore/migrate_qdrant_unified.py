#!/usr/bin/env python3
"""Qdrant 데이터 통합 스크립트 - 모든 임베딩을 하나의 서버로"""

from qdrant_client import QdrantClient
from pathlib import Path

base_path = Path("data/qdrant_data")
unified_path = "data/qdrant_unified"

print("🚀 Qdrant 데이터 통합 시작...\n")

# 통합 서버 클라이언트 (실행 중인 서버에 연결)
unified_client = QdrantClient(url="http://127.0.0.1:6333")

# 각 임베딩 디렉토리 처리
embeddings = {
    "baai-bge-m3": "notion_docs_bge_m3",
    "upstage-solar-embedding": "notion_docs_upstage",
    "openai-large": "notion_docs_openai",
    "gemini-embedding-001": "notion_docs_gemini",
    "qwen3-embedding-4b": "notion_docs_qwen3"
}

for source_dir, new_collection_name in embeddings.items():
    source_path = base_path / source_dir
    if not source_path.exists():
        print(f"⏭️  Skipping {source_dir} (not found)")
        continue

    print(f"🔄 Processing {source_dir}...")

    try:
        # 소스 클라이언트
        source_client = QdrantClient(path=str(source_path))

        # 원본 컬렉션 정보
        source_info = source_client.get_collection("notion_docs")
        print(f"   📊 Found {source_info.points_count} points")

        # 새 컬렉션 생성
        try:
            unified_client.create_collection(
                collection_name=new_collection_name,
                vectors_config=source_info.config.params.vectors
            )
            print(f"   ✅ Created collection: {new_collection_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"   ⚠️  Collection already exists, skipping...")
                continue
            raise

        # 데이터 복사
        offset = None
        total = 0
        batch_size = 100

        while True:
            records, offset = source_client.scroll(
                collection_name="notion_docs",
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            if not records:
                break

            unified_client.upload_points(
                collection_name=new_collection_name,
                points=records
            )

            total += len(records)
            if total % 200 == 0:
                print(f"   📦 Copied {total} points...")

            if offset is None:
                break

        print(f"   ✅ Completed: {new_collection_name} ({total} points)\n")

    except Exception as e:
        print(f"   ❌ Error processing {source_dir}: {e}\n")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*60)
print("✅ Migration complete!")
print("="*60)
print("\n📚 Available collections in unified storage:")
for col in unified_client.get_collections().collections:
    info = unified_client.get_collection(col.name)
    print(f"  - {col.name}: {info.points_count} points")

print("\n💡 Next step:")
print("   Use the collections via: http://127.0.0.1:6333")
