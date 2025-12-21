from typing import List, Set
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from config.settings import QDRANT_COLLECTION

def check_qdrant_data(client: QdrantClient) -> dict:
    """Qdrant 컬렉션의 데이터 확인"""
    try:
        collections = client.get_collections().collections
        exists = any(c.name == QDRANT_COLLECTION for c in collections)
        
        if not exists:
            print(f"❌ Qdrant 컬렉션 없음: {QDRANT_COLLECTION}")
            return {"exists": False, "count": 0, "page_ids": set()}
        
        info = client.get_collection(QDRANT_COLLECTION)
        count = info.points_count
        
        # 모든 page_id 추출
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=["page_id"]
        )
        
        page_ids = {point.payload["page_id"] for point in scroll_result[0]}
        
        print(f"✅ Qdrant 데이터: {count}개 청크, {len(page_ids)}개 페이지")
        
        return {
            "exists": True,
            "count": count,
            "page_ids": page_ids
        }
    except Exception as e:
        print(f"⚠️ Qdrant 확인 실패: {e}")
        return {"exists": False, "count": 0, "page_ids": set()}


def init_qdrant(client: QdrantClient, dimension: int, recreate: bool = False):
    """Qdrant 컬렉션 초기화"""
    collections = client.get_collections().collections
    exists = any(c.name == QDRANT_COLLECTION for c in collections)
    
    if exists and recreate:
        client.delete_collection(QDRANT_COLLECTION)
        exists = False
    
    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )
        for field, ftype in [("page_id", PayloadSchemaType.KEYWORD), 
                            ("page_title", PayloadSchemaType.KEYWORD)]:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=ftype
            )
        print(f"✅ 컬렉션 생성: {QDRANT_COLLECTION}")


def delete_page_from_qdrant(client: QdrantClient, page_id: str):
    """특정 페이지의 모든 청크 삭제"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="page_id",
                    match=MatchValue(value=page_id)
                )
            ]
        )
    )
    print(f"  🗑️ 페이지 삭제: {page_id}")


def store_to_qdrant(chunks, embeddings, client: QdrantClient):
    """Qdrant에 저장"""
    import hashlib
    
    points = []
    for chunk, emb in zip(chunks, embeddings):
        pid = hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:32]
        pid = f"{pid[:8]}-{pid[8:12]}-{pid[12:16]}-{pid[16:20]}-{pid[20:]}"
        
        props = {k: (v if isinstance(v, (str, int, float, bool, list)) else str(v)) 
                for k, v in chunk.properties.items()}
        
        points.append(PointStruct(
            id=pid,
            vector=emb,
            payload={
                "chunk_id": chunk.chunk_id,
                "page_id": chunk.page_id,
                "text": chunk.text,
                "combined_text": chunk.combined_text,
                "has_image": chunk.has_image,
                "image_paths": chunk.image_paths,
                "image_descriptions": chunk.image_descriptions,
                "page_title": chunk.page_title,
                "section_title": chunk.section_title,
                "section_path": chunk.section_path,
                "properties": props
            }
        ))
    
    for i in range(0, len(points), 100):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+100])
    
    print(f"✅ {len(points)}개 청크 저장 완료")