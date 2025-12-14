#!/usr/bin/env python3
"""Vector DB 구축 스크립트 (증분 업데이트 지원)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import *
from models.embeddings.factory import get_embedder
from models.vision.factory import get_vision_model
from core.data_collector import NotionDataSourceCollector
from core.chunker import process_page_data
from core.vector_store import (
    init_qdrant, 
    store_to_qdrant, 
    check_qdrant_data,
    delete_page_from_qdrant
)
from services.incremental_sync import (
    check_existing_data,
    collect_missing_pages,
    update_changed_pages
)
from utils.file_utils import save_json, load_json
from qdrant_client import QdrantClient

def main(force_recreate: bool = False, check_updates: bool = True, limit: int = None):
    """
    Vector DB 구축 메인 함수
    
    Args:
        force_recreate: True면 전체 재생성
        check_updates: True면 수정된 페이지도 확인
        limit: 처리할 페이지 수 제한 (None이면 전체)
    """
    print("=" * 60)
    print("🚀 Vector DB 구축 시작")
    if limit:
        print(f"📊 제한: {limit}개 페이지만 처리")
    print("=" * 60)
    
    data_file = DATA_DIR / "notion_data.json"
    
    # 1. 모델 초기화
    print("\n📦 모델 로딩...")
    embedder = get_embedder()
    vision_model = get_vision_model()
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    
    # 2. 데이터 수집 (증분)
    if force_recreate:
        print("\n♻️ 전체 재수집 모드")
        if data_file.exists():
            from datetime import datetime
            backup = data_file.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            data_file.rename(backup)
            print(f"   백업: {backup}")
        
        collector = NotionDataSourceCollector(NOTION_TOKEN, DATA_SOURCE_ID)
        all_data = collector.collect_all(limit=limit)  # ✅ limit 추가
        save_json(all_data, str(data_file))
        pages_to_index = all_data
        
    else:
        # 기존 데이터 확인
        existing_info = check_existing_data(str(data_file))
        
        if not existing_info["exists"]:
            print("\n📥 초기 수집 시작...")
            collector = NotionDataSourceCollector(NOTION_TOKEN, DATA_SOURCE_ID)
            all_data = collector.collect_all(limit=limit)  # ✅ limit 추가
            save_json(all_data, str(data_file))
            pages_to_index = all_data
            
        else:
            # 새 페이지 수집
            collector = NotionDataSourceCollector(NOTION_TOKEN, DATA_SOURCE_ID)
            new_data = collect_missing_pages(
                collector, 
                existing_info["page_ids"], 
                str(data_file),
                limit=limit  # ✅ limit 추가
            )
            
            # 수정된 페이지 업데이트
            updated_page_ids = set()
            if check_updates:
                all_data = load_json(str(data_file))
                old_data = all_data.copy()
                all_data = update_changed_pages(collector, all_data, str(data_file))
                
                # 어떤 페이지가 업데이트됐는지 추적
                for old, new in zip(old_data, all_data):
                    if old.get("last_edited_time") != new.get("last_edited_time"):
                        updated_page_ids.add(new["page_id"])
            else:
                all_data = load_json(str(data_file))
            
            # 인덱싱할 페이지 결정
            pages_to_index = [
                p for p in all_data 
                if p["page_id"] in updated_page_ids or 
                   p in new_data
            ]
    
    if not pages_to_index:
        print("\n✅ 인덱싱할 페이지 없음 (모두 최신 상태)")
        return
    
    # ✅ limit 적용
    if limit and len(pages_to_index) > limit:
        print(f"\n⚠️  {len(pages_to_index)}개 페이지 중 {limit}개만 처리")
        pages_to_index = pages_to_index[:limit]
    
    print(f"\n📝 {len(pages_to_index)}개 페이지 인덱싱...")
    
    # 3. Qdrant 초기화
    qdrant_info = check_qdrant_data(qdrant_client)
    
    if force_recreate or not qdrant_info["exists"]:
        # 전체 재생성
        all_chunks = []
        for page in pages_to_index:
            chunks = process_page_data(page, embedder, vision_model)
            all_chunks.extend(chunks)
            print(f"  {page.get('title', 'Untitled')}: {len(chunks)}개 청크")
        
        if all_chunks:
            print(f"\n🔢 임베딩 생성 중...")
            texts = [c.combined_text for c in all_chunks]
            embeddings = embedder.embed_texts(texts)
            
            print(f"\n💾 Qdrant 저장 중...")
            init_qdrant(qdrant_client, dimension=len(embeddings[0]), recreate=force_recreate)
            store_to_qdrant(all_chunks, embeddings, qdrant_client)
    else:
        # 증분 업데이트: 변경된 페이지만 재인덱싱
        print("\n🔄 변경된 페이지 재인덱싱...")
        
        for page in pages_to_index:
            page_id = page["page_id"]
            
            # 기존 청크 삭제
            delete_page_from_qdrant(qdrant_client, page_id)
            
            # 새 청크 생성
            chunks = process_page_data(page, embedder, vision_model)
            
            if chunks:
                texts = [c.combined_text for c in chunks]
                embeddings = embedder.embed_texts(texts)
                store_to_qdrant(chunks, embeddings, qdrant_client)
                
                print(f"  ✅ {page.get('title', 'Untitled')}: {len(chunks)}개 청크 업데이트")
    
    print("\n" + "=" * 60)
    print("🎉 Vector DB 구축 완료!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="전체 재생성")
    parser.add_argument("--no-updates", action="store_true", help="수정 체크 안 함")
    parser.add_argument("--limit", type=int, default=None, help="처리할 페이지 수 제한")  # ✅ 추가
    args = parser.parse_args()
    
    main(force_recreate=args.force, check_updates=not args.no_updates, limit=args.limit)