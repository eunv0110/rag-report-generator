import os
import json
from typing import Dict, List, Set
from datetime import datetime
from pathlib import Path

def check_existing_data(filepath: str) -> Dict:
    """기존 데이터 존재 여부 및 통계 확인"""
    if not os.path.exists(filepath):
        print(f"❌ 파일 없음: {filepath}")
        return {
            "exists": False,
            "count": 0,
            "page_ids": set(),
            "last_updated": None
        }
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        page_ids = {item["page_id"] for item in data}
        last_times = [item.get("last_edited_time", "") for item in data]
        last_updated = max(last_times) if last_times else None
        
        print(f"✅ 기존 데이터: {len(data)}개 페이지")
        print(f"   마지막 업데이트: {last_updated}")
        
        return {
            "exists": True,
            "count": len(data),
            "page_ids": page_ids,
            "last_updated": last_updated
        }
    except Exception as e:
        print(f"⚠️ 파일 로드 실패: {e}")
        return {
            "exists": False,
            "count": 0,
            "page_ids": set(),
            "last_updated": None
        }


def collect_missing_pages(collector, existing_page_ids: Set[str], filepath: str, limit: int = None) -> List[Dict]:
    """기존 데이터에 없는 새 페이지만 수집"""
    print("\n🔍 Notion에서 전체 페이지 목록 가져오는 중...")
    all_pages = collector.get_all_pages_from_datasource()
    
    new_pages = [p for p in all_pages if p["id"] not in existing_page_ids]
    
    # ✅ limit 적용
    if limit and len(new_pages) > limit:
        print(f"⚠️  {len(new_pages)}개 새 페이지 중 {limit}개만 수집")
        new_pages = new_pages[:limit]
    
    print(f"\n📊 분석 결과:")
    print(f"   전체 페이지: {len(all_pages)}개")
    print(f"   기존 페이지: {len(existing_page_ids)}개")
    print(f"   새 페이지: {len(new_pages)}개")
    
    if not new_pages:
        print("\n✅ 모든 페이지가 이미 수집되어 있습니다!")
        return []
    
    print(f"\n🚀 {len(new_pages)}개 새 페이지 수집 시작...\n")
    
    new_data = []
    for idx, page in enumerate(new_pages):
        page_id = page["id"]
        properties = collector.extract_page_properties(page)
        title = collector.get_page_title(properties)
        
        print(f"[{idx+1}/{len(new_pages)}] 📄 {title}")
        
        try:
            blocks = collector.get_all_blocks(page_id)
            content_lines = [collector.extract_block_content(b, page_id) for b in blocks]
            full_content = "\n".join(filter(None, content_lines))
            
            new_data.append({
                "page_id": page_id,
                "title": title,
                "created_time": page.get("created_time", ""),
                "last_edited_time": page.get("last_edited_time", ""),
                "properties": properties,
                "content": full_content
            })
            print(f"  → {len(blocks)}개 블록, {len(full_content)}자")
        except Exception as e:
            print(f"  ⚠️ 실패: {e}")
            new_data.append({
                "page_id": page_id,
                "title": title,
                "content": "",
                "created_time": page.get("created_time", ""),
                "last_edited_time": page.get("last_edited_time", ""),
                "properties": properties
            })
    
    # 기존 데이터와 병합
    if os.path.exists(filepath):
        from app.utils.files import load_json, save_json
        existing_data = load_json(filepath)
        merged_data = existing_data + new_data
        save_json(merged_data, filepath)
        print(f"\n💾 병합 완료: {len(existing_data)} + {len(new_data)} = {len(merged_data)}개")
    else:
        from app.utils.files import save_json
        save_json(new_data, filepath)
        print(f"\n💾 새 파일 생성: {len(new_data)}개")
    
    return new_data


def update_changed_pages(collector, existing_data: List[Dict], filepath: str) -> List[Dict]:
    """수정된 페이지 업데이트"""
    print("\n🔄 수정된 페이지 확인 중...")
    
    all_pages = collector.get_all_pages_from_datasource()
    page_map = {p["id"]: p for p in all_pages}
    
    updated_data = []
    update_count = 0
    
    for old_item in existing_data:
        page_id = old_item["page_id"]
        
        if page_id not in page_map:
            print(f"  ⚠️ 삭제된 페이지: {old_item['title']}")
            continue
        
        new_page = page_map[page_id]
        old_time = old_item.get("last_edited_time", "")
        new_time = new_page.get("last_edited_time", "")
        
        if new_time > old_time:
            print(f"  🔄 업데이트: {old_item['title']}")
            
            properties = collector.extract_page_properties(new_page)
            title = collector.get_page_title(properties)
            
            try:
                blocks = collector.get_all_blocks(page_id)
                content_lines = [collector.extract_block_content(b, page_id) for b in blocks]
                full_content = "\n".join(filter(None, content_lines))
                
                updated_data.append({
                    "page_id": page_id,
                    "title": title,
                    "created_time": new_page.get("created_time", ""),
                    "last_edited_time": new_time,
                    "properties": properties,
                    "content": full_content
                })
                update_count += 1
            except Exception as e:
                print(f"    ⚠️ 업데이트 실패: {e}")
                updated_data.append(old_item)
        else:
            updated_data.append(old_item)
    
    if update_count > 0:
        from app.utils.files import save_json
        save_json(updated_data, filepath)
        print(f"\n✅ {update_count}개 페이지 업데이트 완료")
    else:
        print("\n✅ 수정된 페이지 없음")
    
    return updated_data