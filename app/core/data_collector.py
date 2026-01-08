import os
import re
import httpx
from typing import List, Dict, Optional
from pathlib import Path
from notion_client import Client
from app.config.settings import IMAGE_DIR, NOTION_VERSION


class NotionDataSourceCollector:
    """Notion Data Source에서만 데이터 수집"""
    
    def __init__(self, token: str, data_source_id: str):
        if not data_source_id:
            raise ValueError("DATA_SOURCE_ID가 필요합니다!")
        self.client = Client(auth=token, notion_version=os.environ.get("NOTION_VERSION", "2025-09-03"))
        self.data_source_id = data_source_id
        
        # ✅ Path 객체로 변경
        self.image_dir = Path(IMAGE_DIR)
        self.image_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_blocks(self, block_id: str) -> List[Dict]:
        """모든 블록을 재귀적으로 가져오기 (페이지네이션 포함)"""
        all_blocks = []
        cursor = None
        
        while True:
            response = self.client.blocks.children.list(
                block_id=block_id,
                start_cursor=cursor
            )
            all_blocks.extend(response["results"])
            
            if not response.get("has_more"):
                break
            cursor = response["next_cursor"]
        
        # 하위 블록 재귀 탐색
        for block in all_blocks:
            if block.get("has_children"):
                block["children"] = self.get_all_blocks(block["id"])
        
        return all_blocks
    
    def extract_rich_text(self, rich_text_list: List) -> str:
        """rich_text 배열에서 텍스트 추출"""
        if not rich_text_list:
            return ""
        return "".join([t.get("plain_text", "") for t in rich_text_list])
    
    def download_image(self, url: str, page_id: str, block_id: str) -> Optional[str]:
        """이미지 다운로드 후 상대 경로 반환"""
        try:
            response = httpx.get(url, timeout=30)
            if response.status_code == 200:
                ext = "png"
                if "." in url.split("?")[0]:
                    ext = url.split("?")[0].split(".")[-1][:4]
                
                # ✅ 절대 경로로 저장
                filename = self.image_dir / f"{page_id}_{block_id}.{ext}"
                with open(filename, "wb") as f:
                    f.write(response.content)
                
                # ✅ 상대 경로만 반환 (notion_images/xxx.png)
                return f"notion_images/{page_id}_{block_id}.{ext}"
        except Exception as e:
            print(f"  ⚠️ 이미지 다운로드 실패: {e}")
        return None
    
    def extract_block_content(self, block: Dict, page_id: str, depth: int = 0) -> str:
        """블록에서 모든 내용 추출"""
        block_type = block["type"]
        block_id = block["id"]
        indent = "  " * depth
        result = ""
        
        # 텍스트 블록들
        text_types = [
            "paragraph", "heading_1", "heading_2", "heading_3",
            "bulleted_list_item", "numbered_list_item", 
            "quote", "toggle", "to_do", "callout"
        ]
        
        if block_type in text_types:
            text = self.extract_rich_text(block[block_type].get("rich_text", []))
            
            if block_type == "heading_1":
                result = f"{indent}# {text}"
            elif block_type == "heading_2":
                result = f"{indent}## {text}"
            elif block_type == "heading_3":
                result = f"{indent}### {text}"
            elif block_type == "bulleted_list_item":
                result = f"{indent}• {text}"
            elif block_type == "numbered_list_item":
                result = f"{indent}1. {text}"
            elif block_type == "quote":
                result = f"{indent}> {text}"
            elif block_type == "to_do":
                checked = "✅" if block["to_do"].get("checked") else "⬜"
                result = f"{indent}{checked} {text}"
            elif block_type == "callout":
                emoji = block["callout"].get("icon", {}).get("emoji", "💡")
                result = f"{indent}{emoji} {text}"
            else:
                result = f"{indent}{text}"
        
        # 코드 블록
        elif block_type == "code":
            code = self.extract_rich_text(block["code"].get("rich_text", []))
            lang = block["code"].get("language", "")
            caption = self.extract_rich_text(block["code"].get("caption", []))
            result = f"{indent}```{lang}\n{code}\n{indent}```"
            if caption:
                result += f"\n{indent}Caption: {caption}"
        
        # 이미지
        elif block_type == "image":
            image_data = block["image"]
            url = None
            if image_data["type"] == "file":
                url = image_data["file"]["url"]
            elif image_data["type"] == "external":
                url = image_data["external"]["url"]
            
            if url:
                local_path = self.download_image(url, page_id, block_id)
                caption = self.extract_rich_text(image_data.get("caption", []))
                result = f"{indent}[Image: {local_path or url}]"
                if caption:
                    result += f" - {caption}"
        
        # 비디오
        elif block_type == "video":
            video_data = block["video"]
            url = video_data.get("file", {}).get("url") or video_data.get("external", {}).get("url", "unknown")
            result = f"{indent}[Video: {url}]"
        
        # 파일
        elif block_type == "file":
            file_data = block["file"]
            url = file_data.get("file", {}).get("url") or file_data.get("external", {}).get("url", "unknown")
            name = file_data.get("name", "file")
            result = f"{indent}[File: {name} - {url}]"
        
        # 북마크
        elif block_type == "bookmark":
            url = block["bookmark"].get("url", "")
            caption = self.extract_rich_text(block["bookmark"].get("caption", []))
            result = f"{indent}[Bookmark: {url}]"
            if caption:
                result += f" - {caption}"
        
        # 임베드
        elif block_type == "embed":
            url = block["embed"].get("url", "")
            result = f"{indent}[Embed: {url}]"
        
        # ✅ 테이블 처리 추가
        elif block_type == "table":
            # 테이블 자체는 텍스트 없지만, 하위 table_row들을 처리
            result = f"{indent}[Table]"
            
        # 테이블 행
        elif block_type == "table_row":
            cells = block["table_row"].get("cells", [])
            row_data = [self.extract_rich_text(cell) for cell in cells]
            result = f"{indent}| " + " | ".join(row_data) + " |"
        
        # ✅ Column 처리 추가 (2단 레이아웃)
        elif block_type == "column_list":
            result = ""  # 컬럼 리스트는 구분자만
            
        elif block_type == "column":
            result = ""  # 개별 컬럼도 하위 블록만 처리
        
        # 구분선
        elif block_type == "divider":
            result = f"{indent}---"
        
        # 링크 프리뷰
        elif block_type == "link_preview":
            url = block["link_preview"].get("url", "")
            result = f"{indent}[Link: {url}]"
        
        # PDF
        elif block_type == "pdf":
            pdf_data = block["pdf"]
            url = pdf_data.get("file", {}).get("url") or pdf_data.get("external", {}).get("url", "unknown")
            result = f"{indent}[PDF: {url}]"
        
        # 수식
        elif block_type == "equation":
            expr = block["equation"].get("expression", "")
            result = f"{indent}[Equation: {expr}]"
        
        # 자식 페이지
        elif block_type == "child_page":
            title = block["child_page"].get("title", "")
            result = f"{indent}📄 [{title}]"
        
        # 자식 데이터베이스
        elif block_type == "child_database":
            title = block["child_database"].get("title", "")
            result = f"{indent}🗃️ [{title}]"
        
        # ✅ synced_block 처리 추가
        elif block_type == "synced_block":
            result = ""  # 동기화 블록도 하위 내용만
        
        # 기타
        else:
            result = f"{indent}[{block_type}]"
        
        # 하위 블록 처리
        if "children" in block:
            for child in block["children"]:
                child_content = self.extract_block_content(child, page_id, depth + 1)
                if child_content:
                    result += "\n" + child_content
        
        return result
    
    def extract_page_properties(self, page: Dict) -> Dict:
        """페이지 속성(메타데이터) 추출"""
        props = page.get("properties", {})
        extracted = {}
        
        for name, prop in props.items():
            prop_type = prop["type"]
            try:
                if prop_type == "title":
                    extracted[name] = self.extract_rich_text(prop["title"])
                elif prop_type == "rich_text":
                    extracted[name] = self.extract_rich_text(prop["rich_text"])
                elif prop_type == "number":
                    extracted[name] = prop["number"]
                elif prop_type == "select":
                    extracted[name] = prop["select"]["name"] if prop["select"] else None
                elif prop_type == "multi_select":
                    extracted[name] = [s["name"] for s in prop["multi_select"]]
                elif prop_type == "date":
                    if prop["date"]:
                        extracted[name] = {"start": prop["date"].get("start"), "end": prop["date"].get("end")}
                elif prop_type == "checkbox":
                    extracted[name] = prop["checkbox"]
                elif prop_type == "url":
                    extracted[name] = prop["url"]
                elif prop_type == "email":
                    extracted[name] = prop["email"]
                elif prop_type == "phone_number":
                    extracted[name] = prop["phone_number"]
                elif prop_type == "created_time":
                    extracted[name] = prop["created_time"]
                elif prop_type == "last_edited_time":
                    extracted[name] = prop["last_edited_time"]
                elif prop_type == "created_by":
                    extracted[name] = prop["created_by"].get("name", prop["created_by"].get("id"))
                elif prop_type == "last_edited_by":
                    extracted[name] = prop["last_edited_by"].get("name", prop["last_edited_by"].get("id"))
                elif prop_type == "files":
                    extracted[name] = []
                    for f in prop["files"]:
                        if f["type"] == "file":
                            extracted[name].append(f["file"]["url"])
                        elif f["type"] == "external":
                            extracted[name].append(f["external"]["url"])
                elif prop_type == "relation":
                    extracted[name] = [r["id"] for r in prop["relation"]]
                elif prop_type == "formula":
                    formula = prop["formula"]
                    extracted[name] = formula.get(formula["type"])
                elif prop_type == "rollup":
                    rollup = prop["rollup"]
                    extracted[name] = rollup.get(rollup["type"])
                elif prop_type == "status":
                    extracted[name] = prop["status"]["name"] if prop["status"] else None
                else:
                    extracted[name] = f"[{prop_type}]"
            except Exception:
                extracted[name] = None
        
        return extracted
    
    def get_page_title(self, properties: Dict) -> str:
        """속성에서 제목 추출"""
        for name, value in properties.items():
            if isinstance(value, str) and value and name.lower() in ["title", "name", "이름", "제목"]:
                return value
        for name, value in properties.items():
            if isinstance(value, str) and value:
                return value
        return "Untitled"
    
    def get_all_pages_from_datasource(self) -> List[Dict]:
        """Data Source API로 페이지 가져오기 (페이지네이션 포함)"""
        all_pages = []
        cursor = None
        
        while True:
            response = self.client.data_sources.query(
                data_source_id=self.data_source_id,
                start_cursor=cursor,
                page_size=100
            )
            all_pages.extend(response["results"])
            print(f"  📄 {len(all_pages)}개 페이지 로드됨...")
            
            if not response.get("has_more"):
                break
            cursor = response["next_cursor"]
        
        return all_pages
    
    def collect_all(self, limit: int = None) -> List[Dict]:
        """Data Source의 모든 Notion 데이터 수집"""
        print("🚀 Notion 데이터 수집 시작")
        print(f"📊 Data Source ID: {self.data_source_id}\n")
        
        pages = self.get_all_pages_from_datasource()
        print(f"\n✅ 총 {len(pages)}개 페이지 발견\n")

        if limit:
            pages = pages[:limit]   # ✅ 여기서 제한     

        all_data = []
        for idx, page in enumerate(pages):
            page_id = page["id"]
            properties = self.extract_page_properties(page)
            title = self.get_page_title(properties)
            
            print(f"[{idx+1}/{len(pages)}] 📄 {title}")
            
            try:
                blocks = self.get_all_blocks(page_id)
                content_lines = [self.extract_block_content(b, page_id) for b in blocks]
                full_content = "\n".join(filter(None, content_lines))
                
                all_data.append({
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
                all_data.append({
                    "page_id": page_id, 
                    "title": title, 
                    "content": "",
                    "created_time": page.get("created_time", ""),
                    "last_edited_time": page.get("last_edited_time", ""),
                    "properties": properties
                })
        
        print(f"\n🎉 완료! 총 {len(all_data)}개 페이지 수집")
        return all_data