import re
import base64
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from models.base import BaseVisionModel

@dataclass
class Chunk:
    chunk_id: str
    page_id: str
    text: str
    combined_text: str = ""
    has_image: bool = False
    image_paths: list = field(default_factory=list)
    image_descriptions: list = field(default_factory=list)
    page_title: str = ""
    section_title: str = ""
    section_path: list = field(default_factory=list)
    heading_level: int = 0
    chunk_index: int = 0
    created_time: str = ""
    last_edited_time: str = ""
    properties: dict = field(default_factory=dict)

# ========== 청킹 함수들 ==========
def split_by_markdown_headers(content: str, page_id: str, page_title: str) -> List[Dict]:
    """마크다운 헤더 기준으로 섹션 분리"""
    header_pattern = r'^(#{1,3})\s+(.+)$'
    lines = content.split('\n')
    sections = []
    current = {"title": page_title, "level": 0, "path": [page_title], "content": []}
    
    for line in lines:
        match = re.match(header_pattern, line)
        if match:
            if current["content"] or current["title"]:
                sections.append(current.copy())
            
            level = len(match.group(1))
            title = match.group(2).strip()
            path = [title] if level == 1 else [page_title, title]
            current = {"title": title, "level": level, "path": path, "content": []}
        else:
            current["content"].append(line)
    
    if current["content"] or current["title"]:
        sections.append(current)
    
    result = []
    for sec in sections:
        content_text = '\n'.join(sec["content"]).strip()
        if content_text or sec["level"] > 0:
            full_text = f"{'#' * sec['level']} {sec['title']}\n{content_text}".strip() if sec["level"] > 0 else content_text
            result.append({"title": sec["title"], "level": sec["level"], "path": sec["path"], "full_text": full_text})
    
    return result


def estimate_tokens(text: str) -> int:
    """토큰 수 추정 (대략 4자 = 1토큰)"""
    return len(text) // 4


def recursive_split(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """재귀적 텍스트 분할"""
    if estimate_tokens(text) <= chunk_size:
        return [text]
    
    for sep in ["\n## ", "\n### ", "\n\n", "\n• ", "\n", ". ", " "]:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                test = current + sep + part if current else part
                if estimate_tokens(test) <= chunk_size:
                    current = test
                else:
                    if current:
                        chunks.append(current)
                    current = part
            if current:
                chunks.append(current)
            return chunks
    
    char_limit = chunk_size * 4
    return [text[i:i+char_limit] for i in range(0, len(text), char_limit)]


def find_images_in_text(text: str) -> List[Dict]:
    """텍스트에서 이미지 마커 찾기"""
    pattern = r'\[Image:\s*([^\]]+)\]'
    images = []
    for m in re.finditer(pattern, text):
        path = m.group(1).strip()
        
        # ✅ 이미 상대경로로 저장되어 있으므로 그대로 사용
        # 단, notion_images/가 없으면 추가
        if not path.startswith("notion_images/"):
            path = f"notion_images/{path}"
        
        images.append({
            "path": path,
            "start": m.start(),
            "end": m.end(),
            "marker": m.group(0)
        })
    return images


def process_page_data(
    page_data: Dict, 
    embedder,  # BaseEmbedder (실제로는 사용 안 함)
    vision_model: Optional[BaseVisionModel] = None
) -> List[Chunk]:
    """페이지 데이터를 청크로 변환"""
    page_id = page_data["page_id"]
    content = page_data["content"]
    page_title = page_data.get("title", "Untitled")
    
    if not content.strip():
        return []
    
    sections = split_by_markdown_headers(content, page_id, page_title)
    chunks = []
    
    for sec_idx, section in enumerate(sections):
        split_texts = recursive_split(section["full_text"]) if estimate_tokens(section["full_text"]) > CHUNK_SIZE else [section["full_text"]]
        
        for chunk_idx, chunk_text in enumerate(split_texts):
            chunk_id = f"{page_id}_{sec_idx}_{chunk_idx}"
            chunk_images = find_images_in_text(chunk_text)
            image_paths = [img["path"] for img in chunk_images]
            image_descriptions = []
            
            # 이미지 설명 생성
            if chunk_images and vision_model:
                for img in chunk_images:
                    start, end = img["start"], img["end"]
                    text_before = chunk_text[max(0, start-200):start]
                    text_after = chunk_text[end:end+200]
                    
                    context = {
                        "page_title": page_title,
                        "section_title": section["title"],
                        "text_before": text_before,
                        "text_after": text_after
                    }
                    
                    desc = vision_model.describe_image(img["path"], context)
                    image_descriptions.append(desc)
            
            # combined_text 생성
            combined_text = chunk_text
            for img, desc in zip(chunk_images, image_descriptions):
                combined_text = combined_text.replace(img["marker"], f"[이미지: {desc}]")
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                page_id=page_id,
                text=chunk_text,
                combined_text=combined_text,
                has_image=len(chunk_images) > 0,
                image_paths=image_paths,
                image_descriptions=image_descriptions,
                page_title=page_title,
                section_title=section["title"],
                section_path=section["path"],
                heading_level=section["level"],
                chunk_index=chunk_idx,
                created_time=page_data.get("created_time", ""),
                last_edited_time=page_data.get("last_edited_time", ""),
                properties=page_data.get("properties", {})
            ))
    
    return chunks