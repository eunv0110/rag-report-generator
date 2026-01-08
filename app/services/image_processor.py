import re
from typing import List, Dict, Optional
from app.models.base import BaseVisionModel

class ImageProcessor:
    def __init__(self, vision_model: Optional[BaseVisionModel] = None):
        self.vision_model = vision_model
    
    def find_images_in_text(self, text: str) -> List[Dict]:
        """텍스트에서 이미지 마커 찾기"""
        pattern = r'\[Image:\s*([^\]]+)\]'
        images = []
        for m in re.finditer(pattern, text):
            path = m.group(1).strip()
            if not path.startswith("notion_images/"):
                path = f"notion_images/{path}"
            images.append({
                "path": path,
                "start": m.start(),
                "end": m.end(),
                "marker": m.group(0)
            })
        return images
    
    def process_chunk_images(self, chunk_text: str, page_title: str, 
                            section_title: str) -> tuple[str, List[str], List[str]]:
        """청크 내 이미지 처리"""
        chunk_images = self.find_images_in_text(chunk_text)
        image_paths = [img["path"] for img in chunk_images]
        image_descriptions = []
        
        if chunk_images and self.vision_model:
            for img in chunk_images:
                start, end = img["start"], img["end"]
                text_before = chunk_text[max(0, start-200):start]
                text_after = chunk_text[end:end+200]
                
                context = {
                    "page_title": page_title,
                    "section_title": section_title,
                    "text_before": text_before,
                    "text_after": text_after
                }
                
                desc = self.vision_model.describe_image(img["path"], context)
                image_descriptions.append(desc)
        
        # combined_text 생성
        combined_text = chunk_text
        for img, desc in zip(chunk_images, image_descriptions):
            combined_text = combined_text.replace(img["marker"], f"[이미지: {desc}]")
        
        return combined_text, image_paths, image_descriptions