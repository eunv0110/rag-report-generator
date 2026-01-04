import base64
import httpx
from typing import Dict, Any, Optional
from pathlib import Path
from openai import OpenAI
from app.models.base import BaseVisionModel
from app.utils.common import load_prompt

class GPT4Vision(BaseVisionModel):
    def __init__(
        self, 
        model: str, 
        api_key: str, 
        base_url: str, 
        max_tokens: int = 300,
        prompt_template: str = "vision/image_description.txt"
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompt_template_path = f"prompts/templates/{prompt_template}"
    
    def describe_image(self, image_path: str, context: Dict[str, Any]) -> str:
        """이미지 설명 생성"""
        # ✅ 상대 경로를 절대 경로로 변환
        if not image_path.startswith("http") and not image_path.startswith("/"):
            from pathlib import Path
            from app.config.settings import BASE_DIR
            image_path = str(BASE_DIR / "data" / image_path)
        
        # 이미지 로드
        if image_path.startswith("http"):
            try:
                response = httpx.get(image_path, timeout=30)
                base64_image = base64.b64encode(response.content).decode("utf-8")
            except Exception as e:
                return f"[이미지 로드 실패: {e}]"
        else:
            try:
                with open(image_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                return f"[이미지 로드 실패: {e}]"
        
        media_type = "image/png" if ".png" in image_path.lower() else "image/jpeg"

        prompt_template = load_prompt(self.prompt_template_path)
        prompt = prompt_template.format(
            page_title=context.get('page_title', ''),
            section_title=context.get('section_title', ''),
            text_before=context.get('text_before', '')[:100],
            text_after=context.get('text_after', '')[:100]
        )

        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[이미지 설명 생성 실패: {e}]"