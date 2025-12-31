from pathlib import Path
from typing import Dict, Any

class PromptManager:
    def __init__(self, prompts_dir: Path = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent / "templates"
        self.prompts_dir = prompts_dir
    
    def load_prompt(self, template_name: str) -> str:
        """프롬프트 템플릿 로드"""
        template_path = self.prompts_dir / f"{template_name}"
        
        if not template_path.exists():
            raise FileNotFoundError(f"프롬프트 템플릿 없음: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """프롬프트 템플릿을 변수로 채워서 반환"""
        template = self.load_prompt(template_name)
        return template.format(**kwargs)