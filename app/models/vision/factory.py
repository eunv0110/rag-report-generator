import os
import yaml
from pathlib import Path
from app.models.vision.gpt4_vision import GPT4Vision

def get_vision_model(config_path: str = None):
    """설정 파일에서 비전 모델 생성"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vision_config = config['vision']
    provider = vision_config['provider']
    
    if provider == "gpt4":
        api_key = os.getenv(vision_config['api_key_env'])
        if not api_key:
            print("⚠️  Vision 모델 API 키 없음 - 이미지 설명 생성 비활성화")
            return None
        
        return GPT4Vision(
            model=vision_config['model'],
            api_key=api_key,
            base_url=vision_config['base_url'],
            max_tokens=vision_config.get('max_tokens', 300),
            prompt_template=vision_config.get('prompt_template', 'image_description/default')
        )
    else:
        raise ValueError(f"Unknown vision provider: {provider}")