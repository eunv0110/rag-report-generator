import os
import yaml
from pathlib import Path
from models.embeddings.openai_embedder import OpenAIEmbedder

def get_embedder(config_path: str = None):
    """설정 파일에서 임베더 생성"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    emb_config = config['embeddings']
    provider = emb_config['provider']
    
    if provider == "openai":
        return OpenAIEmbedder(
            model=emb_config['model'],
            api_key=os.getenv(emb_config['api_key_env']),
            base_url=emb_config['base_url'],
            batch_size=emb_config.get('batch_size', 100)
        )
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")