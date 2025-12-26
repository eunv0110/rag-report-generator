import os
import yaml
from pathlib import Path
from langchain_core.embeddings import Embeddings
from models.embeddings.openai_embedder import OpenAIEmbedder
from models.embeddings.upstage_embedder import UpstageEmbedder

# Singleton 패턴: 전역 embedder 인스턴스
_embedder_instance = None

def get_embedder(config_path: str = None) -> Embeddings:
    """설정 파일에서 임베더 생성 (Singleton 패턴)"""
    global _embedder_instance

    # 이미 생성된 인스턴스가 있으면 재사용
    if _embedder_instance is not None:
        return _embedder_instance

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    emb_config = config['embeddings']
    provider = emb_config['provider']

    if provider == "openai":
        _embedder_instance = OpenAIEmbedder(
            model=emb_config['model'],
            api_key=os.getenv(emb_config['api_key_env']),
            base_url=emb_config['base_url'],
            batch_size=emb_config.get('batch_size', 100)
        )
        return _embedder_instance
    elif provider == "upstage":
        _embedder_instance = UpstageEmbedder(
            model=emb_config['model'],
            query_model=emb_config.get('query_model'),  # 쿼리 전용 모델 (선택적)
            api_key=os.getenv(emb_config['api_key_env']),
            base_url=emb_config['base_url'],
            batch_size=emb_config.get('batch_size', 100)
        )
        return _embedder_instance
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")