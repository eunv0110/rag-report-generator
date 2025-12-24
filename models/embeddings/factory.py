import os
import yaml
from pathlib import Path
from models.embeddings.openai_embedder import OpenAIEmbedder
from models.embeddings.azure_embedder import AzureEmbedder

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
    elif provider == "azure":
        # endpoint는 환경변수 또는 직접 값 사용
        endpoint = os.getenv(emb_config.get('endpoint_env', '')) or emb_config.get('endpoint', '')
        # 끝의 따옴표와 슬래시 제거
        endpoint = endpoint.strip("'\"").rstrip('/')

        return AzureEmbedder(
            model=emb_config['model'],
            api_key=os.getenv(emb_config['api_key_env']),
            endpoint=endpoint,
            api_version=emb_config.get('api_version', '2024-02-01'),
            batch_size=emb_config.get('batch_size', 100)
        )
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")