import os
import yaml
from pathlib import Path
from langchain_core.embeddings import Embeddings
from models.embeddings.openrouter_embedder import OpenRouterEmbedder
from models.embeddings.upstage_embedder import UpstageEmbedder

# Singleton 패턴: 전역 embedder 인스턴스 (MODEL_PRESET별로 캐싱)
_embedder_cache = {}
_last_preset = None

def get_embedder(config_path: str = None) -> Embeddings:
    """설정 파일에서 임베더 생성 (MODEL_PRESET별 캐싱)"""
    global _embedder_cache, _last_preset

    # 현재 MODEL_PRESET 확인
    current_preset = os.getenv('MODEL_PRESET', 'upstage')

    # 같은 프리셋의 인스턴스가 이미 있으면 재사용
    if current_preset in _embedder_cache:
        return _embedder_cache[current_preset]

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # MODEL_PRESET 환경변수 또는 기본값 사용
    preset_name = os.getenv('MODEL_PRESET', config.get('default_preset', 'upstage'))

    # embedding_presets에서 선택한 프리셋 로드
    if 'embedding_presets' in config:
        if preset_name not in config['embedding_presets']:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(config['embedding_presets'].keys())}")
        emb_config = config['embedding_presets'][preset_name]
        print(f"✅ Using embedding preset: {preset_name}")
    else:
        # 레거시 형식 지원 (embeddings 키가 직접 있는 경우)
        emb_config = config['embeddings']

    provider = emb_config['provider']

    if provider == "openai":
        embedder = OpenRouterEmbedder(
            model=emb_config['model'],
            api_key=os.getenv(emb_config['api_key_env']),
            base_url=emb_config['base_url'],
            batch_size=emb_config.get('batch_size', 100)
        )
        _embedder_cache[current_preset] = embedder
        return embedder
    elif provider == "upstage":
        embedder = UpstageEmbedder(
            model=emb_config['model'],
            query_model=emb_config.get('query_model'),  # 쿼리 전용 모델 (선택적)
            api_key=os.getenv(emb_config['api_key_env']),
            base_url=emb_config['base_url'],
            batch_size=emb_config.get('batch_size', 100)
        )
        _embedder_cache[current_preset] = embedder
        return embedder
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")