import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# 경로 설정
# ============================================================================
# 환경 변수에서 경로 로드 (없으면 기본값 사용)
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(BASE_DIR)))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
QDRANT_DATA_DIR = Path(os.getenv("QDRANT_DATA_DIR", str(DATA_DIR / "qdrant_data")))
PROMPTS_BASE_DIR = Path(os.getenv("PROMPTS_BASE_DIR", str(BASE_DIR / "prompts")))

# 레거시 호환성
IMAGE_DIR = DATA_DIR / "notion_images"
PROMPTS_DIR = PROMPTS_BASE_DIR / "templates"
CONFIG_DIR = BASE_DIR / "config"

# ============================================================================
# 설정 파일 로더
# ============================================================================
def load_yaml_config(config_name: str) -> dict:
    """YAML 설정 파일 로드"""
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_config() -> dict:
    """모델 설정 로드 및 프리셋 적용"""
    config = load_yaml_config("model_config")

    # 환경변수로 모델 프리셋 선택
    preset_name = os.getenv("MODEL_PRESET", config.get("default_preset", "upstage"))

    if "embedding_presets" in config:
        if preset_name not in config["embedding_presets"]:
            available = ", ".join(config["embedding_presets"].keys())
            raise ValueError(f"Unknown MODEL_PRESET: {preset_name}. Available: {available}")

        # 선택된 프리셋을 embeddings로 설정
        config["embeddings"] = config["embedding_presets"][preset_name]
        # print(f"✅ Using embedding preset: {preset_name}")

    return config

# ============================================================================
# 전역 설정 로드
# ============================================================================
MODEL_CONFIG = load_model_config()
# EVALUATION_CONFIG는 experiments/ 디렉토리에서 직접 로드하도록 변경됨
# EVALUATION_CONFIG = load_yaml_config("evaluation_config")
REPORT_CONFIG = load_yaml_config("report_config")

# ============================================================================
# 환경 변수 - API 키 및 인증
# ============================================================================
# Notion
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATA_SOURCE_ID = os.getenv("DATA_SOURCE_ID")
NOTION_VERSION = os.getenv("NOTION_VERSION", "2025-09-03")

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Azure AI
AZURE_AI_CREDENTIAL = os.getenv("AZURE_AI_CREDENTIAL")
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "https://models.inference.ai.azure.com")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ============================================================================
# 청킹 설정
# ============================================================================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
IMAGE_CONTEXT_CHARS = 300

# ============================================================================
# Qdrant 설정
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_USE_SERVER = os.getenv("QDRANT_USE_SERVER", "true").lower() == "true"

# 프리셋별 컬렉션 매핑
PRESET_TO_COLLECTION = {
    "bge-m3": "notion_docs_bge_m3",
    "upstage": "notion_docs_upstage",
    "openai-large": "notion_docs_openai",
    "gemini-embedding-001": "notion_docs_gemini",
    "qwen3-embedding-4b": "notion_docs_qwen3"
}

# 레거시 지원: 로컬 파일 모드
DB_NAME = MODEL_CONFIG['embeddings'].get('db_name', 'default')
QDRANT_PATH = str(DATA_DIR / "qdrant_data" / DB_NAME)
QDRANT_UNIFIED_PATH = str(DATA_DIR / "qdrant_unified")
QDRANT_COLLECTION = "notion_docs"

def get_collection_name(preset: str = None) -> str:
    """프리셋에 따른 컬렉션 이름 반환"""
    if preset is None:
        preset = os.getenv("MODEL_PRESET", "upstage")
    return PRESET_TO_COLLECTION.get(preset, "notion_docs")

def get_qdrant_path() -> str:
    """현재 MODEL_PRESET 환경변수 기반으로 Qdrant 경로 동적 계산 (레거시)"""
    config = load_yaml_config("model_config")
    preset_name = os.getenv("MODEL_PRESET", config.get("default_preset", "upstage"))

    if "embedding_presets" in config and preset_name in config["embedding_presets"]:
        db_name = config["embedding_presets"][preset_name].get('db_name', 'default')
    else:
        db_name = 'default'

    return str(DATA_DIR / "qdrant_data" / db_name)

# ============================================================================
# 평가 데이터셋 설정
# ============================================================================
DEFAULT_NUM_SAMPLES = 20
MAX_TEXT_LENGTH = 1000
MIN_CONTENT_LENGTH = 100
DEFAULT_OUTPUT_DIR = "data/evaluation"
PROMPT_FILE = "prompts/templates/data/qa_generation_prompt.txt"

# ============================================================================
# 디렉토리 생성
# ============================================================================
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
Path(QDRANT_PATH).parent.mkdir(parents=True, exist_ok=True)
