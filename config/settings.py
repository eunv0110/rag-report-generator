import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "data" / "notion_images"
PROMPTS_DIR = BASE_DIR / "prompts" / "templates"

# 설정 파일 로드
def load_model_config():
    config_path = BASE_DIR / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 환경변수로 모델 프리셋 선택
    preset_name = os.getenv("MODEL_PRESET", config.get("default_preset", "upstage"))

    if "embedding_presets" in config:
        if preset_name not in config["embedding_presets"]:
            available = ", ".join(config["embedding_presets"].keys())
            raise ValueError(f"Unknown MODEL_PRESET: {preset_name}. Available: {available}")

        # 선택된 프리셋을 embeddings로 설정
        config["embeddings"] = config["embedding_presets"][preset_name]
        print(f"✅ Using embedding preset: {preset_name}")

    return config

MODEL_CONFIG = load_model_config()

def load_evaluation_config():
    """평가 설정 파일 로드"""
    config_path = BASE_DIR / "config" / "evaluation_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

EVALUATION_CONFIG = load_evaluation_config()

# Notion 설정
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATA_SOURCE_ID = os.getenv("DATA_SOURCE_ID")
NOTION_VERSION = os.getenv("NOTION_VERSION", "2025-09-03")

# API 키
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

AZURE_AI_CREDENTIAL = os.getenv("AZURE_AI_CREDENTIAL")
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "https://models.inference.ai.azure.com")

# Langfuse 설정
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# 청킹 설정
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
IMAGE_CONTEXT_CHARS = 300

# ✅ Qdrant 설정 - 임베딩 설정에서 db_name 사용
DB_NAME = MODEL_CONFIG['embeddings'].get('db_name', 'default')
QDRANT_PATH = str(DATA_DIR / "qdrant_data" / DB_NAME)
QDRANT_COLLECTION = "notion_docs"

def get_qdrant_path():
    """현재 MODEL_PRESET 환경변수 기반으로 Qdrant 경로 동적 계산"""
    config_path = BASE_DIR / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    preset_name = os.getenv("MODEL_PRESET", config.get("default_preset", "upstage"))

    if "embedding_presets" in config and preset_name in config["embedding_presets"]:
        db_name = config["embedding_presets"][preset_name].get('db_name', 'default')
    else:
        db_name = 'default'

    return str(DATA_DIR / "qdrant_data" / db_name)

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
Path(QDRANT_PATH).parent.mkdir(parents=True, exist_ok=True)

#평가 데이터 셋
DEFAULT_NUM_SAMPLES = 20
MAX_TEXT_LENGTH = 1000
MIN_CONTENT_LENGTH = 100
DEFAULT_OUTPUT_DIR = "data/evaluation"
PROMPT_FILE = "prompts/templates/data/qa_generation_prompt.txt"