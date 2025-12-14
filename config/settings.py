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
        return yaml.safe_load(f)

MODEL_CONFIG = load_model_config()

# Notion 설정
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATA_SOURCE_ID = os.getenv("DATA_SOURCE_ID")
NOTION_VERSION = os.getenv("NOTION_VERSION", "2025-09-03")

# API 키
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

AZURE_AI_CREDENTIAL = os.getenv("AZURE_AI_CREDENTIAL")
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "https://models.inference.ai.azure.com")

# 청킹 설정
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
IMAGE_CONTEXT_CHARS = 300

# ✅ Qdrant 설정 - 임베딩 설정에서 db_name 사용
DB_NAME = MODEL_CONFIG['embeddings'].get('db_name', 'default')
QDRANT_PATH = str(DATA_DIR / "qdrant_data" / DB_NAME)
QDRANT_COLLECTION = "notion_docs"

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
Path(QDRANT_PATH).parent.mkdir(parents=True, exist_ok=True)