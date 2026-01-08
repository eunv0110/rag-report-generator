"""FastAPI 서버 - RAG 보고서 생성 API"""
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
load_dotenv()

from app.api.routers import health, reports

# FastAPI 앱 생성
app = FastAPI(
    title="RAG Report Generator API",
    description="RAG 기반 보고서 자동 생성 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router)
app.include_router(reports.router)



if __name__ == "__main__":
    import uvicorn

    # 환경 변수에서 설정 읽기
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=reload
    )
