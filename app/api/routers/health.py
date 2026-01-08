"""Health check router"""
from datetime import datetime

from fastapi import APIRouter

from app.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트 - API 상태 확인"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )
