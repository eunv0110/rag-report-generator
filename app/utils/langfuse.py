"""Langfuse 트레이싱 유틸리티

Langfuse는 LLM 애플리케이션의 트레이싱, 모니터링, 평가를 위한 도구입니다.
"""

from typing import Optional, Dict, Any
from contextlib import contextmanager
from dotenv import load_dotenv
from langfuse import Langfuse
from app.config.settings import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

load_dotenv()

_langfuse_client: Optional[Langfuse] = None


def get_langfuse_client() -> Optional[Langfuse]:
    """Langfuse 클라이언트 싱글톤 반환

    Returns:
        Langfuse 클라이언트 인스턴스 또는 None (설정되지 않은 경우)
    """
    global _langfuse_client

    if _langfuse_client is None:
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            _langfuse_client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            print("✅ Langfuse 연동 완료")
        else:
            print("⚠️  Langfuse 키가 설정되지 않음 (트레이싱 비활성화)")

    return _langfuse_client


@contextmanager
def trace_operation(name: str, metadata: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None):
    """작업을 Langfuse에 트레이싱 (현재 비활성화)

    Args:
        name: 작업 이름
        metadata: 메타데이터
        user_id: 사용자 ID

    Yields:
        None

    Note:
        현재 API 호환성 문제로 트레이싱이 비활성화되어 있습니다.
        추후 활성화 예정입니다.
    """
    # 트레이싱 비활성화 - 추후 활성화 예정
    yield None
