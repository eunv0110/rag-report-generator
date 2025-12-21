"""Langfuse 트레이싱 유틸리티"""

from langfuse import Langfuse
from config.settings import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from typing import Optional
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv() 

_langfuse_client: Optional[Langfuse] = None

def get_langfuse_client() -> Optional[Langfuse]:
    """Langfuse 클라이언트 싱글톤 반환"""
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
def trace_operation(name: str, metadata: dict = None, user_id: str = None):
    """
    작업을 Langfuse에 트레이싱

    Usage:
        with trace_operation("vectordb_build", {"pages": 10}):
            # 작업 수행
            pass
    """
    client = get_langfuse_client()

    if client is None:
        yield None
        return

    trace = client.trace(
        name=name,
        metadata=metadata or {},
        user_id=user_id
    )

    try:
        yield trace
    finally:
        client.flush()
