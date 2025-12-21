"""리트리버 팩토리 - 리트리버 생성 및 관리"""

from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever


class RetrieverFactory:
    """리트리버를 생성하고 관리하는 팩토리 클래스"""

    # 지원하는 리트리버 타입
    SUPPORTED_RETRIEVERS = {
        "bm25": BM25Retriever,
        "bm25_korean": BM25Retriever,
        "bm25_basic": BM25Retriever,
        "dense": DenseRetriever,
        "vector": DenseRetriever,
    }

    @staticmethod
    def create(
        retriever_type: str,
        qdrant_client: QdrantClient,
        **kwargs
    ) -> BaseRetriever:
        """
        리트리버 생성

        Args:
            retriever_type: 리트리버 타입
                - "bm25" 또는 "bm25_korean": BM25 with Korean tokenizer
                - "bm25_basic": BM25 with basic tokenizer
                - "dense" 또는 "vector": Dense retriever
            qdrant_client: Qdrant 클라이언트
            **kwargs: 리트리버별 추가 인자

        Returns:
            생성된 리트리버 인스턴스

        Raises:
            ValueError: 지원하지 않는 리트리버 타입인 경우
        """
        retriever_type = retriever_type.lower()

        if retriever_type not in RetrieverFactory.SUPPORTED_RETRIEVERS:
            raise ValueError(
                f"지원하지 않는 리트리버 타입: {retriever_type}\n"
                f"지원 타입: {list(RetrieverFactory.SUPPORTED_RETRIEVERS.keys())}"
            )

        retriever_class = RetrieverFactory.SUPPORTED_RETRIEVERS[retriever_type]

        # BM25 리트리버 생성
        if retriever_class == BM25Retriever:
            use_korean = retriever_type in ["bm25", "bm25_korean"]
            return BM25Retriever(
                qdrant_client=qdrant_client,
                use_korean_tokenizer=use_korean,
                **kwargs
            )

        # Dense 리트리버 생성
        elif retriever_class == DenseRetriever:
            return DenseRetriever(
                qdrant_client=qdrant_client,
                **kwargs
            )

        else:
            raise ValueError(f"알 수 없는 리트리버 클래스: {retriever_class}")

    @staticmethod
    def create_multiple(
        retriever_configs: List[Dict[str, Any]],
        qdrant_client: QdrantClient
    ) -> List[BaseRetriever]:
        """
        여러 리트리버를 한 번에 생성

        Args:
            retriever_configs: 리트리버 설정 리스트
                예: [
                    {"type": "bm25_korean"},
                    {"type": "dense"},
                    {"type": "bm25_basic"}
                ]
            qdrant_client: Qdrant 클라이언트

        Returns:
            생성된 리트리버 리스트
        """
        retrievers = []

        for config in retriever_configs:
            retriever_type = config.pop("type")
            retriever = RetrieverFactory.create(
                retriever_type=retriever_type,
                qdrant_client=qdrant_client,
                **config
            )
            retrievers.append(retriever)

        return retrievers

    @staticmethod
    def get_all_default_retrievers(qdrant_client: QdrantClient) -> List[BaseRetriever]:
        """
        모든 기본 리트리버를 생성

        Args:
            qdrant_client: Qdrant 클라이언트

        Returns:
            [BM25 Korean, BM25 Basic, Dense Vector] 리트리버 리스트
        """
        configs = [
            {"type": "bm25_korean"},
            {"type": "bm25_basic"},
            {"type": "dense"},
        ]

        return RetrieverFactory.create_multiple(configs, qdrant_client)

    @staticmethod
    def list_available_types() -> List[str]:
        """사용 가능한 리트리버 타입 목록 반환"""
        return list(RetrieverFactory.SUPPORTED_RETRIEVERS.keys())


def create_retriever(
    retriever_type: str,
    qdrant_client: QdrantClient,
    **kwargs
) -> BaseRetriever:
    """
    편의 함수: 리트리버 생성

    Args:
        retriever_type: 리트리버 타입
        qdrant_client: Qdrant 클라이언트
        **kwargs: 리트리버별 추가 인자

    Returns:
        생성된 리트리버
    """
    return RetrieverFactory.create(retriever_type, qdrant_client, **kwargs)
