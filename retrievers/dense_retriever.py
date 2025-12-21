"""Dense Retrieval (벡터 검색) 리트리버"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config.settings import QDRANT_COLLECTION
from models.embeddings.factory import get_embedder
from .base_retriever import BaseRetriever, SearchResult


class DenseRetriever(BaseRetriever):
    """Dense Retrieval (벡터 검색) 엔진"""

    def __init__(self, qdrant_client: QdrantClient, embedder=None):
        """
        Args:
            qdrant_client: Qdrant 클라이언트
            embedder: 임베딩 모델 (None이면 자동 로드)
        """
        self.client = qdrant_client
        self.embedder = embedder or get_embedder()

        print("✅ Dense Retriever 초기화 완료")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        벡터 검색으로 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수

        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩 생성
        query_embedding = self.embedder.embed_texts([query])[0]

        # Qdrant에서 유사도 검색
        search_results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        ).points

        # 결과 변환
        results = []
        for hit in search_results:
            payload = hit.payload
            results.append(SearchResult(
                chunk_id=payload.get("chunk_id"),
                page_id=payload.get("page_id"),
                text=payload.get("text"),
                combined_text=payload.get("combined_text"),
                page_title=payload.get("page_title"),
                section_title=payload.get("section_title"),
                section_path=payload.get("section_path"),
                score=float(hit.score),
                has_image=payload.get("has_image", False),
                image_descriptions=payload.get("image_descriptions", []),
                properties=payload.get("properties", {})
            ))

        return results

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        여러 쿼리를 배치로 검색

        Args:
            queries: 검색 쿼리 리스트
            top_k: 각 쿼리당 반환할 상위 결과 개수

        Returns:
            각 쿼리별 검색 결과 리스트
        """
        # 배치 임베딩 생성
        query_embeddings = self.embedder.embed_texts(queries)

        results = []
        for query_embedding in query_embeddings:
            # Qdrant에서 유사도 검색
            search_results = self.client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            ).points

            # 결과 변환
            query_results = []
            for hit in search_results:
                payload = hit.payload
                query_results.append(SearchResult(
                    chunk_id=payload.get("chunk_id"),
                    page_id=payload.get("page_id"),
                    text=payload.get("text"),
                    combined_text=payload.get("combined_text"),
                    page_title=payload.get("page_title"),
                    section_title=payload.get("section_title"),
                    section_path=payload.get("section_path"),
                    score=float(hit.score),
                    has_image=payload.get("has_image", False),
                    image_descriptions=payload.get("image_descriptions", []),
                    properties=payload.get("properties", {})
                ))

            results.append(query_results)

        return results

    @property
    def name(self) -> str:
        """리트리버 이름 반환"""
        return "Dense_Vector"

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        info = super().get_info()
        info.update({
            "embedder": str(type(self.embedder).__name__)
        })
        return info
