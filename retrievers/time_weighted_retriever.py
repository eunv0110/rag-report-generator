"""Time-Weighted Retriever - 시간 기반 가중치를 적용한 벡터 검색"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import math
from qdrant_client import QdrantClient
from config.settings import QDRANT_COLLECTION
from models.embeddings.factory import get_embedder
from .base_retriever import BaseRetriever, SearchResult


class TimeWeightedRetriever(BaseRetriever):
    """
    시간 가중치를 적용한 Dense Retriever

    최신 문서에 더 높은 가중치를 부여하여 검색 결과를 재정렬합니다.
    decay_rate에 따라 시간이 지날수록 점수가 감소합니다.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder=None,
        decay_rate: float = 0.01,
        time_field: str = "날짜",
        name: str = "TimeWeighted_Vector"
    ):
        """
        Args:
            qdrant_client: Qdrant 클라이언트
            embedder: 임베딩 모델 (None이면 자동 로드)
            decay_rate: 시간 감쇠율 (높을수록 최신 문서 선호도 증가)
                - 0.0: 시간 가중치 없음 (일반 벡터 검색과 동일)
                - 0.01: 약한 시간 가중치 (기본값)
                - 0.05: 중간 시간 가중치
                - 0.1 이상: 강한 시간 가중치 (최신 문서 크게 선호)
            time_field: 메타데이터에서 시간 정보를 가져올 필드명
            name: 리트리버 이름
        """
        self.client = qdrant_client
        self.embedder = embedder or get_embedder()
        self.decay_rate = decay_rate
        self.time_field = time_field
        self._name = name

        print(f"✅ Time-Weighted Retriever 초기화 완료 (decay_rate={decay_rate})")

    def _calculate_time_weight(self, timestamp: Optional[float], current_time: float) -> float:
        """
        시간 가중치 계산

        Args:
            timestamp: 문서의 타임스탬프 (Unix timestamp)
            current_time: 현재 시간 (Unix timestamp)

        Returns:
            시간 가중치 (0.0 ~ 1.0)
        """
        if timestamp is None:
            # 타임스탬프가 없으면 중간 값 반환
            return 0.5

        # 시간 차이 (시간 단위)
        hours_passed = (current_time - timestamp) / 3600.0

        # 지수 감쇠 함수: exp(-decay_rate * hours_passed)
        time_weight = math.exp(-self.decay_rate * hours_passed)

        return time_weight

    def _combine_scores(
        self,
        similarity_score: float,
        time_weight: float,
        alpha: float = 0.5
    ) -> float:
        """
        유사도 점수와 시간 가중치를 결합

        Args:
            similarity_score: 벡터 유사도 점수
            time_weight: 시간 가중치
            alpha: 유사도 점수의 비중 (0.0 ~ 1.0)
                - 1.0: 유사도만 사용 (시간 가중치 무시)
                - 0.5: 유사도와 시간 가중치를 동등하게 고려 (기본값)
                - 0.0: 시간 가중치만 사용 (유사도 무시)

        Returns:
            결합된 점수
        """
        combined = alpha * similarity_score + (1 - alpha) * time_weight
        return combined

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        retrieve_k: int = None
    ) -> List[SearchResult]:
        """
        시간 가중치를 적용한 벡터 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
            alpha: 유사도 점수 비중 (0.0~1.0, 기본값 0.5)
            retrieve_k: 재정렬 전 가져올 결과 개수 (None이면 top_k * 3)

        Returns:
            시간 가중치가 적용된 검색 결과 리스트
        """
        # 재정렬을 위해 더 많은 결과를 가져옴
        if retrieve_k is None:
            retrieve_k = top_k * 3

        # 쿼리 임베딩 생성
        query_embedding = self.embedder.embed_texts([query])[0]

        # Qdrant에서 유사도 검색
        search_results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=retrieve_k,
            with_payload=True
        ).points

        # 현재 시간
        current_time = datetime.now().timestamp()

        # 결과 변환 및 시간 가중치 적용
        results = []
        for hit in search_results:
            payload = hit.payload
            properties = payload.get("properties", {})

            # 타임스탬프 추출
            timestamp = None
            time_value = properties.get(self.time_field)

            if time_value:
                try:
                    # "날짜" 필드인 경우: {'start': '2025-09-26', 'end': None} 형태
                    if isinstance(time_value, str) and time_value.startswith('{'):
                        import ast
                        date_dict = ast.literal_eval(time_value)
                        if 'start' in date_dict and date_dict['start']:
                            # YYYY-MM-DD 형식을 datetime으로 변환
                            timestamp = datetime.fromisoformat(date_dict['start']).timestamp()
                    # ISO 8601 형식 문자열인 경우
                    elif isinstance(time_value, str):
                        timestamp = datetime.fromisoformat(time_value.replace('Z', '+00:00')).timestamp()
                except:
                    timestamp = None

            # 시간 가중치 계산
            time_weight = self._calculate_time_weight(timestamp, current_time)

            # 점수 결합
            similarity_score = float(hit.score)
            combined_score = self._combine_scores(similarity_score, time_weight, alpha)

            result = SearchResult(
                chunk_id=payload.get("chunk_id"),
                page_id=payload.get("page_id"),
                text=payload.get("text"),
                combined_text=payload.get("combined_text"),
                page_title=payload.get("page_title"),
                section_title=payload.get("section_title"),
                section_path=payload.get("section_path"),
                score=combined_score,
                has_image=payload.get("has_image", False),
                image_descriptions=payload.get("image_descriptions", []),
                properties=properties
            )

            # 디버깅 정보 추가
            result.properties["_time_weight"] = time_weight
            result.properties["_similarity_score"] = similarity_score
            result.properties["_timestamp"] = timestamp

            results.append(result)

        # 결합된 점수로 정렬
        results.sort(key=lambda x: x.score, reverse=True)

        # 상위 k개 반환
        return results[:top_k]

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        alpha: float = 0.5,
        retrieve_k: int = None
    ) -> List[List[SearchResult]]:
        """
        여러 쿼리를 배치로 검색

        Args:
            queries: 검색 쿼리 리스트
            top_k: 각 쿼리당 반환할 상위 결과 개수
            alpha: 유사도 점수 비중
            retrieve_k: 재정렬 전 가져올 결과 개수

        Returns:
            각 쿼리별 검색 결과 리스트
        """
        return [
            self.search(query, top_k, alpha, retrieve_k)
            for query in queries
        ]

    @property
    def name(self) -> str:
        """리트리버 이름 반환"""
        return self._name

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        info = super().get_info()
        info.update({
            "embedder": str(type(self.embedder).__name__),
            "decay_rate": self.decay_rate,
            "time_field": self.time_field
        })
        return info
