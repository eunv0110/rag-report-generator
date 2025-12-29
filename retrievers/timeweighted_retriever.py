#!/usr/bin/env python3
"""Time-Weighted Retriever - LangChain 기반 시간 가중치를 적용한 벡터 검색"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing import List, Optional, Dict, Any
from datetime import datetime
import math
import os
from config.settings import QDRANT_COLLECTION, get_qdrant_path
from models.embeddings.factory import get_embedder
from retrievers.dense_retriever import get_langchain_embeddings, _qdrant_client_cache


class TimeWeightedRetriever(BaseRetriever):
    """
    시간 가중치를 적용한 Dense Retriever (LangChain 호환)

    최신 문서에 더 높은 가중치를 부여하여 검색 결과를 재정렬합니다.
    decay_rate에 따라 시간이 지날수록 점수가 감소합니다.
    """

    vectorstore: QdrantVectorStore
    """Qdrant VectorStore"""

    decay_rate: float = 0.01
    """시간 감쇠율 (높을수록 최신 문서 선호도 증가)"""

    time_field: str = "날짜"
    """메타데이터에서 시간 정보를 가져올 필드명"""

    alpha: float = 0.5
    """유사도 점수의 비중 (0.0 ~ 1.0)"""

    k: int = 5
    """반환할 문서 수"""

    retrieve_k: int = 15
    """재정렬 전 가져올 결과 개수"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        decay_rate: float = 0.01,
        time_field: str = "날짜",
        alpha: float = 0.5,
        k: int = 5,
        retrieve_k: int = None,
        **kwargs
    ):
        """
        Args:
            vectorstore: Qdrant VectorStore 인스턴스
            decay_rate: 시간 감쇠율 (높을수록 최신 문서 선호도 증가)
                - 0.0: 시간 가중치 없음 (일반 벡터 검색과 동일)
                - 0.01: 약한 시간 가중치 (기본값)
                - 0.05: 중간 시간 가중치
                - 0.1 이상: 강한 시간 가중치 (최신 문서 크게 선호)
            time_field: 메타데이터에서 시간 정보를 가져올 필드명
            alpha: 유사도 점수 비중 (0.0~1.0, 기본값 0.5)
                - 1.0: 유사도만 사용 (시간 가중치 무시)
                - 0.5: 유사도와 시간 가중치를 동등하게 고려
                - 0.0: 시간 가중치만 사용 (유사도 무시)
            k: 반환할 문서 수
            retrieve_k: 재정렬 전 가져올 결과 개수 (None이면 k * 3)
        """
        if retrieve_k is None:
            retrieve_k = k * 3

        super().__init__(
            vectorstore=vectorstore,
            decay_rate=decay_rate,
            time_field=time_field,
            alpha=alpha,
            k=k,
            retrieve_k=retrieve_k,
            **kwargs
        )

        print(f"✅ Time-Weighted Retriever 초기화 완료 (decay_rate={decay_rate}, alpha={alpha})")

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
        time_weight: float
    ) -> float:
        """
        유사도 점수와 시간 가중치를 결합

        Args:
            similarity_score: 벡터 유사도 점수
            time_weight: 시간 가중치

        Returns:
            결합된 점수
        """
        combined = self.alpha * similarity_score + (1 - self.alpha) * time_weight
        return combined

    def _extract_timestamp(self, time_value: Any) -> Optional[float]:
        """
        메타데이터에서 타임스탬프 추출

        Args:
            time_value: 시간 필드 값

        Returns:
            Unix timestamp 또는 None
        """
        if not time_value:
            return None

        try:
            # "날짜" 필드인 경우: {'start': '2025-09-26', 'end': None} 형태
            if isinstance(time_value, str) and time_value.startswith('{'):
                import ast
                date_dict = ast.literal_eval(time_value)
                if 'start' in date_dict and date_dict['start']:
                    # YYYY-MM-DD 형식을 datetime으로 변환
                    return datetime.fromisoformat(date_dict['start']).timestamp()
            # ISO 8601 형식 문자열인 경우
            elif isinstance(time_value, str):
                return datetime.fromisoformat(time_value.replace('Z', '+00:00')).timestamp()
            # dict 형태인 경우
            elif isinstance(time_value, dict):
                if 'start' in time_value and time_value['start']:
                    return datetime.fromisoformat(str(time_value['start'])).timestamp()
        except Exception as e:
            pass

        return None

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        LangChain 호환 메서드: 시간 가중치를 적용한 관련 문서 검색

        Args:
            query: 검색 쿼리
            run_manager: 콜백 매니저

        Returns:
            시간 가중치가 적용된 LangChain Document 리스트
        """
        # vectorstore에서 더 많은 후보 가져오기
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=self.retrieve_k
        )

        # 현재 시간
        current_time = datetime.now().timestamp()

        # 시간 가중치 적용 및 재정렬
        scored_docs = []
        for doc, similarity_score in docs_with_scores:
            # 메타데이터에서 시간 정보 추출
            # properties 안에 시간 필드가 있을 수 있음
            time_value = doc.metadata.get(self.time_field)
            if not time_value and "properties" in doc.metadata:
                time_value = doc.metadata["properties"].get(self.time_field)

            # 타임스탬프 추출
            timestamp = self._extract_timestamp(time_value)

            # 시간 가중치 계산
            time_weight = self._calculate_time_weight(timestamp, current_time)

            # Qdrant는 거리를 반환하므로 유사도로 변환
            # (cosine distance를 사용하는 경우: similarity = 1 - distance)
            # 이미 score가 유사도인 경우도 있으므로 범위 확인
            if similarity_score <= 1.0:
                # 이미 유사도 점수인 경우
                normalized_similarity = similarity_score
            else:
                # 거리인 경우 (일반적으로 0에 가까울수록 유사)
                normalized_similarity = 1.0 / (1.0 + similarity_score)

            # 점수 결합
            combined_score = self._combine_scores(normalized_similarity, time_weight)

            # 디버깅 정보 추가
            doc.metadata["_time_weight"] = time_weight
            doc.metadata["_similarity_score"] = normalized_similarity
            doc.metadata["_combined_score"] = combined_score
            doc.metadata["_timestamp"] = timestamp

            scored_docs.append((doc, combined_score))

        # 결합된 점수로 정렬
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 반환
        top_docs = [doc for doc, score in scored_docs[:self.k]]

        return top_docs

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        return {
            "name": "TimeWeightedRetriever",
            "type": self.__class__.__name__,
            "decay_rate": self.decay_rate,
            "time_field": self.time_field,
            "alpha": self.alpha,
            "k": self.k,
            "retrieve_k": self.retrieve_k
        }


def get_time_weighted_retriever(
    decay_rate: float = 0.01,
    time_field: str = "날짜",
    alpha: float = 0.5,
    k: int = 5,
    retrieve_k: int = None,
    use_singleton: bool = True
) -> TimeWeightedRetriever:
    """
    Time-Weighted Retriever 생성

    Args:
        decay_rate: 시간 감쇠율 (기본값: 0.01)
        time_field: 메타데이터에서 시간 정보를 가져올 필드명 (기본값: "날짜")
        alpha: 유사도 점수 비중 (기본값: 0.5)
        k: 반환할 문서 수 (기본값: 5)
        retrieve_k: 재정렬 전 가져올 결과 개수 (기본값: k * 3)
        use_singleton: True면 기존 client를 재사용 (Qdrant lock 방지) - 기본값 True

    Returns:
        TimeWeightedRetriever 인스턴스
    """
    # 임베더 로드
    base_embedder = get_embedder()
    langchain_embeddings = get_langchain_embeddings(base_embedder)

    # Qdrant 경로 동적 계산 (현재 MODEL_PRESET 기반)
    qdrant_path = get_qdrant_path()
    model_preset = os.getenv("MODEL_PRESET", "default")

    # Qdrant client 생성 (싱글톤 사용 시 캐시에서 재사용)
    # 캐시 키에 embedding preset 포함 (각 preset마다 별도 클라이언트)
    cache_key = f"{model_preset}_{qdrant_path}_{QDRANT_COLLECTION}"

    if use_singleton and cache_key in _qdrant_client_cache:
        client = _qdrant_client_cache[cache_key]
    else:
        client = QdrantClient(path=qdrant_path)
        if use_singleton:
            _qdrant_client_cache[cache_key] = client

    # Qdrant vectorstore 로드
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=langchain_embeddings,
    )

    # TimeWeightedRetriever 생성
    retriever = TimeWeightedRetriever(
        vectorstore=vectorstore,
        decay_rate=decay_rate,
        time_field=time_field,
        alpha=alpha,
        k=k,
        retrieve_k=retrieve_k
    )

    return retriever


if __name__ == "__main__":
    # 테스트
    print("🔍 Time-Weighted Retriever 테스트")
    print("=" * 60)

    test_query = "RAG 시스템은 어떻게 동작하나요?"

    # 설정: 균형 (유사도:시간 = 50:50)
    print(f"\n{'='*60}")
    print(f"📊 설정: 균형 (유사도:시간 = 50:50)")
    print(f"   - decay_rate: 0.01")
    print(f"   - alpha: 0.5")
    print(f"{'='*60}")

    # Retriever 생성
    retriever = get_time_weighted_retriever(
        decay_rate=0.01,
        alpha=0.5,
        k=5
    )

    print(f"\n📝 Query: {test_query}")
    print("-" * 60)

    # invoke() 메서드 사용
    results = retriever.invoke(test_query)

    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.metadata.get('page_title', 'Unknown')}")
        print(f"    Section: {doc.metadata.get('section_title', 'N/A')}")
        print(f"    유사도: {doc.metadata.get('_similarity_score', 0):.4f}")
        print(f"    시간가중치: {doc.metadata.get('_time_weight', 0):.4f}")
        print(f"    최종점수: {doc.metadata.get('_combined_score', 0):.4f}")
        print(f"    Content: {doc.page_content[:150]}...")

    print("\n" + "=" * 60)
    print("✅ Time-Weighted Retriever 테스트 완료")
    print(f"\n리트리버 정보: {retriever.get_info()}")
