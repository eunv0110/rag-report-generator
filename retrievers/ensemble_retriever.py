"""Ensemble Retriever - Reciprocal Rank Fusion (RRF)"""

from typing import List, Dict, Any
from collections import defaultdict
from .base_retriever import BaseRetriever, SearchResult


class EnsembleRetriever(BaseRetriever):
    """
    앙상블 리트리버 - Reciprocal Rank Fusion (RRF) 방식

    여러 리트리버의 결과를 RRF 알고리즘으로 결합합니다.
    RRF는 각 리트리버의 스코어 스케일에 영향을 받지 않으며,
    단순히 순위 정보만을 사용하여 결과를 결합합니다.

    RRF 스코어 계산:
        score(doc) = Σ 1 / (k + rank_i(doc))

    여기서:
        - rank_i(doc): i번째 리트리버에서 문서의 순위 (1부터 시작)
        - k: 상수 (기본값 60, 일반적으로 사용되는 값)
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: List[float] = None,
        k: int = 60,
        name: str = None
    ):
        """
        Args:
            retrievers: 결합할 리트리버 리스트
            weights: 각 리트리버의 가중치 (기본값: 모두 동일)
            k: RRF 상수 (기본값: 60)
            name: 리트리버 이름 (기본값: 자동 생성)
        """
        if not retrievers:
            raise ValueError("최소 1개 이상의 리트리버가 필요합니다")

        self.retrievers = retrievers
        self.k = k

        # 가중치 설정
        if weights is None:
            self.weights = [1.0] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("가중치 개수와 리트리버 개수가 일치해야 합니다")
            self.weights = weights

        # 이름 설정
        if name is None:
            retriever_names = [r.name for r in retrievers]
            self._name = f"ensemble_rrf({'_'.join(retriever_names)})"
        else:
            self._name = name

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        RRF 방식으로 앙상블 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수

        Returns:
            RRF 스코어 기준 정렬된 검색 결과
        """
        # 각 리트리버에서 결과 가져오기 (top_k의 2배를 가져와 다양성 확보)
        retrieval_k = max(top_k * 2, 10)
        all_results = []

        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.search(query, top_k=retrieval_k)
            all_results.append((results, weight))

        # RRF 스코어 계산
        rrf_scores = defaultdict(float)
        doc_map = {}  # chunk_id -> SearchResult

        for results, weight in all_results:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id

                # RRF 스코어: weight / (k + rank)
                rrf_scores[chunk_id] += weight / (self.k + rank)

                # 첫 번째로 발견된 결과를 저장 (또는 더 높은 원본 스코어를 가진 것)
                if chunk_id not in doc_map or result.score > doc_map[chunk_id].score:
                    doc_map[chunk_id] = result

        # RRF 스코어 기준으로 정렬
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # SearchResult 객체 생성 (RRF 스코어로 업데이트)
        final_results = []
        for chunk_id, rrf_score in sorted_chunks:
            result = doc_map[chunk_id]
            # 새로운 SearchResult 생성 (RRF 스코어로 업데이트)
            final_result = SearchResult(
                chunk_id=result.chunk_id,
                page_id=result.page_id,
                text=result.text,
                combined_text=result.combined_text,
                page_title=result.page_title,
                section_title=result.section_title,
                section_path=result.section_path,
                score=rrf_score,  # RRF 스코어로 대체
                has_image=result.has_image,
                image_descriptions=result.image_descriptions,
                properties=result.properties
            )
            final_results.append(final_result)

        return final_results

    @property
    def name(self) -> str:
        """리트리버 이름 반환"""
        return self._name

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "num_retrievers": len(self.retrievers),
            "retrievers": [r.name for r in self.retrievers],
            "weights": self.weights,
            "k": self.k
        }
