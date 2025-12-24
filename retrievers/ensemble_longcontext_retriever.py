"""Ensemble Retriever with LongContextReorder - RRF + Lost in the Middle mitigation"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_transformers.long_context_reorder import LongContextReorder
from .ensemble_retriever import EnsembleRetriever
from .base_retriever import SearchResult


class EnsembleLongContextRetriever(EnsembleRetriever):
    """
    RRF 앙상블 + LongContextReorder를 결합한 리트리버

    LongContextReorder는 "Lost in the Middle" 현상을 완화합니다:
    - 중요한 문서를 처음과 끝에 배치
    - 덜 중요한 문서를 중간에 배치
    - LLM이 긴 컨텍스트에서 정보를 더 잘 찾도록 도움

    참고: https://arxiv.org/abs/2307.03172
    """

    def __init__(
        self,
        retrievers: List,
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
        super().__init__(retrievers, weights, k, name)
        self.reorderer = LongContextReorder()

        # 이름 재설정
        if name is None:
            retriever_names = [r.name for r in retrievers]
            self._name = f"ensemble_rrf_longcontext({'_'.join(retriever_names)})"

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        RRF로 앙상블 검색 후 LongContextReorder 적용

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수

        Returns:
            LongContextReorder가 적용된 검색 결과
        """
        # 부모 클래스의 RRF 검색 수행
        rrf_results = super().search(query, top_k)

        if not rrf_results:
            return []

        # SearchResult를 LangChain Document로 변환
        documents = []
        for result in rrf_results:
            doc = Document(
                page_content=result.combined_text,
                metadata={
                    "chunk_id": result.chunk_id,
                    "page_id": result.page_id,
                    "text": result.text,
                    "page_title": result.page_title,
                    "section_title": result.section_title,
                    "section_path": result.section_path,
                    "score": result.score,
                    "has_image": result.has_image,
                    "image_descriptions": result.image_descriptions,
                    "properties": result.properties
                }
            )
            documents.append(doc)

        # LongContextReorder 적용
        reordered_docs = self.reorderer.transform_documents(documents)

        # 다시 SearchResult로 변환
        final_results = []
        for doc in reordered_docs:
            meta = doc.metadata
            result = SearchResult(
                chunk_id=meta["chunk_id"],
                page_id=meta["page_id"],
                text=meta["text"],
                combined_text=doc.page_content,
                page_title=meta["page_title"],
                section_title=meta["section_title"],
                section_path=meta["section_path"],
                score=meta["score"],
                has_image=meta["has_image"],
                image_descriptions=meta["image_descriptions"],
                properties=meta["properties"]
            )
            final_results.append(result)

        return final_results

    @property
    def name(self) -> str:
        """리트리버 이름 반환"""
        return self._name

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        info = super().get_info()
        info.update({
            "uses_long_context_reorder": True,
            "reorder_strategy": "Lost in the Middle mitigation"
        })
        return info
