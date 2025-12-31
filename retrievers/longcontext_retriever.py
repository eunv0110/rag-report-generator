#!/usr/bin/env python3
"""Long Context Retriever - LangChain 기반 문서 재정렬을 통한 긴 컨텍스트 최적화"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.document_transformers.long_context_reorder import LongContextReorder
from typing import List, Optional, Dict, Any


class LongContextRetriever(BaseRetriever):
    """
    Long Context Retriever - "Lost in the Middle" 현상 완화

    LLM은 긴 컨텍스트의 중간 부분에 있는 정보를 놓치는 경향이 있습니다.
    이 리트리버는 중요한 문서를 처음과 끝에 배치하여 LLM이 정보를 더 잘 찾도록 합니다.

    재정렬 패턴:
    - 가장 관련성 높은 문서 → 맨 처음
    - 두 번째로 관련성 높은 문서 → 맨 끝
    - 세 번째로 관련성 높은 문서 → 두 번째 위치
    - 네 번째로 관련성 높은 문서 → 끝에서 두 번째
    - ... (이런 식으로 중요도가 낮은 문서들이 중간으로)

    참고: "Lost in the Middle" 논문
    https://arxiv.org/abs/2307.03172
    """

    base_retriever: BaseRetriever
    """기본 리트리버 (Dense, BM25, Ensemble 등)"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        **kwargs
    ):
        """
        Args:
            base_retriever: 기본 리트리버 (Dense, BM25, Ensemble 등)
        """
        super().__init__(
            base_retriever=base_retriever,
            **kwargs
        )

        print(f"✅ Long Context Retriever 초기화 완료")
        print(f"   기본 리트리버: {type(base_retriever).__name__}")
        print(f"   재정렬 전략: Lost in the Middle mitigation")

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        LangChain 호환 메서드: 검색 후 문서 재정렬

        Args:
            query: 검색 쿼리
            run_manager: 콜백 매니저

        Returns:
            재정렬된 LangChain Document 리스트
        """
        # 기본 리트리버로 검색
        documents = self.base_retriever.invoke(query)

        if not documents:
            return []

        print(f"\n[Long Context Reorder]")
        print(f"  검색된 문서 수: {len(documents)}")

        # 원본 순서 출력
        print(f"  원본 순서: [", end="")
        for i, doc in enumerate(documents):
            if i > 0:
                print(", ", end="")
            title = doc.metadata.get('page_title', 'Unknown')[:20]
            print(f"{i+1}:{title}", end="")
        print("]")

        # LongContextReorder 적용
        reorderer = LongContextReorder()
        reordered_docs = reorderer.transform_documents(documents)

        # 재정렬된 순서 출력
        print(f"  재정렬 순서: [", end="")
        for i, doc in enumerate(reordered_docs):
            if i > 0:
                print(", ", end="")
            title = doc.metadata.get('page_title', 'Unknown')[:20]
            # 원본에서 몇 번째였는지 찾기
            orig_idx = documents.index(doc) + 1
            print(f"{orig_idx}→{i+1}", end="")
        print("]")

        return reordered_docs

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        return {
            "name": "LongContextRetriever",
            "type": self.__class__.__name__,
            "base_retriever": type(self.base_retriever).__name__,
            "reorder_strategy": "Lost in the Middle mitigation",
            "description": "중요한 문서를 처음/끝에 배치하여 LLM 성능 향상"
        }


def get_longcontext_retriever(
    base_retriever: Optional[BaseRetriever] = None,
    k: int = 5
) -> LongContextRetriever:
    """
    Long Context Retriever 생성

    Args:
        base_retriever: 기본 리트리버 (None이면 Dense Retriever 사용)
        k: 기본 리트리버에서 반환할 문서 수

    Returns:
        LongContextRetriever 인스턴스
    """
    if base_retriever is None:
        from retrievers.dense_retriever import get_dense_retriever
        base_retriever = get_dense_retriever(k=k)

    retriever = LongContextRetriever(
        base_retriever=base_retriever
    )

    return retriever


if __name__ == "__main__":
    print("🔍 Long Context Retriever 테스트")
    print("=" * 60)

    # 테스트 쿼리
    test_queries = [
        "RAG 시스템의 구성 요소는?",
        "임베딩 모델 선택 기준",
    ]

    # Long Context Retriever 테스트
    print("\n" + "=" * 60)
    print("📝 Long Context Retriever")
    print("=" * 60)

    # Dense Retriever 기반으로 생성
    retriever = get_longcontext_retriever(k=10)

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-" * 60)

        results = retriever.invoke(query)

        print(f"\n재정렬된 결과 ({len(results)}개):")
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.metadata.get('page_title', 'Unknown')}")
            print(f"    Section: {doc.metadata.get('section_title', 'N/A')}")
            print(f"    Content: {doc.page_content[:150]}...")

    print("\n" + "=" * 60)
    print("✅ Long Context Retriever 테스트 완료")
    print(f"\n리트리버 정보: {retriever.get_info()}")
