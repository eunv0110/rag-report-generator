#!/usr/bin/env python3
"""Ensemble Retriever - BM25 + Dense Retriever with RRF (Reciprocal Rank Fusion)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from app.retrievers.bm25_retriever import get_bm25_retriever
from app.retrievers.dense_retriever import get_dense_retriever


def get_ensemble_retriever(
    k: int = 5,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    date_filter: tuple = None,
    use_mmr: bool = True,
    lambda_mult: float = 0.5
) -> BaseRetriever:
    """
    Ensemble Retriever 생성 (BM25 + Dense with RRF)

    RRF (Reciprocal Rank Fusion)는 여러 검색 결과를 결합하는 알고리즘입니다.
    각 문서의 순위를 역수로 변환하여 점수를 계산하고, 가중치를 적용하여 최종 순위를 결정합니다.

    Args:
        k: 반환할 문서 수
        bm25_weight: BM25 결과의 가중치 (0.0 ~ 1.0)
        dense_weight: Dense 결과의 가중치 (0.0 ~ 1.0)
        date_filter: (start_date, end_date) 튜플 (ISO 형식)
        use_mmr: Dense Retriever에서 MMR 사용 여부 (기본값 True)
        lambda_mult: MMR lambda 파라미터 (0~1, 기본값 0.5)

    Returns:
        EnsembleRetriever 인스턴스
    """
    # 각 retriever 생성 (더 많은 후보를 가져와서 RRF로 재순위화)
    bm25_retriever = get_bm25_retriever(k=k * 2, date_filter=date_filter)
    dense_retriever = get_dense_retriever(
        k=k * 2,
        date_filter=date_filter,
        use_mmr=use_mmr,
        lambda_mult=lambda_mult
    )

    # EnsembleRetriever로 결합 (c 파라미터는 RRF의 랭크 상수)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight],
        c=60  # RRF 파라미터: 기본값 60
    )

    return ensemble_retriever


if __name__ == "__main__":
    # 테스트
    print("🔍 Ensemble Retriever (RRF) 테스트")
    print("=" * 60)

    # Retriever 생성 (BM25: 50%, Dense: 50%)
    retriever = get_ensemble_retriever(k=5, bm25_weight=0.5, dense_weight=0.5)

    # 테스트 쿼리
    test_queries = [
        "RAG 시스템은 어떻게 동작하나요?",
        "임베딩 모델에 대해 알려주세요",
        "벡터 데이터베이스란 무엇인가요?"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)

        results = retriever.invoke(query)

        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.metadata.get('page_title', 'Unknown')}")
            print(f"    Section: {doc.metadata.get('section_title', 'N/A')}")
            print(f"    Content: {doc.page_content[:200]}...")

    print("\n" + "=" * 60)
    print("✅ Ensemble Retriever 테스트 완료")
