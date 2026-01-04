#!/usr/bin/env python3
"""리트리버 팩토리

다양한 리트리버 조합을 생성하는 팩토리 함수를 제공합니다.

지원하는 리트리버 타입:
- rrf_ensemble: RRF (Reciprocal Rank Fusion) 앙상블
- rrf_multiquery: RRF + MultiQuery
- rrf_multiquery_longcontext: RRF + MultiQuery + LongContext
- rrf_longcontext_timeweighted: RRF + LongContext + TimeWeighted
"""

import os
from typing import Tuple, List, Dict, Any

from retrievers.ensemble_retriever import get_ensemble_retriever
from retrievers.multiquery_retriever import get_multiquery_retriever
from retrievers.longcontext_retriever import get_longcontext_retriever
from retrievers.timeweighted_retriever import get_time_weighted_retriever


# 지원하는 리트리버 타입 상수
RETRIEVER_TYPE_RRF_ENSEMBLE = "rrf_ensemble"
RETRIEVER_TYPE_RRF_MULTIQUERY = "rrf_multiquery"
RETRIEVER_TYPE_RRF_MULTIQUERY_LONGCONTEXT = "rrf_multiquery_longcontext"
RETRIEVER_TYPE_RRF_LONGCONTEXT_TIMEWEIGHTED = "rrf_longcontext_timeweighted"


def create_retriever_from_config(
    retriever_config: Dict[str, Any],
    top_k: int = 10
) -> Tuple[Any, List[str]]:
    """설정에서 리트리버 생성

    Args:
        retriever_config: 리트리버 설정 딕셔너리
            - retriever_type: 리트리버 타입 (예: "rrf_multiquery_longcontext")
            - embedding_preset: 임베딩 프리셋 (예: "openai_large")
            - k (optional): Top-K 값
        top_k: Top-K 값 (retriever_config에 'k' 값이 있으면 그것을 우선 사용)

    Returns:
        (retriever, retriever_tags) 튜플
            - retriever: 생성된 리트리버 인스턴스
            - retriever_tags: 리트리버 구성 요소 태그 리스트

    Raises:
        ValueError: 알 수 없는 리트리버 타입일 때
    """
    retriever_type = retriever_config['retriever_type']
    embedding_preset = retriever_config['embedding_preset']

    # retriever_config에서 k 값 가져오기 (없으면 top_k 사용)
    k = retriever_config.get('k', top_k)

    # 환경 변수로 임베딩 프리셋 설정
    os.environ['MODEL_PRESET'] = embedding_preset

    if retriever_type == "rrf_multiquery_longcontext":
        # RRF + MultiQuery + LongContext
        base_retriever = get_ensemble_retriever(k=k)
        multiquery_retriever = get_multiquery_retriever(
            base_retriever=base_retriever,
            num_queries=3,
            k=k
        )
        retriever = get_longcontext_retriever(
            base_retriever=multiquery_retriever,
            k=k
        )
        retriever_tags = ["longcontext", "multiquery", "ensemble", "rrf"]

    elif retriever_type == "rrf_ensemble":
        # RRF Ensemble
        retriever = get_ensemble_retriever(k=k)
        retriever_tags = ["ensemble", "rrf", "bm25", "dense"]

    elif retriever_type == "rrf_multiquery":
        # RRF + MultiQuery
        base_retriever = get_ensemble_retriever(k=k)
        retriever = get_multiquery_retriever(
            base_retriever=base_retriever,
            num_queries=3,
            k=k
        )
        retriever_tags = ["multiquery", "ensemble", "rrf"]

    elif retriever_type == "rrf_longcontext_timeweighted":
        # RRF + LongContext + TimeWeighted
        time_weighted = get_time_weighted_retriever(
            decay_rate=0.01,
            k=k
        )
        retriever = get_longcontext_retriever(
            base_retriever=time_weighted,
            k=k
        )
        retriever_tags = ["longcontext", "time_weighted", "decay_0.01"]

    else:
        raise ValueError(f"알 수 없는 리트리버 타입: {retriever_type}")

    return retriever, retriever_tags


def create_strategy_retriever(strategy_num: int, top_k: int = 10) -> Tuple[Any, str, List[str]]:
    """전략 번호로 리트리버 생성

    Args:
        strategy_num: 전략 번호 (1-4)
            1: RRF 앙상블 (베이스라인)
            2: RRF + MultiQuery
            3: RRF + MultiQuery + LongContext
            4: RRF + LongContext + TimeWeighted
        top_k: Top-K 값 (검색할 문서 개수)

    Returns:
        (retriever, retriever_name, retriever_tags) 튜플
            - retriever: 생성된 리트리버 인스턴스
            - retriever_name: 리트리버 이름
            - retriever_tags: 리트리버 구성 요소 태그 리스트

    Raises:
        ValueError: 지원하지 않는 전략 번호일 때 (1-4 외의 값)
    """
    if strategy_num == 1:
        # Strategy 1: RRF (Baseline)
        retriever = get_ensemble_retriever(k=top_k)
        retriever_name = "ensemble_rrf"
        retriever_tags = ["ensemble", "rrf", "bm25", "dense"]

    elif strategy_num == 2:
        # Strategy 2: RRF + MultiQuery
        base_retriever = get_ensemble_retriever(k=top_k)
        retriever = get_multiquery_retriever(
            base_retriever=base_retriever,
            num_queries=3,
            k=top_k
        )
        retriever_name = "rrf_multiquery"
        retriever_tags = ["multiquery", "ensemble", "rrf", "bm25", "dense"]

    elif strategy_num == 3:
        # Strategy 3: RRF + MultiQuery + LongContext
        base_retriever = get_ensemble_retriever(k=top_k)
        multiquery_retriever = get_multiquery_retriever(
            base_retriever=base_retriever,
            num_queries=3,
            k=top_k
        )
        retriever = get_longcontext_retriever(
            base_retriever=multiquery_retriever,
            k=top_k
        )
        retriever_name = "rrf_multiquery_longcontext"
        retriever_tags = ["longcontext", "multiquery", "ensemble", "rrf", "bm25", "dense"]

    elif strategy_num == 4:
        # Strategy 4: RRF + LongContext + TimeWeighted
        time_weighted = get_time_weighted_retriever(
            decay_rate=0.01,
            k=top_k
        )
        retriever = get_longcontext_retriever(
            base_retriever=time_weighted,
            k=top_k
        )
        retriever_name = "rrf_longcontext_timeweighted"
        retriever_tags = ["longcontext", "time_weighted", "decay_0.01"]

    else:
        raise ValueError(f"알 수 없는 전략 번호: {strategy_num}")

    return retriever, retriever_name, retriever_tags
