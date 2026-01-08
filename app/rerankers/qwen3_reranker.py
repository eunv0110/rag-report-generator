#!/usr/bin/env python3
"""Qwen3-Reranker-4B 모델을 사용한 문서 재순위화

evaluate_reranker.py의 reranker 로직을 추출하여 모듈화
"""

import torch
from typing import List, Any
from sentence_transformers import CrossEncoder


# Reranker 모델 인스턴스 (전역 변수로 한 번만 로드)
QWEN3_RERANKER = None


def format_query(query: str, instruction: str = None) -> str:
    """Qwen3-Reranker를 위한 쿼리 포맷팅

    Args:
        query: 검색 쿼리
        instruction: 검색 지시문 (기본값: 일반적인 검색 지시문)

    Returns:
        포맷팅된 쿼리 문자열
    """
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = "Given a query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document: str) -> str:
    """Qwen3-Reranker를 위한 문서 포맷팅

    Args:
        document: 문서 텍스트

    Returns:
        포맷팅된 문서 문자열
    """
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


def get_optimal_batch_size() -> int:
    """GPU 메모리에 따른 최적 배치 크기 계산

    Returns:
        최적 배치 크기 (기본: 16)
    """
    # VLLM과 공존하기 위해 배치 크기를 16으로 고정
    return 16


def get_qwen3_reranker() -> CrossEncoder:
    """Qwen3-Reranker-4B 모델 로드 (캐싱)

    모델이 이미 로드되어 있으면 재사용하고, 없으면 새로 로드합니다.

    Returns:
        로드된 Qwen3-Reranker-4B 모델
    """
    global QWEN3_RERANKER

    # 이미 로드된 모델이 있으면 재사용
    if QWEN3_RERANKER is not None:
        return QWEN3_RERANKER

    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"🔄 Qwen3-Reranker-4B 모델 로딩 중... (device: {device})")
    QWEN3_RERANKER = CrossEncoder(
        "tomaarsen/Qwen3-Reranker-4B-seq-cls",
        max_length=8192,
        device=device,
        trust_remote_code=True
    )
    print("✅ Qwen3-Reranker-4B 모델 로드 완료!")

    # 최적 배치 크기 출력
    optimal_bs = get_optimal_batch_size()
    print(f"💡 권장 배치 크기: {optimal_bs}")

    return QWEN3_RERANKER


def rerank_documents(
    query: str,
    docs: List[Any],
    top_k: int = 6,
    batch_size: int = None,
    initial_k: int = None
) -> List[Any]:
    """Qwen3-Reranker-4B 모델로 문서 재순위화

    Args:
        query: 검색 쿼리
        docs: 검색된 문서 리스트 (langchain Document 객체)
        top_k: 최종 반환할 문서 수
        batch_size: 배치 처리 크기 (None이면 자동 계산)
        initial_k: 초기 검색 문서 수 (재순위화 전, None이면 docs 길이 사용)

    Returns:
        재순위화된 상위 k개 문서
    """
    # Reranker 모델 로드
    reranker = get_qwen3_reranker()

    # 배치 크기 자동 설정
    if batch_size is None:
        batch_size = get_optimal_batch_size()

    # 초기 문서 수 설정
    if initial_k is None:
        initial_k = len(docs)

    # Qwen3-Reranker 포맷으로 query-document 쌍 생성
    formatted_query = format_query(query)
    pairs = [
        [formatted_query, format_document(doc.page_content)]
        for doc in docs
    ]

    # 배치 단위로 재순위화 점수 계산 (메모리 절약)
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        batch_scores = reranker.predict(batch_pairs)
        all_scores.extend(batch_scores)

        # 배치 처리 후 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 점수와 문서를 함께 정렬 (내림차순)
    doc_score_pairs = list(zip(docs, all_scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 상위 k개 문서만 반환
    reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

    print(f"\n🔄 Qwen3 Reranking 완료: {initial_k}개 → {len(reranked_docs)}개 (배치 크기: {batch_size})")
    print("Top 3 Reranked Scores:")
    for i, (doc, score) in enumerate(doc_score_pairs[:3], 1):
        print(f"  {i}. {doc.metadata.get('page_title', 'Unknown')}: {score:.4f}")

    return reranked_docs
