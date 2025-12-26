#!/usr/bin/env python3
"""통합 RAG 리트리버 평가 스크립트

사용 예시:
    # BM25 리트리버 평가
    python scripts/evaluate.py --retriever bm25_korean

    # Dense 리트리버 평가
    python scripts/evaluate.py --retriever dense

    # RRF Ensemble 리트리버 평가
    python scripts/evaluate.py --retriever ensemble_rrf

    # RRF + LongContext 리트리버 평가
    python scripts/evaluate.py --retriever ensemble_rrf_longcontext

    # MultiQuery 리트리버 평가 (기본 리트리버: ensemble)
    python scripts/evaluate.py --retriever multiquery --base-retriever ensemble --num-queries 3

    # RRF + MultiQuery 평가
    python scripts/evaluate.py --retriever multiquery --base-retriever ensemble_rrf --num-queries 3

    # RRF + LongContext + MultiQuery 평가
    python scripts/evaluate.py --retriever multiquery --base-retriever ensemble_rrf_longcontext --num-queries 3

    # QueryRewrite 리트리버 평가 (기본 리트리버: ensemble_rrf)
    python scripts/evaluate.py --retriever query_rewrite --base-retriever ensemble_rrf

    # QueryRewrite + Dense 평가
    python scripts/evaluate.py --retriever query_rewrite --base-retriever dense

    # TimeWeighted 리트리버 평가
    python scripts/evaluate.py --retriever time_weighted --decay-rate 0.01

    # RRF + TimeWeighted 평가
    python scripts/evaluate.py --retriever ensemble_rrf_timeweighted --decay-rate 0.01

    # S11: Summary-Level 평가
    python scripts/evaluate.py --retriever summary

    # S12: Mixed Retrieval 평가
    python scripts/evaluate.py --retriever mixed

    # RRF + Summary (S11) 평가
    python scripts/evaluate.py --retriever ensemble_rrf_summary

    # RRF + Mixed (S12) 평가
    python scripts/evaluate.py --retriever ensemble_rrf_mixed

    # 사용 가능한 리트리버 목록 확인
    python scripts/evaluate.py --list-retrievers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from langchain.chat_models import init_chat_model

from config.settings import (
    QDRANT_PATH,
    AZURE_AI_CREDENTIAL,
    AZURE_AI_ENDPOINT,
    MODEL_CONFIG
)
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.ensemble_retriever import EnsembleRetriever
from retrievers.ensemble_longcontext_retriever import EnsembleLongContextRetriever
from retrievers.multiquery_retriever import MultiQueryRetriever
from retrievers.time_weighted_retriever import TimeWeightedRetriever
from retrievers.raptor_retriever import RaptorRetriever
from retrievers.summary_retriever import SummaryRetriever
from retrievers.mixed_retriever import MixedRetriever
from retrievers.query_rewrite_retriever import QueryRewriteRetriever
from utils.langfuse_utils import get_langfuse_client
from utils.embedding_cache import EmbeddingCache, CachedEmbedder
from models.embeddings.factory import get_embedder

# 상수 정의
DEFAULT_DATASET_PATH = "/home/work/rag/Project/rag-report-generator/data/evaluation/merged_qa_dataset.json"
DEFAULT_NUM_CONTEXTS_FOR_ANSWER = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 500
DEFAULT_TOP_K = 10
DEFAULT_NUM_QUERIES = 3
DEFAULT_DECAY_RATE = 0.01

SYSTEM_PROMPT_FILE = "prompts/templates/evaluation/system_prompt.txt"
ANSWER_PROMPT_FILE = "prompts/templates/evaluation/answer_generation_prompt.txt"

# 사용 가능한 리트리버 타입
AVAILABLE_RETRIEVERS = {
    "bm25_basic": "BM25 리트리버 (기본)",
    "bm25_korean": "BM25 리트리버 (한국어 토크나이저)",
    "dense": "Dense 벡터 리트리버",
    "ensemble_rrf": "RRF Ensemble (BM25 + Dense)",
    "ensemble_rrf_longcontext": "RRF + LongContextReorder (BM25 + Dense)",
    "ensemble_rrf_timeweighted": "RRF + TimeWeighted (BM25 + TimeWeighted)",
    "ensemble_rrf_timeweighted_longcontext": "RRF + TimeWeighted + LongContext (BM25 + TimeWeighted)",
    "multiquery": "MultiQuery 리트리버 (기본 리트리버 위에 래핑)",
    "query_rewrite": "QueryRewrite 리트리버 (쿼리 최적화 + 기본 리트리버)",
    "time_weighted": "TimeWeighted 리트리버",
    "raptor": "RAPTOR Tree 리트리버 (계층적 문서 구조)",
    "raptor_refine": "RAPTOR Tree 리트리버 with Refine Summarizer (문맥 일관성 강화)",
    "ensemble_rrf_raptor": "RRF Ensemble (BM25 + Dense + RAPTOR)",
    "ensemble_rrf_raptor_refine": "RRF Ensemble (BM25 + Dense + RAPTOR Refine)",
    "ensemble_rrf_summary": "S11: RRF Ensemble (BM25 + Dense[notion_summary])",
    "ensemble_rrf_mixed": "S12: RRF Ensemble (BM25 + Dense[notion_mixed])",
}


def generate_version_tag(retriever_name: str, version: str = "v1") -> str:
    """버전 태그 생성"""
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{retriever_name}_{date_str}_{version}"


def load_prompt(prompt_file: str) -> str:
    """프롬프트 템플릿 로드"""
    prompt_path = Path(__file__).parent.parent / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_evaluation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """평가용 데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_llm_answer(question: str, contexts: List[str]) -> str:
    """LLM API를 호출하여 답변 생성"""
    if not AZURE_AI_CREDENTIAL or not AZURE_AI_ENDPOINT:
        return "Azure OpenAI 설정이 올바르지 않습니다. .env 파일을 확인하세요."

    os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
    os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

    answer_prompt_template = load_prompt(ANSWER_PROMPT_FILE)
    context_text = "\n\n".join(contexts[:DEFAULT_NUM_CONTEXTS_FOR_ANSWER]) if contexts else "관련 문서를 찾을 수 없습니다."
    prompt = answer_prompt_template.replace("{{context}}", context_text).replace("{{question}}", question)

    try:
        model = init_chat_model(
            "azure_ai:gpt-4.1",
            temperature=DEFAULT_TEMPERATURE,
            max_completion_tokens=DEFAULT_MAX_TOKENS
        )
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        error_msg = f"답변 생성 실패: {str(e)}"
        print(f"  ⚠️ LLM API 호출 실패: {e}")
        return error_msg


def create_trace_and_generation(
    langfuse,
    retriever_name: str,
    question: str,
    contexts: List[str],
    answer: str,
    ground_truth: str,
    context_metadata: List[Dict],
    item_metadata: Dict,
    total_time: float,
    idx: int,
    context_page_id: Optional[str] = None,
    version_tag: str = "v1",
    retriever_tags: List[str] = None
) -> str:
    """Langfuse Trace와 Generation 생성"""
    context_text = "\n\n---\n\n".join(contexts) if contexts else ""

    if retriever_tags is None:
        retriever_tags = []

    all_tags = [
        f"{retriever_name}_{version_tag}",
        version_tag,
        "evaluation"
    ] + retriever_tags

    with langfuse.start_as_current_observation(
        as_type='generation',
        name=f"generation_{retriever_name}_{version_tag}",
        model="gpt-4.1",
        input={
            "question": question,
            "context": context_text
        },
        output={
            "answer": answer
        },
        metadata={
            "ground_truth": ground_truth,
            "contexts": contexts,
            "context_metadata": context_metadata,
            "retriever_type": retriever_name,
            "version": version_tag,
            "retriever_tags": retriever_tags
        }
    ) as generation:
        trace_id = generation.trace_id

        langfuse.update_current_trace(
            name=f"eval_{retriever_name}_{version_tag}_q{idx}",
            tags=all_tags,
            input={
                "question": question,
                "context": context_text
            },
            output={
                "answer": answer
            },
            metadata={
                "retriever": retriever_name,
                "version": version_tag,
                "total_time_ms": total_time * 1000,
                "num_retrieved_contexts": len(contexts),
                "context_page_id": context_page_id,
                "question_id": idx,
                "category": item_metadata.get("category", "unknown"),
                "difficulty": item_metadata.get("difficulty", "unknown"),
                "retriever_components": retriever_tags
            }
        )

    print(f"\n[DEBUG] Trace {idx}:")
    print(f"  - ID: {trace_id}")
    print(f"  - Question: {question[:50]}...")
    print(f"  - Context length: {len(context_text)} chars")
    print(f"  - Answer length: {len(answer)} chars")

    return trace_id


def add_retrieval_quality_score(langfuse, trace_id: str, context_metadata: List[Dict]):
    """검색 품질 스코어 추가"""
    if not context_metadata:
        return

    avg_score = sum(m["score"] for m in context_metadata) / len(context_metadata)
    langfuse.create_score(
        trace_id=trace_id,
        name="retrieval_quality",
        value=avg_score,
        comment=f"Average retrieval score from {len(context_metadata)} contexts"
    )


def evaluate_single_query(
    retriever,
    item: Dict[str, Any],
    langfuse,
    idx: int,
    top_k: int,
    base_version: str = "v1",
    retriever_tags: List[str] = None
) -> Dict[str, Any]:
    """단일 쿼리 평가"""
    question = item["question"]
    ground_truth = item["ground_truth"]
    context_page_id = item.get("context_page_id")
    item_metadata = item.get("metadata", {})

    version_tag = generate_version_tag(retriever.name, base_version)
    start_time = time.time()

    # 검색 수행
    search_results = retriever.search(question, top_k=top_k)
    contexts = [result.combined_text for result in search_results]

    if not contexts:
        print(f"  ⚠️ [{idx}] No contexts found for question!")
        contexts = ["검색 결과가 없습니다."]

    context_metadata = [
        {
            "score": result.score,
            "page_title": result.page_title,
            "section_title": result.section_title,
            "chunk_id": result.chunk_id
        }
        for result in search_results
    ]

    # LLM 답변 생성
    answer = generate_llm_answer(question, contexts)

    if not answer or answer.startswith("답변 생성 실패") or answer.startswith("Azure OpenAI 설정"):
        print(f"  ⚠️ [{idx}] LLM answer generation failed!")
        if not answer:
            answer = "답변을 생성할 수 없습니다."

    total_time = time.time() - start_time

    # Langfuse Trace & Generation
    trace_id = create_trace_and_generation(
        langfuse=langfuse,
        retriever_name=retriever.name,
        question=question,
        contexts=contexts,
        answer=answer,
        ground_truth=ground_truth,
        context_metadata=context_metadata,
        item_metadata=item_metadata,
        total_time=total_time,
        idx=idx,
        context_page_id=context_page_id,
        version_tag=version_tag,
        retriever_tags=retriever_tags
    )

    # 검색 품질 스코어 추가
    add_retrieval_quality_score(langfuse, trace_id, context_metadata)

    print(f"  [{idx}] {question[:50]}... ({len(contexts)}개 문서, {total_time*1000:.0f}ms)")

    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "num_contexts": len(contexts),
        "time": total_time,
        "trace_id": trace_id
    }


def evaluate_retriever(
    retriever,
    eval_data: List[Dict[str, Any]],
    langfuse,
    top_k: int = DEFAULT_TOP_K,
    base_version: str = "v1",
    retriever_tags: List[str] = None
) -> Dict[str, Any]:
    """리트리버 평가"""
    print(f"\n{'=' * 60}")
    print(f"🔍 {retriever.name} 평가 중...")
    print(f"{'=' * 60}")

    stats = {
        "total_queries": len(eval_data),
        "total_time": 0,
        "evaluations": []
    }

    for idx, item in enumerate(eval_data, 1):
        eval_result = evaluate_single_query(
            retriever=retriever,
            item=item,
            langfuse=langfuse,
            idx=idx,
            top_k=top_k,
            base_version=base_version,
            retriever_tags=retriever_tags
        )

        stats["evaluations"].append(eval_result)
        stats["total_time"] += eval_result["time"]

    stats["avg_time"] = stats["total_time"] / stats["total_queries"]
    stats["avg_contexts"] = sum(e["num_contexts"] for e in stats["evaluations"]) / stats["total_queries"]

    return stats


def create_retriever(
    retriever_type: str,
    qdrant_client: QdrantClient,
    embedding_cache: Optional[EmbeddingCache] = None,
    base_retriever_type: str = "ensemble_rrf",
    num_queries: int = DEFAULT_NUM_QUERIES,
    decay_rate: float = DEFAULT_DECAY_RATE
):
    """
    리트리버 생성 팩토리 함수

    Args:
        retriever_type: 리트리버 타입
        qdrant_client: Qdrant 클라이언트
        embedding_cache: 임베딩 캐시 (Dense 리트리버용)
        base_retriever_type: MultiQuery의 기본 리트리버 타입
        num_queries: MultiQuery에서 생성할 쿼리 수
        decay_rate: TimeWeighted 리트리버의 decay rate

    Returns:
        (retriever, retriever_tags)
    """
    retriever_tags = []

    # 임베더 생성 (Dense 리트리버 필요시)
    def get_cached_embedder():
        base_embedder = get_embedder()
        if embedding_cache:
            # 설정 파일에서 모델명 가져오기
            embedding_model = MODEL_CONFIG.get('embeddings', {}).get('model', 'text-embedding-3-large')
            return CachedEmbedder(base_embedder, embedding_cache, model_name=embedding_model)
        return base_embedder

    # BM25 리트리버
    if retriever_type == "bm25_basic":
        retriever = BM25Retriever(qdrant_client, use_korean_tokenizer=False)
        retriever_tags = ["bm25"]

    elif retriever_type == "bm25_korean":
        retriever = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        retriever_tags = ["bm25", "korean"]

    # Dense 리트리버
    elif retriever_type == "dense":
        embedder = get_cached_embedder()
        retriever = DenseRetriever(qdrant_client, embedder=embedder)
        retriever_tags = ["dense"]

    # RRF Ensemble
    elif retriever_type == "ensemble_rrf":
        bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        embedder = get_cached_embedder()
        dense = DenseRetriever(qdrant_client, embedder=embedder)
        retriever = EnsembleRetriever(
            retrievers=[bm25, dense],
            name="ensemble_rrf"
        )
        retriever_tags = ["ensemble", "rrf", "bm25", "dense"]

    # RRF + LongContext
    elif retriever_type == "ensemble_rrf_longcontext":
        bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        embedder = get_cached_embedder()
        dense = DenseRetriever(qdrant_client, embedder=embedder)
        retriever = EnsembleLongContextRetriever(
            retrievers=[bm25, dense],
            name="ensemble_rrf_longcontext"
        )
        retriever_tags = ["ensemble", "rrf", "longcontext", "bm25", "dense"]

    # TimeWeighted 리트리버
    elif retriever_type == "time_weighted":
        embedder = get_cached_embedder()
        retriever = TimeWeightedRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            decay_rate=decay_rate,
            name=f"time_weighted_decay{decay_rate}"
        )
        retriever_tags = ["time_weighted", f"decay_{decay_rate}"]

    # RRF + TimeWeighted
    elif retriever_type == "ensemble_rrf_timeweighted":
        bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        embedder = get_cached_embedder()
        tw = TimeWeightedRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            decay_rate=decay_rate,
            name=f"time_weighted_decay{decay_rate}"
        )
        retriever = EnsembleRetriever(
            retrievers=[bm25, tw],
            name=f"ensemble_rrf_timeweighted_{decay_rate}"
        )
        retriever_tags = ["ensemble", "rrf", "bm25", "time_weighted", f"decay_{decay_rate}"]

    # RRF + TimeWeighted + LongContext
    elif retriever_type == "ensemble_rrf_timeweighted_longcontext":
        bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        embedder = get_cached_embedder()
        tw = TimeWeightedRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            decay_rate=decay_rate,
            name=f"time_weighted_decay{decay_rate}"
        )
        retriever = EnsembleLongContextRetriever(
            retrievers=[bm25, tw],
            name=f"ensemble_rrf_timeweighted_longcontext_{decay_rate}"
        )
        retriever_tags = ["ensemble", "rrf", "longcontext", "bm25", "time_weighted", f"decay_{decay_rate}"]

    # MultiQuery 리트리버
    elif retriever_type == "multiquery":
        # 기본 리트리버 생성
        base_retriever, base_tags = create_retriever(
            base_retriever_type,
            qdrant_client,
            embedding_cache,
            decay_rate=decay_rate
        )
        retriever = MultiQueryRetriever(
            base_retriever=base_retriever,
            num_queries=num_queries,
            name=f"multiquery_{base_retriever.name}"
        )
        retriever_tags = ["multiquery", f"num_queries_{num_queries}"] + base_tags

    # RAPTOR 리트리버
    elif retriever_type == "raptor":
        embedder = get_cached_embedder()
        retriever = RaptorRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name="notion_raptor"
        )
        retriever_tags = ["raptor", "tree", "hierarchical"]

    # RAPTOR Refine 리트리버
    elif retriever_type == "raptor_refine":
        embedder = get_cached_embedder()
        retriever = RaptorRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name="notion_raptor_refine",
            name="raptor_refine"
        )
        retriever_tags = ["raptor", "tree", "hierarchical", "refine"]

    # RRF + RAPTOR
    elif retriever_type == "ensemble_rrf_raptor":
        bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        embedder = get_cached_embedder()
        dense = DenseRetriever(qdrant_client, embedder=embedder)
        raptor = RaptorRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name="notion_raptor"
        )
        retriever = EnsembleRetriever(
            retrievers=[bm25, dense, raptor],
            name="ensemble_rrf_raptor"
        )
        retriever_tags = ["ensemble", "rrf", "bm25", "dense", "raptor"]

    # RRF + RAPTOR Refine
    elif retriever_type == "ensemble_rrf_raptor_refine":
        bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        embedder = get_cached_embedder()
        dense = DenseRetriever(qdrant_client, embedder=embedder)
        raptor = RaptorRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name="notion_raptor_refine",
            name="raptor_refine"
        )
        retriever = EnsembleRetriever(
            retrievers=[bm25, dense, raptor],
            name="ensemble_rrf_raptor_refine"
        )
        retriever_tags = ["ensemble", "rrf", "bm25", "dense", "raptor", "refine"]

    # RRF + Summary (S11) - BM25 + Dense(notion_summary)
    elif retriever_type == "ensemble_rrf_summary":
        bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        embedder = get_cached_embedder()
        dense_summary = DenseRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name="notion_summary"
        )
        retriever = EnsembleRetriever(
            retrievers=[bm25, dense_summary],
            name="ensemble_rrf_summary"
        )
        retriever_tags = ["ensemble", "rrf", "bm25", "dense", "summary", "s11"]

    # RRF + Mixed (S12) - BM25 + Dense(notion_mixed)
    elif retriever_type == "ensemble_rrf_mixed":
        bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True)
        embedder = get_cached_embedder()
        dense_mixed = DenseRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name="notion_mixed"
        )
        retriever = EnsembleRetriever(
            retrievers=[bm25, dense_mixed],
            name="ensemble_rrf_mixed"
        )
        retriever_tags = ["ensemble", "rrf", "bm25", "dense", "mixed", "s12"]

    # QueryRewrite 리트리버
    elif retriever_type == "query_rewrite":
        # 기본 리트리버 생성
        base_retriever, base_tags = create_retriever(
            base_retriever_type,
            qdrant_client,
            embedding_cache,
            decay_rate=decay_rate
        )
        retriever = QueryRewriteRetriever(
            base_retriever=base_retriever,
            name=f"query_rewrite_{base_retriever.name}"
        )
        retriever_tags = ["query_rewrite", "llm_optimization"] + base_tags

    else:
        raise ValueError(f"알 수 없는 리트리버 타입: {retriever_type}")

    return retriever, retriever_tags


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="통합 RAG 리트리버 평가 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=False,
        choices=list(AVAILABLE_RETRIEVERS.keys()),
        help="평가할 리트리버 타입"
    )
    parser.add_argument(
        "--list-retrievers",
        action="store_true",
        help="사용 가능한 리트리버 타입 목록 출력"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="평가 데이터셋 경로"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="검색할 상위 k개 문서"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="버전 태그"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="임베딩 캐시 비활성화"
    )

    # MultiQuery 관련 옵션
    parser.add_argument(
        "--base-retriever",
        type=str,
        choices=list(AVAILABLE_RETRIEVERS.keys()),
        default="ensemble_rrf",
        help="MultiQuery의 기본 리트리버 타입 (기본값: ensemble_rrf)"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=DEFAULT_NUM_QUERIES,
        help="MultiQuery에서 생성할 쿼리 수 (기본값: 3)"
    )

    # TimeWeighted 관련 옵션
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=DEFAULT_DECAY_RATE,
        help="TimeWeighted 리트리버의 decay rate (기본값: 0.01)"
    )

    args = parser.parse_args()

    # 리트리버 목록 출력
    if args.list_retrievers:
        print("사용 가능한 리트리버 타입:")
        for ret_type, description in AVAILABLE_RETRIEVERS.items():
            print(f"  {ret_type:<30} - {description}")
        return

    # 리트리버 타입 필수 확인
    if not args.retriever:
        parser.error("--retriever 옵션이 필요합니다. --list-retrievers로 사용 가능한 타입을 확인하세요.")

    print("=" * 60)
    print(f"📊 {AVAILABLE_RETRIEVERS[args.retriever]} 평가")
    print("=" * 60)

    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()
    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        return

    # Qdrant 클라이언트 초기화
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    # 평가 데이터셋 로드
    eval_data = load_evaluation_dataset(args.dataset)
    print(f"\n✅ 평가 데이터: {len(eval_data)}개 질문")

    # 임베딩 캐시 초기화
    use_cache = not args.no_cache
    embedding_cache = EmbeddingCache() if use_cache else None

    if use_cache:
        print("💾 임베딩 캐시 활성화")

    # 리트리버 생성
    print(f"\n📦 리트리버 초기화 중...")
    retriever, retriever_tags = create_retriever(
        retriever_type=args.retriever,
        qdrant_client=qdrant_client,
        embedding_cache=embedding_cache,
        base_retriever_type=args.base_retriever,
        num_queries=args.num_queries,
        decay_rate=args.decay_rate
    )

    print(f"✅ 리트리버 초기화 완료: {retriever.name}")
    print(f"   - 타입: {args.retriever}")
    print(f"   - 태그: {', '.join(retriever_tags)}")

    if args.retriever == "multiquery":
        print(f"   - 기본 리트리버: {args.base_retriever}")
        print(f"   - 생성할 쿼리 수: {args.num_queries}")

    if "time_weighted" in retriever_tags:
        print(f"   - Decay rate: {args.decay_rate}")

    # 평가 수행
    stats = evaluate_retriever(
        retriever,
        eval_data,
        langfuse,
        args.top_k,
        args.version,
        retriever_tags=retriever_tags
    )

    # 임베딩 캐시 저장
    if use_cache and embedding_cache:
        print("\n💾 임베딩 캐시 저장 중...")
        embedding_cache.save()
        embedding_cache.print_stats()

    # Langfuse flush
    print("\n⏳ Langfuse에 데이터 전송 중...")
    langfuse.flush()

    # 결과 출력
    print("\n" + "=" * 60)
    print("📈 평가 결과")
    print("=" * 60)
    print(f"리트리버: {retriever.name}")
    print(f"총 쿼리: {stats['total_queries']}")
    print(f"평균 컨텍스트 수: {stats['avg_contexts']:.2f}")
    print(f"평균 시간: {stats['avg_time']*1000:.2f}ms")

    print("\n" + "=" * 60)
    print("✅ 평가 완료!")
    print("=" * 60)
    print(f"\n💡 다음 단계:")
    print(f"   1. 🌐 Langfuse 대시보드: https://cloud.langfuse.com")
    print(f"   2. 📊 Traces 탭에서 생성된 trace 확인")
    print(f"   3. 🔧 Settings → Evaluations → RAGAS 메트릭 설정")
    print(f"   4. ⚙️  Evaluations 탭에서 자동 평가 결과 확인")

    # 결과 저장
    output_file = Path(args.dataset).parent / f"{args.retriever}_evaluation_stats.json"
    save_result = {k: v for k, v in stats.items() if k != "evaluations"}
    save_result["num_evaluations"] = len(stats.get("evaluations", []))
    save_result["config"] = {
        "retriever_type": args.retriever,
        "retriever_name": retriever.name,
        "retriever_tags": retriever_tags,
        "top_k": args.top_k,
        "version": args.version,
    }

    # 추가 설정 정보
    if args.retriever == "multiquery":
        save_result["config"]["base_retriever"] = args.base_retriever
        save_result["config"]["num_queries"] = args.num_queries

    if "time_weighted" in retriever_tags:
        save_result["config"]["decay_rate"] = args.decay_rate

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 통계 저장: {output_file}")


if __name__ == "__main__":
    main()
