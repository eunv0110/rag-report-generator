#!/usr/bin/env python3
"""4가지 Retrieval 전략 성능 평가 스크립트

평가 전략:
  1. RRF (Baseline)
  2. RRF + MultiQuery
  3. RRF + MultiQuery + LongContext
  4. RRF + LongContext + TimeWeighted

사용법:
    python evaluators/evaluate_4_strategies_new.py

    # 특정 전략만 평가
    python evaluators/evaluate_4_strategies_new.py --strategies 1 2

    # Top-K 설정
    python evaluators/evaluate_4_strategies_new.py --top-k 10
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
)
from retrievers.ensemble_retriever import get_ensemble_retriever
from retrievers.multiquery_retriever import get_multiquery_retriever
from retrievers.longcontext_retriever import get_longcontext_retriever
from retrievers.timeweighted_retriever import get_time_weighted_retriever
from utils.langfuse_utils import get_langfuse_client
from models.embeddings.factory import get_embedder

# 상수 정의
DEFAULT_DATASET_PATH = "/home/work/rag/Project/rag-report-generator/data/evaluation/merged_qa_dataset.json"
DEFAULT_NUM_CONTEXTS_FOR_ANSWER = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 500
DEFAULT_TOP_K = 10

SYSTEM_PROMPT_FILE = "prompts/templates/evaluation/system_prompt.txt"
ANSWER_PROMPT_FILE = "prompts/templates/evaluation/answer_generation_prompt.txt"


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


def generate_version_tag(retriever_name: str, version: str = "v1") -> str:
    """버전 태그 생성"""
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{retriever_name}_{date_str}_{version}"


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

    # metadata에 score가 있는 경우에만 계산
    scores = [m.get("score") for m in context_metadata if m.get("score") is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        langfuse.create_score(
            trace_id=trace_id,
            name="retrieval_quality",
            value=avg_score,
            comment=f"Average retrieval score from {len(scores)} contexts"
        )


def evaluate_single_query(
    retriever,
    retriever_name: str,
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

    version_tag = generate_version_tag(retriever_name, base_version)
    start_time = time.time()

    # 검색 수행 (invoke 메서드 사용)
    search_results = retriever.invoke(question)

    # LangChain Document를 contexts로 변환
    contexts = []
    context_metadata = []

    for result in search_results[:top_k]:
        contexts.append(result.page_content)
        context_metadata.append({
            "page_title": result.metadata.get('page_title', 'Unknown'),
            "section_title": result.metadata.get('section_title', 'N/A'),
            "chunk_id": result.metadata.get('chunk_id', 'unknown'),
            "score": result.metadata.get('_combined_score') or result.metadata.get('_similarity_score')
        })

    if not contexts:
        print(f"  ⚠️ [{idx}] No contexts found for question!")
        contexts = ["검색 결과가 없습니다."]

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
        retriever_name=retriever_name,
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
    retriever_name: str,
    eval_data: List[Dict[str, Any]],
    langfuse,
    top_k: int = DEFAULT_TOP_K,
    base_version: str = "v1",
    retriever_tags: List[str] = None
) -> Dict[str, Any]:
    """리트리버 평가"""
    print(f"\n{'=' * 60}")
    print(f"🔍 {retriever_name} 평가 중...")
    print(f"{'=' * 60}")

    stats = {
        "total_queries": len(eval_data),
        "total_time": 0,
        "evaluations": []
    }

    for idx, item in enumerate(eval_data, 1):
        eval_result = evaluate_single_query(
            retriever=retriever,
            retriever_name=retriever_name,
            item=item,
            langfuse=langfuse,
            idx=idx,
            top_k=top_k,
            base_version=base_version,
            retriever_tags=retriever_tags
        )

        stats["evaluations"].append(eval_result)
        stats["total_time"] += eval_result["time"]

        # 각 질문 평가 후 캐시 저장 (중단되어도 캐시 보존)
        try:
            embedder = get_embedder()
            if hasattr(embedder, 'use_cache') and embedder.use_cache and embedder.cache:
                embedder.cache.save()
        except:
            pass  # 캐시 저장 실패는 무시 (평가 진행에 영향 없음)

    stats["avg_time"] = stats["total_time"] / stats["total_queries"]
    stats["avg_contexts"] = sum(e["num_contexts"] for e in stats["evaluations"]) / stats["total_queries"]

    return stats


def create_strategy_retriever(strategy_num: int, top_k: int = 10):
    """
    전략별 리트리버 생성

    Returns:
        (retriever, retriever_name, retriever_tags)
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
        # TimeWeighted를 먼저 적용
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


def run_evaluation(
    strategy_num: int,
    dataset: str,
    top_k: int,
    version: str,
    langfuse
) -> bool:
    """단일 평가 전략 실행

    Args:
        strategy_num: 전략 번호
        dataset: 평가 데이터셋 경로
        top_k: Top-K 값
        version: 버전 태그
        langfuse: Langfuse 클라이언트

    Returns:
        성공 여부
    """
    STRATEGIES = {
        1: "RRF (Baseline)",
        2: "RRF + MultiQuery",
        3: "RRF + MultiQuery + LongContext",
        4: "RRF + LongContext + TimeWeighted"
    }

    strategy_name = STRATEGIES[strategy_num]

    print("\n" + "=" * 70)
    print(f"{strategy_num}️⃣  {strategy_name} 평가 중...")
    print("=" * 70)

    try:
        # 평가 데이터셋 로드
        eval_data = load_evaluation_dataset(dataset)

        # 리트리버 생성
        retriever, retriever_name, retriever_tags = create_strategy_retriever(strategy_num, top_k)

        print(f"✅ 리트리버 초기화 완료: {retriever_name}")
        print(f"   - 태그: {', '.join(retriever_tags)}")

        # 평가 수행
        stats = evaluate_retriever(
            retriever=retriever,
            retriever_name=retriever_name,
            eval_data=eval_data,
            langfuse=langfuse,
            top_k=top_k,
            base_version=version,
            retriever_tags=retriever_tags
        )

        # 결과 저장
        output_file = Path(dataset).parent / f"{retriever_name}_evaluation_stats.json"
        save_result = {k: v for k, v in stats.items() if k != "evaluations"}
        save_result["num_evaluations"] = len(stats.get("evaluations", []))
        save_result["config"] = {
            "strategy_num": strategy_num,
            "strategy_name": strategy_name,
            "retriever_name": retriever_name,
            "retriever_tags": retriever_tags,
            "top_k": top_k,
            "version": version,
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n✅ {strategy_name} 평가 완료")
        print(f"   - 평균 시간: {stats['avg_time']*1000:.2f}ms")
        print(f"   - 평균 컨텍스트 수: {stats['avg_contexts']:.2f}")
        print(f"   - 결과 저장: {output_file}")

        return True

    except Exception as e:
        print(f"❌ {strategy_name} 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="4가지 Retrieval 전략 성능 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--strategies",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help="평가할 전략 번호 (기본값: 모두)"
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
        help="Top-K 값 (기본값: 10)"
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="버전 태그 (기본값: v1)"
    )

    args = parser.parse_args()

    # 시작 메시지
    print("=" * 70)
    print("🚀 4가지 Retrieval 전략 성능 평가 시작")
    print("=" * 70)
    print(f"\n📊 평가 설정:")
    print(f"   - Dataset: {args.dataset}")
    print(f"   - Top-K: {args.top_k}")
    print(f"   - Version: {args.version}")
    print(f"   - 평가할 전략: {args.strategies}")

    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()
    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        return

    # 각 전략 평가
    results = {}
    for strategy_num in sorted(args.strategies):
        success = run_evaluation(
            strategy_num=strategy_num,
            dataset=args.dataset,
            top_k=args.top_k,
            version=args.version,
            langfuse=langfuse
        )
        results[strategy_num] = success

    # Langfuse flush
    print("\n⏳ Langfuse에 데이터 전송 중...")
    langfuse.flush()

    # 임베딩 캐시 저장
    print("\n💾 임베딩 캐시 저장 중...")
    try:
        embedder = get_embedder()
        if hasattr(embedder, 'save_cache'):
            embedder.save_cache()
        else:
            print("  ℹ️  캐시 저장 메서드 없음 (캐시 비활성화 상태)")
    except Exception as e:
        print(f"  ⚠️  캐시 저장 실패: {e}")

    # 결과 요약
    STRATEGY_NAMES = {
        1: "RRF (Baseline)",
        2: "RRF + MultiQuery",
        3: "RRF + MultiQuery + LongContext",
        4: "RRF + LongContext + TimeWeighted"
    }

    print("\n" + "=" * 70)
    print("📊 평가 결과 요약")
    print("=" * 70)

    for strategy_num in sorted(args.strategies):
        strategy_name = STRATEGY_NAMES[strategy_num]
        status = "✅ 성공" if results[strategy_num] else "❌ 실패"
        print(f"{strategy_num}. {strategy_name:<40} {status}")

    # 완료 메시지
    success_count = sum(results.values())
    total_count = len(results)

    print("\n" + "=" * 70)
    if success_count == total_count:
        print("✅ 모든 평가 완료!")
    else:
        print(f"⚠️  일부 평가 실패 ({success_count}/{total_count} 성공)")
    print("=" * 70)

    print("\n📊 다음 단계:")
    print("   1. Langfuse 대시보드에서 결과 확인: https://cloud.langfuse.com")
    print("   2. 각 전략별 stats 파일 확인:")
    print("      - data/evaluation/ensemble_rrf_evaluation_stats.json")
    print("      - data/evaluation/rrf_multiquery_evaluation_stats.json")
    print("      - data/evaluation/rrf_multiquery_longcontext_evaluation_stats.json")
    print("      - data/evaluation/rrf_longcontext_timeweighted_evaluation_stats.json")
    print()

    # 실패한 경우 종료 코드 1 반환
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
