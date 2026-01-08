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

from dotenv import load_dotenv
load_dotenv()

import json
import time
from datetime import datetime
from typing import List, Dict, Any

from utils.langfuse import get_langfuse_client
from models.embeddings.factory import get_embedder
from utils.common import (
    load_prompt,
    load_evaluation_dataset,
    generate_llm_answer,
    create_trace_and_generation,
    add_retrieval_quality_score,
    save_embedding_cache
)
from utils.retriever_factory import create_strategy_retriever

# 상수 정의
DEFAULT_DATASET_PATH = "/home/work/rag/Project/rag-report-generator/data/evaluation/merged_qa_dataset.json"
DEFAULT_NUM_CONTEXTS_FOR_ANSWER = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 500
DEFAULT_TOP_K = 10


def generate_version_tag(retriever_name: str, version: str = "v1") -> str:
    """버전 태그 생성"""
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{retriever_name}_{date_str}_{version}"


def evaluate_single_query(
    retriever,
    retriever_name: str,
    item: Dict[str, Any],
    langfuse,
    idx: int,
    top_k: int,
    base_version: str = "v1",
    retriever_tags: List[str] = None,
    prompt_version: int = 5
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

    # 답변 생성 프롬프트 로드
    answer_prompt_file = f"prompts/templates/evaluation/weekly_report/answer_generation_prompt_ver{prompt_version}.txt"
    answer_prompt_template = load_prompt(answer_prompt_file)
    system_prompt = ""  # 시스템 프롬프트는 answer_generation_prompt에 포함되어 있음

    # LLM 답변 생성
    answer = generate_llm_answer(
        question=question,
        contexts=contexts,
        system_prompt=system_prompt,
        answer_generation_prompt=answer_prompt_template,
        num_contexts=DEFAULT_NUM_CONTEXTS_FOR_ANSWER,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS
    )

    if not answer or answer.startswith("답변 생성 실패") or answer.startswith("Azure OpenAI 설정"):
        print(f"  ⚠️ [{idx}] LLM answer generation failed!")
        if not answer:
            answer = "답변을 생성할 수 없습니다."

    total_time = time.time() - start_time

    # Langfuse Trace & Generation
    additional_metadata = {"context_page_id": context_page_id}
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
        version_tag=version_tag,
        retriever_tags=retriever_tags,
        additional_metadata=additional_metadata
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
    retriever_tags: List[str] = None,
    prompt_version: int = 5
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
            retriever_tags=retriever_tags,
            prompt_version=prompt_version
        )

        stats["evaluations"].append(eval_result)
        stats["total_time"] += eval_result["time"]

        # 각 질문 평가 후 캐시 저장
        save_embedding_cache()

    stats["avg_time"] = stats["total_time"] / stats["total_queries"]
    stats["avg_contexts"] = sum(e["num_contexts"] for e in stats["evaluations"]) / stats["total_queries"]

    return stats




def run_evaluation(
    strategy_num: int,
    dataset: str,
    top_k: int,
    version: str,
    langfuse,
    prompt_version: int = 5
) -> bool:
    """단일 평가 전략 실행

    Args:
        strategy_num: 전략 번호
        dataset: 평가 데이터셋 경로
        top_k: Top-K 값
        version: 버전 태그
        langfuse: Langfuse 클라이언트
        prompt_version: 프롬프트 버전 (2-6)

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
            retriever_tags=retriever_tags,
            prompt_version=prompt_version
        )

        # 결과 저장
        output_file = Path(dataset).parent / f"{retriever_name}_prompt_v{prompt_version}_evaluation_stats.json"
        save_result = {k: v for k, v in stats.items() if k != "evaluations"}
        save_result["num_evaluations"] = len(stats.get("evaluations", []))
        save_result["config"] = {
            "strategy_num": strategy_num,
            "strategy_name": strategy_name,
            "retriever_name": retriever_name,
            "retriever_tags": retriever_tags,
            "top_k": top_k,
            "version": version,
            "prompt_version": prompt_version,
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

    parser.add_argument(
        "--prompt-version",
        type=int,
        choices=[2, 3, 4, 5, 6, 7],
        default=5,
        help="프롬프트 버전 (2-7, 기본값: 5)"
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
    print(f"   - Prompt Version: {args.prompt_version}")
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
            langfuse=langfuse,
            prompt_version=args.prompt_version
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
