#!/usr/bin/env python3
"""주간 보고서 vs 임원 보고서 평가 스크립트

각 보고서 타입에 최적화된 리트리버 조합을 평가합니다.

사용법:
    # 모든 평가 실행 (주간 + 임원)
    python evaluators/evaluate_report_types.py --report-type both

    # 주간 보고서만 평가
    python evaluators/evaluate_report_types.py --report-type weekly

    # 임원 보고서만 평가
    python evaluators/evaluate_report_types.py --report-type executive

    # 특정 리트리버 조합만 평가
    python evaluators/evaluate_report_types.py --report-type weekly --retrievers upstage_rrf_multiquery_lc
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import json
import time
import yaml
from datetime import datetime
from typing import List, Dict, Any

from utils.langfuse_utils import get_langfuse_client
from utils.common_utils import (
    load_prompt,
    load_evaluation_dataset,
    generate_llm_answer,
    add_retrieval_quality_score,
    save_embedding_cache
)
from utils.retriever_factory import create_retriever_from_config

# 상수 정의
DEFAULT_DATASET_PATH = "/home/work/rag/Project/rag-report-generator/data/evaluation/merged_qa_dataset.json"
DEFAULT_NUM_CONTEXTS_FOR_ANSWER = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TOP_K = 10


def load_evaluation_config() -> Dict[str, Any]:
    """평가 설정 파일 로드"""
    config_path = Path(__file__).parent.parent / "config" / "evaluation_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)




def evaluate_single_query(
    retriever,
    retriever_config: Dict[str, Any],
    report_type: str,
    item: Dict[str, Any],
    langfuse,
    idx: int,
    top_k: int,
    system_prompt: str,
    answer_generation_prompt: str,
    version_tag: str = "v1"
) -> Dict[str, Any]:
    """단일 쿼리 평가"""
    question = item["question"]
    ground_truth = item["ground_truth"]
    context_page_id = item.get("context_page_id")
    item_metadata = item.get("metadata", {})

    start_time = time.time()

    # 검색 수행
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
    answer = generate_llm_answer(
        question=question,
        contexts=contexts,
        system_prompt=system_prompt,
        answer_generation_prompt=answer_generation_prompt,
        num_contexts=DEFAULT_NUM_CONTEXTS_FOR_ANSWER,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS
    )

    if not answer or answer.startswith("답변 생성 실패") or answer.startswith("Azure OpenAI 설정"):
        print(f"  ⚠️ [{idx}] LLM answer generation failed!")
        if not answer:
            answer = "답변을 생성할 수 없습니다."

    total_time = time.time() - start_time

    # Langfuse Trace & Generation - common_utils 함수 사용
    retriever_name = retriever_config['name']
    retriever_tags = [
        retriever_config['embedding_preset'],
        retriever_config['retriever_type'],
        f"top_k_{top_k}",
        report_type
    ]

    additional_metadata = {
        "context_page_id": context_page_id,
        "retriever_name": retriever_name,
        "display_name": retriever_config['display_name'],
        "report_type": report_type,
        "top_k": top_k,
        "embedding_preset": retriever_config['embedding_preset'],
        "retriever_type": retriever_config['retriever_type']
    }

    from utils.common_utils import create_trace_and_generation as create_trace
    trace_id = create_trace(
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
    retriever_config: Dict[str, Any],
    report_type: str,
    eval_data: List[Dict[str, Any]],
    langfuse,
    top_k: int,
    system_prompt: str,
    answer_generation_prompt: str,
    version_tag: str = "v1"
) -> Dict[str, Any]:
    """리트리버 평가"""
    print(f"\n{'=' * 80}")
    print(f"🔍 {retriever_config['display_name']} - {report_type} 평가 중...")
    print(f"{'=' * 80}")

    stats = {
        "total_queries": len(eval_data),
        "total_time": 0,
        "evaluations": []
    }

    for idx, item in enumerate(eval_data, 1):
        eval_result = evaluate_single_query(
            retriever=retriever,
            retriever_config=retriever_config,
            report_type=report_type,
            item=item,
            langfuse=langfuse,
            idx=idx,
            top_k=top_k,
            system_prompt=system_prompt,
            answer_generation_prompt=answer_generation_prompt,
            version_tag=version_tag
        )

        stats["evaluations"].append(eval_result)
        stats["total_time"] += eval_result["time"]

        # 캐시 저장
        save_embedding_cache()

    stats["avg_time"] = stats["total_time"] / stats["total_queries"]
    stats["avg_contexts"] = sum(e["num_contexts"] for e in stats["evaluations"]) / stats["total_queries"]

    return stats


def run_report_evaluation(
    report_type: str,
    config: Dict[str, Any],
    dataset_path: str,
    top_k: int,
    version: str,
    langfuse,
    selected_retrievers: List[str] = None
) -> Dict[str, Any]:
    """특정 보고서 타입에 대한 평가 실행

    Args:
        report_type: 'weekly_report' or 'executive_report'
        config: 전체 설정 딕셔너리
        dataset_path: 평가 데이터셋 경로
        top_k: Top-K 값
        version: 버전 태그
        langfuse: Langfuse 클라이언트
        selected_retrievers: 평가할 리트리버 이름 리스트 (None이면 모두)

    Returns:
        평가 결과 딕셔너리
    """
    report_config = config[report_type]

    print("\n" + "=" * 80)
    print(f"📊 {report_config['name']} 평가 시작")
    print("=" * 80)
    print(f"우선순위: {' > '.join(report_config['priority'])}")

    # 프롬프트 로드
    system_prompt = load_prompt(report_config['system_prompt_path'])
    answer_generation_prompt = load_prompt(report_config['answer_generation_prompt_path'])

    # 데이터셋 로드
    eval_data = load_evaluation_dataset(dataset_path)
    print(f"📋 데이터셋: {len(eval_data)} 개 샘플")

    # 평가할 리트리버 필터링
    retrievers_to_eval = report_config['retrievers']
    if selected_retrievers:
        retrievers_to_eval = [
            r for r in retrievers_to_eval
            if r['name'] in selected_retrievers
        ]

    print(f"🔍 평가 대상: {len(retrievers_to_eval)}개 리트리버\n")

    results = {}

    for retriever_config in retrievers_to_eval:
        # Top-K 리스트 가져오기 (없으면 커맨드라인 top_k 사용)
        top_k_list = retriever_config.get('top_k_list', [top_k])

        for current_top_k in top_k_list:
            print(f"\n{'=' * 80}")
            print(f"🚀 {retriever_config['display_name']}")
            print(f"   임베딩: {retriever_config['embedding_preset']}")
            print(f"   리트리버: {retriever_config['retriever_type']}")
            print(f"   Top-K: {current_top_k}")
            print(f"   {retriever_config['description']}")
            print(f"{'=' * 80}")

            try:
                # 리트리버 생성
                retriever, retriever_tags = create_retriever_from_config(retriever_config, current_top_k)

                # 평가 수행
                stats = evaluate_retriever(
                    retriever=retriever,
                    retriever_config=retriever_config,
                    report_type=report_type,
                    eval_data=eval_data,
                    langfuse=langfuse,
                    top_k=current_top_k,
                    system_prompt=system_prompt,
                    answer_generation_prompt=answer_generation_prompt,
                    version_tag=version
                )

                # 결과 저장
                output_dir = Path(config['evaluation']['output_dir']) / report_type
                output_dir.mkdir(parents=True, exist_ok=True)

                # 파일명에 top-k 포함
                output_file = output_dir / f"{retriever_config['name']}_k{current_top_k}_stats.json"
                save_result = {k: v for k, v in stats.items() if k != "evaluations"}
                save_result["num_evaluations"] = len(stats.get("evaluations", []))
                save_result["config"] = {
                    "retriever_name": retriever_config['name'],
                    "display_name": retriever_config['display_name'],
                    "report_type": report_type,
                    "embedding_preset": retriever_config['embedding_preset'],
                    "retriever_type": retriever_config['retriever_type'],
                    "top_k": current_top_k,
                    "version": version,
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(save_result, f, indent=2, ensure_ascii=False, default=str)

                print(f"\n✅ 평가 완료 (Top-K={current_top_k})")
                print(f"   - 평균 시간: {stats['avg_time']*1000:.2f}ms")
                print(f"   - 평균 컨텍스트 수: {stats['avg_contexts']:.2f}")
                print(f"   - 결과 저장: {output_file}")

                result_key = f"{retriever_config['name']}_k{current_top_k}"
                results[result_key] = {
                    "success": True,
                    "stats": stats,
                    "output_file": str(output_file),
                    "top_k": current_top_k
                }

            except Exception as e:
                print(f"❌ 평가 실패 (Top-K={current_top_k}): {e}")
                import traceback
                traceback.print_exc()

                result_key = f"{retriever_config['name']}_k{current_top_k}"
                results[result_key] = {
                    "success": False,
                    "error": str(e),
                    "top_k": current_top_k
                }

    return results


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="주간 보고서 vs 임원 보고서 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--report-type",
        type=str,
        choices=["weekly", "executive", "both"],
        default="both",
        help="평가할 보고서 타입 (기본값: both)"
    )

    parser.add_argument(
        "--retrievers",
        type=str,
        nargs="+",
        help="평가할 리트리버 이름 (미지정 시 모두 평가)"
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

    # 설정 로드
    config = load_evaluation_config()

    print("=" * 80)
    print("🎯 주간 보고서 vs 임원 보고서 평가 시스템")
    print("=" * 80)
    print(f"\n📊 평가 설정:")
    print(f"   - 보고서 타입: {args.report_type}")
    print(f"   - Dataset: {args.dataset}")
    print(f"   - Top-K: {args.top_k}")
    print(f"   - Version: {args.version}")
    if args.retrievers:
        print(f"   - 선택된 리트리버: {', '.join(args.retrievers)}")

    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()
    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        return

    # 평가 실행
    all_results = {}

    if args.report_type in ["weekly", "both"]:
        weekly_results = run_report_evaluation(
            report_type="weekly_report",
            config=config,
            dataset_path=args.dataset,
            top_k=args.top_k,
            version=args.version,
            langfuse=langfuse,
            selected_retrievers=args.retrievers
        )
        all_results["weekly_report"] = weekly_results

    if args.report_type in ["executive", "both"]:
        exec_results = run_report_evaluation(
            report_type="executive_report",
            config=config,
            dataset_path=args.dataset,
            top_k=args.top_k,
            version=args.version,
            langfuse=langfuse,
            selected_retrievers=args.retrievers
        )
        all_results["executive_report"] = exec_results

    # Langfuse flush
    print("\n⏳ Langfuse에 데이터 전송 중...")
    langfuse.flush()

    # 임베딩 캐시 저장
    print("\n💾 임베딩 캐시 저장 중...")
    save_embedding_cache()

    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 평가 결과 요약")
    print("=" * 80)

    for report_type, results in all_results.items():
        report_name = "주간 보고서 (운영팀)" if report_type == "weekly_report" else "임원 보고서 (의사결정)"
        print(f"\n[{report_name}]")

        for retriever_name, result in results.items():
            status = "✅ 성공" if result['success'] else "❌ 실패"
            print(f"  {retriever_name}: {status}")

    print("\n" + "=" * 80)
    print("✅ 모든 평가 완료!")
    print("=" * 80)

    print("\n📊 다음 단계:")
    print("   1. Langfuse 대시보드에서 결과 확인: https://cloud.langfuse.com")
    print("   2. 결과 파일 확인:")
    print(f"      - {config['evaluation']['output_dir']}/weekly_report/")
    print(f"      - {config['evaluation']['output_dir']}/executive_report/")
    print()


if __name__ == "__main__":
    main()
