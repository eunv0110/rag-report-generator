#!/usr/bin/env python3
"""리트리버 평가 결과 비교 (Langfuse 태그 기반)

Langfuse에 저장된 평가 결과를 태그로 필터링하여
여러 리트리버의 성능을 비교하는 스크립트
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
from collections import defaultdict
import json
from utils.langfuse_utils import get_langfuse_client


def fetch_traces_by_tag(langfuse, tag: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    특정 태그로 traces 가져오기

    Args:
        langfuse: Langfuse 클라이언트
        tag: 필터링할 태그
        limit: 최대 가져올 trace 수

    Returns:
        trace 리스트
    """
    print(f"🔍 태그 '{tag}'로 traces 가져오는 중...")

    try:
        traces = langfuse.fetch_traces(
            tags=[tag],
            limit=limit
        )

        trace_list = list(traces.data)
        print(f"   ✅ {len(trace_list)}개 traces 가져옴")
        return trace_list

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return []


def get_trace_scores(langfuse, trace_id: str) -> Dict[str, float]:
    """
    특정 trace의 모든 scores 가져오기

    Args:
        langfuse: Langfuse 클라이언트
        trace_id: trace ID

    Returns:
        {score_name: score_value} 딕셔너리
    """
    try:
        trace = langfuse.fetch_trace(trace_id)
        scores = {}

        if hasattr(trace, 'scores') and trace.scores:
            for score in trace.scores:
                scores[score.name] = score.value

        return scores

    except Exception as e:
        print(f"   ⚠️  Trace {trace_id} scores 가져오기 실패: {e}")
        return {}


def analyze_retriever_performance(
    langfuse,
    version_tag: str = "3",
    retriever_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    각 리트리버의 성능 분석

    Args:
        langfuse: Langfuse 클라이언트
        version_tag: 버전 태그 (예: "v2")
        retriever_names: 분석할 리트리버 이름 리스트 (None이면 자동 탐지)

    Returns:
        리트리버별 성능 통계
    """
    if retriever_names is None:
        retriever_names = ["BM25_Basic", "Dense_Vector", "rrf_ensemble"]

    results = {}

    for retriever_name in retriever_names:
        tag = f"{retriever_name}_{version_tag}"
        print(f"\n{'=' * 60}")
        print(f"📊 {retriever_name} 분석 중...")
        print(f"{'=' * 60}")

        # 1. 해당 리트리버의 모든 traces 가져오기
        traces = fetch_traces_by_tag(langfuse, tag)

        if not traces:
            print(f"   ⚠️  '{tag}' 태그로 traces를 찾을 수 없습니다.")
            continue

        # 2. 각 trace의 메타데이터 및 scores 수집
        trace_data = []
        scores_summary = defaultdict(list)

        for trace in traces:
            trace_info = {
                "trace_id": trace.id,
                "name": trace.name,
                "timestamp": trace.timestamp,
            }

            # 메타데이터 추출
            if hasattr(trace, 'metadata') and trace.metadata:
                trace_info["metadata"] = trace.metadata
                trace_info["total_time_ms"] = trace.metadata.get("total_time_ms", 0)
                trace_info["num_contexts"] = trace.metadata.get("num_retrieved_contexts", 0)

            # Scores 가져오기
            scores = get_trace_scores(langfuse, trace.id)
            trace_info["scores"] = scores

            # 통계를 위한 scores 수집
            for score_name, score_value in scores.items():
                scores_summary[score_name].append(score_value)

            trace_data.append(trace_info)

        # 3. 통계 계산
        stats = {
            "retriever": retriever_name,
            "version": version_tag,
            "total_traces": len(traces),
            "avg_time_ms": sum(t.get("total_time_ms", 0) for t in trace_data) / len(trace_data) if trace_data else 0,
            "avg_contexts": sum(t.get("num_contexts", 0) for t in trace_data) / len(trace_data) if trace_data else 0,
            "scores": {}
        }

        # Scores 평균 계산
        for score_name, values in scores_summary.items():
            if values:
                stats["scores"][score_name] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        results[retriever_name] = {
            "stats": stats,
            "traces": trace_data
        }

        # 결과 출력
        print(f"\n📈 {retriever_name} 통계:")
        print(f"   Total traces: {stats['total_traces']}")
        print(f"   Avg time: {stats['avg_time_ms']:.2f} ms")
        print(f"   Avg contexts: {stats['avg_contexts']:.2f}")

        if stats["scores"]:
            print(f"\n   Scores:")
            for score_name, score_stats in stats["scores"].items():
                print(f"      {score_name}:")
                print(f"         평균: {score_stats['avg']:.4f}")
                print(f"         범위: {score_stats['min']:.4f} ~ {score_stats['max']:.4f}")
                print(f"         개수: {score_stats['count']}")

    return results


def compare_retrievers(
    langfuse,
    version_tag: str = "v3",
    retriever_names: Optional[List[str]] = None
):
    """리트리버 성능 비교 및 출력"""
    results = analyze_retriever_performance(langfuse, version_tag, retriever_names)

    if not results:
        print("\n❌ 분석할 데이터가 없습니다.")
        return

    # 비교 테이블 출력
    print("\n" + "=" * 80)
    print("🏆 리트리버 성능 비교")
    print("=" * 80)

    # 헤더
    print(f"\n{'Retriever':<20} {'Traces':<10} {'Avg Time (ms)':<15} {'Avg Contexts':<15}")
    print("-" * 70)

    # 각 리트리버 통계
    for retriever_name, data in results.items():
        stats = data["stats"]
        print(
            f"{retriever_name:<20} "
            f"{stats['total_traces']:<10} "
            f"{stats['avg_time_ms']:<15.2f} "
            f"{stats['avg_contexts']:<15.2f}"
        )

    # Scores 비교 (있는 경우)
    all_score_names = set()
    for data in results.values():
        all_score_names.update(data["stats"]["scores"].keys())

    if all_score_names:
        print("\n" + "=" * 80)
        print("📊 Scores 비교")
        print("=" * 80)

        for score_name in sorted(all_score_names):
            print(f"\n{score_name}:")
            print(f"{'Retriever':<20} {'평균':<15} {'최소':<15} {'최대':<15}")
            print("-" * 70)

            for retriever_name, data in results.items():
                if score_name in data["stats"]["scores"]:
                    score_stats = data["stats"]["scores"][score_name]
                    print(
                        f"{retriever_name:<20} "
                        f"{score_stats['avg']:<15.4f} "
                        f"{score_stats['min']:<15.4f} "
                        f"{score_stats['max']:<15.4f}"
                    )

    # 결과 저장
    output_file = Path(__file__).parent.parent / "data" / "evaluation" / f"retriever_comparison_{version_tag}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # traces 데이터는 제외하고 stats만 저장
    save_data = {
        retriever: data["stats"]
        for retriever, data in results.items()
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 결과 저장: {output_file}")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Langfuse에서 태그로 리트리버 성능 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        help="비교할 버전 태그 (기본값: v2)"
    )
    parser.add_argument(
        "--retrievers",
        type=str,
        nargs="+",
        default=None,
        help="비교할 리트리버 이름 (기본값: BM25_Basic Dense_Vector rrf_ensemble)"
    )

    args = parser.parse_args()

    # Langfuse 클라이언트 초기화
    print("🔧 Langfuse 클라이언트 초기화 중...")
    langfuse = get_langfuse_client()

    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        print("   .env 파일에서 LANGFUSE_PUBLIC_KEY와 LANGFUSE_SECRET_KEY를 확인하세요.")
        return

    print("✅ Langfuse 연결 성공\n")

    # 리트리버 비교
    compare_retrievers(
        langfuse=langfuse,
        version_tag=args.version,
        retriever_names=args.retrievers
    )


if __name__ == "__main__":
    main()
