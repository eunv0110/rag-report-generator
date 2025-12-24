#!/usr/bin/env python3
"""평가 결과 비교 (CSV 파일 기반)

두 개의 Langfuse CSV 평가 결과 파일을 로드하여
메트릭 별로 성능을 비교하는 스크립트
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
import json


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    CSV 파일 로드

    Args:
        csv_path: CSV 파일 경로

    Returns:
        pandas DataFrame
    """
    print(f"📂 파일 로딩 중: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ✅ {len(df)} 행 로드됨")
    return df


def extract_metrics_by_trace(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Trace별로 메트릭 추출

    Args:
        df: Langfuse 데이터프레임

    Returns:
        {trace_id: {metric_name: value}} 딕셔너리
    """
    trace_metrics = defaultdict(dict)

    for _, row in df.iterrows():
        trace_id = row['traceId']
        metric_name = row['name']

        # value가 숫자형인 경우만 처리
        if pd.notna(row['value']):
            try:
                metric_value = float(row['value'])
                trace_metrics[trace_id][metric_name] = metric_value
            except (ValueError, TypeError):
                pass

    return dict(trace_metrics)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    통계 계산

    Args:
        values: 값 리스트

    Returns:
        통계 딕셔너리 (평균, 최소, 최대, 중앙값, 표준편차)
    """
    if not values:
        return {
            "count": 0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "std": 0.0
        }

    return {
        "count": len(values),
        "avg": np.mean(values),
        "min": np.min(values),
        "max": np.max(values),
        "median": np.median(values),
        "std": np.std(values)
    }


def analyze_single_file(df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
    """
    단일 CSV 파일 분석

    Args:
        df: 데이터프레임
        file_name: 파일 이름

    Returns:
        분석 결과 딕셔너리
    """
    print(f"\n{'=' * 60}")
    print(f"📊 {file_name} 분석 중...")
    print(f"{'=' * 60}")

    # Trace별 메트릭 추출
    trace_metrics = extract_metrics_by_trace(df)

    # 메트릭별 통계 계산
    metrics_summary = defaultdict(list)

    for trace_id, metrics in trace_metrics.items():
        for metric_name, value in metrics.items():
            metrics_summary[metric_name].append(value)

    # 통계 계산
    stats = {}
    for metric_name, values in metrics_summary.items():
        stats[metric_name] = calculate_statistics(values)

    # 결과 출력
    print(f"\n📈 총 Trace 수: {len(trace_metrics)}")
    print(f"📈 총 평가 항목 수: {len(df)}")

    if stats:
        print(f"\n메트릭 통계:")
        for metric_name, metric_stats in sorted(stats.items()):
            print(f"\n   {metric_name}:")
            print(f"      개수: {metric_stats['count']}")
            print(f"      평균: {metric_stats['avg']:.4f}")
            print(f"      중앙값: {metric_stats['median']:.4f}")
            print(f"      표준편차: {metric_stats['std']:.4f}")
            print(f"      범위: {metric_stats['min']:.4f} ~ {metric_stats['max']:.4f}")

    return {
        "file_name": file_name,
        "total_traces": len(trace_metrics),
        "total_evaluations": len(df),
        "metrics": stats,
        "trace_metrics": trace_metrics
    }


def compare_two_files(
    result1: Dict[str, Any],
    result2: Dict[str, Any]
) -> None:
    """
    두 파일의 결과 비교 및 출력

    Args:
        result1: 첫 번째 파일 분석 결과
        result2: 두 번째 파일 분석 결과
    """
    print("\n" + "=" * 80)
    print("🏆 두 파일 비교 결과")
    print("=" * 80)

    # 기본 통계 비교
    print(f"\n{'항목':<30} {result1['file_name']:<25} {result2['file_name']:<25}")
    print("-" * 80)
    print(f"{'총 Trace 수':<30} {result1['total_traces']:<25} {result2['total_traces']:<25}")
    print(f"{'총 평가 항목 수':<30} {result1['total_evaluations']:<25} {result2['total_evaluations']:<25}")

    # 메트릭별 비교
    all_metrics = set(result1["metrics"].keys()) | set(result2["metrics"].keys())

    if all_metrics:
        print("\n" + "=" * 80)
        print("📊 메트릭별 비교")
        print("=" * 80)

        for metric_name in sorted(all_metrics):
            print(f"\n[{metric_name}]")
            print(f"{'통계':<20} {result1['file_name']:<25} {result2['file_name']:<25} {'차이':<15}")
            print("-" * 85)

            stats1 = result1["metrics"].get(metric_name, {})
            stats2 = result2["metrics"].get(metric_name, {})

            if stats1 and stats2:
                # 개수
                print(f"{'개수':<20} {stats1.get('count', 0):<25} {stats2.get('count', 0):<25} {stats2.get('count', 0) - stats1.get('count', 0):<15}")

                # 평균
                avg1 = stats1.get('avg', 0)
                avg2 = stats2.get('avg', 0)
                diff = avg2 - avg1
                diff_pct = (diff / avg1 * 100) if avg1 != 0 else 0
                print(f"{'평균':<20} {avg1:<25.4f} {avg2:<25.4f} {diff:+.4f} ({diff_pct:+.2f}%)")

                # 중앙값
                med1 = stats1.get('median', 0)
                med2 = stats2.get('median', 0)
                diff = med2 - med1
                print(f"{'중앙값':<20} {med1:<25.4f} {med2:<25.4f} {diff:+.4f}")

                # 표준편차
                std1 = stats1.get('std', 0)
                std2 = stats2.get('std', 0)
                diff = std2 - std1
                print(f"{'표준편차':<20} {std1:<25.4f} {std2:<25.4f} {diff:+.4f}")

                # 최소/최대
                print(f"{'최소값':<20} {stats1.get('min', 0):<25.4f} {stats2.get('min', 0):<25.4f}")
                print(f"{'최대값':<20} {stats1.get('max', 0):<25.4f} {stats2.get('max', 0):<25.4f}")
            else:
                if stats1:
                    print(f"   ⚠️  {result2['file_name']}에 '{metric_name}' 메트릭 없음")
                else:
                    print(f"   ⚠️  {result1['file_name']}에 '{metric_name}' 메트릭 없음")

    # 공통 Trace 분석
    traces1 = set(result1["trace_metrics"].keys())
    traces2 = set(result2["trace_metrics"].keys())

    common_traces = traces1 & traces2
    only_in_1 = traces1 - traces2
    only_in_2 = traces2 - traces1

    print("\n" + "=" * 80)
    print("🔍 Trace 비교")
    print("=" * 80)
    print(f"공통 Trace 수: {len(common_traces)}")
    print(f"{result1['file_name']}에만 있는 Trace: {len(only_in_1)}")
    print(f"{result2['file_name']}에만 있는 Trace: {len(only_in_2)}")

    # 공통 Trace에 대한 메트릭 차이 분석
    if common_traces and all_metrics:
        print("\n" + "=" * 80)
        print("📈 공통 Trace 메트릭 개선/저하 분석")
        print("=" * 80)

        for metric_name in sorted(all_metrics):
            improvements = 0
            degradations = 0
            unchanged = 0

            for trace_id in common_traces:
                val1 = result1["trace_metrics"][trace_id].get(metric_name)
                val2 = result2["trace_metrics"][trace_id].get(metric_name)

                if val1 is not None and val2 is not None:
                    if val2 > val1:
                        improvements += 1
                    elif val2 < val1:
                        degradations += 1
                    else:
                        unchanged += 1

            if improvements + degradations + unchanged > 0:
                print(f"\n{metric_name}:")
                print(f"   개선: {improvements} ({improvements/len(common_traces)*100:.1f}%)")
                print(f"   저하: {degradations} ({degradations/len(common_traces)*100:.1f}%)")
                print(f"   동일: {unchanged} ({unchanged/len(common_traces)*100:.1f}%)")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="두 개의 Langfuse CSV 평가 결과 파일 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file1",
        type=str,
        help="첫 번째 CSV 파일 경로"
    )
    parser.add_argument(
        "file2",
        type=str,
        help="두 번째 CSV 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 경로 (JSON 형식)"
    )

    args = parser.parse_args()

    # CSV 파일 로드
    df1 = load_csv_data(args.file1)
    df2 = load_csv_data(args.file2)

    # 각 파일 분석
    result1 = analyze_single_file(df1, Path(args.file1).name)
    result2 = analyze_single_file(df2, Path(args.file2).name)

    # 두 파일 비교
    compare_two_files(result1, result2)

    # 결과 저장 (선택적)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "file1": {
                "name": result1["file_name"],
                "total_traces": result1["total_traces"],
                "total_evaluations": result1["total_evaluations"],
                "metrics": result1["metrics"]
            },
            "file2": {
                "name": result2["file_name"],
                "total_traces": result2["total_traces"],
                "total_evaluations": result2["total_evaluations"],
                "metrics": result2["metrics"]
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
