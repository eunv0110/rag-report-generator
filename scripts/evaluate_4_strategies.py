#!/usr/bin/env python3
"""4가지 Retrieval 전략 성능 평가 스크립트

평가 전략:
  1. RRF (Baseline)
  2. RRF + MultiQuery
  3. RRF + MultiQuery + LongContext
  4. RRF + LongContext + TimeWeighted

사용법:
    python scripts/evaluate_4_strategies.py

    # 특정 전략만 평가
    python scripts/evaluate_4_strategies.py --strategies 1 2 3

    # Top-K 설정
    python scripts/evaluate_4_strategies.py --top-k 15
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import argparse
from typing import List

# 평가 전략 정의
STRATEGIES = {
    1: {
        "name": "RRF (Baseline)",
        "cmd": [
            "python", "scripts/evaluate_retriever.py",
            "--retriever", "ensemble_rrf"
        ]
    },
    2: {
        "name": "RRF + MultiQuery",
        "cmd": [
            "python", "scripts/evaluate_retriever.py",
            "--retriever", "multiquery",
            "--base-retriever", "ensemble_rrf",
            "--num-queries", "3"
        ]
    },
    3: {
        "name": "RRF + MultiQuery + LongContext",
        "cmd": [
            "python", "scripts/evaluate_retriever.py",
            "--retriever", "multiquery",
            "--base-retriever", "ensemble_rrf_longcontext",
            "--num-queries", "3"
        ]
    },
    4: {
        "name": "RRF + LongContext + TimeWeighted",
        "cmd": [
            "python", "scripts/evaluate_retriever.py",
            "--retriever", "ensemble_rrf_timeweighted_longcontext",
            "--decay-rate", "0.01"
        ]
    }
}


def run_evaluation(
    strategy_num: int,
    dataset: str,
    top_k: int,
    version: str
) -> bool:
    """단일 평가 전략 실행

    Args:
        strategy_num: 전략 번호
        dataset: 평가 데이터셋 경로
        top_k: Top-K 값
        version: 버전 태그

    Returns:
        성공 여부
    """
    strategy = STRATEGIES[strategy_num]

    print("\n" + "=" * 70)
    print(f"{strategy_num}️⃣  {strategy['name']} 평가 중...")
    print("=" * 70)

    # 명령어 구성
    cmd = strategy["cmd"] + [
        "--dataset", dataset,
        "--top-k", str(top_k),
        "--version", version
    ]

    try:
        # 평가 실행
        result = subprocess.run(cmd, check=True)
        print(f"✅ {strategy['name']} 평가 완료")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ {strategy['name']} 평가 실패: {e}")
        return False


def main():
    """메인 함수"""
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
        default="/home/work/rag/Project/rag-report-generator/data/evaluation/merged_qa_dataset.json",
        help="평가 데이터셋 경로"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
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

    # 각 전략 평가
    results = {}
    for strategy_num in sorted(args.strategies):
        success = run_evaluation(
            strategy_num=strategy_num,
            dataset=args.dataset,
            top_k=args.top_k,
            version=args.version
        )
        results[strategy_num] = success

    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 평가 결과 요약")
    print("=" * 70)

    for strategy_num in sorted(args.strategies):
        strategy_name = STRATEGIES[strategy_num]["name"]
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
    print("      - data/evaluation/multiquery_ensemble_rrf_evaluation_stats.json")
    print("      - data/evaluation/multiquery_ensemble_rrf_longcontext_evaluation_stats.json")
    print("      - data/evaluation/ensemble_rrf_timeweighted_longcontext_0.01_evaluation_stats.json")
    print("\n   3. 비교 분석을 위해 Langfuse에서 CSV export 후:")
    print("      python scripts/compare_evaluation_results.py")
    print()

    # 실패한 경우 종료 코드 1 반환
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
