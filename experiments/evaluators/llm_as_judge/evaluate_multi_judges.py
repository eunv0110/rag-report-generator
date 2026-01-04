#!/usr/bin/env python3
"""여러 Judge 모델로 병렬 평가 수행

여러 Judge LLM을 사용하여 동시에 평가를 수행하고 결과를 비교합니다.
"""

import sys
from pathlib import Path

# Python 경로 설정을 가장 먼저 수행
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from evaluate_combinations import evaluate_all_questions


def evaluate_with_single_judge(
    combinations_file: str,
    judge_model: str,
    provider: str,
    report_type: str,
    output_base_dir: str
) -> Dict[str, Any]:
    """단일 Judge 모델로 평가 수행 (병렬 실행용)

    Args:
        combinations_file: all_combinations.json 파일 경로
        judge_model: Judge 모델 이름
        provider: LLM 제공자
        report_type: 보고서 타입
        output_base_dir: 결과 저장 기본 디렉토리

    Returns:
        평가 결과
    """
    # 각 프로세스에서 Python 경로 재설정
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(Path(__file__).parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent.parent))

    # Judge 모델 이름을 파일명에 안전하게 사용
    safe_model_name = judge_model.replace('/', '_').replace(':', '_')
    output_dir = Path(output_base_dir) / f"evaluation_{safe_model_name}"

    print(f"\n{'='*100}")
    print(f"🚀 Judge 모델 '{judge_model}' 평가 시작")
    print(f"{'='*100}")

    try:
        result = evaluate_all_questions(
            combinations_file=combinations_file,
            judge_model=judge_model,
            provider=provider,
            report_type=report_type,
            output_dir=str(output_dir)
        )

        print(f"\n✅ Judge 모델 '{judge_model}' 평가 완료!")
        return {
            "judge_model": judge_model,
            "provider": provider,
            "success": True,
            "result": result,
            "output_dir": str(output_dir)
        }

    except Exception as e:
        print(f"\n❌ Judge 모델 '{judge_model}' 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            "judge_model": judge_model,
            "provider": provider,
            "success": False,
            "error": str(e),
            "output_dir": str(output_dir)
        }


def evaluate_with_multiple_judges(
    combinations_file: str,
    judge_configs: List[Dict[str, str]],
    report_type: str,
    output_base_dir: str,
    max_workers: int = 3
) -> Dict[str, Any]:
    """여러 Judge 모델로 병렬 평가 수행

    Args:
        combinations_file: all_combinations.json 파일 경로
        judge_configs: Judge 설정 리스트 [{"model": "gpt-5.1", "provider": "azure_ai"}, ...]
        report_type: 보고서 타입
        output_base_dir: 결과 저장 기본 디렉토리
        max_workers: 최대 동시 실행 프로세스 수 (기본: 3)

    Returns:
        전체 평가 결과
    """
    print(f"\n{'='*100}")
    print(f"🔥 다중 Judge 모델 병렬 평가 시작")
    print(f"{'='*100}")
    print(f"📊 평가할 Judge 모델 수: {len(judge_configs)}")
    for config in judge_configs:
        print(f"  - {config['model']} ({config['provider']})")
    print(f"⚡ 최대 동시 실행 수: {max_workers}")

    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 병렬 실행
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 모든 Judge 모델에 대해 평가 작업 제출
        future_to_config = {}
        for config in judge_configs:
            future = executor.submit(
                evaluate_with_single_judge,
                combinations_file=combinations_file,
                judge_model=config['model'],
                provider=config['provider'],
                report_type=report_type,
                output_base_dir=output_base_dir
            )
            future_to_config[future] = config

        # 완료된 작업 결과 수집
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"\n❌ {config['model']} 평가 중 예외 발생: {e}")
                all_results.append({
                    "judge_model": config['model'],
                    "provider": config['provider'],
                    "success": False,
                    "error": str(e)
                })

    # 결과 요약
    print(f"\n{'='*100}")
    print(f"📊 전체 평가 결과 요약")
    print(f"{'='*100}")

    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]

    print(f"✅ 성공: {len(successful_results)}/{len(all_results)}")
    print(f"❌ 실패: {len(failed_results)}/{len(all_results)}")

    if failed_results:
        print("\n실패한 Judge 모델:")
        for r in failed_results:
            print(f"  - {r['judge_model']}: {r.get('error', 'Unknown error')}")

    # 전체 결과 저장
    summary_result = {
        "combinations_file": combinations_file,
        "report_type": report_type,
        "num_judges": len(judge_configs),
        "num_successful": len(successful_results),
        "num_failed": len(failed_results),
        "judge_configs": judge_configs,
        "results": all_results,
        "timestamp": datetime.now().isoformat()
    }

    summary_file = output_base_path / "multi_judge_evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)

    print(f"\n💾 전체 결과 저장: {summary_file}")

    # Judge 간 비교 분석
    if len(successful_results) > 1:
        compare_judges(successful_results, output_base_path, report_type)

    return summary_result


def compare_judges(
    judge_results: List[Dict[str, Any]],
    output_dir: Path,
    report_type: str
):
    """여러 Judge 모델의 평가 결과 비교

    Args:
        judge_results: 성공한 Judge 평가 결과 리스트
        output_dir: 결과 저장 디렉토리
        report_type: 보고서 타입
    """
    print(f"\n{'='*100}")
    print(f"🔍 Judge 모델 간 비교 분석")
    print(f"{'='*100}")

    # LLM별로 각 Judge가 준 점수 수집
    llm_scores_by_judge = {}

    for judge_result in judge_results:
        judge_model = judge_result['judge_model']
        evaluations = judge_result['result']['evaluations']

        for evaluation in evaluations:
            question_id = evaluation['question_id']

            for llm_name, llm_result in evaluation['results'].items():
                key = (llm_name, question_id)

                if key not in llm_scores_by_judge:
                    llm_scores_by_judge[key] = {}

                llm_scores_by_judge[key][judge_model] = llm_result['final_score']

    # 데이터프레임 생성
    rows = []
    for (llm_name, question_id), judge_scores in llm_scores_by_judge.items():
        row = {
            'llm_name': llm_name,
            'question_id': question_id,
        }

        # 각 Judge의 점수 추가
        for judge_model, score in judge_scores.items():
            safe_judge_name = judge_model.replace('/', '_').replace(':', '_')
            row[safe_judge_name] = score

        # 평균, 표준편차, 범위 계산
        scores = list(judge_scores.values())
        row['avg_score'] = sum(scores) / len(scores)
        row['std_score'] = pd.Series(scores).std()
        row['min_score'] = min(scores)
        row['max_score'] = max(scores)
        row['score_range'] = max(scores) - min(scores)

        rows.append(row)

    df = pd.DataFrame(rows)

    # LLM별 평균 계산
    llm_avg = df.groupby('llm_name').agg({
        'avg_score': 'mean',
        'std_score': 'mean',
        'score_range': 'mean'
    }).round(3)

    llm_avg = llm_avg.sort_values('avg_score', ascending=False)

    # CSV 저장
    detail_csv = output_dir / "judge_comparison_detailed.csv"
    df.to_csv(detail_csv, index=False, encoding='utf-8-sig')
    print(f"📊 상세 비교 저장: {detail_csv}")

    summary_csv = output_dir / "judge_comparison_summary.csv"
    llm_avg.to_csv(summary_csv, encoding='utf-8-sig')
    print(f"📊 요약 비교 저장: {summary_csv}")

    # 마크다운 리포트
    md_file = output_dir / "judge_comparison.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Judge 모델 간 비교 분석\n\n")
        f.write(f"**보고서 타입**: {report_type}\n\n")
        f.write(f"**비교한 Judge 모델 수**: {len(judge_results)}\n\n")
        f.write(f"**분석 일시**: {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")

        f.write("## Judge 모델 목록\n\n")
        for jr in judge_results:
            f.write(f"- **{jr['judge_model']}** ({jr['provider']})\n")
        f.write("\n")

        f.write("## LLM별 평균 점수 (모든 Judge 평균)\n\n")
        f.write(llm_avg.to_markdown())
        f.write("\n\n")

        # Judge 간 일치도 분석
        f.write("## Judge 간 일치도 분석\n\n")
        f.write(f"- **평균 점수 차이 범위**: {df['score_range'].mean():.3f}\n")
        f.write(f"- **최대 점수 차이**: {df['score_range'].max():.3f}\n")
        f.write(f"- **최소 점수 차이**: {df['score_range'].min():.3f}\n\n")

        if df['score_range'].mean() > 1.0:
            f.write("⚠️ **경고**: Judge 모델 간 평가 차이가 큽니다 (평균 범위 > 1.0)\n\n")
        else:
            f.write("✅ **양호**: Judge 모델 간 평가가 비교적 일치합니다.\n\n")

    print(f"📝 마크다운 리포트 저장: {md_file}")

    # 콘솔 출력
    print("\n" + "="*100)
    print("🏆 LLM별 평균 점수 (모든 Judge 평균)")
    print("="*100)
    print(llm_avg.to_string())
    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="여러 Judge 모델로 병렬 평가 수행"
    )

    parser.add_argument(
        "--combinations-file",
        type=str,
        required=True,
        help="all_combinations.json 파일 경로"
    )

    parser.add_argument(
        "--report-type",
        type=str,
        default="weekly_report",
        choices=["weekly_report", "executive_report"],
        help="보고서 타입 (기본: weekly_report)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="결과 저장 기본 디렉토리"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="최대 동시 실행 프로세스 수 (기본: 3)"
    )

    parser.add_argument(
        "--judges",
        type=str,
        nargs='+',
        default=["gpt-5.1:azure_ai", "anthropic/claude-opus-4.5:openrouter", "DeepSeek-V3.1:azure_ai"],
        help="Judge 모델 설정 (형식: model:provider, 예: gpt-5.1:azure_ai)"
    )

    args = parser.parse_args()

    # Judge 설정 파싱
    judge_configs = []
    for judge_spec in args.judges:
        parts = judge_spec.split(':')
        if len(parts) != 2:
            print(f"⚠️  잘못된 Judge 설정 형식: {judge_spec} (형식: model:provider)")
            continue

        model, provider = parts
        judge_configs.append({
            "model": model,
            "provider": provider
        })

    if not judge_configs:
        print("❌ 유효한 Judge 모델 설정이 없습니다.")
        sys.exit(1)

    # 병렬 평가 실행
    result = evaluate_with_multiple_judges(
        combinations_file=args.combinations_file,
        judge_configs=judge_configs,
        report_type=args.report_type,
        output_base_dir=args.output_dir,
        max_workers=args.max_workers
    )

    print("\n✅ 모든 Judge 모델 평가 완료!")

    if result['num_failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
