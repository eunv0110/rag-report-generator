#!/usr/bin/env python3
"""육안 평가와 Judge 평가 비교 분석

질문별로 흩어진 Judge 평가 결과를 통합하고, 육안 평가와 비교합니다.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy.stats import spearmanr, kendalltau
from datetime import datetime


def aggregate_judge_evaluations(judge_eval_dir: Path) -> Dict[str, Any]:
    """Judge 평가 결과 통합

    질문별로 흩어진 평가 결과를 LLM별로 통합하여 평균 점수 계산
    """

    # 모든 Judge 모델 디렉토리
    judge_dirs = [d for d in judge_eval_dir.iterdir() if d.is_dir() and d.name.startswith('evaluation_')]

    if not judge_dirs:
        return None

    print(f"\n{'='*80}")
    print(f"Judge 평가 결과 통합")
    print(f"{'='*80}")
    print(f"발견된 Judge 모델: {len(judge_dirs)}개")

    # 각 Judge 모델별로 결과 수집
    all_judge_results = {}

    for judge_dir in judge_dirs:
        judge_name = judge_dir.name.replace('evaluation_', '')
        print(f"\n📊 Judge: {judge_name}")

        # 질문별 평가 파일들
        question_files = sorted(judge_dir.glob('question_*.json'))

        if not question_files:
            continue

        # LLM별 점수 수집 (질문별로)
        llm_scores_by_question = {}

        for qfile in question_files:
            with open(qfile, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for llm_name, result in data.get('results', {}).items():
                if llm_name not in llm_scores_by_question:
                    llm_scores_by_question[llm_name] = []

                llm_scores_by_question[llm_name].append({
                    'question': data.get('question', ''),
                    'final_score': result.get('final_score', 0),
                    'criterion_results': result.get('criterion_results', [])
                })

        # LLM별 평균 점수 계산
        llm_avg_scores = {}
        for llm_name, scores_list in llm_scores_by_question.items():
            avg_score = np.mean([s['final_score'] for s in scores_list])
            llm_avg_scores[llm_name] = {
                'final_score': avg_score,
                'num_questions': len(scores_list),
                'scores_by_question': scores_list
            }
            print(f"  {llm_name}: {avg_score:.2f} (질문 {len(scores_list)}개)")

        all_judge_results[judge_name] = llm_avg_scores

    return all_judge_results


def map_llm_names(name: str) -> str:
    """LLM 이름 표준화"""
    name_mapping = {
        'phi4': 'Phi-4',
        'deepseek_v31': 'DeepSeek-V3.1',
        'gpt41': 'OpenAI GPT-4.1',
        'gpt51': 'OpenAI GPT-5.1',
        'llama33_70b': 'Llama-3.3-70B-Instruct',
        'claude_opus_45': 'Claude 4.5 Opus',
        'claude_sonnet_45': 'Claude 4.5 Sonnet'
    }
    return name_mapping.get(name, name)


def compare_human_vs_judges(
    human_eval_path: Path,
    judge_results: Dict[str, Any],
    report_type: str
) -> Dict[str, Any]:
    """육안 평가와 각 Judge 평가 비교"""

    # 육안 평가 로드
    with open(human_eval_path, 'r', encoding='utf-8') as f:
        human_data = json.load(f)

    human_results = human_data['results']

    print(f"\n{'='*80}")
    print(f"{report_type.upper()} - 육안 평가 vs Judge 평가 비교")
    print(f"{'='*80}")

    comparisons = {}

    for judge_name, judge_scores in judge_results.items():
        print(f"\n🤖 Judge: {judge_name}")

        # 공통 LLM 찾기
        comparison_data = []

        for llm_code, judge_data in judge_scores.items():
            llm_name = map_llm_names(llm_code)

            if llm_name not in human_results:
                print(f"  ⚠️ {llm_name}: 육안 평가 없음")
                continue

            human_score = human_results[llm_name]['final_score']
            judge_score = judge_data['final_score']

            comparison_data.append({
                'llm': llm_name,
                'human_score': human_score,
                'judge_score': judge_score,
                'score_diff': human_score - judge_score,
                'num_questions': judge_data['num_questions']
            })

        if not comparison_data:
            print("  ⚠️ 비교 가능한 LLM 없음")
            continue

        # DataFrame 생성
        df = pd.DataFrame(comparison_data)

        # 순위 계산
        df['human_rank'] = df['human_score'].rank(ascending=False, method='min').astype(int)
        df['judge_rank'] = df['judge_score'].rank(ascending=False, method='min').astype(int)
        df['rank_diff'] = abs(df['human_rank'] - df['judge_rank'])

        df = df.sort_values('human_rank')

        # 상관관계
        if len(df) > 2:
            spearman_corr, spearman_p = spearmanr(df['human_score'], df['judge_score'])
            kendall_corr, kendall_p = kendalltau(df['human_score'], df['judge_score'])
            pearson_corr = np.corrcoef(df['human_score'], df['judge_score'])[0, 1]

            print(f"\n  📊 상관관계:")
            print(f"    Pearson: {pearson_corr:.4f}")
            print(f"    Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})")
            print(f"    Kendall Tau: {kendall_corr:.4f} (p={kendall_p:.4f})")

            # 해석
            if spearman_corr > 0.8:
                interpretation = "✅ 매우 강한 상관관계 - 육안 평가와 매우 유사"
            elif spearman_corr > 0.6:
                interpretation = "✓ 강한 상관관계 - 대체로 일치"
            elif spearman_corr > 0.4:
                interpretation = "⚠️ 중간 정도 - 일부 차이"
            else:
                interpretation = "❌ 약한 상관관계 - 큰 차이"

            print(f"    → {interpretation}")

        # 순위 차이
        print(f"\n  📈 순위 차이:")
        print(f"    평균: {df['rank_diff'].mean():.2f}")
        print(f"    최대: {df['rank_diff'].max()}")

        # 비교표
        print(f"\n  📋 상세 비교:")
        for _, row in df.iterrows():
            print(f"    {row['llm']:<25} | 육안: {row['human_rank']}위({row['human_score']:.2f}) | Judge: {row['judge_rank']}위({row['judge_score']:.2f}) | 차이: {row['rank_diff']}")

        # 큰 차이
        big_diff = df[df['rank_diff'] >= 2]
        if len(big_diff) > 0:
            print(f"\n  ⚠️ 큰 순위 차이 (>= 2):")
            for _, row in big_diff.iterrows():
                print(f"    {row['llm']}: 육안 {row['human_rank']}위 → Judge {row['judge_rank']}위 (차이: {row['rank_diff']})")

        comparisons[judge_name] = {
            'df': df,
            'correlations': {
                'pearson': pearson_corr if len(df) > 2 else None,
                'spearman': spearman_corr if len(df) > 2 else None,
                'kendall': kendall_corr if len(df) > 2 else None
            } if len(df) > 2 else None,
            'rank_diff_stats': {
                'mean': df['rank_diff'].mean(),
                'max': df['rank_diff'].max(),
                'std': df['rank_diff'].std()
            }
        }

    return comparisons


def generate_comparison_report(
    comparisons: Dict[str, Any],
    report_type: str,
    output_dir: Path
):
    """비교 리포트 생성"""

    report_path = output_dir / f"human_vs_judge_comparison_{report_type}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 육안 평가 vs LLM Judge 평가 비교\n\n")
        f.write(f"**보고서 타입**: {report_type}\n\n")
        f.write(f"**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        for judge_name, comp_data in comparisons.items():
            f.write(f"## Judge: {judge_name}\n\n")

            df = comp_data['df']

            # 상관관계
            if comp_data['correlations']:
                corr = comp_data['correlations']
                f.write(f"### 상관관계\n\n")
                f.write(f"- **Pearson**: {corr['pearson']:.4f}\n")
                f.write(f"- **Spearman**: {corr['spearman']:.4f}\n")
                f.write(f"- **Kendall Tau**: {corr['kendall']:.4f}\n\n")

                # 해석
                spearman = corr['spearman']
                if spearman > 0.8:
                    f.write("✅ **매우 강한 상관관계** - 육안 평가와 매우 유사합니다.\n\n")
                elif spearman > 0.6:
                    f.write("✓ **강한 상관관계** - 대체로 일치합니다.\n\n")
                elif spearman > 0.4:
                    f.write("⚠️ **중간 정도의 상관관계** - 일부 차이가 있습니다.\n\n")
                else:
                    f.write("❌ **약한 상관관계** - 큰 차이가 있습니다.\n\n")

            # 순위 차이
            rank_diff = comp_data['rank_diff_stats']
            f.write(f"### 순위 차이\n\n")
            f.write(f"- **평균**: {rank_diff['mean']:.2f}\n")
            f.write(f"- **최대**: {rank_diff['max']}\n")
            f.write(f"- **표준편차**: {rank_diff['std']:.2f}\n\n")

            # 비교표
            f.write(f"### 상세 비교\n\n")
            f.write("| LLM | 육안 점수 | 육안 순위 | Judge 점수 | Judge 순위 | 순위 차이 |\n")
            f.write("|-----|-----------|-----------|------------|-----------|----------|\n")
            for _, row in df.iterrows():
                f.write(f"| {row['llm']} | {row['human_score']:.2f} | {row['human_rank']} | ")
                f.write(f"{row['judge_score']:.2f} | {row['judge_rank']} | {row['rank_diff']} |\n")
            f.write("\n")

            # 큰 차이
            big_diff = df[df['rank_diff'] >= 2]
            if len(big_diff) > 0:
                f.write(f"### ⚠️ 큰 순위 차이 (>= 2)\n\n")
                for _, row in big_diff.iterrows():
                    f.write(f"- **{row['llm']}**\n")
                    f.write(f"  - 육안 평가: {row['human_rank']}위 ({row['human_score']:.2f}점)\n")
                    f.write(f"  - Judge 평가: {row['judge_rank']}위 ({row['judge_score']:.2f}점)\n")
                    f.write(f"  - 순위 차이: {row['rank_diff']}\n\n")

            f.write("---\n\n")

    print(f"\n📝 비교 리포트 생성: {report_path}")

    # CSV도 저장
    for judge_name, comp_data in comparisons.items():
        csv_path = output_dir / f"{report_type}_{judge_name.replace('/', '_')}_comparison.csv"
        comp_data['df'].to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 CSV 저장: {csv_path}")


def main():
    """메인 함수"""

    base_dir = Path("data/results/multi_llm_test")

    # Weekly 보고서
    print("\n" + "="*80)
    print("주간 보고서 (WEEKLY) 분석")
    print("="*80)

    weekly_judge_dir = base_dir / "weekly/20260102_052136/multi_judge_evaluation"
    weekly_human = base_dir / "human_evaluation/weekly_human_evaluation_judge_format.json"

    if weekly_judge_dir.exists():
        weekly_judge_results = aggregate_judge_evaluations(weekly_judge_dir)

        if weekly_judge_results:
            weekly_comparisons = compare_human_vs_judges(
                weekly_human,
                weekly_judge_results,
                "weekly"
            )

            # 리포트 생성
            output_dir = base_dir / "evaluation_comparison"
            output_dir.mkdir(parents=True, exist_ok=True)

            generate_comparison_report(weekly_comparisons, "weekly", output_dir)

    # Executive 보고서
    print("\n" + "="*80)
    print("최종 보고서 (EXECUTIVE) 분석")
    print("="*80)

    executive_judge_dir = base_dir / "executive/20260102_053212/multi_judge_evaluation"
    executive_human = base_dir / "human_evaluation/executive_human_evaluation_judge_format.json"

    if executive_judge_dir.exists():
        executive_judge_results = aggregate_judge_evaluations(executive_judge_dir)

        if executive_judge_results:
            executive_comparisons = compare_human_vs_judges(
                executive_human,
                executive_judge_results,
                "executive"
            )

            # 리포트 생성
            output_dir = base_dir / "evaluation_comparison"
            output_dir.mkdir(parents=True, exist_ok=True)

            generate_comparison_report(executive_comparisons, "executive", output_dir)

    print("\n✅ 모든 비교 분석 완료!")


if __name__ == "__main__":
    main()
