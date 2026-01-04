#!/usr/bin/env python3
"""육안 평가와 LLM Judge 평가의 차이 분석

육안 평가 결과와 LLM as Judge 평가 결과를 비교하여:
1. 순위 차이 분석
2. 평가 기준별 점수 차이
3. 평가 방식의 특징과 편향
4. 개선 방안 제안
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy.stats import spearmanr, kendalltau
from datetime import datetime


def load_human_evaluation(file_path: str) -> Dict[str, Any]:
    """육안 평가 결과 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_judge_evaluation(file_path: str) -> Dict[str, Any]:
    """Judge 평가 결과 로드 (있다면)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def analyze_human_evaluation_patterns(
    human_eval: Dict[str, Any],
    report_type: str
) -> Dict[str, Any]:
    """육안 평가의 패턴 분석"""

    print(f"\n{'='*80}")
    print(f"{report_type.upper()} - 육안 평가 분석")
    print(f"{'='*80}")

    results = human_eval['results']

    # 순위 정보
    rankings = []
    for llm_name, data in results.items():
        final_score = data['final_score']
        rankings.append({
            'llm': llm_name,
            'final_score': final_score,
            'criterion_scores': {
                cr['criterion']: cr['score']
                for cr in data['criterion_results']
            }
        })

    rankings.sort(key=lambda x: x['final_score'], reverse=True)

    # 점수 분포
    scores = [r['final_score'] for r in rankings]
    score_range = max(scores) - min(scores)

    print(f"\n📊 점수 분포:")
    print(f"  최고점: {max(scores):.2f}")
    print(f"  최저점: {min(scores):.2f}")
    print(f"  평균: {np.mean(scores):.2f}")
    print(f"  중앙값: {np.median(scores):.2f}")
    print(f"  점수 범위: {score_range:.2f}")
    print(f"  표준편차: {np.std(scores):.2f}")

    # 순위별 점수
    print(f"\n🏆 순위별 점수:")
    for i, r in enumerate(rankings, 1):
        print(f"  {i}위: {r['llm']:<25} - {r['final_score']:.2f}점")

    # 평가 기준별 평균 점수
    criteria_names = list(rankings[0]['criterion_scores'].keys())

    print(f"\n📋 평가 기준별 평균 점수 (10점 만점):")
    for criterion in criteria_names:
        criterion_scores = [r['criterion_scores'][criterion] for r in rankings]
        print(f"  {criterion}: {np.mean(criterion_scores):.2f}")

    # 평가 기준별 변별력 (표준편차)
    print(f"\n🔍 평가 기준별 변별력 (표준편차):")
    for criterion in criteria_names:
        criterion_scores = [r['criterion_scores'][criterion] for r in rankings]
        std = np.std(criterion_scores)
        print(f"  {criterion}: {std:.2f} (변별력: {'높음' if std > 1.0 else '보통' if std > 0.5 else '낮음'})")

    # LLM별 강점/약점 분석
    print(f"\n💡 LLM별 특징:")
    for r in rankings[:3]:  # 상위 3개만
        llm_name = r['llm']
        llm_data = results[llm_name]

        print(f"\n  {llm_name}:")
        if llm_data.get('strengths'):
            print(f"    강점:")
            for strength in llm_data['strengths']:
                print(f"      - {strength}")
        if llm_data.get('weaknesses'):
            print(f"    약점:")
            for weakness in llm_data['weaknesses']:
                print(f"      - {weakness}")

    return {
        'rankings': rankings,
        'score_stats': {
            'max': max(scores),
            'min': min(scores),
            'mean': np.mean(scores),
            'median': np.median(scores),
            'range': score_range,
            'std': np.std(scores)
        },
        'criteria_stats': {
            criterion: {
                'mean': np.mean([r['criterion_scores'][criterion] for r in rankings]),
                'std': np.std([r['criterion_scores'][criterion] for r in rankings])
            }
            for criterion in criteria_names
        }
    }


def compare_with_judge(
    human_eval: Dict[str, Any],
    judge_eval: Dict[str, Any],
    report_type: str
) -> Dict[str, Any]:
    """육안 평가와 Judge 평가 비교"""

    if judge_eval is None:
        print(f"\n⚠️  Judge 평가 결과가 없습니다.")
        return None

    print(f"\n{'='*80}")
    print(f"{report_type.upper()} - 육안 평가 vs Judge 평가 비교")
    print(f"{'='*80}")

    human_results = human_eval['results']
    judge_results = judge_eval['results']

    # 공통 LLM
    common_llms = set(human_results.keys()) & set(judge_results.keys())

    if not common_llms:
        print("⚠️  공통 LLM이 없습니다.")
        return None

    print(f"공통 LLM 수: {len(common_llms)}")

    # 비교 데이터
    comparison = []
    for llm in common_llms:
        human_score = human_results[llm]['final_score']
        judge_score = judge_results[llm]['final_score']

        comparison.append({
            'llm': llm,
            'human_score': human_score,
            'judge_score': judge_score,
            'score_diff': human_score - judge_score
        })

    # DataFrame 생성
    df = pd.DataFrame(comparison)

    # 순위 계산
    df['human_rank'] = df['human_score'].rank(ascending=False, method='min').astype(int)
    df['judge_rank'] = df['judge_score'].rank(ascending=False, method='min').astype(int)
    df['rank_diff'] = abs(df['human_rank'] - df['judge_rank'])

    df = df.sort_values('human_rank')

    # 상관관계
    spearman_corr, spearman_p = spearmanr(df['human_score'], df['judge_score'])
    kendall_corr, kendall_p = kendalltau(df['human_score'], df['judge_score'])
    pearson_corr = np.corrcoef(df['human_score'], df['judge_score'])[0, 1]

    print(f"\n📊 상관관계:")
    print(f"  Pearson: {pearson_corr:.4f}")
    print(f"  Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})")
    print(f"  Kendall Tau: {kendall_corr:.4f} (p={kendall_p:.4f})")

    # 순위 차이
    print(f"\n📈 순위 차이:")
    print(f"  평균: {df['rank_diff'].mean():.2f}")
    print(f"  최대: {df['rank_diff'].max()}")

    # 비교표
    print(f"\n📋 상세 비교:")
    print(df.to_string(index=False))

    # 큰 차이
    big_diff = df[df['rank_diff'] >= 2]
    if len(big_diff) > 0:
        print(f"\n⚠️  큰 순위 차이 (>= 2):")
        for _, row in big_diff.iterrows():
            print(f"  {row['llm']}:")
            print(f"    육안: {row['human_rank']}위 ({row['human_score']:.2f}점)")
            print(f"    Judge: {row['judge_rank']}위 ({row['judge_score']:.2f}점)")
            print(f"    차이: {row['rank_diff']}")

    return {
        'comparison_df': df,
        'correlations': {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr
        },
        'rank_diff_stats': {
            'mean': df['rank_diff'].mean(),
            'max': df['rank_diff'].max(),
            'std': df['rank_diff'].std()
        }
    }


def generate_insights_and_recommendations(
    weekly_analysis: Dict[str, Any],
    executive_analysis: Dict[str, Any],
    output_dir: Path
):
    """인사이트 및 개선 방안 리포트 생성"""

    report_path = output_dir / "evaluation_insights_and_recommendations.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 육안 평가 vs LLM Judge 평가 - 인사이트 및 개선 방안\n\n")
        f.write(f"**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 주간 보고서 인사이트
        f.write("## 📊 주간 보고서 평가 인사이트\n\n")

        if weekly_analysis:
            rankings = weekly_analysis['rankings']
            stats = weekly_analysis['score_stats']

            f.write(f"### 점수 분포\n\n")
            f.write(f"- 최고점: {stats['max']:.2f}\n")
            f.write(f"- 최저점: {stats['min']:.2f}\n")
            f.write(f"- 점수 범위: {stats['range']:.2f}\n")
            f.write(f"- 표준편차: {stats['std']:.2f}\n\n")

            f.write(f"### 순위\n\n")
            for i, r in enumerate(rankings, 1):
                f.write(f"{i}. **{r['llm']}**: {r['final_score']:.2f}점\n")
            f.write("\n")

            f.write(f"### 평가 기준별 변별력\n\n")
            for criterion, crit_stats in weekly_analysis['criteria_stats'].items():
                discriminability = "높음" if crit_stats['std'] > 1.0 else "보통" if crit_stats['std'] > 0.5 else "낮음"
                f.write(f"- **{criterion}**: 평균 {crit_stats['mean']:.2f}, 표준편차 {crit_stats['std']:.2f} (변별력: {discriminability})\n")
            f.write("\n")

        # 최종 보고서 인사이트
        f.write("## 📋 최종 보고서 평가 인사이트\n\n")

        if executive_analysis:
            rankings = executive_analysis['rankings']
            stats = executive_analysis['score_stats']

            f.write(f"### 점수 분포\n\n")
            f.write(f"- 최고점: {stats['max']:.2f}\n")
            f.write(f"- 최저점: {stats['min']:.2f}\n")
            f.write(f"- 점수 범위: {stats['range']:.2f}\n")
            f.write(f"- 표준편차: {stats['std']:.2f}\n\n")

            f.write(f"### 순위\n\n")
            for i, r in enumerate(rankings, 1):
                f.write(f"{i}. **{r['llm']}**: {r['final_score']:.2f}점\n")
            f.write("\n")

            f.write(f"### 평가 기준별 변별력\n\n")
            for criterion, crit_stats in executive_analysis['criteria_stats'].items():
                discriminability = "높음" if crit_stats['std'] > 1.0 else "보통" if crit_stats['std'] > 0.5 else "낮음"
                f.write(f"- **{criterion}**: 평균 {crit_stats['mean']:.2f}, 표준편차 {crit_stats['std']:.2f} (변별력: {discriminability})\n")
            f.write("\n")

        # 주요 발견사항
        f.write("## 💡 주요 발견사항\n\n")

        f.write("### 1. 보고서 타입에 따른 LLM 성능 차이\n\n")
        f.write("- **주간 보고서**: GPT-4.1 (91점), DeepSeek-V3.1 (90점), Claude Sonnet 4.5 (88점)\n")
        f.write("  - 간결성과 실무성이 중요\n")
        f.write("  - GPT-4.1이 가장 균형잡힌 성능\n")
        f.write("  - DeepSeek-V3.1은 간결성에서 최고점\n\n")

        f.write("- **최종 보고서**: GPT-4.1 (87.5점), GPT-5.1 (85.5점), DeepSeek-V3.1 (79점)\n")
        f.write("  - 완성도와 정확성이 중요\n")
        f.write("  - GPT-5.1은 완벽하지만 너무 길어서 2위\n")
        f.write("  - 주간보고서와 다른 순위 변동 (예: Claude 모델들이 하위권)\n\n")

        f.write("### 2. 평가 기준별 특징\n\n")

        f.write("**주간 보고서:**\n")
        f.write("- **간결성**이 가장 중요한 변별력 (가중치 35%)\n")
        f.write("- GPT-5.1은 간결성 부족으로 순위 하락\n")
        f.write("- DeepSeek-V3.1은 간결성에서 최고 점수\n\n")

        f.write("**최종 보고서:**\n")
        f.write("- **간결성**의 중요도가 상대적으로 낮아짐 (가중치 25%)\n")
        f.write("- **완성도**, **정확성**, **실용성**이 균등하게 중요\n")
        f.write("- GPT-5.1은 완성도에서 최고점이지만 간결성 부족으로 종합 2위\n\n")

        f.write("### 3. LLM별 특징\n\n")

        f.write("**GPT-4.1:**\n")
        f.write("- 두 보고서 타입 모두에서 1위\n")
        f.write("- 균형잡힌 성능 (간결성, 구조, 정확성 모두 우수)\n")
        f.write("- 실무 활용도가 가장 높음\n\n")

        f.write("**GPT-5.1:**\n")
        f.write("- 완성도와 정확성은 최고 수준\n")
        f.write("- 주간보고서에서는 6위로 저조 (너무 장황함)\n")
        f.write("- 최종보고서에서는 2위 (상세함이 장점으로 작용)\n")
        f.write("- 용도에 따라 성능 편차가 큼\n\n")

        f.write("**DeepSeek-V3.1:**\n")
        f.write("- 간결성에서 최고 점수\n")
        f.write("- 주간보고서 2위, 최종보고서 3위\n")
        f.write("- 빠른 파악용으로 적합\n")
        f.write("- 때때로 맥락 부족\n\n")

        f.write("**Claude 모델들 (Opus 4.5, Sonnet 4.5):**\n")
        f.write("- 주간보고서에서는 중상위권 (3-4위)\n")
        f.write("- 최종보고서에서는 하위권 (4-5위)\n")
        f.write("- 구조화와 일관성은 좋으나 실용성 부족\n")
        f.write("- '문서에 명시되지 않음' 과다 사용 경향\n\n")

        f.write("**Llama-3.3-70B:**\n")
        f.write("- 두 보고서 타입 모두 최하위 (7위)\n")
        f.write("- 특히 최종보고서에서 매우 저조 (28점)\n")
        f.write("- 모든 평가 기준에서 부족\n\n")

        # 개선 방안
        f.write("## 🔧 개선 방안 제안\n\n")

        f.write("### 1. 육안 평가 프로세스 개선\n\n")
        f.write("**현재 장점:**\n")
        f.write("- 실무 관점에서의 실용성 평가\n")
        f.write("- 맥락과 뉘앙스 파악 가능\n")
        f.write("- 명확한 평가 기준과 가중치\n\n")

        f.write("**개선 필요 사항:**\n")
        f.write("- ✅ 평가자 간 일관성 확보 (현재는 단일 평가자)\n")
        f.write("- ✅ 평가 기준에 대한 구체적인 예시 제공\n")
        f.write("- ✅ 블라인드 평가 도입 (LLM 이름 숨김)\n")
        f.write("- ✅ 복수 평가자의 평균 점수 사용\n\n")

        f.write("### 2. LLM Judge 평가 개선\n\n")
        f.write("**필요한 개선:**\n")
        f.write("- 현재 Judge 평가가 실패한 상태 (tabulate 오류)\n")
        f.write("- Judge 평가 완료 후 육안 평가와 비교 필요\n")
        f.write("- Judge 모델의 편향 분석 필요\n")
        f.write("- 여러 Judge 모델 사용하여 합의 평가\n\n")

        f.write("### 3. 하이브리드 평가 시스템\n\n")
        f.write("**제안:**\n")
        f.write("1. **1차 평가 (LLM Judge)**\n")
        f.write("   - 자동화된 대량 평가\n")
        f.write("   - 객관적 기준 (정확성, 구조 등)\n")
        f.write("   - 빠른 스크리닝\n\n")

        f.write("2. **2차 평가 (육안 평가)**\n")
        f.write("   - 상위권 LLM에 대한 정밀 평가\n")
        f.write("   - 실무 활용도 평가\n")
        f.write("   - 최종 의사결정\n\n")

        f.write("3. **평가 결과 통합**\n")
        f.write("   - Judge 점수 (40%) + 육안 점수 (60%)\n")
        f.write("   - 또는 Judge로 후보군 선정 → 육안으로 최종 선택\n\n")

        f.write("### 4. 평가 기준 세분화\n\n")

        f.write("**주간 보고서:**\n")
        f.write("- 간결성을 '적절한 길이'와 '불필요한 반복 제거'로 분리\n")
        f.write("- 실무성을 '즉시 활용 가능성'과 '액션 아이템 명확성'으로 분리\n\n")

        f.write("**최종 보고서:**\n")
        f.write("- 완성도를 '구조 완성도'와 '내용 완성도'로 분리\n")
        f.write("- 실용성을 '의사결정 지원'과 '인사이트 제공'으로 분리\n\n")

        f.write("### 5. LLM별 최적 활용 방안\n\n")

        f.write("**GPT-4.1:**\n")
        f.write("- 범용 보고서 생성에 최적\n")
        f.write("- 실무 활용도가 높음\n")
        f.write("- 기본 선택지로 추천\n\n")

        f.write("**GPT-5.1:**\n")
        f.write("- 외부 제출용 상세 문서에 최적\n")
        f.write("- 완벽한 정확성이 필요한 경우\n")
        f.write("- 주간보고서에는 부적합 (요약 프롬프트 추가 필요)\n\n")

        f.write("**DeepSeek-V3.1:**\n")
        f.write("- 빠른 상황 파악용\n")
        f.write("- 간결한 요약이 필요한 경우\n")
        f.write("- 비용 효율적\n\n")

        f.write("**Claude 모델들:**\n")
        f.write("- 구조화된 문서에 적합\n")
        f.write("- 프롬프트 개선 필요 ('문서에 명시되지 않음' 과다 사용 방지)\n")
        f.write("- 테이블 활용 능력 우수\n\n")

    print(f"\n📝 인사이트 리포트 생성: {report_path}")


def main():
    """메인 함수"""

    # 데이터 디렉토리
    data_dir = Path("data/results/multi_llm_test")
    human_eval_dir = data_dir / "human_evaluation"

    # 육안 평가 로드
    weekly_human = load_human_evaluation(
        human_eval_dir / "weekly_human_evaluation_judge_format.json"
    )
    executive_human = load_human_evaluation(
        human_eval_dir / "executive_human_evaluation_judge_format.json"
    )

    # Judge 평가 로드 (있다면)
    weekly_judge = load_judge_evaluation(
        data_dir / "weekly/20260102_052136/multi_judge_evaluation/llm_judge_evaluation_weekly.json"
    )
    executive_judge = load_judge_evaluation(
        data_dir / "executive/20260102_053212/multi_judge_evaluation/llm_judge_evaluation_executive.json"
    )

    # 육안 평가 분석
    weekly_analysis = analyze_human_evaluation_patterns(weekly_human, "weekly")
    executive_analysis = analyze_human_evaluation_patterns(executive_human, "executive")

    # Judge 평가와 비교 (있다면)
    if weekly_judge:
        compare_with_judge(weekly_human, weekly_judge, "weekly")

    if executive_judge:
        compare_with_judge(executive_human, executive_judge, "executive")

    # 출력 디렉토리
    output_dir = data_dir / "evaluation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 인사이트 및 개선 방안 리포트 생성
    generate_insights_and_recommendations(
        weekly_analysis,
        executive_analysis,
        output_dir
    )

    # 분석 결과 저장
    with open(output_dir / "weekly_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(weekly_analysis, f, indent=2, ensure_ascii=False, default=str)

    with open(output_dir / "executive_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(executive_analysis, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 분석 완료!")
    print(f"📁 결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
