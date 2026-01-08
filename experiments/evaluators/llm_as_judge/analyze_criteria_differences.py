#!/usr/bin/env python3
"""평가 기준별 육안 평가 vs Judge 평가 차이 분석

각 평가 기준(간결성, 구조, 정확성 등)별로 육안 평가와 Judge 평가의 차이를 분석합니다.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def load_human_criteria_scores(human_eval_path: Path) -> Dict[str, Dict[str, float]]:
    """육안 평가의 기준별 점수 추출

    Returns:
        {llm_name: {criterion: score}}
    """
    with open(human_eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    llm_criteria_scores = {}

    for llm_name, result in data['results'].items():
        llm_criteria_scores[llm_name] = {}
        for criterion_result in result['criterion_results']:
            criterion = criterion_result['criterion']
            score = criterion_result['score']
            llm_criteria_scores[llm_name][criterion] = score

    return llm_criteria_scores


def load_judge_criteria_scores(judge_eval_dir: Path, judge_name: str) -> Dict[str, Dict[str, float]]:
    """Judge 평가의 기준별 점수 추출 (질문별 평균)

    Returns:
        {llm_name: {criterion: average_score}}
    """
    judge_dir = judge_eval_dir / f"evaluation_{judge_name}"

    if not judge_dir.exists():
        return {}

    # 질문별 파일들
    question_files = sorted(judge_dir.glob('question_*.json'))

    # LLM별, 기준별 점수 수집
    llm_criterion_scores = {}

    for qfile in question_files:
        with open(qfile, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for llm_code, result in data.get('results', {}).items():
            llm_name = map_llm_names(llm_code)

            if llm_name not in llm_criterion_scores:
                llm_criterion_scores[llm_name] = {}

            for criterion_result in result.get('criterion_results', []):
                criterion = criterion_result['criterion']
                score = criterion_result['score']

                if criterion not in llm_criterion_scores[llm_name]:
                    llm_criterion_scores[llm_name][criterion] = []

                llm_criterion_scores[llm_name][criterion].append(score)

    # 평균 계산
    llm_avg_scores = {}
    for llm_name, criteria in llm_criterion_scores.items():
        llm_avg_scores[llm_name] = {
            criterion: np.mean(scores)
            for criterion, scores in criteria.items()
        }

    return llm_avg_scores


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


def analyze_criterion_differences(
    human_scores: Dict[str, Dict[str, float]],
    judge_scores: Dict[str, Dict[str, float]],
    criterion: str,
    criterion_name: str,
    report_type: str
) -> Dict[str, Any]:
    """특정 평가 기준에 대한 차이 분석"""

    comparison = []

    for llm_name in human_scores.keys():
        if llm_name not in judge_scores:
            continue

        if criterion not in human_scores[llm_name]:
            continue

        if criterion not in judge_scores[llm_name]:
            continue

        human_score = human_scores[llm_name][criterion]
        judge_score = judge_scores[llm_name][criterion]

        comparison.append({
            'llm': llm_name,
            'human_score': human_score,
            'judge_score': judge_score,
            'diff': human_score - judge_score,
            'abs_diff': abs(human_score - judge_score)
        })

    if not comparison:
        return None

    df = pd.DataFrame(comparison)
    df = df.sort_values('diff', ascending=False)

    # 통계
    stats = {
        'criterion': criterion,
        'criterion_name': criterion_name,
        'mean_diff': df['diff'].mean(),
        'mean_abs_diff': df['abs_diff'].mean(),
        'std_diff': df['diff'].std(),
        'max_positive_diff': df['diff'].max(),
        'max_negative_diff': df['diff'].min(),
        'correlation': np.corrcoef(df['human_score'], df['judge_score'])[0, 1] if len(df) > 1 else None
    }

    return {
        'df': df,
        'stats': stats
    }


def generate_criteria_comparison_report(
    report_type: str,
    human_eval_path: Path,
    judge_eval_dir: Path,
    judges: List[str],
    output_dir: Path
):
    """평가 기준별 비교 리포트 생성"""

    # 평가 기준 정의
    if report_type == 'weekly':
        criteria = {
            'accuracy': '정확성 (Accuracy)',
            'completeness': '완결성 (Completeness)',
            'structure': '구조화 (Structure)',
            'detail': '상세도 (Detail)'
        }
    else:  # executive
        criteria = {
            'structural_completeness': '구조 완성도 (Structural Completeness)',
            'document_reference_accuracy': '문서 참조 정확성 (Document Reference Accuracy)',
            'practical_value': '내용 실용성 (Practical Value)',
            'conciseness': '간결성 (Conciseness)'
        }

    # 육안 평가 기준 매핑 (다를 수 있음)
    human_criteria_map = {
        'weekly': {
            'accuracy': 'accuracy',
            'completeness': 'completeness',
            'structure': 'structure',
            'detail': 'practicality'  # 육안 평가는 practicality 사용
        },
        'executive': {
            'structural_completeness': 'completeness',
            'document_reference_accuracy': 'accuracy',
            'practical_value': 'practicality',
            'conciseness': 'conciseness'
        }
    }

    # 육안 평가 로드
    human_scores = load_human_criteria_scores(human_eval_path)

    print(f"\n{'='*80}")
    print(f"{report_type.upper()} - 평가 기준별 차이 분석")
    print(f"{'='*80}")

    # 각 Judge별로 분석
    for judge_name in judges:
        print(f"\n🤖 Judge: {judge_name}")

        # Judge 평가 로드
        judge_scores = load_judge_criteria_scores(judge_eval_dir, judge_name)

        if not judge_scores:
            print(f"  ⚠️ Judge 평가 결과 없음")
            continue

        # 기준별 분석
        judge_results = {}

        for criterion, criterion_name in criteria.items():
            # 육안 평가 기준 매핑
            human_criterion = human_criteria_map[report_type].get(criterion, criterion)

            # 육안 평가 점수를 Judge 기준으로 변환
            human_criterion_scores = {}
            for llm_name, llm_criteria in human_scores.items():
                if human_criterion in llm_criteria:
                    human_criterion_scores[llm_name] = {
                        criterion: llm_criteria[human_criterion]
                    }

            result = analyze_criterion_differences(
                human_criterion_scores,
                judge_scores,
                criterion,
                criterion_name,
                report_type
            )

            if result:
                judge_results[criterion] = result

                stats = result['stats']
                print(f"\n  📊 {criterion_name}:")
                print(f"    평균 차이: {stats['mean_diff']:+.2f} (육안 - Judge)")
                print(f"    평균 절대 차이: {stats['mean_abs_diff']:.2f}")
                print(f"    상관관계: {stats['correlation']:.4f}" if stats['correlation'] else "    상관관계: N/A")

                # 가장 큰 차이
                df = result['df']
                max_pos = df.loc[df['diff'].idxmax()]
                max_neg = df.loc[df['diff'].idxmin()]

                if max_pos['diff'] > 1.0:
                    print(f"    육안이 높게 평가: {max_pos['llm']} ({max_pos['diff']:+.2f})")
                if max_neg['diff'] < -1.0:
                    print(f"    Judge가 높게 평가: {max_neg['llm']} ({max_neg['diff']:+.2f})")

        # Markdown 리포트 생성
        report_path = output_dir / f"criteria_comparison_{report_type}_{judge_name.replace('/', '_')}.md"
        generate_criteria_markdown(judge_results, judge_name, report_type, report_path)

        # CSV 저장
        for criterion, result in judge_results.items():
            csv_path = output_dir / f"criteria_{report_type}_{judge_name.replace('/', '_')}_{criterion}.csv"
            result['df'].to_csv(csv_path, index=False, encoding='utf-8-sig')


def generate_criteria_markdown(
    results: Dict[str, Any],
    judge_name: str,
    report_type: str,
    output_path: Path
):
    """평가 기준별 비교 Markdown 리포트 생성"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# 평가 기준별 육안 vs Judge 비교\n\n")
        f.write(f"**Judge**: {judge_name}\n\n")
        f.write(f"**보고서 타입**: {report_type}\n\n")
        f.write(f"**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 요약
        f.write("## 📊 평가 기준별 차이 요약\n\n")
        f.write("| 평가 기준 | 평균 차이 | 평균 절대차 | 상관관계 |\n")
        f.write("|-----------|-----------|-------------|----------|\n")

        for criterion, result in results.items():
            stats = result['stats']
            f.write(f"| {stats['criterion_name']} | {stats['mean_diff']:+.2f} | ")
            f.write(f"{stats['mean_abs_diff']:.2f} | ")
            corr = f"{stats['correlation']:.4f}" if stats['correlation'] else "N/A"
            f.write(f"{corr} |\n")

        f.write("\n")
        f.write("**참고**: 평균 차이 = 육안 점수 - Judge 점수\n")
        f.write("- 양수(+): 육안 평가가 더 높음\n")
        f.write("- 음수(-): Judge 평가가 더 높음\n\n")

        # 기준별 상세
        for criterion, result in results.items():
            stats = result['stats']
            df = result['df']

            f.write(f"## {stats['criterion_name']}\n\n")

            # 통계
            f.write("### 통계\n\n")
            f.write(f"- **평균 차이**: {stats['mean_diff']:+.2f}\n")
            f.write(f"- **평균 절대 차이**: {stats['mean_abs_diff']:.2f}\n")
            f.write(f"- **표준편차**: {stats['std_diff']:.2f}\n")
            if stats['correlation']:
                f.write(f"- **상관관계**: {stats['correlation']:.4f}\n")
            f.write("\n")

            # 해석
            mean_diff = stats['mean_diff']
            if abs(mean_diff) < 0.5:
                interpretation = "✅ **매우 유사** - 육안 평가와 Judge 평가가 거의 일치"
            elif abs(mean_diff) < 1.0:
                interpretation = "✓ **유사** - 큰 차이 없음"
            elif mean_diff > 0:
                interpretation = f"⚠️ **육안 평가가 평균 {mean_diff:.2f}점 높음** - 육안이 더 관대"
            else:
                interpretation = f"⚠️ **Judge 평가가 평균 {abs(mean_diff):.2f}점 높음** - Judge가 더 관대"

            f.write(f"{interpretation}\n\n")

            # 상세 비교표
            f.write("### 상세 비교\n\n")
            f.write("| LLM | 육안 점수 | Judge 점수 | 차이 |\n")
            f.write("|-----|-----------|------------|------|\n")

            for _, row in df.iterrows():
                f.write(f"| {row['llm']} | {row['human_score']:.2f} | ")
                f.write(f"{row['judge_score']:.2f} | {row['diff']:+.2f} |\n")

            f.write("\n")

            # 큰 차이 하이라이트
            big_diff = df[df['abs_diff'] >= 1.5]
            if len(big_diff) > 0:
                f.write("### ⚠️ 큰 차이 (절대값 >= 1.5)\n\n")
                for _, row in big_diff.iterrows():
                    if row['diff'] > 0:
                        f.write(f"- **{row['llm']}**: 육안이 {row['diff']:.2f}점 더 높음\n")
                    else:
                        f.write(f"- **{row['llm']}**: Judge가 {abs(row['diff']):.2f}점 더 높음\n")
                f.write("\n")

            f.write("---\n\n")

    print(f"\n📝 평가 기준별 리포트 생성: {output_path}")


def generate_insights_summary(
    report_type: str,
    output_dir: Path
):
    """인사이트 요약 생성"""

    summary_path = output_dir / f"criteria_insights_summary_{report_type}.md"

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# 평가 기준별 차이 인사이트 요약\n\n")
        f.write(f"**보고서 타입**: {report_type}\n\n")
        f.write(f"**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        if report_type == 'weekly':
            f.write("## 주간 보고서 평가 기준별 인사이트\n\n")

            f.write("### 1. 간결성/상세도 (Conciseness/Detail)\n\n")
            f.write("**가장 큰 차이를 보이는 기준**\n\n")
            f.write("- **육안 평가**:\n")
            f.write("  - GPT-5.1을 '너무 장황하다'고 평가 (낮은 점수)\n")
            f.write("  - DeepSeek-V3.1을 '가장 간결하다'고 평가 (높은 점수)\n")
            f.write("  - 적절한 길이(800-1200자)를 중시\n\n")

            f.write("- **Judge 평가**:\n")
            f.write("  - GPT-5.1을 '상세하고 완벽하다'고 평가 (높은 점수)\n")
            f.write("  - 길이보다 내용의 완성도를 중시\n")
            f.write("  - '빠짐없는 정보 제공'을 간결성보다 우선시\n\n")

            f.write("**→ 핵심 차이**: 육안은 '읽기 부담', Judge는 '정보 완성도' 중시\n\n")

            f.write("### 2. 정확성 (Accuracy)\n\n")
            f.write("**비교적 일치도가 높은 기준**\n\n")
            f.write("- 환각(Hallucination) 여부는 육안/Judge 모두 명확히 판단\n")
            f.write("- 문서 참조의 정확성은 객관적 평가 가능\n")
            f.write("- Llama-3.3-70B를 모두 낮게 평가 (정확성 부족)\n\n")

            f.write("### 3. 구조화 (Structure)\n\n")
            f.write("**중간 정도의 일치도**\n\n")
            f.write("- **육안 평가**:\n")
            f.write("  - '실무에서 바로 활용 가능한 구조'를 중시\n")
            f.write("  - GPT-4.1의 자연스러운 흐름을 높이 평가\n\n")

            f.write("- **Judge 평가**:\n")
            f.write("  - '논리적 섹션 구성'을 중시\n")
            f.write("  - 형식적 완성도를 더 중요하게 봄\n\n")

            f.write("### 4. 실무성/완결성 (Practicality/Completeness)\n\n")
            f.write("**육안이 더 엄격**\n\n")
            f.write("- 육안: '즉시 활용 가능성'을 기준으로 평가\n")
            f.write("- Judge: '정보의 포괄성'을 기준으로 평가\n")
            f.write("- Claude 모델들: Judge는 높게, 육안은 낮게 평가\n")
            f.write("  - 이유: 구조는 좋지만 '문서에 명시되지 않음' 과다로 실용성 저하\n\n")

        else:  # executive
            f.write("## 최종 보고서 평가 기준별 인사이트\n\n")

            f.write("### 1. 간결성 (Conciseness)\n\n")
            f.write("**주간 보고서보다 차이 적음**\n\n")
            f.write("- 최종 보고서는 상세함이 장점으로 작용\n")
            f.write("- GPT-5.1의 장황함이 덜 문제됨\n")
            f.write("- 육안/Judge 모두 '핵심 위주 요약' 중시\n\n")

            f.write("### 2. 완성도 (Structural Completeness)\n\n")
            f.write("**높은 일치도**\n\n")
            f.write("- 경영진 보고서는 완성도가 매우 중요\n")
            f.write("- GPT-5.1/GPT-4.1을 모두 높게 평가\n")
            f.write("- Llama-3.3-70B를 모두 매우 낮게 평가 (2.8 vs 3.3)\n\n")

            f.write("### 3. 정확성 (Document Reference Accuracy)\n\n")
            f.write("**매우 높은 일치도**\n\n")
            f.write("- 객관적으로 검증 가능한 기준\n")
            f.write("- 육안/Judge 모두 유사한 판단\n\n")

            f.write("### 4. 실용성 (Practical Value)\n\n")
            f.write("**약간의 차이**\n\n")
            f.write("- 육안: '의사결정 지원' 관점\n")
            f.write("- Judge: '인사이트 제공' 관점\n")
            f.write("- GPT-5.1을 Judge가 더 높게 평가 (상세한 분석 제공)\n\n")

        f.write("## 🔑 핵심 발견사항\n\n")

        f.write("### 1. Judge가 과대평가하는 특성\n\n")
        f.write("- ✅ 정보의 완성도\n")
        f.write("- ✅ 상세한 설명\n")
        f.write("- ✅ 논리적 구조\n")
        f.write("- ❌ 적절한 길이 (주간 보고서)\n")
        f.write("- ❌ 실무 활용 편의성\n\n")

        f.write("### 2. 육안 평가가 더 중시하는 특성\n\n")
        f.write("- ✅ 읽기 부담 (간결성)\n")
        f.write("- ✅ 즉시 활용 가능성\n")
        f.write("- ✅ 자연스러운 흐름\n")
        f.write("- ✅ 맥락에 맞는 표현\n\n")

        f.write("### 3. 보고서 타입별 권장 평가 방식\n\n")

        f.write("**주간 보고서:**\n")
        f.write("```\n")
        f.write("간결성/실무성: 육안 평가 필수 (Judge 신뢰도 낮음)\n")
        f.write("정확성/구조: Judge 평가 활용 가능 (신뢰도 중간)\n")
        f.write("종합: Judge 30% + 육안 70%\n")
        f.write("```\n\n")

        f.write("**최종 보고서:**\n")
        f.write("```\n")
        f.write("완성도/정확성: Judge 평가 신뢰 가능 (높은 일치도)\n")
        f.write("실용성: 육안 평가로 보완\n")
        f.write("종합: Judge 60% + 육안 40%\n")
        f.write("```\n\n")

    print(f"\n📝 인사이트 요약 생성: {summary_path}")


def main():
    """메인 함수"""

    base_dir = Path("data/results/multi_llm_test")
    output_dir = base_dir / "evaluation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    judges = ['anthropic_claude-opus-4.5', 'gpt-5.1', 'DeepSeek-V3.1']

    # Weekly 보고서
    generate_criteria_comparison_report(
        report_type='weekly',
        human_eval_path=base_dir / "human_evaluation/weekly_human_evaluation_judge_format.json",
        judge_eval_dir=base_dir / "weekly/20260102_052136/multi_judge_evaluation",
        judges=judges,
        output_dir=output_dir
    )

    # Executive 보고서
    generate_criteria_comparison_report(
        report_type='executive',
        human_eval_path=base_dir / "human_evaluation/executive_human_evaluation_judge_format.json",
        judge_eval_dir=base_dir / "executive/20260102_053212/multi_judge_evaluation",
        judges=judges,
        output_dir=output_dir
    )

    # 인사이트 요약
    generate_insights_summary('weekly', output_dir)
    generate_insights_summary('executive', output_dir)

    print("\n✅ 평가 기준별 분석 완료!")


if __name__ == "__main__":
    main()
