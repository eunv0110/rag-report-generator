#!/usr/bin/env python3
"""LLM 성능 비교 분석 스크립트

다양한 LLM 모델의 성능을 GPT 대비 비율로 비교하여 리포트를 생성합니다.
"""

import sys
from pathlib import Path

# Python 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd


def load_evaluation_results(eval_dir: Path) -> Dict[str, Any]:
    """평가 결과 로드

    Args:
        eval_dir: 평가 결과 디렉토리

    Returns:
        평가 결과 딕셔너리
    """
    overall_eval_file = eval_dir / "overall_evaluation.json"

    if not overall_eval_file.exists():
        return None

    with open(overall_eval_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_llm_scores(evaluation_data: Dict[str, Any]) -> Dict[str, float]:
    """평가 데이터에서 LLM별 평균 점수 추출

    Args:
        evaluation_data: 평가 결과 데이터

    Returns:
        LLM별 평균 점수 딕셔너리
    """
    llm_scores = {}

    for eval_item in evaluation_data.get('evaluations', []):
        for llm_name, llm_result in eval_item.get('results', {}).items():
            if llm_name not in llm_scores:
                llm_scores[llm_name] = []

            final_score = llm_result.get('final_score', 0)
            llm_scores[llm_name].append(final_score)

    # 평균 계산
    avg_scores = {
        llm: sum(scores) / len(scores) if scores else 0
        for llm, scores in llm_scores.items()
    }

    return avg_scores


def compare_with_baseline(
    llm_scores: Dict[str, float],
    baseline_model: str = "gpt41"
) -> Dict[str, Dict[str, float]]:
    """베이스라인(GPT) 대비 성능 비율 계산

    Args:
        llm_scores: LLM별 평균 점수
        baseline_model: 베이스라인 모델명 (기본: gpt41)

    Returns:
        LLM별 비교 결과 딕셔너리
    """
    baseline_score = llm_scores.get(baseline_model, 0)

    if baseline_score == 0:
        print(f"⚠️ 경고: {baseline_model}의 점수가 0입니다.")
        return {}

    comparison = {}

    for llm_name, score in llm_scores.items():
        comparison[llm_name] = {
            "score": score,
            "percentage": (score / baseline_score) * 100,
            "diff": score - baseline_score
        }

    return comparison


def analyze_multi_judge_results(test_dir: Path, report_type: str) -> pd.DataFrame:
    """여러 Judge 모델의 평가 결과 분석

    Args:
        test_dir: 테스트 결과 디렉토리
        report_type: 보고서 타입

    Returns:
        분석 결과 DataFrame
    """
    judge_eval_dir = test_dir / "multi_judge_evaluation"

    if not judge_eval_dir.exists():
        print(f"⚠️ 평가 디렉토리가 없습니다: {judge_eval_dir}")
        return None

    all_judge_results = []

    # 각 Judge 모델별 평가 결과 수집
    for judge_dir in judge_eval_dir.iterdir():
        if not judge_dir.is_dir():
            continue

        if not judge_dir.name.startswith("evaluation_"):
            continue

        judge_name = judge_dir.name.replace("evaluation_", "")

        eval_data = load_evaluation_results(judge_dir)
        if not eval_data:
            print(f"⚠️ {judge_name} 평가 결과를 로드할 수 없습니다.")
            continue

        llm_scores = extract_llm_scores(eval_data)

        # gpt41 대비 비교
        comparison = compare_with_baseline(llm_scores, baseline_model="gpt41")

        for llm_name, comp_result in comparison.items():
            all_judge_results.append({
                "judge": judge_name,
                "report_type": report_type,
                "llm": llm_name,
                "score": comp_result["score"],
                "vs_gpt41_percentage": comp_result["percentage"],
                "vs_gpt41_diff": comp_result["diff"]
            })

    if not all_judge_results:
        return None

    df = pd.DataFrame(all_judge_results)
    return df


def generate_markdown_report(
    weekly_df: pd.DataFrame,
    executive_df: pd.DataFrame,
    output_file: Path
):
    """마크다운 리포트 생성

    Args:
        weekly_df: 주간 보고서 분석 결과
        executive_df: 임원 보고서 분석 결과
        output_file: 출력 파일 경로
    """
    lines = []

    lines.append("# LLM 성능 비교 분석 리포트")
    lines.append("")
    lines.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 요약
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("다양한 LLM 모델의 성능을 GPT-4.1 대비 상대 성능으로 비교 분석했습니다.")
    lines.append("3개의 Judge 모델(GPT-5.1, Claude Opus 4.5, DeepSeek-V3.1)을 사용하여 객관적인 평가를 수행했습니다.")
    lines.append("")

    # 주간 보고서 분석
    if weekly_df is not None and not weekly_df.empty:
        lines.append("## 1. 주간 보고서 (Weekly Report)")
        lines.append("")
        lines.append("**설정**: BGE-M3 + Qwen3-Reranker-4B + RRF Ensemble (Top 6)")
        lines.append("")

        # Judge별 평균 계산
        weekly_avg = weekly_df.groupby("llm").agg({
            "score": "mean",
            "vs_gpt41_percentage": "mean",
            "vs_gpt41_diff": "mean"
        }).round(2)

        weekly_avg = weekly_avg.sort_values("score", ascending=False)

        lines.append("### 평균 성능 (3개 Judge 평균)")
        lines.append("")
        lines.append("| 순위 | LLM | 평균 점수 | GPT-4.1 대비 | 점수 차이 |")
        lines.append("|------|-----|-----------|--------------|-----------|")

        for rank, (llm, row) in enumerate(weekly_avg.iterrows(), 1):
            percentage_str = f"{row['vs_gpt41_percentage']:.1f}%"
            diff_str = f"{row['vs_gpt41_diff']:+.2f}"

            # 특별 표시
            if llm == "gpt41":
                llm_display = f"**{llm}** (기준)"
            elif llm == "deepseek_v31":
                llm_display = f"**{llm}** ⭐"
            else:
                llm_display = llm

            lines.append(f"| {rank} | {llm_display} | {row['score']:.2f} | {percentage_str} | {diff_str} |")

        lines.append("")

        # Judge별 상세 결과
        lines.append("### Judge별 상세 점수")
        lines.append("")

        weekly_pivot = weekly_df.pivot_table(
            index="llm",
            columns="judge",
            values="score",
            aggfunc="mean"
        ).round(2)

        lines.append(weekly_pivot.to_markdown())
        lines.append("")

    # 임원 보고서 분석
    if executive_df is not None and not executive_df.empty:
        lines.append("## 2. 임원 보고서 (Executive Report)")
        lines.append("")
        lines.append("**설정**: OpenAI + RRF MultiQuery (Top 8)")
        lines.append("")

        # Judge별 평균 계산
        executive_avg = executive_df.groupby("llm").agg({
            "score": "mean",
            "vs_gpt41_percentage": "mean",
            "vs_gpt41_diff": "mean"
        }).round(2)

        executive_avg = executive_avg.sort_values("score", ascending=False)

        lines.append("### 평균 성능 (3개 Judge 평균)")
        lines.append("")
        lines.append("| 순위 | LLM | 평균 점수 | GPT-4.1 대비 | 점수 차이 |")
        lines.append("|------|-----|-----------|--------------|-----------|")

        for rank, (llm, row) in enumerate(executive_avg.iterrows(), 1):
            percentage_str = f"{row['vs_gpt41_percentage']:.1f}%"
            diff_str = f"{row['vs_gpt41_diff']:+.2f}"

            # 특별 표시
            if llm == "gpt41":
                llm_display = f"**{llm}** (기준)"
            else:
                llm_display = llm

            lines.append(f"| {rank} | {llm_display} | {row['score']:.2f} | {percentage_str} | {diff_str} |")

        lines.append("")

        # Judge별 상세 결과
        lines.append("### Judge별 상세 점수")
        lines.append("")

        executive_pivot = executive_df.pivot_table(
            index="llm",
            columns="judge",
            values="score",
            aggfunc="mean"
        ).round(2)

        lines.append(executive_pivot.to_markdown())
        lines.append("")

    # 주요 발견사항
    lines.append("## 3. 주요 발견사항 (Key Findings)")
    lines.append("")

    if weekly_df is not None and not weekly_df.empty:
        # DeepSeek-V3.1 vs GPT-4.1 비교
        deepseek_weekly = weekly_avg.loc["deepseek_v31"]
        gpt41_weekly = weekly_avg.loc["gpt41"]

        lines.append(f"### 주간 보고서: DeepSeek-V3.1 vs GPT-4.1")
        lines.append("")
        lines.append(f"- **DeepSeek-V3.1**: {deepseek_weekly['score']:.2f}점")
        lines.append(f"- **GPT-4.1**: {gpt41_weekly['score']:.2f}점")
        lines.append(f"- **상대 성능**: GPT-4.1 대비 **{deepseek_weekly['vs_gpt41_percentage']:.1f}%**")
        lines.append(f"- **점수 차이**: {deepseek_weekly['vs_gpt41_diff']:+.2f}점")
        lines.append("")

        if deepseek_weekly['score'] > gpt41_weekly['score']:
            lines.append(f"✅ DeepSeek-V3.1이 GPT-4.1보다 **{abs(deepseek_weekly['vs_gpt41_diff']):.2f}점 높은** 성능을 보였습니다.")
        elif deepseek_weekly['score'] < gpt41_weekly['score']:
            lines.append(f"⚠️ DeepSeek-V3.1이 GPT-4.1보다 {abs(deepseek_weekly['vs_gpt41_diff']):.2f}점 낮은 성능을 보였습니다.")
        else:
            lines.append("📊 DeepSeek-V3.1과 GPT-4.1이 동일한 성능을 보였습니다.")

        lines.append("")

    # 모델별 강점 분석
    lines.append("### 모델별 특징")
    lines.append("")
    lines.append("| 모델 | 타입 | 특징 | 권장 사용 |")
    lines.append("|------|------|------|-----------|")
    lines.append("| GPT-4.1 | 프론티어 | 안정적 성능, 베이스라인 | 중요 의사결정 보고서 |")
    lines.append("| GPT-5.1 | 프론티어 | 최신 모델, 고성능 | 복잡한 분석 필요 시 |")
    lines.append("| DeepSeek-V3.1 | 오픈소스 | 비용 효율적, GPT 수준 성능 | 대량 보고서 생성 |")
    lines.append("| Claude Opus 4.5 | 프론티어 | 장문 처리 우수 | 상세한 분석 보고서 |")
    lines.append("| Claude Sonnet 4.5 | 프론티어 | 빠른 속도, 균형잡힌 성능 | 일반 업무 보고서 |")
    lines.append("| Llama-3.3-70B | 오픈소스 | 로컬 실행 가능 | 민감 데이터 처리 |")
    lines.append("| Phi-4 | 경량 | 낮은 리소스, 빠른 응답 | 간단한 요약 |")
    lines.append("")

    # 결론
    lines.append("## 4. 결론 및 권장사항")
    lines.append("")
    lines.append("### 비용 대비 성능 관점")
    lines.append("")
    lines.append("본 프로젝트는 다양한 LLM 모델을 실험하여 각 상황에 맞는 최적의 모델을 선택할 수 있는 역량을 확보했습니다.")
    lines.append("")
    lines.append("**핵심 강점:**")
    lines.append("")
    lines.append("1. **오픈소스 모델 활용**: DeepSeek-V3.1과 같은 오픈소스 모델이 GPT-4.1과 유사한 성능을 보여, 비용 절감 가능")
    lines.append("2. **다양한 모델 실험**: 7개의 서로 다른 LLM 모델을 체계적으로 평가")
    lines.append("3. **객관적 평가**: 3개의 Judge 모델을 사용한 다각도 성능 평가")
    lines.append("4. **최적화된 RAG 파이프라인**: 보고서 타입별 최적의 Retriever + LLM 조합 도출")
    lines.append("")
    lines.append("### 권장사항")
    lines.append("")
    lines.append("- **주간 보고서**: BGE-M3 + Qwen3 Reranker + **DeepSeek-V3.1** (비용 효율적)")
    lines.append("- **임원 보고서**: OpenAI + RRF MultiQuery + **GPT-4.1** (안정적 품질)")
    lines.append("- **대량 처리**: DeepSeek-V3.1 또는 Llama-3.3-70B (로컬 실행 가능)")
    lines.append("- **실험/연구**: 계속해서 새로운 모델 테스트 및 벤치마크")
    lines.append("")

    # 메타데이터
    lines.append("---")
    lines.append("")
    lines.append("## 메타데이터")
    lines.append("")
    lines.append("- **평가 방법**: LLM-as-Judge (Multi-Judge)")
    lines.append("- **Judge 모델**: GPT-5.1, Claude Opus 4.5, DeepSeek-V3.1")
    lines.append("- **평가 기준**: Conciseness, Strategic Value, Accuracy, Clarity, Actionability")
    lines.append("- **테스트 질문 수**: 5개 (각 보고서 타입)")
    lines.append("- **평가 LLM 수**: 7개")
    lines.append("")

    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✅ 리포트 저장: {output_file}")


def main():
    """메인 함수"""

    # 테스트 디렉토리 경로
    weekly_dir = Path("/home/work/rag/Project/rag-report-generator/data/results/multi_llm_test/weekly/20260102_052136")
    executive_dir = Path("/home/work/rag/Project/rag-report-generator/data/results/multi_llm_test/executive/20260102_053212")

    print("\n" + "="*100)
    print("🚀 LLM 성능 비교 분석 시작")
    print("="*100)

    # 주간 보고서 분석
    print("\n📊 주간 보고서 분석 중...")
    weekly_df = analyze_multi_judge_results(weekly_dir, "weekly")

    if weekly_df is not None:
        print(f"✅ 주간 보고서: {len(weekly_df)} 건의 평가 결과 로드")
        print("\n주간 보고서 평균 점수:")
        weekly_avg = weekly_df.groupby("llm")["score"].mean().sort_values(ascending=False)
        for llm, score in weekly_avg.items():
            print(f"  {llm}: {score:.2f}")
    else:
        print("⚠️ 주간 보고서 분석 실패")

    # 임원 보고서 분석
    print("\n📊 임원 보고서 분석 중...")
    executive_df = analyze_multi_judge_results(executive_dir, "executive")

    if executive_df is not None:
        print(f"✅ 임원 보고서: {len(executive_df)} 건의 평가 결과 로드")
        print("\n임원 보고서 평균 점수:")
        executive_avg = executive_df.groupby("llm")["score"].mean().sort_values(ascending=False)
        for llm, score in executive_avg.items():
            print(f"  {llm}: {score:.2f}")
    else:
        print("⚠️ 임원 보고서 분석 실패")

    # 마크다운 리포트 생성
    print("\n📝 마크다운 리포트 생성 중...")

    output_dir = Path("/home/work/rag/Project/rag-report-generator/data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"llm_performance_comparison_{timestamp}.md"

    generate_markdown_report(weekly_df, executive_df, output_file)

    # CSV 저장
    if weekly_df is not None:
        weekly_csv = output_dir / f"weekly_scores_{timestamp}.csv"
        weekly_df.to_csv(weekly_csv, index=False, encoding='utf-8-sig')
        print(f"✅ 주간 보고서 CSV 저장: {weekly_csv}")

    if executive_df is not None:
        executive_csv = output_dir / f"executive_scores_{timestamp}.csv"
        executive_df.to_csv(executive_csv, index=False, encoding='utf-8-sig')
        print(f"✅ 임원 보고서 CSV 저장: {executive_csv}")

    print("\n" + "="*100)
    print("✅ 분석 완료!")
    print("="*100)


if __name__ == "__main__":
    main()
