#!/usr/bin/env python3
"""육안 평가(Human Evaluation)와 LLM as Judge 평가 비교 분석 도구

사람의 육안 평가 결과와 LLM Judge의 자동 평가 결과를 비교하여
평가 방식의 차이, 상관관계, 편향 등을 분석합니다.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns


class HumanVsJudgeComparator:
    """육안 평가와 Judge 평가 비교 분석 클래스"""

    def __init__(self):
        self.human_scores = {}
        self.judge_scores = {}

    def load_human_evaluation(
        self,
        report_type: str,
        scores: Dict[str, float]
    ):
        """육안 평가 결과 로드

        Args:
            report_type: 'weekly' or 'executive'
            scores: {llm_name: score} 딕셔너리
        """
        self.human_scores[report_type] = scores
        print(f"✅ {report_type} 육안 평가 결과 로드: {len(scores)}개 LLM")

    def load_judge_evaluation(
        self,
        report_type: str,
        judge_results_path: str
    ):
        """Judge 평가 결과 로드

        Args:
            report_type: 'weekly' or 'executive'
            judge_results_path: Judge 평가 결과 JSON 파일 경로
        """
        with open(judge_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        scores = {}
        for llm_name, result in data.get('results', {}).items():
            scores[llm_name] = result.get('final_score', 0)

        self.judge_scores[report_type] = scores
        print(f"✅ {report_type} Judge 평가 결과 로드: {len(scores)}개 LLM")

    def normalize_llm_names(self, report_type: str):
        """LLM 이름 정규화 (매칭을 위해)

        육안 평가와 Judge 평가의 LLM 이름을 통일합니다.
        """
        # LLM 이름 매핑 규칙
        name_mapping = {
            # 육안 평가 이름 -> 표준 이름
            "GPT-4.1": "OpenAI GPT-4.1",
            "GPT-5.1": "OpenAI GPT-5.1",
            "DeepSeek-V3.1": "DeepSeek-V3.1",
            "Claude Sonnet 4.5": "Claude 4.5 Sonnet",
            "Claude Opus 4.5": "Claude 4.5 Opus",
            "Phi-4": "Phi-4",
            "Llama-3.3-70B": "Llama-3.3-70B-Instruct"
        }

        # 육안 평가 이름 변환
        if report_type in self.human_scores:
            normalized = {}
            for name, score in self.human_scores[report_type].items():
                std_name = name_mapping.get(name, name)
                normalized[std_name] = score
            self.human_scores[report_type] = normalized

    def compare_rankings(
        self,
        report_type: str
    ) -> Dict[str, Any]:
        """육안 평가와 Judge 평가의 순위 비교

        Args:
            report_type: 'weekly' or 'executive'

        Returns:
            비교 결과 딕셔너리
        """
        if report_type not in self.human_scores or report_type not in self.judge_scores:
            raise ValueError(f"{report_type}에 대한 평가 데이터가 없습니다.")

        human = self.human_scores[report_type]
        judge = self.judge_scores[report_type]

        # 공통 LLM만 비교
        common_llms = set(human.keys()) & set(judge.keys())

        if not common_llms:
            raise ValueError("육안 평가와 Judge 평가에 공통 LLM이 없습니다.")

        print(f"\n{'='*80}")
        print(f"{report_type.upper()} - 육안 평가 vs Judge 평가 비교")
        print(f"{'='*80}")
        print(f"공통 LLM 수: {len(common_llms)}")

        # 순위 계산
        human_ranks = self._get_rankings(human)
        judge_ranks = self._get_rankings(judge)

        # 비교 데이터 생성
        comparison = []
        for llm in common_llms:
            comparison.append({
                'llm': llm,
                'human_score': human[llm],
                'human_rank': human_ranks[llm],
                'judge_score': judge[llm],
                'judge_rank': judge_ranks[llm],
                'rank_diff': abs(human_ranks[llm] - judge_ranks[llm]),
                'score_diff': human[llm] - judge[llm]
            })

        # DataFrame 생성
        df = pd.DataFrame(comparison)
        df = df.sort_values('human_rank')

        # 통계 계산
        human_scores_list = [human[llm] for llm in common_llms]
        judge_scores_list = [judge[llm] for llm in common_llms]

        spearman_corr, spearman_p = spearmanr(human_scores_list, judge_scores_list)
        kendall_corr, kendall_p = kendalltau(human_scores_list, judge_scores_list)

        # Pearson 상관계수 (점수 기반)
        pearson_corr = np.corrcoef(human_scores_list, judge_scores_list)[0, 1]

        print(f"\n📊 상관관계 분석:")
        print(f"  Pearson 상관계수: {pearson_corr:.4f} (점수 기반)")
        print(f"  Spearman 상관계수: {spearman_corr:.4f} (p={spearman_p:.4f}) (순위 기반)")
        print(f"  Kendall Tau: {kendall_corr:.4f} (p={kendall_p:.4f}) (순위 일치도)")

        # 순위 차이 분석
        avg_rank_diff = df['rank_diff'].mean()
        max_rank_diff = df['rank_diff'].max()

        print(f"\n📈 순위 차이 분석:")
        print(f"  평균 순위 차이: {avg_rank_diff:.2f}")
        print(f"  최대 순위 차이: {max_rank_diff}")

        # 가장 큰 차이를 보인 LLM
        biggest_diff_llm = df.loc[df['rank_diff'].idxmax()]
        print(f"  가장 큰 차이: {biggest_diff_llm['llm']}")
        print(f"    육안: {biggest_diff_llm['human_rank']}위 ({biggest_diff_llm['human_score']}점)")
        print(f"    Judge: {biggest_diff_llm['judge_rank']}위 ({biggest_diff_llm['judge_score']:.2f}점)")

        # 상세 비교표 출력
        print(f"\n📋 상세 비교표:")
        print(df.to_string(index=False))

        return {
            'report_type': report_type,
            'num_llms': len(common_llms),
            'comparison_df': df,
            'correlations': {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'spearman_pvalue': spearman_p,
                'kendall': kendall_corr,
                'kendall_pvalue': kendall_p
            },
            'rank_differences': {
                'mean': avg_rank_diff,
                'max': max_rank_diff,
                'std': df['rank_diff'].std()
            }
        }

    def _get_rankings(self, scores: Dict[str, float]) -> Dict[str, int]:
        """점수를 기반으로 순위 계산"""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {llm: rank + 1 for rank, (llm, _) in enumerate(sorted_items)}

    def analyze_evaluation_criteria_differences(
        self,
        report_type: str,
        human_criteria_scores: Dict[str, Dict[str, float]],
        judge_results_path: str
    ):
        """평가 기준별 차이 분석

        Args:
            report_type: 'weekly' or 'executive'
            human_criteria_scores: {llm_name: {criterion: score}} 형식
            judge_results_path: Judge 평가 결과 JSON 파일 경로
        """
        with open(judge_results_path, 'r', encoding='utf-8') as f:
            judge_data = json.load(f)

        print(f"\n{'='*80}")
        print(f"{report_type.upper()} - 평가 기준별 차이 분석")
        print(f"{'='*80}")

        # Judge의 평가 기준별 점수 추출
        judge_criteria_scores = {}
        for llm_name, result in judge_data.get('results', {}).items():
            judge_criteria_scores[llm_name] = {}
            for criterion_result in result.get('criterion_results', []):
                criterion_key = criterion_result['criterion']
                # 10점 만점으로 정규화
                judge_criteria_scores[llm_name][criterion_key] = criterion_result['score']

        # 비교 분석
        common_llms = set(human_criteria_scores.keys()) & set(judge_criteria_scores.keys())

        for llm in common_llms:
            print(f"\n🤖 {llm}:")
            print(f"  육안 평가:")
            for criterion, score in human_criteria_scores[llm].items():
                print(f"    {criterion}: {score}")
            print(f"  Judge 평가:")
            for criterion, score in judge_criteria_scores[llm].items():
                print(f"    {criterion}: {score}")

    def identify_disagreements(
        self,
        report_type: str,
        threshold: int = 2
    ) -> List[Dict[str, Any]]:
        """큰 의견 차이를 보이는 LLM 식별

        Args:
            report_type: 'weekly' or 'executive'
            threshold: 순위 차이 임계값 (기본: 2)

        Returns:
            의견 차이가 큰 LLM 리스트
        """
        result = self.compare_rankings(report_type)
        df = result['comparison_df']

        disagreements = df[df['rank_diff'] >= threshold].to_dict('records')

        print(f"\n⚠️  큰 의견 차이 (순위 차이 >= {threshold}):")
        for item in disagreements:
            print(f"\n  {item['llm']}:")
            print(f"    육안: {item['human_rank']}위 ({item['human_score']}점)")
            print(f"    Judge: {item['judge_rank']}위 ({item['judge_score']:.2f}점)")
            print(f"    순위 차이: {item['rank_diff']}")

        return disagreements

    def generate_comparison_report(
        self,
        output_dir: str
    ):
        """종합 비교 리포트 생성

        Args:
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 각 보고서 타입별로 비교
        for report_type in self.human_scores.keys():
            if report_type not in self.judge_scores:
                continue

            # 비교 분석
            result = self.compare_rankings(report_type)

            # CSV 저장
            csv_path = output_path / f"comparison_{report_type}.csv"
            result['comparison_df'].to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 비교표 저장: {csv_path}")

            # Markdown 리포트 생성
            md_path = output_path / f"comparison_report_{report_type}.md"
            self._generate_markdown_report(result, md_path)
            print(f"📝 리포트 저장: {md_path}")

            # JSON 저장
            json_path = output_path / f"comparison_{report_type}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'report_type': result['report_type'],
                    'num_llms': result['num_llms'],
                    'correlations': result['correlations'],
                    'rank_differences': result['rank_differences'],
                    'comparison': result['comparison_df'].to_dict('records')
                }, f, indent=2, ensure_ascii=False)
            print(f"💾 JSON 저장: {json_path}")

    def _generate_markdown_report(
        self,
        result: Dict[str, Any],
        output_path: Path
    ):
        """Markdown 리포트 생성"""
        df = result['comparison_df']
        report_type = result['report_type']

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# 육안 평가 vs LLM Judge 비교 리포트\n\n")
            f.write(f"**보고서 타입**: {report_type}\n\n")
            f.write(f"**분석 LLM 수**: {result['num_llms']}\n\n")
            f.write("---\n\n")

            # 상관관계
            f.write("## 📊 상관관계 분석\n\n")
            f.write(f"- **Pearson 상관계수**: {result['correlations']['pearson']:.4f}\n")
            f.write(f"- **Spearman 순위 상관계수**: {result['correlations']['spearman']:.4f} (p={result['correlations']['spearman_pvalue']:.4f})\n")
            f.write(f"- **Kendall Tau**: {result['correlations']['kendall']:.4f} (p={result['correlations']['kendall_pvalue']:.4f})\n\n")

            # 해석
            spearman = result['correlations']['spearman']
            if spearman > 0.8:
                interpretation = "✅ **매우 강한 양의 상관관계** - 육안 평가와 Judge 평가가 매우 유사합니다."
            elif spearman > 0.6:
                interpretation = "✓ **강한 양의 상관관계** - 육안 평가와 Judge 평가가 대체로 일치합니다."
            elif spearman > 0.4:
                interpretation = "⚠️ **중간 정도의 상관관계** - 일부 차이가 있습니다."
            else:
                interpretation = "❌ **약한 상관관계** - 육안 평가와 Judge 평가의 차이가 큽니다."

            f.write(f"{interpretation}\n\n")

            # 순위 차이
            f.write("## 📈 순위 차이 분석\n\n")
            f.write(f"- **평균 순위 차이**: {result['rank_differences']['mean']:.2f}\n")
            f.write(f"- **최대 순위 차이**: {result['rank_differences']['max']}\n")
            f.write(f"- **표준편차**: {result['rank_differences']['std']:.2f}\n\n")

            # 상세 비교표
            f.write("## 📋 상세 비교표\n\n")
            f.write("| LLM | 육안 점수 | 육안 순위 | Judge 점수 | Judge 순위 | 순위 차이 |\n")
            f.write("|-----|-----------|-----------|------------|-----------|----------|\n")
            for _, row in df.iterrows():
                f.write(f"| {row['llm']} | {row['human_score']:.1f} | {row['human_rank']} | ")
                f.write(f"{row['judge_score']:.2f} | {row['judge_rank']} | {row['rank_diff']} |\n")

            f.write("\n")

            # 큰 차이를 보인 LLM
            big_diff = df[df['rank_diff'] >= 2]
            if len(big_diff) > 0:
                f.write("## ⚠️ 큰 의견 차이를 보인 LLM\n\n")
                for _, row in big_diff.iterrows():
                    f.write(f"### {row['llm']}\n\n")
                    f.write(f"- 육안 평가: {row['human_rank']}위 ({row['human_score']}점)\n")
                    f.write(f"- Judge 평가: {row['judge_rank']}위 ({row['judge_score']:.2f}점)\n")
                    f.write(f"- 순위 차이: {row['rank_diff']}\n\n")


def main():
    """메인 함수 - 예제 사용법"""
    comparator = HumanVsJudgeComparator()

    # 육안 평가 결과 입력 (귀하의 평가 결과)
    weekly_human_scores = {
        "OpenAI GPT-4.1": 91,
        "DeepSeek-V3.1": 90,
        "Claude 4.5 Sonnet": 88,
        "Claude 4.5 Opus": 86,
        "Phi-4": 82,
        "OpenAI GPT-5.1": 81,
        "Llama-3.3-70B-Instruct": 72
    }

    executive_human_scores = {
        "OpenAI GPT-4.1": 87.5,
        "OpenAI GPT-5.1": 85.5,
        "DeepSeek-V3.1": 79.0,
        "Claude 4.5 Opus": 64.0,
        "Claude 4.5 Sonnet": 63.5,
        "Phi-4": 54.5,
        "Llama-3.3-70B-Instruct": 28.0
    }

    # 육안 평가 로드
    comparator.load_human_evaluation('weekly', weekly_human_scores)
    comparator.load_human_evaluation('executive', executive_human_scores)

    # Judge 평가 결과 로드 (실제 파일이 있다면)
    # comparator.load_judge_evaluation(
    #     'weekly',
    #     'data/results/multi_llm_test/weekly/20260102_052136/judge_evaluation.json'
    # )

    # 비교 분석
    # comparator.compare_rankings('weekly')

    # 리포트 생성
    # comparator.generate_comparison_report('data/results/evaluation_comparison')

    print("\n✅ 비교 분석 도구 로드 완료!")
    print("📖 사용 예시:")
    print("  1. comparator.load_judge_evaluation('weekly', 'judge_results.json')")
    print("  2. comparator.compare_rankings('weekly')")
    print("  3. comparator.generate_comparison_report('output_dir')")


if __name__ == "__main__":
    main()
