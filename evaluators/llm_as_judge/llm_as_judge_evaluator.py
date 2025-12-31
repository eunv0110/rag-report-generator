#!/usr/bin/env python3
"""LLM as Judge 평가기 - 생성된 보고서의 품질을 LLM으로 평가

여러 LLM이 생성한 주간/월간 보고서를 평가하여 어떤 보고서가 더 적합한지 판단합니다.
Azure AI 및 OpenRouter를 통한 다양한 Judge 모델 지원.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.chat_models import init_chat_model
from openai import OpenAI
import pandas as pd


class LLMAsJudgeEvaluator:
    """LLM as Judge 평가 클래스

    Azure AI 또는 OpenRouter를 통해 다양한 Judge 모델을 사용하여
    생성된 보고서의 품질을 평가합니다.
    """

    # 평가 기준 정의
    EVALUATION_CRITERIA = {
        "weekly_report": {
            "completeness": {
                "name": "완전성 (Completeness)",
                "description": "주요 지표, 활동, 이슈, 다음 주 계획이 모두 포함되었는가?",
                "weight": 0.25
            },
            "relevance": {
                "name": "관련성 (Relevance)",
                "description": "질문과 관련된 정보만 포함하고 불필요한 정보는 제외되었는가?",
                "weight": 0.20
            },
            "accuracy": {
                "name": "정확성 (Accuracy)",
                "description": "검색된 문서의 내용을 정확하게 반영하고 있는가? 환각은 없는가?",
                "weight": 0.25
            },
            "structure": {
                "name": "구조화 (Structure)",
                "description": "주간 보고서 형식에 적합하게 구조화되었는가?",
                "weight": 0.15
            },
            "readability": {
                "name": "가독성 (Readability)",
                "description": "읽기 쉽고 이해하기 쉬운가? 적절한 포맷팅이 되어 있는가?",
                "weight": 0.15
            }
        },
        "executive_report": {
            "conciseness": {
                "name": "간결성 (Conciseness)",
                "description": "핵심만 간결하게 요약되었는가? 불필요한 세부사항은 없는가?",
                "weight": 0.25
            },
            "strategic_value": {
                "name": "전략적 가치 (Strategic Value)",
                "description": "경영진이 의사결정에 활용할 수 있는 인사이트를 제공하는가?",
                "weight": 0.25
            },
            "accuracy": {
                "name": "정확성 (Accuracy)",
                "description": "검색된 문서의 내용을 정확하게 반영하고 있는가? 환각은 없는가?",
                "weight": 0.25
            },
            "clarity": {
                "name": "명확성 (Clarity)",
                "description": "명확하고 이해하기 쉬운 언어로 작성되었는가?",
                "weight": 0.15
            },
            "priority": {
                "name": "우선순위 (Priority)",
                "description": "중요한 정보가 먼저 제시되고 우선순위가 명확한가?",
                "weight": 0.10
            }
        }
    }

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        provider: str = "azure_ai",
        temperature: float = 0
    ):
        """
        Args:
            judge_model: 평가에 사용할 LLM 모델
                - Azure AI: gpt-4o, gpt-4.5, o1 등
                - OpenRouter: anthropic/claude-opus-4.5 등
            provider: 사용할 제공자 ("azure_ai" 또는 "openrouter")
            temperature: 생성 온도 (기본: 0 - 일관된 평가를 위함)
        """
        self.judge_model = judge_model
        self.provider = provider
        self.temperature = temperature

        # Judge LLM 초기화
        if provider == "azure_ai":
            from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT

            self.llm = init_chat_model(
                model=f"azure_ai:{judge_model}",
                api_key=AZURE_AI_CREDENTIAL,
                azure_endpoint=AZURE_AI_ENDPOINT,
                temperature=temperature
            )
            self.client = None
        elif provider == "openrouter":
            from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

            self.llm = None
            self.client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL
            )
        else:
            raise ValueError(f"지원하지 않는 provider: {provider}. 'azure_ai' 또는 'openrouter'를 사용하세요.")

    def _load_prompt_template(self, template_name: str) -> str:
        """프롬프트 템플릿 로드

        Args:
            template_name: 템플릿 파일 이름

        Returns:
            프롬프트 템플릿 문자열
        """
        template_path = Path(__file__).parent.parent.parent / "prompts" / "templates" / "evaluation" / "llm_judge" / template_name
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _create_evaluation_prompt(
        self,
        question: str,
        answer: str,
        report_type: str,
        criteria_name: str,
        criteria_desc: str
    ) -> str:
        """평가 프롬프트 생성

        Args:
            question: 원본 질문
            answer: 평가할 답변
            report_type: 보고서 타입 (weekly_report, executive_report)
            criteria_name: 평가 기준 이름
            criteria_desc: 평가 기준 설명

        Returns:
            프롬프트 문자열
        """
        report_type_ko = "주간 보고서" if report_type == "weekly_report" else "경영진 보고서"

        # 프롬프트 템플릿 로드
        template = self._load_prompt_template("criterion_evaluation_prompt.txt")

        # 템플릿에 값 채우기
        prompt = template.replace("{report_type_ko}", report_type_ko)\
                        .replace("{criteria_name}", criteria_name)\
                        .replace("{criteria_desc}", criteria_desc)\
                        .replace("{question}", question)\
                        .replace("{answer}", answer)

        return prompt

    def _evaluate_single_criterion(
        self,
        question: str,
        answer: str,
        report_type: str,
        criterion_key: str,
        criterion_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """단일 평가 기준으로 답변 평가

        Args:
            question: 질문
            answer: 답변
            report_type: 보고서 타입
            criterion_key: 평가 기준 키
            criterion_info: 평가 기준 정보

        Returns:
            평가 결과
        """
        prompt = self._create_evaluation_prompt(
            question=question,
            answer=answer,
            report_type=report_type,
            criteria_name=criterion_info["name"],
            criteria_desc=criterion_info["description"]
        )

        messages = [
            {"role": "system", "content": "당신은 보고서 품질을 평가하는 전문가입니다. 공정하고 객관적으로 평가해주세요."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Provider에 따라 다른 방식으로 호출
            if self.provider == "azure_ai":
                response = self.llm.invoke(messages)
                result = json.loads(response.content)
            else:  # openrouter
                response = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=messages,
                    max_tokens=4000,
                    temperature=self.temperature
                )
                content = response.choices[0].message.content

                # JSON 추출 (```json ... ``` 형식 지원)
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    result = json.loads(content)

            return {
                "criterion": criterion_key,
                "criterion_name": criterion_info["name"],
                "weight": criterion_info["weight"],
                "score": result["score"],
                "weighted_score": result["score"] * criterion_info["weight"],
                "reasoning": result["reasoning"],
                "strengths": result.get("strengths", []),
                "weaknesses": result.get("weaknesses", [])
            }
        except Exception as e:
            print(f"⚠️  평가 중 오류 발생 ({criterion_key}): {e}")
            return {
                "criterion": criterion_key,
                "criterion_name": criterion_info["name"],
                "weight": criterion_info["weight"],
                "score": 0,
                "weighted_score": 0,
                "reasoning": f"평가 실패: {str(e)}",
                "strengths": [],
                "weaknesses": []
            }

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        report_type: str = "weekly_report"
    ) -> Dict[str, Any]:
        """답변을 모든 기준으로 평가

        Args:
            question: 질문
            answer: 답변
            report_type: 보고서 타입 (weekly_report, executive_report)

        Returns:
            종합 평가 결과
        """
        if report_type not in self.EVALUATION_CRITERIA:
            raise ValueError(f"지원하지 않는 보고서 타입: {report_type}")

        criteria = self.EVALUATION_CRITERIA[report_type]
        criterion_results = []

        print(f"\n{'='*80}")
        print(f"📊 {report_type} 평가 시작")
        print(f"{'='*80}")

        # 각 평가 기준별로 평가
        for criterion_key, criterion_info in criteria.items():
            print(f"\n평가 기준: {criterion_info['name']}")

            result = self._evaluate_single_criterion(
                question=question,
                answer=answer,
                report_type=report_type,
                criterion_key=criterion_key,
                criterion_info=criterion_info
            )

            criterion_results.append(result)
            print(f"  점수: {result['score']}/10 (가중치: {result['weight']}, 가중 점수: {result['weighted_score']:.2f})")

        # 종합 점수 계산
        total_weighted_score = sum(r["weighted_score"] for r in criterion_results)
        total_max_score = sum(10 * r["weight"] for r in criterion_results)
        final_score = (total_weighted_score / total_max_score) * 10

        print(f"\n{'='*80}")
        print(f"✅ 최종 점수: {final_score:.2f}/10")
        print(f"{'='*80}")

        return {
            "report_type": report_type,
            "final_score": final_score,
            "total_weighted_score": total_weighted_score,
            "criterion_results": criterion_results,
            "timestamp": datetime.now().isoformat()
        }

    def compare_multiple_answers(
        self,
        question: str,
        answers: Dict[str, str],
        report_type: str = "weekly_report"
    ) -> Dict[str, Any]:
        """여러 답변을 비교 평가

        Args:
            question: 질문
            answers: {llm_name: answer} 형식의 딕셔너리
            report_type: 보고서 타입

        Returns:
            비교 평가 결과
        """
        results = {}

        print(f"\n{'='*80}")
        print(f"🔍 {len(answers)}개 답변 비교 평가")
        print(f"{'='*80}")

        # 각 답변 평가
        for llm_name, answer in answers.items():
            print(f"\n[{llm_name}] 평가 중...")
            results[llm_name] = self.evaluate_answer(question, answer, report_type)

        # 랭킹 생성
        ranking = sorted(
            results.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True
        )

        print(f"\n{'='*80}")
        print("🏆 최종 순위")
        print(f"{'='*80}")
        for rank, (llm_name, result) in enumerate(ranking, 1):
            print(f"{rank}위: {llm_name:<20} - {result['final_score']:.2f}/10")

        return {
            "question": question,
            "report_type": report_type,
            "num_answers": len(answers),
            "results": results,
            "ranking": [(name, result["final_score"]) for name, result in ranking],
            "timestamp": datetime.now().isoformat()
        }

    def evaluate_from_results_dir(
        self,
        results_dir: str,
        report_type: str = "weekly_report",
        output_path: str = None
    ) -> Dict[str, Any]:
        """결과 디렉토리에서 모든 LLM 답변을 로드하고 평가

        Args:
            results_dir: 결과 디렉토리 경로 (예: .../llm_comparison/bge_m3_rrf_ensemble_20251228_114425)
            report_type: 보고서 타입
            output_path: 결과 저장 경로 (None이면 자동 생성)

        Returns:
            평가 결과
        """
        results_path = Path(results_dir)

        if not results_path.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {results_dir}")

        # 각 LLM 디렉토리에서 results.json 로드
        answers = {}
        question = None

        for llm_dir in results_path.iterdir():
            if not llm_dir.is_dir():
                continue

            result_file = llm_dir / "results.json"
            if not result_file.exists():
                print(f"⚠️  {llm_dir.name}의 results.json을 찾을 수 없습니다.")
                continue

            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data.get("results") and len(data["results"]) > 0:
                    result = data["results"][0]

                    if question is None:
                        question = result.get("question", "")

                    if result.get("result", {}).get("success"):
                        answers[data["llm_name"]] = result["result"]["answer"]
                        print(f"✅ {data['llm_name']} 답변 로드 완료")
            except Exception as e:
                print(f"⚠️  {llm_dir.name} 로드 중 오류: {e}")

        if not answers:
            raise ValueError("평가할 답변을 찾을 수 없습니다.")

        if not question:
            raise ValueError("질문을 찾을 수 없습니다.")

        # 비교 평가 실행
        comparison_result = self.compare_multiple_answers(question, answers, report_type)

        # 결과 저장
        if output_path is None:
            output_path = results_path / f"llm_judge_evaluation_{report_type}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False)

        print(f"\n💾 평가 결과 저장: {output_path}")

        # 상세 보고서 생성 (CSV)
        self._save_detailed_report(comparison_result, output_path.parent / f"llm_judge_report_{report_type}.csv")

        return comparison_result

    def batch_evaluate_from_dir(
        self,
        base_dir: str,
        report_type: str = "weekly_report"
    ) -> Dict[str, Any]:
        """여러 retriever 결과를 일괄 평가

        Args:
            base_dir: llm_comparison 디렉토리 경로
            report_type: 보고서 타입

        Returns:
            전체 평가 결과
        """
        base_path = Path(base_dir)

        if not base_path.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {base_dir}")

        all_results = {}

        # 각 retriever 디렉토리 순회
        for retriever_dir in sorted(base_path.iterdir()):
            if not retriever_dir.is_dir():
                continue

            retriever_name = retriever_dir.name
            print(f"\n{'='*100}")
            print(f"📂 {retriever_name} 평가 중...")
            print(f"{'='*100}")

            try:
                result = self.evaluate_from_results_dir(
                    results_dir=str(retriever_dir),
                    report_type=report_type
                )
                all_results[retriever_name] = result
                print(f"✅ {retriever_name} 평가 완료")
            except Exception as e:
                print(f"❌ {retriever_name} 평가 실패: {e}")
                all_results[retriever_name] = {"error": str(e)}

        # 전체 결과 저장
        output_path = base_path / f"all_evaluations_{report_type}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 전체 평가 결과 저장: {output_path}")

        # 요약 리포트 생성
        self._create_summary_report(all_results, base_path / f"summary_{report_type}.csv")

        return all_results

    def _create_summary_report(self, all_results: Dict[str, Any], output_path: Path):
        """요약 리포트 생성

        Args:
            all_results: 전체 평가 결과
            output_path: CSV 출력 경로
        """
        rows = []

        for retriever_name, result in all_results.items():
            if "error" in result:
                continue

            for llm_name, llm_result in result.get("results", {}).items():
                row = {
                    "retriever": retriever_name,
                    "llm_name": llm_name,
                    "final_score": llm_result["final_score"]
                }

                # 각 평가 기준별 점수 추가
                for criterion_result in llm_result["criterion_results"]:
                    criterion_key = criterion_result["criterion"]
                    row[f"{criterion_key}_score"] = criterion_result["score"]
                    row[f"{criterion_key}_weighted"] = criterion_result["weighted_score"]

                rows.append(row)

        df = pd.DataFrame(rows)

        # 정렬: retriever별, 최종 점수 내림차순
        df = df.sort_values(["retriever", "final_score"], ascending=[True, False])

        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"📊 요약 리포트 저장: {output_path}")

        # 콘솔 출력
        print(f"\n{'='*100}")
        print("🏆 Retriever별 최고 점수 LLM")
        print(f"{'='*100}")

        for retriever in df["retriever"].unique():
            retriever_df = df[df["retriever"] == retriever]
            top_llm = retriever_df.iloc[0]
            print(f"\n{retriever}:")
            print(f"  1위: {top_llm['llm_name']:<20} - {top_llm['final_score']:.2f}/10")

            if len(retriever_df) > 1:
                second_llm = retriever_df.iloc[1]
                print(f"  2위: {second_llm['llm_name']:<20} - {second_llm['final_score']:.2f}/10")

        # 전체 LLM별 평균 점수
        print(f"\n{'='*100}")
        print("🎯 LLM별 평균 점수 (모든 retriever)")
        print(f"{'='*100}")

        llm_avg = df.groupby("llm_name")["final_score"].mean().sort_values(ascending=False)
        for llm_name, avg_score in llm_avg.items():
            print(f"  {llm_name:<20} - {avg_score:.2f}/10")

    def _save_detailed_report(self, comparison_result: Dict[str, Any], output_path: Path):
        """상세 평가 보고서를 CSV로 저장

        Args:
            comparison_result: 비교 평가 결과
            output_path: CSV 출력 경로
        """
        rows = []

        for llm_name, result in comparison_result["results"].items():
            for criterion_result in result["criterion_results"]:
                rows.append({
                    "llm_name": llm_name,
                    "final_score": result["final_score"],
                    "criterion": criterion_result["criterion"],
                    "criterion_name": criterion_result["criterion_name"],
                    "weight": criterion_result["weight"],
                    "score": criterion_result["score"],
                    "weighted_score": criterion_result["weighted_score"],
                    "reasoning": criterion_result["reasoning"],
                    "strengths": " | ".join(criterion_result["strengths"]),
                    "weaknesses": " | ".join(criterion_result["weaknesses"])
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"📊 상세 보고서 저장: {output_path}")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM as Judge 평가기 - 생성된 보고서의 품질을 LLM으로 평가"
    )

    # 모드 선택
    subparsers = parser.add_subparsers(dest="mode", help="실행 모드")

    # 단일 디렉토리 평가
    single_parser = subparsers.add_parser("single", help="단일 결과 디렉토리 평가")
    single_parser.add_argument("--results-dir", required=True, help="결과 디렉토리 경로")
    single_parser.add_argument("--report-type", default="weekly_report",
                              choices=["weekly_report", "executive_report"],
                              help="보고서 타입")
    single_parser.add_argument("--output", help="결과 저장 경로 (선택)")

    # 배치 평가
    batch_parser = subparsers.add_parser("batch", help="여러 retriever 결과 일괄 평가")
    batch_parser.add_argument("--base-dir", required=True,
                             help="llm_comparison 디렉토리 경로")
    batch_parser.add_argument("--report-type", default="weekly_report",
                             choices=["weekly_report", "executive_report"],
                             help="보고서 타입")

    # 공통 옵션
    for p in [single_parser, batch_parser]:
        p.add_argument("--judge-model", default="gpt-4o",
                      help="평가에 사용할 LLM 모델 (예: gpt-4o, anthropic/claude-opus-4.5)")
        p.add_argument("--provider", default="azure_ai",
                      choices=["azure_ai", "openrouter"],
                      help="LLM 제공자")
        p.add_argument("--temperature", type=float, default=0,
                      help="생성 온도 (기본: 0)")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    # 평가기 생성
    evaluator = LLMAsJudgeEvaluator(
        judge_model=args.judge_model,
        provider=args.provider,
        temperature=args.temperature
    )

    print(f"\n{'='*100}")
    print(f"🤖 Judge 모델: {args.judge_model} (provider: {args.provider})")
    print(f"{'='*100}\n")

    # 모드별 실행
    if args.mode == "single":
        evaluator.evaluate_from_results_dir(
            results_dir=args.results_dir,
            report_type=args.report_type,
            output_path=args.output
        )
    elif args.mode == "batch":
        evaluator.batch_evaluate_from_dir(
            base_dir=args.base_dir,
            report_type=args.report_type
        )

    print("\n✅ 모든 평가 완료!")


if __name__ == "__main__":
    main()
