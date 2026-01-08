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
            "accuracy": {
                "name": "정확성 (Accuracy)",
                "description": "검색된 문서의 내용을 정확하게 반영하고 있는가? 환각은 없는가?",
                "weight": 0.30
            },
            "completeness": {
                "name": "완결성 (Completeness)",
                "description": "주요 지표, 활동, 이슈, 다음 주 계획 등 필요한 정보가 빠짐없이 포함되었는가?",
                "weight": 0.25
            },
            "structure": {
                "name": "구조화 (Structure)",
                "description": "주간 보고서 형식에 적합하게 논리적으로 구조화되었는가?",
                "weight": 0.25
            },
            "detail": {
                "name": "상세도 (Detail)",
                "description": "적절한 수준의 상세 정보를 제공하는가? 너무 간략하거나 지나치게 장황하지 않은가?",
                "weight": 0.20
            }
        },
        "executive_report": {
            "structural_completeness": {
                "name": "구조 완성도 (Structural Completeness)",
                "description": "보고서 구조가 논리적이고 완성도가 높은가? 경영진 보고서로서 적절한 구조인가?",
                "weight": 0.30
            },
            "document_reference_accuracy": {
                "name": "문서 참조 정확성 (Document Reference Accuracy)",
                "description": "검색된 문서의 내용을 정확하게 참조하고 반영하고 있는가? 환각은 없는가?",
                "weight": 0.25
            },
            "practical_value": {
                "name": "내용 실용성 (Practical Value)",
                "description": "경영진이 의사결정에 실제로 활용할 수 있는 실용적인 정보와 인사이트를 제공하는가?",
                "weight": 0.25
            },
            "conciseness": {
                "name": "간결성 (Conciseness)",
                "description": "핵심만 간결하게 요약되었는가? 불필요한 세부사항 없이 명료한가?",
                "weight": 0.20
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
        criterion_info: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """단일 평가 기준으로 답변 평가

        Args:
            question: 질문
            answer: 답변
            report_type: 보고서 타입
            criterion_key: 평가 기준 키
            criterion_info: 평가 기준 정보
            max_retries: 최대 재시도 횟수

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
            {"role": "system", "content": "당신은 보고서 품질을 평가하는 전문가입니다. 공정하고 객관적으로 평가해주세요. 반드시 유효한 JSON 형식으로 응답해야 합니다."},
            {"role": "user", "content": prompt}
        ]

        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"   🔄 재시도 {attempt}/{max_retries-1}...")
                    import time
                    time.sleep(2)  # 재시도 전 2초 대기

                # Provider에 따라 다른 방식으로 호출
                if self.provider == "azure_ai":
                    # Azure AI의 경우 - langchain을 통한 호출
                    # max_completion_tokens 또는 max_tokens를 시도
                    try:
                        # GPT-5.1 등 새 모델은 max_completion_tokens 사용
                        response = self.llm.invoke(messages, max_completion_tokens=8000)
                        content = response.content
                    except Exception as e:
                        error_msg = str(e)
                        # max_tokens로 재시도
                        if "max_completion_tokens" in error_msg or "unsupported" in error_msg.lower():
                            try:
                                response = self.llm.invoke(messages, max_tokens=8000)
                                content = response.content
                            except:
                                # 둘 다 안되면 파라미터 없이 호출
                                response = self.llm.invoke(messages)
                                content = response.content
                        else:
                            # 다른 에러면 파라미터 없이 재시도
                            response = self.llm.invoke(messages)
                            content = response.content
                else:  # openrouter
                    # JSON 모드를 시도하되, 실패하면 일반 모드로 폴백
                    try:
                        response = self.client.chat.completions.create(
                            model=self.judge_model,
                            messages=messages,
                            max_tokens=8000,  # 토큰 제한 증가 (4000 -> 8000)
                            temperature=self.temperature,
                            response_format={"type": "json_object"}  # JSON 모드 활성화
                        )
                        content = response.choices[0].message.content
                    except Exception as json_mode_error:
                        # JSON 모드 실패 시 일반 모드로 재시도
                        if attempt == 0:
                            print(f"⚠️  JSON 모드 실패, 일반 모드로 재시도: {json_mode_error}")
                        response = self.client.chat.completions.create(
                            model=self.judge_model,
                            messages=messages,
                            max_tokens=8000,
                            temperature=self.temperature
                        )
                        content = response.choices[0].message.content

                # 응답 내용 확인
                if not content or content.strip() == "":
                    raise ValueError("LLM이 빈 응답을 반환했습니다")

                # JSON 추출 - 다양한 형식 지원
                result = None

                # 1. ```json ... ``` 형식
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass

                # 2. ``` ... ``` 형식 (json 키워드 없이)
                if result is None:
                    json_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass

                # 3. 순수 JSON (코드 블록 없이)
                if result is None:
                    try:
                        result = json.loads(content)
                    except json.JSONDecodeError:
                        # 4. JSON 객체 부분만 추출 시도 (non-greedy)
                        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(0))
                            except json.JSONDecodeError:
                                pass

                        # 5. 가장 긴 JSON 객체 추출 시도 (greedy, 불완전한 JSON 처리)
                        if result is None:
                            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                            if json_match:
                                try:
                                    result = json.loads(json_match.group(0))
                                except json.JSONDecodeError:
                                    pass

                # JSON 파싱 실패 시 - 부분 데이터 수동 추출 시도
                if result is None:
                    if attempt == 0:
                        print(f"⚠️  JSON 파싱 실패. 수동 추출 시도 중...")
                        print(f"원본 응답:\n{content[:500]}")

                    # 수동으로 필드 추출
                    score_match = re.search(r'"score"\s*:\s*(\d+)', content)
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', content, re.DOTALL)

                    if score_match and reasoning_match:
                        result = {
                            "score": int(score_match.group(1)),
                            "reasoning": reasoning_match.group(1),
                            "strengths": [],
                            "weaknesses": []
                        }

                        # strengths 추출 시도
                        strengths_match = re.search(r'"strengths"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                        if strengths_match:
                            strengths_str = strengths_match.group(1)
                            strengths = re.findall(r'"([^"]*)"', strengths_str)
                            result["strengths"] = strengths

                        # weaknesses 추출 시도
                        weaknesses_match = re.search(r'"weaknesses"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                        if weaknesses_match:
                            weaknesses_str = weaknesses_match.group(1)
                            weaknesses = re.findall(r'"([^"]*)"', weaknesses_str)
                            result["weaknesses"] = weaknesses

                        print(f"✅ 수동 추출 성공: score={result['score']}")
                    else:
                        raise ValueError("JSON 파싱 실패 - score 또는 reasoning을 찾을 수 없음")

                # 성공 시 결과 반환
                return {
                    "criterion": criterion_key,
                    "criterion_name": criterion_info["name"],
                    "weight": criterion_info["weight"],
                    "score": result.get("score", 0),
                    "weighted_score": result.get("score", 0) * criterion_info["weight"],
                    "reasoning": result.get("reasoning", ""),
                    "strengths": result.get("strengths", []),
                    "weaknesses": result.get("weaknesses", [])
                }

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"⚠️  시도 {attempt + 1} 실패: {e}")
                continue

        # 모든 재시도 실패 시
        print(f"❌ 평가 중 오류 발생 ({criterion_key}): {last_error}")
        print(f"   모델: {self.judge_model}, Provider: {self.provider}")
        print(f"   {max_retries}번 재시도 모두 실패")
        import traceback
        print(f"   상세 오류:\n{traceback.format_exc()}")
        return {
            "criterion": criterion_key,
            "criterion_name": criterion_info["name"],
            "weight": criterion_info["weight"],
            "score": 0,
            "weighted_score": 0,
            "reasoning": f"평가 실패 ({max_retries}번 재시도): {str(last_error)}",
            "strengths": [],
            "weaknesses": [],
            "error": str(last_error)
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

    def analyze_judge_bias(
        self,
        all_evaluations: List[Dict[str, Any]],
        judge_model: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Judge 모델의 평가 편향 분석

        자사 모델 선호 편향, 일관성 등을 분석합니다.

        Args:
            all_evaluations: 모든 평가 결과 리스트
            judge_model: 평가에 사용한 모델
            output_dir: 결과 저장 디렉토리

        Returns:
            편향 분석 결과
        """
        print(f"\n{'='*100}")
        print(f"🔍 Judge 모델 편향 분석: {judge_model}")
        print(f"{'='*100}")

        # 제조사별 LLM 분류
        vendor_map = {
            "openai": ["gpt", "o1"],
            "anthropic": ["claude"],
            "meta": ["llama"],
            "microsoft": ["phi"],
            "deepseek": ["deepseek"],
            "google": ["gemini"],
            "alibaba": ["qwen"]
        }

        # Judge 모델의 제조사 식별
        judge_vendor = None
        for vendor, keywords in vendor_map.items():
            if any(keyword in judge_model.lower() for keyword in keywords):
                judge_vendor = vendor
                break

        print(f"📌 Judge 모델 제조사: {judge_vendor or '알 수 없음'}")

        # LLM별 점수 및 순위 수집
        llm_rankings = {}
        llm_scores = {}
        llm_vendors = {}

        for evaluation in all_evaluations:
            for rank, (llm_name, score) in enumerate(evaluation['ranking'], 1):
                if llm_name not in llm_rankings:
                    llm_rankings[llm_name] = []
                    llm_scores[llm_name] = []

                    # LLM 제조사 식별
                    llm_vendor = None
                    for vendor, keywords in vendor_map.items():
                        if any(keyword in llm_name.lower() for keyword in keywords):
                            llm_vendor = vendor
                            break
                    llm_vendors[llm_name] = llm_vendor

                llm_rankings[llm_name].append(rank)
                llm_scores[llm_name].append(score)

        # 통계 계산
        stats = []
        for llm_name in llm_scores.keys():
            scores = llm_scores[llm_name]
            rankings = llm_rankings[llm_name]
            llm_vendor = llm_vendors[llm_name]
            is_same_vendor = (llm_vendor == judge_vendor) if (llm_vendor and judge_vendor) else False

            stats.append({
                "llm_name": llm_name,
                "llm_vendor": llm_vendor or "unknown",
                "is_same_vendor": is_same_vendor,
                "avg_score": sum(scores) / len(scores),
                "std_score": pd.Series(scores).std(),
                "avg_rank": sum(rankings) / len(rankings),
                "first_place_count": rankings.count(1),
                "last_place_count": rankings.count(max(rankings)),
                "num_evaluations": len(scores)
            })

        # DataFrame 생성 및 정렬
        df_stats = pd.DataFrame(stats)
        df_stats = df_stats.sort_values("avg_score", ascending=False)

        # 자사 vs 타사 비교
        if judge_vendor:
            same_vendor_df = df_stats[df_stats["is_same_vendor"] == True]
            other_vendor_df = df_stats[df_stats["is_same_vendor"] == False]

            bias_analysis = {
                "judge_model": judge_model,
                "judge_vendor": judge_vendor,
                "same_vendor_avg_score": same_vendor_df["avg_score"].mean() if len(same_vendor_df) > 0 else 0,
                "other_vendor_avg_score": other_vendor_df["avg_score"].mean() if len(other_vendor_df) > 0 else 0,
                "same_vendor_avg_rank": same_vendor_df["avg_rank"].mean() if len(same_vendor_df) > 0 else 0,
                "other_vendor_avg_rank": other_vendor_df["avg_rank"].mean() if len(other_vendor_df) > 0 else 0,
                "score_difference": (same_vendor_df["avg_score"].mean() - other_vendor_df["avg_score"].mean()) if (len(same_vendor_df) > 0 and len(other_vendor_df) > 0) else 0,
                "num_same_vendor_llms": len(same_vendor_df),
                "num_other_vendor_llms": len(other_vendor_df)
            }

            print(f"\n📊 자사 모델 편향 분석:")
            print(f"  자사 모델 평균 점수: {bias_analysis['same_vendor_avg_score']:.3f}")
            print(f"  타사 모델 평균 점수: {bias_analysis['other_vendor_avg_score']:.3f}")
            print(f"  점수 차이: {bias_analysis['score_difference']:.3f}")
            print(f"  자사 모델 평균 순위: {bias_analysis['same_vendor_avg_rank']:.2f}")
            print(f"  타사 모델 평균 순위: {bias_analysis['other_vendor_avg_rank']:.2f}")

            if bias_analysis['score_difference'] > 0.5:
                print(f"\n⚠️  경고: 자사 모델 선호 편향이 감지되었습니다 (차이: {bias_analysis['score_difference']:.3f})")
            elif bias_analysis['score_difference'] < -0.5:
                print(f"\n⚠️  경고: 자사 모델에 대한 부정적 편향이 감지되었습니다 (차이: {bias_analysis['score_difference']:.3f})")
            else:
                print(f"\n✅ 자사 모델 편향이 적절한 수준입니다 (차이: {bias_analysis['score_difference']:.3f})")
        else:
            bias_analysis = {
                "judge_model": judge_model,
                "judge_vendor": "unknown",
                "message": "Judge 모델의 제조사를 식별할 수 없어 편향 분석을 수행할 수 없습니다."
            }

        # 상세 통계 저장
        stats_csv = output_dir / f"bias_analysis_stats_{judge_model.replace('/', '_')}.csv"
        df_stats.to_csv(stats_csv, index=False, encoding='utf-8-sig')
        print(f"\n💾 상세 통계 저장: {stats_csv}")

        # 편향 분석 결과 저장
        bias_json = output_dir / f"bias_analysis_{judge_model.replace('/', '_')}.json"
        bias_result = {
            **bias_analysis,
            "detailed_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        with open(bias_json, 'w', encoding='utf-8') as f:
            json.dump(bias_result, f, indent=2, ensure_ascii=False)
        print(f"💾 편향 분석 결과 저장: {bias_json}")

        # 시각화를 위한 마크다운 리포트
        md_file = output_dir / f"bias_analysis_{judge_model.replace('/', '_')}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Judge 모델 편향 분석 리포트\n\n")
            f.write(f"**Judge 모델**: {judge_model}\n\n")
            f.write(f"**Judge 제조사**: {judge_vendor or '알 수 없음'}\n\n")
            f.write(f"**분석 일시**: {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")

            if judge_vendor:
                f.write("## 자사 모델 편향 분석\n\n")
                f.write(f"- **자사 모델 평균 점수**: {bias_analysis['same_vendor_avg_score']:.3f}\n")
                f.write(f"- **타사 모델 평균 점수**: {bias_analysis['other_vendor_avg_score']:.3f}\n")
                f.write(f"- **점수 차이**: {bias_analysis['score_difference']:.3f}\n")
                f.write(f"- **자사 모델 평균 순위**: {bias_analysis['same_vendor_avg_rank']:.2f}\n")
                f.write(f"- **타사 모델 평균 순위**: {bias_analysis['other_vendor_avg_rank']:.2f}\n\n")

                if abs(bias_analysis['score_difference']) > 0.5:
                    f.write(f"⚠️ **편향 경고**: 점수 차이가 0.5를 초과합니다.\n\n")
                else:
                    f.write(f"✅ **편향 적정**: 점수 차이가 허용 범위 내입니다.\n\n")

            f.write("## LLM별 상세 통계\n\n")
            f.write(df_stats.to_markdown(index=False))
            f.write("\n")

        print(f"📝 마크다운 리포트 저장: {md_file}")

        return bias_result


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
