#!/usr/bin/env python3
"""육안 평가(Human Evaluation) 템플릿 및 데이터 관리

육안 평가 결과를 구조화하여 저장하고, Judge 평가와 비교 가능한 형태로 관리합니다.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class HumanEvaluationManager:
    """육안 평가 관리 클래스"""

    # 주간 보고서 평가 기준
    WEEKLY_CRITERIA = {
        "conciseness": {
            "name": "간결성",
            "description": "불필요한 반복 없이 핵심만 간결하게 전달하는가?",
            "weight": 0.35,
            "max_score": 35
        },
        "structure": {
            "name": "구조",
            "description": "주간 보고서 형식에 적합하게 논리적으로 구조화되었는가?",
            "weight": 0.25,
            "max_score": 25
        },
        "practicality": {
            "name": "실무성",
            "description": "실무에서 바로 활용 가능한 형태인가?",
            "weight": 0.25,
            "max_score": 25
        },
        "accuracy": {
            "name": "정확성",
            "description": "검색된 문서의 내용을 정확하게 반영하고 있는가?",
            "weight": 0.15,
            "max_score": 15
        }
    }

    # 최종 보고서 평가 기준
    EXECUTIVE_CRITERIA = {
        "completeness": {
            "name": "완성도",
            "description": "경영진 보고서로서 구조와 내용이 완성도가 높은가?",
            "weight": 0.25,
            "max_score": 25
        },
        "accuracy": {
            "name": "정확성",
            "description": "검색된 문서의 내용을 정확하게 반영하고 있는가?",
            "weight": 0.25,
            "max_score": 25
        },
        "practicality": {
            "name": "실용성",
            "description": "경영진이 의사결정에 실제로 활용할 수 있는 내용인가?",
            "weight": 0.25,
            "max_score": 25
        },
        "conciseness": {
            "name": "간결성",
            "description": "핵심만 간결하게 요약되었는가?",
            "weight": 0.25,
            "max_score": 25
        }
    }

    def __init__(self):
        self.evaluations = {}

    def create_evaluation_template(
        self,
        report_type: str,
        llm_names: List[str]
    ) -> Dict[str, Any]:
        """평가 템플릿 생성

        Args:
            report_type: 'weekly' or 'executive'
            llm_names: 평가할 LLM 이름 리스트

        Returns:
            평가 템플릿 딕셔너리
        """
        criteria = self.WEEKLY_CRITERIA if report_type == 'weekly' else self.EXECUTIVE_CRITERIA

        template = {
            "report_type": report_type,
            "evaluation_date": datetime.now().isoformat(),
            "criteria": criteria,
            "evaluations": {}
        }

        for llm_name in llm_names:
            template["evaluations"][llm_name] = {
                "criteria_scores": {
                    criterion_key: {
                        "score": 0,
                        "max_score": criterion_info["max_score"],
                        "comment": ""
                    }
                    for criterion_key, criterion_info in criteria.items()
                },
                "total_score": 0,
                "rank": 0,
                "overall_comment": "",
                "strengths": [],
                "weaknesses": []
            }

        return template

    def load_evaluation_from_data(
        self,
        report_type: str,
        evaluation_data: Dict[str, Any]
    ):
        """귀하의 평가 결과를 구조화된 형태로 변환

        Args:
            report_type: 'weekly' or 'executive'
            evaluation_data: {llm_name: {total_score, criteria_scores, comments}}
        """
        criteria = self.WEEKLY_CRITERIA if report_type == 'weekly' else self.EXECUTIVE_CRITERIA

        structured_eval = {
            "report_type": report_type,
            "evaluation_date": datetime.now().isoformat(),
            "criteria": criteria,
            "evaluations": {}
        }

        for llm_name, data in evaluation_data.items():
            structured_eval["evaluations"][llm_name] = {
                "criteria_scores": data.get("criteria_scores", {}),
                "total_score": data.get("total_score", 0),
                "rank": data.get("rank", 0),
                "overall_comment": data.get("overall_comment", ""),
                "strengths": data.get("strengths", []),
                "weaknesses": data.get("weaknesses", [])
            }

        self.evaluations[report_type] = structured_eval
        return structured_eval

    def save_evaluation(
        self,
        report_type: str,
        output_path: str
    ):
        """평가 결과 저장

        Args:
            report_type: 'weekly' or 'executive'
            output_path: 저장 경로
        """
        if report_type not in self.evaluations:
            raise ValueError(f"{report_type} 평가 데이터가 없습니다.")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations[report_type], f, indent=2, ensure_ascii=False)

        print(f"✅ 평가 결과 저장: {output_file}")

    def export_for_judge_comparison(
        self,
        report_type: str,
        output_path: str
    ):
        """Judge 평가와 비교 가능한 형식으로 내보내기

        Args:
            report_type: 'weekly' or 'executive'
            output_path: 저장 경로
        """
        if report_type not in self.evaluations:
            raise ValueError(f"{report_type} 평가 데이터가 없습니다.")

        eval_data = self.evaluations[report_type]

        # Judge 평가 형식으로 변환
        judge_format = {
            "report_type": report_type,
            "evaluation_date": eval_data["evaluation_date"],
            "evaluator": "Human",
            "results": {}
        }

        for llm_name, llm_eval in eval_data["evaluations"].items():
            judge_format["results"][llm_name] = {
                "final_score": llm_eval["total_score"] / 10,  # 100점을 10점 만점으로 변환
                "criterion_results": []
            }

            for criterion_key, criterion_score in llm_eval["criteria_scores"].items():
                criterion_info = eval_data["criteria"][criterion_key]
                judge_format["results"][llm_name]["criterion_results"].append({
                    "criterion": criterion_key,
                    "criterion_name": criterion_info["name"],
                    "weight": criterion_info["weight"],
                    "score": criterion_score["score"] / criterion_score["max_score"] * 10,  # 10점 만점으로
                    "weighted_score": criterion_score["score"] / 100,
                    "reasoning": criterion_score.get("comment", ""),
                    "strengths": llm_eval.get("strengths", []),
                    "weaknesses": llm_eval.get("weaknesses", [])
                })

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(judge_format, f, indent=2, ensure_ascii=False)

        print(f"✅ Judge 비교용 형식으로 저장: {output_file}")
        return judge_format


def create_your_evaluation_data():
    """귀하의 육안 평가 결과를 구조화된 데이터로 변환"""

    # 주간 보고서 평가 결과
    weekly_evaluation = {
        "OpenAI GPT-4.1": {
            "total_score": 91,
            "rank": 1,
            "criteria_scores": {
                "conciseness": {"score": 30, "max_score": 35, "comment": "평균 1,000자 내외로 가장 적절한 길이"},
                "structure": {"score": 23, "max_score": 25, "comment": "주요 지표 → 활동 → 이슈 → 다음주 계획 흐름이 자연스러움"},
                "practicality": {"score": 24, "max_score": 25, "comment": "실무에서 바로 활용 가능한 형태"},
                "accuracy": {"score": 14, "max_score": 15, "comment": "불필요한 반복이나 장황한 설명 없음"}
            },
            "strengths": [
                "평균 1,000자 내외로 가장 적절한 길이",
                "주요 지표 → 활동 → 이슈 → 다음주 계획 흐름이 자연스러움",
                "실무에서 바로 활용 가능한 형태",
                "불필요한 반복이나 장황한 설명 없음"
            ],
            "weaknesses": [],
            "overall_comment": "1위: 실무에서 바로 활용 가능한 최적의 주간 보고서"
        },
        "DeepSeek-V3.1": {
            "total_score": 90,
            "rank": 2,
            "criteria_scores": {
                "conciseness": {"score": 32, "max_score": 35, "comment": "가장 간결함 (평균 800자)"},
                "structure": {"score": 22, "max_score": 25, "comment": "명확한 구조"},
                "practicality": {"score": 22, "max_score": 25, "comment": "핵심 수치와 액션만 정리"},
                "accuracy": {"score": 14, "max_score": 15, "comment": "불필요한 설명 최소화"}
            },
            "strengths": [
                "가장 간결함 (평균 800자)",
                "불필요한 설명 최소화",
                "핵심 수치와 액션만 정리"
            ],
            "weaknesses": [
                "가끔 너무 간결해서 맥락 부족"
            ],
            "overall_comment": "2위: 빠른 파악용 간결한 보고서"
        },
        "Claude 4.5 Sonnet": {
            "total_score": 88,
            "rank": 3,
            "criteria_scores": {
                "conciseness": {"score": 31, "max_score": 35, "comment": "적절한 길이"},
                "structure": {"score": 22, "max_score": 25, "comment": "구조가 명확하고 일관성 있음"},
                "practicality": {"score": 22, "max_score": 25, "comment": "테이블 활용으로 가독성 좋음"},
                "accuracy": {"score": 13, "max_score": 15, "comment": "가끔 '문서에 명시되지 않음' 과다"}
            },
            "strengths": [
                "구조가 명확하고 일관성 있음",
                "테이블 활용으로 가독성 좋음"
            ],
            "weaknesses": [
                "때때로 '문서에 명시되지 않음' 과다"
            ],
            "overall_comment": "3위: 구조적으로 우수"
        },
        "Claude 4.5 Opus": {
            "total_score": 86,
            "rank": 4,
            "criteria_scores": {
                "conciseness": {"score": 30, "max_score": 35},
                "structure": {"score": 22, "max_score": 25},
                "practicality": {"score": 21, "max_score": 25},
                "accuracy": {"score": 13, "max_score": 15}
            },
            "strengths": [],
            "weaknesses": [],
            "overall_comment": "4위"
        },
        "Phi-4": {
            "total_score": 82,
            "rank": 5,
            "criteria_scores": {
                "conciseness": {"score": 28, "max_score": 35},
                "structure": {"score": 21, "max_score": 25},
                "practicality": {"score": 20, "max_score": 25},
                "accuracy": {"score": 13, "max_score": 15}
            },
            "strengths": [],
            "weaknesses": [],
            "overall_comment": "5위"
        },
        "OpenAI GPT-5.1": {
            "total_score": 81,
            "rank": 6,
            "criteria_scores": {
                "conciseness": {"score": 18, "max_score": 35, "comment": "너무 장황함"},
                "structure": {"score": 24, "max_score": 25},
                "practicality": {"score": 24, "max_score": 25},
                "accuracy": {"score": 15, "max_score": 15}
            },
            "strengths": [],
            "weaknesses": ["너무 장황함"],
            "overall_comment": "6위: 내용은 좋으나 너무 길어서 주간 보고서로는 부적합"
        },
        "Llama-3.3-70B-Instruct": {
            "total_score": 72,
            "rank": 7,
            "criteria_scores": {
                "conciseness": {"score": 25, "max_score": 35},
                "structure": {"score": 18, "max_score": 25},
                "practicality": {"score": 18, "max_score": 25},
                "accuracy": {"score": 11, "max_score": 15}
            },
            "strengths": [],
            "weaknesses": [],
            "overall_comment": "7위"
        }
    }

    # 최종 보고서 평가 결과
    executive_evaluation = {
        "OpenAI GPT-4.1": {
            "total_score": 87.5,
            "rank": 1,
            "criteria_scores": {
                "completeness": {"score": 22.5, "max_score": 25, "comment": "임원/실무진이 읽기 가장 좋은 형태"},
                "accuracy": {"score": 22.5, "max_score": 25, "comment": "높은 완성도"},
                "practicality": {"score": 22.5, "max_score": 25, "comment": "핵심 파악 용이"},
                "conciseness": {"score": 20.0, "max_score": 25, "comment": "적절한 길이"}
            },
            "strengths": [
                "임원/실무진이 읽기 가장 좋은 형태",
                "핵심 파악 용이",
                "적절한 길이",
                "높은 완성도"
            ],
            "weaknesses": [],
            "overall_comment": "1위: 임원/실무진이 읽기 가장 좋은 형태"
        },
        "OpenAI GPT-5.1": {
            "total_score": 85.5,
            "rank": 2,
            "criteria_scores": {
                "completeness": {"score": 24.0, "max_score": 25, "comment": "완벽한 구조"},
                "accuracy": {"score": 24.0, "max_score": 25, "comment": "완벽한 정확성"},
                "practicality": {"score": 24.0, "max_score": 25, "comment": "빠짐없는 정보"},
                "conciseness": {"score": 13.5, "max_score": 25, "comment": "너무 길어서 핵심 파악이 어려움"}
            },
            "strengths": [
                "완벽한 구조",
                "완벽한 정확성",
                "빠짐없는 정보"
            ],
            "weaknesses": [
                "너무 길어서 핵심 파악이 어려움"
            ],
            "overall_comment": "2위: 외부 제출용 상세 기술 문서에 최적"
        },
        "DeepSeek-V3.1": {
            "total_score": 79.0,
            "rank": 3,
            "criteria_scores": {
                "completeness": {"score": 19.5, "max_score": 25},
                "accuracy": {"score": 19.5, "max_score": 25},
                "practicality": {"score": 17.5, "max_score": 25, "comment": "세부 내용 부족할 수 있음"},
                "conciseness": {"score": 22.5, "max_score": 25, "comment": "효율적, 핵심만 추출"}
            },
            "strengths": [
                "효율적",
                "핵심만 추출"
            ],
            "weaknesses": [
                "세부 내용 부족할 수 있음"
            ],
            "overall_comment": "3위: 빠른 상황 파악용 요약 보고서"
        },
        "Claude 4.5 Opus": {
            "total_score": 64.0,
            "rank": 4,
            "criteria_scores": {
                "completeness": {"score": 16.0, "max_score": 25},
                "accuracy": {"score": 16.0, "max_score": 25},
                "practicality": {"score": 13.5, "max_score": 25},
                "conciseness": {"score": 18.5, "max_score": 25}
            },
            "strengths": [],
            "weaknesses": [],
            "overall_comment": "4위"
        },
        "Claude 4.5 Sonnet": {
            "total_score": 63.5,
            "rank": 5,
            "criteria_scores": {
                "completeness": {"score": 15.5, "max_score": 25},
                "accuracy": {"score": 15.5, "max_score": 25},
                "practicality": {"score": 13.0, "max_score": 25},
                "conciseness": {"score": 18.5, "max_score": 25}
            },
            "strengths": [],
            "weaknesses": [],
            "overall_comment": "5위"
        },
        "Phi-4": {
            "total_score": 54.5,
            "rank": 6,
            "criteria_scores": {
                "completeness": {"score": 14.0, "max_score": 25},
                "accuracy": {"score": 13.0, "max_score": 25},
                "practicality": {"score": 12.0, "max_score": 25},
                "conciseness": {"score": 16.0, "max_score": 25}
            },
            "strengths": [],
            "weaknesses": [],
            "overall_comment": "6위"
        },
        "Llama-3.3-70B-Instruct": {
            "total_score": 28.0,
            "rank": 7,
            "criteria_scores": {
                "completeness": {"score": 6.5, "max_score": 25},
                "accuracy": {"score": 5.5, "max_score": 25},
                "practicality": {"score": 5.5, "max_score": 25},
                "conciseness": {"score": 10.5, "max_score": 25}
            },
            "strengths": [],
            "weaknesses": [],
            "overall_comment": "7위"
        }
    }

    return weekly_evaluation, executive_evaluation


def main():
    """메인 함수 - 육안 평가 데이터 저장"""
    manager = HumanEvaluationManager()

    # 육안 평가 데이터 로드
    weekly_eval, executive_eval = create_your_evaluation_data()

    # 구조화된 형태로 변환
    manager.load_evaluation_from_data('weekly', weekly_eval)
    manager.load_evaluation_from_data('executive', executive_eval)

    # 저장 디렉토리
    output_dir = Path("data/results/multi_llm_test/human_evaluation")

    # 평가 결과 저장
    manager.save_evaluation('weekly', output_dir / "weekly_human_evaluation.json")
    manager.save_evaluation('executive', output_dir / "executive_human_evaluation.json")

    # Judge 비교용 형식으로 저장
    manager.export_for_judge_comparison('weekly', output_dir / "weekly_human_evaluation_judge_format.json")
    manager.export_for_judge_comparison('executive', output_dir / "executive_human_evaluation_judge_format.json")

    print("\n✅ 육안 평가 데이터 저장 완료!")
    print(f"📁 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
