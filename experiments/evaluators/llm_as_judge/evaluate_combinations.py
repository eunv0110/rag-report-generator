#!/usr/bin/env python3
"""조합 결과 평가 스크립트 - LLM as Judge로 여러 LLM 성능 비교

all_combinations.json 파일을 읽어서 각 질문에 대해 여러 LLM의 답변을 평가하고 순위를 매깁니다.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, Any, List
from datetime import datetime
from llm_as_judge_evaluator import LLMAsJudgeEvaluator
import pandas as pd


def load_combinations_result(combinations_file: str) -> Dict[str, Any]:
    """all_combinations.json 파일 로드"""
    with open(combinations_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_answers_by_question(combinations_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """질문별로 답변을 그룹화

    Returns:
        [
            {
                "question_id": 1,
                "question": "질문 내용",
                "date_filter": "날짜 범위",
                "answers": {
                    "phi4": "답변 내용",
                    "gpt41": "답변 내용",
                    ...
                }
            },
            ...
        ]
    """
    # 질문별로 데이터를 저장할 딕셔너리
    questions_dict = {}

    # 각 조합(LLM)의 결과를 순회
    for combination in combinations_data['combinations']:
        llm_name = combination['llm_name']
        llm_display_name = combination['llm_display_name']

        # 해당 LLM의 각 질문 결과를 순회
        for result_item in combination['results']:
            question_id = result_item['question_id']
            question = result_item['question']
            date_filter = result_item.get('date_filter')

            # 질문별 딕셔너리 초기화
            if question_id not in questions_dict:
                questions_dict[question_id] = {
                    "question_id": question_id,
                    "question": question,
                    "date_filter": date_filter,
                    "answers": {},
                    "display_names": {},
                    "num_docs": result_item.get('num_docs', 0),
                    "doc_titles": result_item.get('doc_titles', [])
                }

            # 답변 추가 (성공한 경우만)
            if result_item['result'].get('success'):
                questions_dict[question_id]['answers'][llm_name] = result_item['result']['answer']
                questions_dict[question_id]['display_names'][llm_name] = llm_display_name

    # 질문 ID 순서대로 정렬하여 리스트로 반환
    return [questions_dict[qid] for qid in sorted(questions_dict.keys())]


def evaluate_all_questions(
    combinations_file: str,
    judge_model: str = "gpt-4o",
    provider: str = "azure_ai",
    report_type: str = "weekly_report",
    output_dir: str = None
) -> Dict[str, Any]:
    """모든 질문에 대해 LLM 답변 평가

    Args:
        combinations_file: all_combinations.json 파일 경로
        judge_model: 평가에 사용할 모델 (gpt-4o, anthropic/claude-opus-4.5 등)
        provider: LLM 제공자 (azure_ai 또는 openrouter)
        report_type: 보고서 타입 (weekly_report, executive_report)
        output_dir: 결과 저장 디렉토리 (None이면 자동 생성)

    Returns:
        전체 평가 결과
    """
    # 조합 결과 로드
    print(f"\n{'='*100}")
    print(f"📂 조합 결과 로드: {combinations_file}")
    print(f"{'='*100}")

    combinations_data = load_combinations_result(combinations_file)

    print(f"✅ 로드 완료:")
    print(f"  - Retriever: {combinations_data['retriever_display_name']}")
    print(f"  - LLM 개수: {combinations_data['num_llms']}")
    print(f"  - 질문 개수: {combinations_data['num_questions']}")
    print(f"  - 완료된 질문: {combinations_data['num_completed_questions']}")

    # 질문별로 답변 추출
    questions_with_answers = extract_answers_by_question(combinations_data)

    print(f"\n📊 평가 가능한 질문 수: {len(questions_with_answers)}")
    for q in questions_with_answers:
        print(f"  Q{q['question_id']}: {len(q['answers'])}개 LLM 답변")

    # 평가기 초기화
    print(f"\n🤖 Judge 모델 초기화: {judge_model} (provider: {provider})")
    evaluator = LLMAsJudgeEvaluator(
        judge_model=judge_model,
        provider=provider
    )

    # 출력 디렉토리 설정
    if output_dir is None:
        combinations_path = Path(combinations_file).parent
        output_dir = combinations_path / "evaluation_results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모든 질문에 대해 평가
    all_evaluations = []

    for q_data in questions_with_answers:
        question_id = q_data['question_id']
        question = q_data['question']
        answers = q_data['answers']
        display_names = q_data['display_names']

        print(f"\n{'='*100}")
        print(f"📋 질문 {question_id}/{len(questions_with_answers)}")
        print(f"{'='*100}")
        print(f"❓ {question}")
        print(f"📅 날짜 필터: {q_data.get('date_filter', 'N/A')}")
        print(f"📄 검색 문서 수: {q_data['num_docs']}")
        print(f"💬 평가할 LLM: {', '.join(answers.keys())}")

        # 비교 평가 실행
        try:
            comparison_result = evaluator.compare_multiple_answers(
                question=question,
                answers=answers,
                report_type=report_type
            )

            # 질문 메타데이터 추가
            comparison_result['question_id'] = question_id
            comparison_result['date_filter'] = q_data.get('date_filter')
            comparison_result['num_docs'] = q_data['num_docs']
            comparison_result['doc_titles'] = q_data['doc_titles']
            comparison_result['display_names'] = display_names

            all_evaluations.append(comparison_result)

            # 질문별 결과 저장
            question_output_file = output_dir / f"question_{question_id}_evaluation.json"
            with open(question_output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)

            print(f"\n💾 질문 {question_id} 평가 결과 저장: {question_output_file}")

        except Exception as e:
            print(f"\n❌ 질문 {question_id} 평가 중 오류: {e}")
            import traceback
            traceback.print_exc()

    # 전체 평가 결과 집계
    print(f"\n{'='*100}")
    print(f"📊 전체 평가 결과 집계")
    print(f"{'='*100}")

    final_result = {
        "combinations_file": str(combinations_file),
        "retriever_name": combinations_data['retriever_name'],
        "retriever_display_name": combinations_data['retriever_display_name'],
        "judge_model": judge_model,
        "judge_provider": provider,
        "report_type": report_type,
        "num_questions_evaluated": len(all_evaluations),
        "num_llms": combinations_data['num_llms'],
        "evaluations": all_evaluations,
        "timestamp": datetime.now().isoformat()
    }

    # 전체 결과 저장
    overall_output_file = output_dir / "overall_evaluation.json"
    with open(overall_output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"\n💾 전체 평가 결과 저장: {overall_output_file}")

    # 종합 순위표 생성
    generate_summary_ranking(final_result, output_dir)

    # Judge 모델 편향 분석
    print(f"\n{'='*100}")
    print(f"🔍 Judge 모델 편향 분석 시작")
    print(f"{'='*100}")

    bias_analysis = evaluator.analyze_judge_bias(
        all_evaluations=all_evaluations,
        judge_model=judge_model,
        output_dir=output_dir
    )

    final_result['bias_analysis'] = bias_analysis

    # 편향 분석 포함한 최종 결과 재저장
    with open(overall_output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    return final_result


def generate_summary_ranking(final_result: Dict[str, Any], output_dir: Path):
    """종합 순위표 생성 (CSV + 마크다운)

    각 LLM의 평균 점수, 질문별 순위, 승률 등을 계산하여 종합 순위를 생성합니다.
    """
    print(f"\n{'='*100}")
    print(f"🏆 종합 순위표 생성")
    print(f"{'='*100}")

    # LLM별 점수 수집
    llm_scores = {}
    llm_rankings = {}

    for evaluation in final_result['evaluations']:
        question_id = evaluation['question_id']

        for llm_name, result in evaluation['results'].items():
            if llm_name not in llm_scores:
                llm_scores[llm_name] = []
                llm_rankings[llm_name] = []

            llm_scores[llm_name].append({
                'question_id': question_id,
                'score': result['final_score']
            })

        # 순위 계산
        ranking = evaluation['ranking']
        for rank, (llm_name, score) in enumerate(ranking, 1):
            llm_rankings[llm_name].append(rank)

    # 종합 통계 계산
    summary_rows = []

    for llm_name in llm_scores.keys():
        scores = [s['score'] for s in llm_scores[llm_name]]
        rankings = llm_rankings[llm_name]

        summary_rows.append({
            'LLM': llm_name,
            '평균 점수': sum(scores) / len(scores) if scores else 0,
            '최고 점수': max(scores) if scores else 0,
            '최저 점수': min(scores) if scores else 0,
            '평균 순위': sum(rankings) / len(rankings) if rankings else 0,
            '1위 횟수': rankings.count(1),
            '2위 횟수': rankings.count(2),
            '3위 횟수': rankings.count(3),
            '평가 질문 수': len(scores)
        })

    # 평균 점수로 정렬
    summary_rows.sort(key=lambda x: x['평균 점수'], reverse=True)

    # DataFrame 생성
    df = pd.DataFrame(summary_rows)

    # CSV 저장
    csv_file = output_dir / "summary_ranking.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"📊 CSV 저장: {csv_file}")

    # 마크다운 표 생성
    md_file = output_dir / "summary_ranking.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# LLM 종합 순위표\n\n")
        f.write(f"**평가 일시**: {final_result['timestamp']}\n\n")
        f.write(f"**Retriever**: {final_result['retriever_display_name']}\n\n")
        f.write(f"**Judge Model**: {final_result['judge_model']} ({final_result['judge_provider']})\n\n")
        f.write(f"**평가 질문 수**: {final_result['num_questions_evaluated']}\n\n")
        f.write("---\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n---\n\n")

        # 질문별 상세 순위
        f.write("## 질문별 순위\n\n")
        for evaluation in final_result['evaluations']:
            f.write(f"### 질문 {evaluation['question_id']}: {evaluation['question']}\n\n")
            f.write(f"**날짜 필터**: {evaluation.get('date_filter', 'N/A')}\n\n")

            ranking_data = []
            for rank, (llm_name, score) in enumerate(evaluation['ranking'], 1):
                display_name = evaluation['display_names'].get(llm_name, llm_name)
                ranking_data.append({
                    '순위': rank,
                    'LLM': display_name,
                    '점수': f"{score:.2f}"
                })

            ranking_df = pd.DataFrame(ranking_data)
            f.write(ranking_df.to_markdown(index=False))
            f.write("\n\n")

    print(f"📝 마크다운 저장: {md_file}")

    # 콘솔 출력
    print("\n" + "="*100)
    print("🏆 최종 종합 순위")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="조합 결과 평가 - LLM as Judge로 여러 LLM 성능 비교"
    )
    parser.add_argument(
        "--combinations-file",
        type=str,
        required=True,
        help="all_combinations.json 파일 경로"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.1",
        help="평가에 사용할 LLM 모델 (기본: gpt-4o)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="azure_ai",
        choices=["azure_ai", "openrouter"],
        help="LLM 제공자 (기본: azure_ai)"
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
        default=None,
        help="결과 저장 디렉토리 (기본: combinations_file과 같은 디렉토리/evaluation_results)"
    )

    args = parser.parse_args()

    # 평가 실행
    result = evaluate_all_questions(
        combinations_file=args.combinations_file,
        judge_model=args.judge_model,
        provider=args.provider,
        report_type=args.report_type,
        output_dir=args.output_dir
    )

    print("\n✅ 모든 평가 완료!")


if __name__ == "__main__":
    main()
