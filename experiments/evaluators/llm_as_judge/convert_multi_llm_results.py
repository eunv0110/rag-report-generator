#!/usr/bin/env python3
"""multi_llm_test 결과를 all_combinations.json 형식으로 변환

results_final.json 파일을 읽어서 evaluate_combinations.py에서 사용할 수 있는 형식으로 변환합니다.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def convert_results_to_combinations(results_file: str, output_file: str = None) -> Dict[str, Any]:
    """results_final.json을 all_combinations.json 형식으로 변환

    Args:
        results_file: results_final.json 파일 경로
        output_file: 출력 파일 경로 (None이면 같은 디렉토리에 all_combinations.json 생성)

    Returns:
        변환된 데이터
    """
    # 입력 파일 로드
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {results_file}")

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📂 파일 로드: {results_file}")
    print(f"  - Report Type: {data['report_type']}")
    print(f"  - Retriever: {data['retriever_config']['display_name']}")
    print(f"  - LLMs: {data['num_llms']}")
    print(f"  - Questions: {data['num_questions']}")

    # all_combinations.json 형식으로 변환
    combinations = []

    for llm_config in data['llm_configs']:
        llm_name = llm_config['name']
        llm_display_name = llm_config['display_name']

        # 해당 LLM의 모든 질문 결과 수집
        llm_results = []

        for question_data in data['results']:
            question_id = question_data['question_id']
            question = question_data['question']
            num_docs = question_data['num_docs']

            # 해당 LLM의 결과 찾기
            llm_result_data = None
            doc_titles = []

            for llm_result in question_data['llms']:
                if llm_result['llm_name'] == llm_name:
                    llm_result_data = llm_result
                    doc_titles = llm_result.get('doc_titles', [])
                    break

            if llm_result_data:
                # 결과 추가
                result_item = {
                    "question_id": question_id,
                    "question": question,
                    "date_filter": None,  # multi_llm_test에는 date_filter가 없음
                    "num_docs": num_docs,
                    "doc_titles": doc_titles,
                    "result": {
                        "success": llm_result_data['success'],
                        "answer": llm_result_data.get('answer', ''),
                        "answer_length": llm_result_data.get('answer_length', 0)
                    }
                }
                llm_results.append(result_item)

        # LLM 조합 추가
        combinations.append({
            "llm_name": llm_name,
            "llm_display_name": llm_display_name,
            "model_id": llm_config.get('model_id', ''),
            "num_questions": len(llm_results),
            "num_completed_questions": sum(1 for r in llm_results if r['result']['success']),
            "results": llm_results
        })

    # 최종 데이터 구조
    output_data = {
        "test_date": data['test_date'],
        "report_type": data['report_type'],
        "retriever_name": data['retriever_config']['name'],
        "retriever_display_name": data['retriever_config']['display_name'],
        "num_llms": data['num_llms'],
        "num_questions": data['num_questions'],
        "num_completed_questions": data['num_questions'],  # 모든 질문이 완료됨
        "combinations": combinations
    }

    # 출력 파일 경로 결정
    if output_file is None:
        output_file = results_path.parent / "all_combinations.json"
    else:
        output_file = Path(output_file)

    # 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 변환 완료!")
    print(f"💾 저장 위치: {output_file}")
    print(f"\n📊 변환 결과:")
    print(f"  - 총 LLM 조합: {len(combinations)}")
    print(f"  - 질문당 LLM 수: {data['num_llms']}")

    return output_data


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="multi_llm_test 결과를 all_combinations.json 형식으로 변환"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="results_final.json 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 파일 경로 (기본: 같은 디렉토리에 all_combinations.json)"
    )

    args = parser.parse_args()

    try:
        convert_results_to_combinations(args.results_file, args.output)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
