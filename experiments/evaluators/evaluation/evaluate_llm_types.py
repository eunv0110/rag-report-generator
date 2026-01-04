#!/usr/bin/env python3
"""여러 LLM 모델을 사용한 보고서 테스트 (Langfuse 연동)

evaluation_config.yaml의 설정을 사용하여 각 보고서 타입별로 지정된 리트리버와 LLM 조합을 테스트합니다:
- weekly: bge_m3_rrf_ensemble + Qwen3-Reranker-4B (Top 6)
- executive: openai_rrf_multiquery (Top 8)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import json
import yaml
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.langfuse import get_langfuse_client
from utils.common import save_embedding_cache

# Load evaluation config from experiments directory
def load_evaluation_config():
    """experiments 디렉토리의 evaluation_config.yaml 파일 로드"""
    config_path = Path(__file__).parent.parent.parent / "evaluation_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

EVALUATION_CONFIG = load_evaluation_config()

# Reranker 테스트 스크립트에서 함수 재사용
from evaluators.evaluation.evaluate_reranker import (
    retrieve_and_rerank_documents,
    generate_answer_with_llm,
    get_report_config,
    load_prompt
)


def get_test_llms():
    """evaluation_config.yaml에서 테스트 LLM 목록 가져오기"""
    return EVALUATION_CONFIG.get('test_llms', [])


def get_retriever_config_from_yaml(report_type: str) -> Dict[str, Any]:
    """evaluation_config.yaml에서 보고서 타입별 리트리버 설정 반환

    Args:
        report_type: 'weekly' 또는 'executive'

    Returns:
        리트리버 설정 딕셔너리
    """
    key = f"{report_type}_report"
    retrievers = EVALUATION_CONFIG.get('simple_test_retrievers', {}).get(key, [])

    if not retrievers:
        raise ValueError(f"No retrievers found for {report_type}_report in evaluation_config.yaml")

    # 리트리버 선택
    if report_type == "weekly":
        # weekly: 첫 번째 (bge_m3_rrf_ensemble, Top 6)
        retriever = retrievers[0]
    elif report_type == "executive":
        # executive: openai_rrf_multiquery (Top 8) 선택
        # config에서 name이 "openai_rrf_multiquery"인 것 찾기
        retriever = next(
            (r for r in retrievers if r['name'] == 'openai_rrf_multiquery'),
            retrievers[0]  # 못 찾으면 첫 번째 사용
        )
    else:
        retriever = retrievers[0]

    # 리트리버 타입 매핑
    retriever_type_map = {
        "rrf_ensemble": "bge-m3",
        "rrf_multiquery_lc": "openai-large-rrf-multiquery",
        "rrf_multiquery": "openai-rrf-multiquery"
    }

    retriever_type = retriever_type_map.get(retriever['type'], "bge-m3")

    # Reranker 사용 여부 판단 (rrf_ensemble 타입이면 reranker 사용)
    use_reranker = retriever['type'] == 'rrf_ensemble'

    return {
        "name": retriever['name'],
        "display_name": retriever['display_name'],
        "retriever_type": retriever_type,
        "reranker_type": "qwen3" if use_reranker else None,
        "top_k": retriever['top_k'],
        "description": retriever['description'],
        "embedding": retriever['embedding']
    }


def retrieve_with_multiquery_llm(
    question: str,
    retriever_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    langfuse,
    question_id: int = None
) -> list:
    """특정 LLM을 사용하여 MultiQuery 검색 수행

    Args:
        question: 검색 질문
        retriever_config: 리트리버 설정
        llm_config: MultiQuery에 사용할 LLM 설정
        langfuse: Langfuse 클라이언트
        question_id: 질문 ID

    Returns:
        검색된 문서 리스트
    """
    from utils.retriever_factory import create_retriever_from_config
    from retrievers.multiquery_retriever import MultiQueryRetriever

    # 기본 리트리버 설정
    base_retriever_cfg = {
        "name": f"{retriever_config['name']}_base",
        "display_name": retriever_config['display_name'],
        "description": retriever_config['description'],
        "embedding_preset": "openai-large",
        "retriever_type": "rrf_ensemble",  # MultiQuery의 base는 rrf_ensemble 사용
        "bm25_weight": 0.5,
        "dense_weight": 0.5,
    }

    # Langfuse span으로 검색 과정 기록
    if langfuse and question_id:
        with langfuse.start_as_current_observation(
            as_type='span',
            name=f"retrieval_multiquery_{llm_config['name']}_q{question_id}",
            input={"question": question},
            metadata={
                "retriever_name": retriever_config['name'],
                "retriever_type": "rrf_multiquery",
                "top_k": retriever_config['top_k'],
                "multiquery_llm": llm_config['model_id'],
                "multiquery_llm_name": llm_config['name']
            }
        ) as span:
            # Base retriever 생성
            base_retriever, _ = create_retriever_from_config(base_retriever_cfg, retriever_config['top_k'] * 2)

            # MultiQueryRetriever 생성 (LLM 지정)
            multiquery_retriever = MultiQueryRetriever(
                base_retriever=base_retriever,
                num_queries=3,
                temperature=0.7,
                llm_model=llm_config['model_id']
            )

            print(f"🔍 {llm_config['display_name']}로 MultiQuery 검색 중 (k={retriever_config['top_k']})...")
            docs = multiquery_retriever.invoke(question)

            # Top-K로 제한
            docs = docs[:retriever_config['top_k']]

            span.update(output={
                "num_docs": len(docs),
                "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in docs]
            })

            print(f"📄 검색된 문서 수: {len(docs)}")

            return docs
    else:
        # Langfuse 없이 실행
        base_retriever, _ = create_retriever_from_config(base_retriever_cfg, retriever_config['top_k'] * 2)
        multiquery_retriever = MultiQueryRetriever(
            base_retriever=base_retriever,
            num_queries=3,
            temperature=0.7,
            llm_model=llm_config['model_id']
        )

        print(f"🔍 {llm_config['display_name']}로 MultiQuery 검색 중 (k={retriever_config['top_k']})...")
        docs = multiquery_retriever.invoke(question)
        docs = docs[:retriever_config['top_k']]
        print(f"📄 검색된 문서 수: {len(docs)}")

        return docs


def test_single_llm_simple(
    llm_config: Dict[str, Any],
    question: str,
    docs: list,
    report_type: str,
    langfuse,
    question_id: int = None,
    retriever_config: Dict[str, Any] = None,
    use_multiquery_per_llm: bool = False,
    version: str = "v1"
) -> Dict[str, Any]:
    """단일 LLM으로 답변 생성 (간단 버전)

    Args:
        llm_config: LLM 설정
        question: 질문
        docs: 사전 검색된 문서 (use_multiquery_per_llm=False일 때 사용)
        report_type: 보고서 타입
        langfuse: Langfuse 클라이언트
        question_id: 질문 ID
        retriever_config: 리트리버 설정 (use_multiquery_per_llm=True일 때 필요)
        use_multiquery_per_llm: 각 LLM마다 별도로 MultiQuery 검색 수행 여부
        version: 평가 버전 태그
    """

    print(f"\n{'=' * 100}")
    print(f"💬 {llm_config['display_name']}")
    print(f"   {llm_config['description']}")
    print(f"{'=' * 100}")

    try:
        # MultiQuery를 각 LLM마다 사용하는 경우
        if use_multiquery_per_llm and retriever_config:
            docs = retrieve_with_multiquery_llm(
                question=question,
                retriever_config=retriever_config,
                llm_config=llm_config,
                langfuse=langfuse,
                question_id=question_id
            )

        print("💬 답변 생성 중...")
        answer = generate_answer_with_llm(
            question,
            docs,
            llm_config,
            report_type,
            langfuse,
            question_id,
            version
        )

        print(f"\n✅ 생성된 답변 (앞부분):\n{answer[:500]}...\n")

        return {
            "success": True,
            "llm_name": llm_config['name'],
            "llm_display_name": llm_config['display_name'],
            "model_id": llm_config['model_id'],
            "answer": answer,
            "answer_length": len(answer),
            "num_docs": len(docs),
            "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in docs]
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}\n")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "llm_name": llm_config['name'],
            "llm_display_name": llm_config['display_name'],
            "model_id": llm_config.get('model_id', 'unknown'),
            "error": str(e)
        }


def run_multi_llm_test(
    questions: List[str],
    report_type: str,
    output_dir: str = None,
    version: str = "v1"
):
    """여러 LLM으로 보고서 테스트 실행

    Args:
        questions: 테스트할 질문 리스트
        report_type: 보고서 타입 ('weekly' 또는 'executive')
        output_dir: 결과 저장 디렉토리
        version: 평가 버전 태그 (기본: v1)
    """

    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()
    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        return

    # evaluation_config.yaml에서 리트리버 및 LLM 설정 가져오기
    retriever_config = get_retriever_config_from_yaml(report_type)
    test_llms = get_test_llms()

    if not test_llms:
        print("❌ evaluation_config.yaml에 test_llms 설정이 없습니다.")
        return

    report_type_display = "주간보고서" if report_type == "weekly" else "최종보고서"

    print("\n" + "=" * 100)
    print(f"🧪 {report_type_display} - 여러 LLM 테스트 (Langfuse 연동)")
    print("=" * 100)
    print(f"\n🔍 리트리버: {retriever_config['display_name']}")
    print(f"   설명: {retriever_config['description']}")
    print(f"📊 Top-K: {retriever_config['top_k']}")
    print(f"\n💬 테스트할 LLM ({len(test_llms)}개):")
    for i, llm_config in enumerate(test_llms, 1):
        print(f"  {i}. {llm_config['display_name']} - {llm_config['description']}")
    print(f"\n📋 테스트 질문 ({len(questions)}개):")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    print()

    # 결과 저장용
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 각 질문에 대해 테스트
    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'#' * 100}")
        print(f"📋 질문 {q_idx}/{len(questions)}")
        print(f"{'#' * 100}")
        print(f"❓ {question}\n")

        # MultiQuery 사용 여부 결정
        use_multiquery_per_llm = retriever_config['reranker_type'] is None  # executive는 True

        # 문서 검색 (리트리버에 따라 다르게 처리)
        if retriever_config['reranker_type']:
            # Reranker 사용 (weekly - BGE-M3 + RRF Ensemble)
            # 한 번만 검색하고 모든 LLM이 동일한 문서 사용
            docs = retrieve_and_rerank_documents(
                question=question,
                report_type=report_type,
                top_k=retriever_config['top_k'],
                date_filter=None,
                langfuse=langfuse,
                question_id=q_idx,
                batch_size=16,
                retriever_type=retriever_config['retriever_type'],
                reranker_type=retriever_config['reranker_type']
            )
        else:
            # MultiQuery 사용 (executive)
            # 각 LLM마다 별도로 검색하므로 여기서는 빈 리스트
            print(f"🔍 각 LLM이 독립적으로 MultiQuery 검색을 수행합니다.")
            docs = []  # 각 LLM이 자체적으로 검색

        question_result = {
            "question_id": q_idx,
            "question": question,
            "retriever": retriever_config['display_name'],
            "top_k": retriever_config['top_k'],
            "use_multiquery_per_llm": use_multiquery_per_llm,
            "num_docs": len(docs) if docs else 0,
            "llms": []
        }

        # 각 LLM으로 병렬 테스트
        if use_multiquery_per_llm:
            print(f"\n🚀 {len(test_llms)}개 LLM으로 병렬 MultiQuery 검색 + 답변 생성 시작...\n")
        else:
            print(f"\n🚀 {len(test_llms)}개 LLM으로 병렬 답변 생성 시작...\n")

        with ThreadPoolExecutor(max_workers=min(len(test_llms), 7)) as executor:
            future_to_llm = {
                executor.submit(
                    test_single_llm_simple,
                    llm_config,
                    question,
                    docs,
                    report_type,
                    langfuse,
                    q_idx,
                    retriever_config,
                    use_multiquery_per_llm,
                    version
                ): (rank, llm_config)
                for rank, llm_config in enumerate(test_llms, 1)
            }

            for future in as_completed(future_to_llm):
                rank, llm_config = future_to_llm[future]
                try:
                    result = future.result()
                    result["rank"] = rank
                    question_result["llms"].append(result)
                except Exception as e:
                    print(f"❌ {llm_config['display_name']} 실행 중 오류: {e}")
                    result = {
                        "success": False,
                        "llm_name": llm_config['name'],
                        "llm_display_name": llm_config['display_name'],
                        "error": str(e),
                        "rank": rank
                    }
                    question_result["llms"].append(result)

        # rank 순으로 정렬
        question_result["llms"].sort(key=lambda x: x["rank"])
        all_results.append(question_result)

        # 중간 저장
        save_results(all_results, report_type, retriever_config, test_llms, timestamp, output_dir)

    # 최종 저장
    final_result = save_results(all_results, report_type, retriever_config, test_llms, timestamp, output_dir, is_final=True)

    print(f"\n{'=' * 100}")
    print(f"✅ 테스트 완료!")
    print(f"📁 결과 저장: {final_result}")
    print(f"{'=' * 100}\n")

    # Langfuse flush
    print("\n⏳ Langfuse에 데이터 전송 중...")
    langfuse.flush()

    # 임베딩 캐시 저장
    print("\n💾 임베딩 캐시 저장 중...")
    save_embedding_cache()


def save_results(
    all_results: List[Dict],
    report_type: str,
    retriever_config: Dict,
    test_llms: List[Dict],
    timestamp: str,
    output_dir: str = None,
    is_final: bool = False
) -> str:
    """결과 저장

    Args:
        all_results: 전체 결과 리스트
        report_type: 보고서 타입
        retriever_config: 리트리버 설정
        test_llms: 테스트 LLM 목록
        timestamp: 타임스탬프
        output_dir: 출력 디렉토리
        is_final: 최종 저장 여부

    Returns:
        저장된 파일 경로
    """
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path("data/results/multi_llm_test")

    # 보고서 타입별 디렉토리
    output_path = base_dir / report_type / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    output_file = output_path / ("results_final.json" if is_final else "results_temp.json")

    result_data = {
        "test_date": datetime.now().isoformat(),
        "report_type": report_type,
        "retriever_config": retriever_config,
        "num_llms": len(test_llms),
        "llm_configs": test_llms,
        "num_questions": len(all_results),
        "results": all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 텍스트 요약본 저장
    if is_final:
        summary_file = output_path / "summary.txt"
        summary_lines = []

        summary_lines.append("=" * 100)
        summary_lines.append(f"{retriever_config['display_name']} - 여러 LLM 테스트 결과 요약")
        summary_lines.append("=" * 100)
        summary_lines.append("")

        for q_result in all_results:
            summary_lines.append(f"{'#' * 100}")
            summary_lines.append(f"질문 {q_result['question_id']}: {q_result['question']}")
            summary_lines.append(f"{'#' * 100}")
            summary_lines.append(f"검색된 문서: {q_result['num_docs']}개")
            summary_lines.append("")

            for llm_result in q_result['llms']:
                summary_lines.append(f"\n{'=' * 80}")
                summary_lines.append(f"LLM: {llm_result['llm_display_name']}")
                summary_lines.append(f"{'=' * 80}")

                if llm_result.get('success'):
                    answer = llm_result.get('answer', 'N/A')
                    summary_lines.append(f"답변 (길이: {llm_result.get('answer_length', 0)}자):")
                    summary_lines.append(answer)
                else:
                    summary_lines.append(f"❌ 오류: {llm_result.get('error', 'Unknown error')}")

                summary_lines.append("")

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))

    return str(output_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="여러 LLM 모델을 사용한 보고서 테스트 (Langfuse 연동)"
    )
    parser.add_argument(
        "--report-type",
        type=str,
        choices=["weekly", "executive"],
        required=True,
        help="보고서 타입: weekly(주간보고서), executive(최종보고서)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        default=None,
        help="테스트할 질문 리스트 (공백으로 구분). 미지정시 기본 질문 사용"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="결과 저장 디렉토리 (기본: data/results/multi_llm_test)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="평가 버전 태그 (기본: v1)"
    )

    args = parser.parse_args()

    # 기본 질문 설정
    if args.questions:
        questions = args.questions
    else:
        # evaluation_config.yaml에서 기본 질문 로드
        config = get_report_config(args.report_type)
        questions = config['test_questions']  # 모든 질문 사용 (5개)

        print(f"\n✅ 기본 테스트 질문을 사용합니다 ({len(questions)}개):")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")

    # 테스트 실행
    run_multi_llm_test(
        questions=questions,
        report_type=args.report_type,
        output_dir=args.output_dir,
        version=args.version
    )


if __name__ == "__main__":
    main()
