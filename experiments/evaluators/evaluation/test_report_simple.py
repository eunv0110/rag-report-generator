#!/usr/bin/env python3
"""보고서용 Top 3 리트리버 보고서 자동 생성 및 비교 (Langfuse 연동)

세 가지 최고 성능 리트리버로 실제 보고서를 생성하고 결과를 비교합니다.
Langfuse를 사용하여 모든 실행을 추적하고 비교합니다.

보고서 타입:
- executive: 최종보고서 (Gemini, BGE-M3, OpenAI)
- weekly: 주간보고서 (BGE-M3, Upstage, OpenAI)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from langchain.chat_models import init_chat_model
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT, EVALUATION_CONFIG
from utils.langfuse import get_langfuse_client
from utils.dates import parse_date_range, extract_date_filter_from_question
from utils.common import load_prompt

# 리트리버 임포트
from retrievers.ensemble_retriever import get_ensemble_retriever


def get_report_config(report_type: str) -> Dict[str, Any]:
    """보고서 타입에 따른 설정 반환"""
    if report_type not in ['executive', 'weekly']:
        raise ValueError(f"Invalid report_type: {report_type}. Must be 'executive' or 'weekly'")

    return {
        'test_questions': EVALUATION_CONFIG['test_questions'][f'{report_type}_report'],
        'retriever_configs': EVALUATION_CONFIG['simple_test_retrievers'][f'{report_type}_report'],
        'default_retriever_index': EVALUATION_CONFIG['default_retriever_index'][f'{report_type}_report'],
        'llm_configs': EVALUATION_CONFIG['test_llms'],
        'prompt_dir': f'{report_type}_report',
        'qdrant_locks': {
            'executive': [
                "/home/work/rag/Project/rag-report-generator/data/qdrant_data/gemini-embedding-001/.lock",
                "/home/work/rag/Project/rag-report-generator/data/qdrant_data/baai-bge-m3/.lock",
                "/home/work/rag/Project/rag-report-generator/data/qdrant_data/openai-large/.lock"
            ],
            'weekly': [
                "/home/work/rag/Project/rag-report-generator/data/qdrant_data/baai-bge-m3/.lock",
                "/home/work/rag/Project/rag-report-generator/data/qdrant_data/upstage-solar-embedding/.lock",
                "/home/work/rag/Project/rag-report-generator/data/qdrant_data/openai-large/.lock"
            ]
        }[report_type]
    }


def _load_evaluation_prompt(prompt_file: str, report_type: str) -> str:
    """평가용 프롬프트 파일 로드 (헬퍼 함수)"""
    prompt_path = f"prompts/templates/evaluation/{report_type}_report/{prompt_file}"
    return load_prompt(prompt_path)


def load_test_questions(n: int = 5) -> List[Dict[str, Any]]:
    """테스트용 질문 로드"""
    qa_file = Path(__file__).parent.parent / "data" / "evaluation" / "llm_generated_qa_v2.json"

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    return qa_data[:n]


def generate_answer_with_llm(query: str, docs: list, llm_config: dict, report_type: str, langfuse=None, question_id: int = None) -> str:
    """지정된 LLM으로 답변 생성"""
    from openai import OpenAI
    from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

    # LLM별 컨텍스트 제한 처리
    if 'max_docs' in llm_config and llm_config['max_docs']:
        max_docs = llm_config['max_docs']
        if len(docs) > max_docs:
            docs = docs[:max_docs]
            print(f"  ⚠️ {llm_config['display_name']} 컨텍스트 제한으로 문서 수를 {len(docs)}개로 줄였습니다.")

    # Context 구성
    context_parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get('page_title', 'Unknown')
        content = doc.page_content
        context_parts.append(f"[문서 {i}] {title}\n{content}\n")

    context_text = "\n".join(context_parts)

    # 프롬프트 로드
    system_prompt = _load_evaluation_prompt("system_prompt.txt", report_type)
    answer_generation_template = _load_evaluation_prompt("answer_generation_prompt.txt", report_type)

    # 템플릿에 변수 대입
    user_prompt = answer_generation_template.replace("{context}", context_text).replace("{question}", query)

    # Azure AI 설정
    os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
    os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # LLM 타입에 따라 다르게 호출
    if llm_config['model_id'].startswith('anthropic:'):
        # Anthropic Claude 모델 - OpenRouter 사용
        model_name = llm_config['model_id'].split(':')[1]
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )

        # Langfuse로 답변 생성 기록
        if langfuse and question_id:
            with langfuse.start_as_current_observation(
                as_type='generation',
                name=f"generation_{llm_config['name']}_q{question_id}",
                model=llm_config['model_id'],
                input={"question": query, "context": context_text[:500] + "..." if len(context_text) > 500 else context_text},
                metadata={"llm": llm_config['name'], "num_docs": len(docs)}
            ) as generation:
                response = client.chat.completions.create(
                    model=f"anthropic/{model_name}",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0
                )
                answer = response.choices[0].message.content

                # 토큰 사용량 추출 (OpenAI API 형식)
                usage_dict = None
                if hasattr(response, 'usage') and response.usage:
                    usage_dict = {
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }

                generation.update(
                    output={"answer": answer},
                    usage=usage_dict if usage_dict else None
                )

                # Trace 업데이트 (LLM과 retriever 정보 포함)
                langfuse.update_current_trace(
                    name=f"llm_comparison_{llm_config['name']}_q{question_id}",
                    tags=[llm_config['name'], "llm_comparison", f"{report_type}_report"],
                    input={"question": query},
                    output={"answer": answer},
                    metadata={
                        "llm_name": llm_config['name'],
                        "llm_display_name": llm_config['display_name'],
                        "llm_description": llm_config['description'],
                        "model_id": llm_config['model_id'],
                        "question_id": question_id,
                        "num_docs": len(docs),
                        "report_type": report_type
                    }
                )
        else:
            response = client.chat.completions.create(
                model=f"anthropic/{model_name}",
                messages=messages,
                max_tokens=1000,
                temperature=0
            )
            answer = response.choices[0].message.content
    else:
        # Azure AI 모델 (LangChain)
        model = init_chat_model(
            llm_config['model_id'],
            temperature=0,
            max_completion_tokens=1000
        )

        # Langfuse로 답변 생성 기록
        if langfuse and question_id:
            with langfuse.start_as_current_observation(
                as_type='generation',
                name=f"generation_{llm_config['name']}_q{question_id}",
                model=llm_config['model_id'],
                input={"question": query, "context": context_text[:500] + "..." if len(context_text) > 500 else context_text},
                metadata={"llm": llm_config['name'], "num_docs": len(docs)}
            ) as generation:
                response = model.invoke(messages)
                answer = response.content

                # 토큰 사용량 추출 (LangChain response)
                usage_dict = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage_dict = {
                        "input": response.usage_metadata.get('input_tokens', 0),
                        "output": response.usage_metadata.get('output_tokens', 0),
                        "total": response.usage_metadata.get('total_tokens', 0)
                    }
                elif hasattr(response, 'response_metadata') and response.response_metadata:
                    # 일부 LangChain 모델은 response_metadata에 token_usage 포함
                    token_usage = response.response_metadata.get('token_usage', {})
                    if token_usage:
                        usage_dict = {
                            "input": token_usage.get('prompt_tokens', 0),
                            "output": token_usage.get('completion_tokens', 0),
                            "total": token_usage.get('total_tokens', 0)
                        }

                generation.update(
                    output={"answer": answer},
                    usage=usage_dict if usage_dict else None
                )

                # Trace 업데이트 (LLM과 retriever 정보 포함)
                langfuse.update_current_trace(
                    name=f"llm_comparison_{llm_config['name']}_q{question_id}",
                    tags=[llm_config['name'], "llm_comparison", f"{report_type}_report"],
                    input={"question": query},
                    output={"answer": answer},
                    metadata={
                        "llm_name": llm_config['name'],
                        "llm_display_name": llm_config['display_name'],
                        "llm_description": llm_config['description'],
                        "model_id": llm_config['model_id'],
                        "question_id": question_id,
                        "num_docs": len(docs),
                        "report_type": report_type
                    }
                )
        else:
            response = model.invoke(messages)
            answer = response.content

    return answer


def retrieve_documents(question: str, retriever_config: dict, qdrant_locks: List[str], date_filter: tuple = None, langfuse=None, question_id: int = None):
    """리트리버로 문서 검색 (1번만 수행)"""
    import time
    import gc

    # Qdrant 락 파일 정리 (매번 실행 전)
    for lock_file in qdrant_locks:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                pass

    # 가비지 컬렉션 강제 실행
    gc.collect()
    time.sleep(0.5)

    # 환경 변수 설정
    os.environ["MODEL_PRESET"] = retriever_config['embedding']

    # 임베딩 캐시 활성화
    os.environ["USE_EMBEDDING_CACHE"] = "true"

    # 리트리버 생성
    if retriever_config['type'] == 'rrf_ensemble':
        retriever = get_ensemble_retriever(
            k=retriever_config['top_k'],
            bm25_weight=0.5,
            dense_weight=0.5,
            date_filter=date_filter
        )
    elif retriever_config['type'] == 'rrf_multiquery_lc':
        # MultiQuery + RRF Ensemble
        from retrievers.multiquery_retriever import get_multiquery_retriever

        # 기본 RRF Ensemble 리트리버 생성
        base_retriever = get_ensemble_retriever(
            k=retriever_config['top_k'],
            bm25_weight=0.5,
            dense_weight=0.5,
            date_filter=date_filter
        )

        # MultiQuery로 래핑
        retriever = get_multiquery_retriever(
            base_retriever=base_retriever,
            num_queries=3,
            temperature=0.7
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_config['type']}")

    # 문서 검색
    print("🔍 검색 중...")

    # Langfuse로 검색 과정 기록
    if langfuse and question_id:
        with langfuse.start_as_current_observation(
            as_type='span',
            name=f"retrieval_{retriever_config['name']}_q{question_id}",
            input={"question": question},
            metadata={
                "retriever_name": retriever_config['name'],
                "retriever_display_name": retriever_config['display_name'],
                "retriever_type": retriever_config['type'],
                "embedding": retriever_config['embedding'],
                "top_k": retriever_config['top_k'],
                "description": retriever_config['description']
            }
        ) as span:
            docs = retriever.invoke(question)
            span.update(output={
                "num_docs": len(docs),
                "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in docs]
            })
    else:
        docs = retriever.invoke(question)

    print(f"\n📄 검색된 문서 수: {len(docs)}")
    print("\n검색된 문서 제목:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.metadata.get('page_title', 'Unknown')}")

    return docs


def test_single_llm(
    llm_config: Dict[str, Any],
    question: str,
    docs: list,
    report_type: str,
    langfuse=None,
    question_id: int = None
) -> Dict[str, Any]:
    """단일 LLM으로 답변 생성"""

    print(f"\n{'=' * 100}")
    print(f"💬 {llm_config['display_name']}")
    print(f"   {llm_config['description']}")
    print(f"{'=' * 100}")

    try:
        # 답변 생성
        print("💬 답변 생성 중...")
        answer = generate_answer_with_llm(
            question,
            docs,
            llm_config,
            report_type,
            langfuse,
            question_id
        )

        print(f"\n✅ 생성된 답변:\n{answer}\n")

        return {
            "success": True,
            "llm_name": llm_config['name'],
            "llm_display_name": llm_config['display_name'],
            "model_id": llm_config['model_id'],
            "answer": answer
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}\n")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "llm_name": llm_config['name'],
            "llm_display_name": llm_config['display_name'],
            "error": str(e)
        }


def save_combination_results(all_results: List[Dict], retriever_config: Dict, llm_configs: List[Dict], timestamp: str, combination_base_dir: Path, questions: List[str]):
    """중간 결과를 조합별로 저장하는 함수"""

    # 전체 조합 결과를 모을 리스트
    all_combinations_data = []
    all_answers_text = []

    for llm_config in llm_configs:
        llm_name = llm_config['name']

        # Retriever + LLM 조합 폴더명 생성
        combination_name = f"{retriever_config['name']}_{llm_name}"
        combination_dir = combination_base_dir / combination_name
        combination_dir.mkdir(parents=True, exist_ok=True)

        # 해당 LLM의 결과만 필터링
        llm_results = []
        for q_result in all_results:
            for llm_result in q_result['llms']:
                if llm_result['llm_name'] == llm_name:
                    llm_results.append({
                        "question_id": q_result['question_id'],
                        "question": q_result['question'],
                        "date_filter": q_result['date_filter'],
                        "num_docs": q_result['num_docs'],
                        "doc_titles": q_result['doc_titles'],
                        "result": llm_result
                    })

        # Retriever + LLM 조합별 JSON 저장
        combination_output = {
            "combination_name": combination_name,
            "retriever_name": retriever_config['name'],
            "retriever_display_name": retriever_config['display_name'],
            "retriever_description": retriever_config['description'],
            "llm_name": llm_name,
            "llm_display_name": llm_config['display_name'],
            "llm_description": llm_config['description'],
            "num_questions": len(llm_results),
            "results": llm_results
        }

        # 전체 조합 데이터에 추가
        all_combinations_data.append(combination_output)

        combination_file = combination_dir / "results.json"
        with open(combination_file, 'w', encoding='utf-8') as f:
            json.dump(combination_output, f, ensure_ascii=False, indent=2)

        # 답변만 따로 텍스트 파일로 저장
        answer_file = combination_dir / "answers.txt"
        answer_text_parts = []
        answer_text_parts.append(f"=" * 100)
        answer_text_parts.append(f"{retriever_config['display_name']} + {llm_config['display_name']} - 답변 모음")
        answer_text_parts.append(f"=" * 100)
        answer_text_parts.append("")

        for i, result in enumerate(llm_results, 1):
            answer_text_parts.append(f"{'=' * 100}")
            answer_text_parts.append(f"질문 {i}/{len(llm_results)}")
            answer_text_parts.append(f"{'=' * 100}")
            answer_text_parts.append(f"❓ {result['question']}")
            answer_text_parts.append("")

            if result['result'].get('success'):
                answer_text_parts.append(f"✅ 답변:")
                answer_text_parts.append(f"{result['result'].get('answer', 'N/A')}")
                answer_text_parts.append("")
            else:
                answer_text_parts.append(f"❌ 오류 발생:")
                answer_text_parts.append(f"{result['result'].get('error', 'Unknown error')}")
                answer_text_parts.append("")

        answer_text = "\n".join(answer_text_parts)
        with open(answer_file, 'w', encoding='utf-8') as f:
            f.write(answer_text)

        # 전체 답변 텍스트에 추가
        all_answers_text.append(answer_text)
        all_answers_text.append("\n\n")  # 조합 간 구분

    # 전체 조합 결과를 하나의 JSON 파일로 저장
    all_combinations_file = combination_base_dir / "all_combinations.json"
    all_combinations_output = {
        "test_date": datetime.now().isoformat(),
        "retriever_name": retriever_config['name'],
        "retriever_display_name": retriever_config['display_name'],
        "num_llms": len(llm_configs),
        "num_questions": len(questions),
        "num_completed_questions": len(all_results),
        "total_combinations": len(all_combinations_data),
        "combinations": all_combinations_data
    }

    with open(all_combinations_file, 'w', encoding='utf-8') as f:
        json.dump(all_combinations_output, f, ensure_ascii=False, indent=2)

    # 전체 답변을 하나의 텍스트 파일로 저장
    all_answers_file = combination_base_dir / "all_answers.txt"
    with open(all_answers_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_answers_text))

    print(f"✅ 중간 결과 저장 완료: {combination_base_dir}")


def run_comparison_test(
    questions: List[str],
    report_type: str,
    output_file: str = None,
    global_date_filter: tuple = None,
    retriever_index: int = None
):
    """여러 질문에 대해 여러 LLM 비교 테스트 (1개 리트리버 사용)

    Args:
        questions: 질문 리스트
        report_type: 보고서 타입 ('executive' 또는 'weekly')
        output_file: 결과 저장 경로
        global_date_filter: 전역 날짜 필터 (명시적으로 지정된 경우, 질문별 추출보다 우선)
        retriever_index: 사용할 리트리버 인덱스 (None이면 DEFAULT_RETRIEVER_INDEX 사용)
    """

    # 보고서 타입에 맞는 설정 로드
    config = get_report_config(report_type)

    # 사용할 리트리버 결정
    if retriever_index is None:
        retriever_index = config['default_retriever_index']

    retriever_config = config['retriever_configs'][retriever_index]
    llm_configs = config['llm_configs']

    # Langfuse 초기화
    langfuse = get_langfuse_client()

    report_type_display = "최종보고서용" if report_type == "executive" else "주간보고서용"

    print("\n" + "=" * 100)
    print(f"🧪 {report_type_display} LLM 비교 테스트 (Langfuse 연동)")
    print("=" * 100)
    print(f"\n🔍 리트리버: {retriever_config['display_name']} - {retriever_config['description']}")
    if global_date_filter:
        print(f"📅 전역 날짜 필터: {global_date_filter[0][:10]} ~ {global_date_filter[1][:10]}")
    print("\n💬 평가 대상 LLM:")
    for i, llm_config in enumerate(llm_configs, 1):
        print(f"  {i}. {llm_config['display_name']} - {llm_config['description']}")
    print()

    all_results = []

    # 타임스탬프를 먼저 생성 (모든 저장에 동일한 타임스탬프 사용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combination_base_dir = Path("data/results/combinations") / timestamp
    combination_base_dir.mkdir(parents=True, exist_ok=True)

    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'=' * 100}")
        print(f"📋 질문 {q_idx}/{len(questions)}")
        print(f"{'=' * 100}")
        print(f"❓ {question}\n")

        # 날짜 필터 결정: 전역 필터가 있으면 그것 사용, 없으면 질문에서 추출
        if global_date_filter:
            date_filter = global_date_filter
        else:
            date_filter = extract_date_filter_from_question(question)
            if date_filter:
                print(f"📅 감지된 날짜 필터: {date_filter[0][:10]} ~ {date_filter[1][:10]}\n")

        # 문서 검색 (1번만 수행)
        docs = retrieve_documents(question, retriever_config, config['qdrant_locks'], date_filter, langfuse, q_idx)

        question_result = {
            "question_id": q_idx,
            "question": question,
            "date_filter": f"{date_filter[0][:10]} ~ {date_filter[1][:10]}" if date_filter else None,
            "retriever": retriever_config['name'],
            "num_docs": len(docs),
            "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in docs],
            "llms": []
        }

        # 각 LLM으로 병렬 테스트
        print(f"\n🚀 7개 LLM으로 병렬 답변 생성 시작...\n")

        with ThreadPoolExecutor(max_workers=7) as executor:
            # 모든 LLM 작업을 동시에 제출
            future_to_llm = {
                executor.submit(test_single_llm, llm_config, question, docs, report_type, langfuse, q_idx): (rank, llm_config)
                for rank, llm_config in enumerate(llm_configs, 1)
            }

            # 완료되는 순서대로 결과 수집
            for future in as_completed(future_to_llm):
                rank, llm_config = future_to_llm[future]
                try:
                    result = future.result()
                    result["rank"] = rank
                    question_result["llms"].append(result)

                    # LLM 답변이 생성될 때마다 즉시 저장 (중간에 멈춰도 결과 보존)
                    temp_results = all_results + [question_result]
                    try:
                        save_combination_results(temp_results, retriever_config, llm_configs, timestamp, combination_base_dir, questions)
                        print(f"💾 {llm_config['display_name']} 결과 저장 완료")
                    except Exception as e:
                        print(f"⚠️ 저장 실패: {e}")

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

        # rank 순서대로 정렬 (병렬 실행으로 순서가 섞였을 수 있음)
        question_result["llms"].sort(key=lambda x: x["rank"])

        all_results.append(question_result)

        # 질문이 끝날 때마다 최종 저장
        print(f"\n💾 질문 {q_idx} 완료 - 최종 결과 저장 중...")
        save_combination_results(all_results, retriever_config, llm_configs, timestamp, combination_base_dir, questions)

    # 최종 저장 (이미 중간 저장되었으므로 summary.json만 추가 저장)
    base_output_dir = Path("data/results/llm_comparison") / f"{retriever_config['name']}_{timestamp}"

    # 전체 결과 저장
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = base_output_dir / "summary.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_result = {
        "test_date": datetime.now().isoformat(),
        "report_type": report_type,
        "num_questions": len(questions),
        "retriever_config": retriever_config,
        "llm_configs": llm_configs,
        "results": all_results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 100}")
    print(f"✅ 모든 테스트 완료!")
    print(f"📁 Summary 저장: {output_path}")
    print(f"📁 조합별 결과: {combination_base_dir}")

    print(f"\n🔗 Langfuse에서 상세 추적 정보를 확인하세요")
    print(f"{'=' * 100}")

    # Langfuse flush
    langfuse.flush()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="보고서용 Top 3 리트리버로 보고서 생성 및 비교 (Langfuse 연동)"
    )
    parser.add_argument(
        "--report-type",
        type=str,
        choices=["executive", "weekly"],
        default="executive",
        help="보고서 타입: executive(최종보고서), weekly(주간보고서) (기본: executive)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi", "interactive"],
        default="interactive",
        help="테스트 모드: single(단일 질문), multi(파일에서 여러 질문), interactive(대화형 입력)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="multi 모드에서 테스트할 질문 개수 (기본: 5)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="single 모드에서 사용할 질문"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 경로 (지정하지 않으면 저장하지 않음)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="시작 날짜 (YYYY-MM-DD 형식)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="종료 날짜 (YYYY-MM-DD 형식)"
    )
    parser.add_argument(
        "--date-range",
        type=str,
        default=None,
        help="자연어 날짜 범위 (예: '이번 주', '지난주', '12월 2주차')"
    )
    parser.add_argument(
        "--retriever",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="사용할 리트리버 인덱스 (0~2, 보고서 타입별로 다름)"
    )

    args = parser.parse_args()

    # 날짜 필터 파싱
    date_filter = parse_date_range(
        date_input=args.date_range,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 보고서 타입에 맞는 설정 로드
    config = get_report_config(args.report_type)

    if args.mode == "interactive":
        # 대화형 모드 - 사용자가 직접 질문 입력
        print("\n" + "=" * 100)
        print(f"🎯 대화형 모드: 질문을 직접 입력하세요 (보고서 타입: {args.report_type})")
        print("=" * 100)
        print("💡 팁:")
        print("   - 여러 질문을 입력하려면 세미콜론(;)으로 구분하세요")
        print("   - 입력 없이 엔터를 누르면 기본 테스트 질문 5개가 사용됩니다")
        print()

        user_input = input("📝 질문을 입력하세요 (엔터: 기본 질문 사용): ").strip()

        if not user_input:
            print("\n✅ 기본 테스트 질문을 사용합니다:")
            questions = config['test_questions']
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")
        else:
            # 세미콜론으로 구분된 여러 질문 처리
            questions = [q.strip() for q in user_input.split(';') if q.strip()]

            print(f"\n✅ 총 {len(questions)}개의 질문이 입력되었습니다.")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")

        run_comparison_test(questions, args.report_type, args.output, date_filter, args.retriever)

    elif args.mode == "single":
        # 단일 질문 모드
        if not args.question:
            print("❌ --question 인자가 필요합니다.")
            return
        run_comparison_test([args.question], args.report_type, args.output, date_filter, args.retriever)

    else:
        # 여러 질문 모드 (파일에서 로드)
        test_questions_data = load_test_questions(n=args.num_questions)
        questions = [q["question"] for q in test_questions_data]

        if args.output is None:
            args.output = "data/final/llm_comparison_langfuse.json"

        run_comparison_test(questions, args.report_type, args.output, date_filter, args.retriever)


if __name__ == "__main__":
    main()
