#!/usr/bin/env python3
"""BGE-M3 + RRF Ensemble + Qwen3-Reranker-4B 보고서 자동 생성 및 비교 (Langfuse 연동)

BGE-M3 RRF Ensemble로 문서를 검색한 후, Qwen3-Reranker-4B로 재순위화하여
최종 k개(6, 8, 10)의 문서를 LLM에 전달하여 답변을 생성합니다.

보고서 타입:
- executive: 최종보고서
- weekly: 주간보고서
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
from sentence_transformers import CrossEncoder

from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT, EVALUATION_CONFIG
from utils.langfuse_utils import get_langfuse_client
from utils.date_utils import parse_date_range, extract_date_filter_from_question
from utils.common_utils import (
    load_evaluation_dataset,
    create_trace_and_generation,
    add_retrieval_quality_score,
    save_embedding_cache
)

# 리트리버 임포트
from retrievers.ensemble_retriever import get_ensemble_retriever


# Qwen3 Reranker 초기화 (전역 변수로 한 번만 로드)
RERANKER_MODEL = None

def format_queries(query: str, instruction: str = None) -> str:
    """Qwen3-Reranker를 위한 쿼리 포맷팅"""
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = "Given a query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document: str) -> str:
    """Qwen3-Reranker를 위한 문서 포맷팅"""
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


def get_reranker():
    """Qwen3-Reranker-4B-seq-cls 모델 로드 (캐싱)"""
    global RERANKER_MODEL
    if RERANKER_MODEL is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔄 Qwen3-Reranker-4B-seq-cls 모델 로딩 중... (device: {device})")
        RERANKER_MODEL = CrossEncoder(
            "tomaarsen/Qwen3-Reranker-4B-seq-cls",
            max_length=8192,
            device=device,
            trust_remote_code=True
        )
        print("✅ Qwen3-Reranker-4B-seq-cls 모델 로드 완료!")
    return RERANKER_MODEL


def rerank_documents(query: str, docs: list, top_k: int = 6) -> list:
    """Qwen3-Reranker-4B로 문서 재순위화

    Args:
        query: 검색 쿼리
        docs: 검색된 문서 리스트 (langchain Document 객체)
        top_k: 최종 반환할 문서 수

    Returns:
        재순위화된 상위 k개 문서
    """
    reranker = get_reranker()

    # Qwen3-Reranker 포맷에 맞게 쿼리와 문서 변환
    formatted_query = format_queries(query)
    pairs = [
        [formatted_query, format_document(doc.page_content)]
        for doc in docs
    ]

    # 재순위화 점수 계산
    scores = reranker.predict(pairs)

    # 점수와 문서를 함께 정렬
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 상위 k개 문서만 반환
    reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

    print(f"\n🔄 Reranking 완료: {len(docs)}개 → {len(reranked_docs)}개")
    print("Top 3 Reranked Scores:")
    for i, (doc, score) in enumerate(doc_score_pairs[:3], 1):
        print(f"  {i}. {doc.metadata.get('page_title', 'Unknown')}: {score:.4f}")

    return reranked_docs


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


def load_prompt(prompt_file: str, report_type: str) -> str:
    """프롬프트 파일 로드"""
    # evaluators/evaluation/test_report_reranker.py에서 프로젝트 루트로 이동
    project_root = Path(__file__).parent.parent.parent
    prompt_path = project_root / "prompts" / "templates" / "evaluation" / f"{report_type}_report" / prompt_file
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_test_questions(n: int = 5) -> List[Dict[str, Any]]:
    """테스트용 질문 로드 (레거시, interactive 모드용)"""
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
    system_prompt = load_prompt("system_prompt.txt", report_type)
    answer_generation_template = load_prompt("answer_generation_prompt.txt", report_type)

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

                # 토큰 사용량 추출
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

                langfuse.update_current_trace(
                    name=f"llm_comparison_{llm_config['name']}_q{question_id}",
                    tags=[llm_config['name'], "reranker_test", f"{report_type}_report"],
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

                # 토큰 사용량 추출
                usage_dict = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage_dict = {
                        "input": response.usage_metadata.get('input_tokens', 0),
                        "output": response.usage_metadata.get('output_tokens', 0),
                        "total": response.usage_metadata.get('total_tokens', 0)
                    }
                elif hasattr(response, 'response_metadata') and response.response_metadata:
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

                langfuse.update_current_trace(
                    name=f"llm_comparison_{llm_config['name']}_q{question_id}",
                    tags=[llm_config['name'], "reranker_test", f"{report_type}_report"],
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


def retrieve_and_rerank_documents(
    question: str,
    qdrant_locks: List[str],
    top_k: int = 6,
    date_filter: tuple = None,
    langfuse=None,
    question_id: int = None
):
    """BGE-M3 RRF Ensemble로 검색 후 Qwen3-Reranker로 재순위화"""
    import time
    import gc

    # Qdrant 락 파일 정리
    for lock_file in qdrant_locks:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                pass

    gc.collect()
    time.sleep(0.5)

    # BGE-M3 임베딩 설정
    os.environ["MODEL_PRESET"] = "bge-m3"
    os.environ["USE_EMBEDDING_CACHE"] = "true"

    # RRF Ensemble 리트리버 생성 (더 많은 후보 검색)
    initial_k = max(20, top_k * 3)  # 재순위화를 위해 더 많은 문서 검색
    retriever = get_ensemble_retriever(
        k=initial_k,
        bm25_weight=0.5,
        dense_weight=0.5,
        date_filter=date_filter
    )

    print(f"🔍 BGE-M3 RRF Ensemble 검색 중 (초기 k={initial_k})...")

    # Langfuse로 검색 과정 기록
    if langfuse and question_id:
        with langfuse.start_as_current_observation(
            as_type='span',
            name=f"retrieval_bge-m3-rrf_q{question_id}",
            input={"question": question},
            metadata={
                "retriever_name": "bge-m3-rrf-reranker",
                "initial_k": initial_k,
                "final_k": top_k,
                "reranker": "Qwen3-Reranker-4B"
            }
        ) as span:
            # 문서 검색
            initial_docs = retriever.invoke(question)
            print(f"📄 초기 검색 문서 수: {len(initial_docs)}")

            # Reranking
            reranked_docs = rerank_documents(question, initial_docs, top_k=top_k)

            span.update(output={
                "initial_num_docs": len(initial_docs),
                "final_num_docs": len(reranked_docs),
                "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in reranked_docs]
            })
    else:
        initial_docs = retriever.invoke(question)
        print(f"📄 초기 검색 문서 수: {len(initial_docs)}")
        reranked_docs = rerank_documents(question, initial_docs, top_k=top_k)

    print(f"\n📄 최종 문서 수: {len(reranked_docs)}")
    print("\n최종 문서 제목:")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"  {i}. {doc.metadata.get('page_title', 'Unknown')}")

    return reranked_docs


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


def save_combination_results(
    all_results: List[Dict],
    top_k: int,
    llm_configs: List[Dict],
    timestamp: str,
    combination_base_dir: Path,
    questions: List[str],
    report_type: str
):
    """중간 결과를 조합별로 저장하는 함수"""

    all_combinations_data = []
    all_answers_text = []

    for llm_config in llm_configs:
        llm_name = llm_config['name']

        # Retriever + LLM 조합 폴더명 생성
        combination_name = f"bge-m3-rrf-reranker-k{top_k}_{llm_name}"
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
            "retriever_name": f"bge-m3-rrf-reranker-k{top_k}",
            "retriever_display_name": f"BGE-M3 + RRF + Qwen3-Reranker (Top {top_k})",
            "retriever_description": f"BGE-M3 embedding with RRF ensemble and Qwen3-Reranker-4B reranking (final k={top_k})",
            "llm_name": llm_name,
            "llm_display_name": llm_config['display_name'],
            "llm_description": llm_config['description'],
            "num_questions": len(llm_results),
            "results": llm_results
        }

        all_combinations_data.append(combination_output)

        combination_file = combination_dir / "results.json"
        with open(combination_file, 'w', encoding='utf-8') as f:
            json.dump(combination_output, f, ensure_ascii=False, indent=2)

        # 답변만 따로 텍스트 파일로 저장
        answer_file = combination_dir / "answers.txt"
        answer_text_parts = []
        answer_text_parts.append(f"=" * 100)
        answer_text_parts.append(f"BGE-M3 + RRF + Qwen3-Reranker (Top {top_k}) + {llm_config['display_name']} - 답변 모음")
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

        all_answers_text.append(answer_text)
        all_answers_text.append("\n\n")

    # 전체 조합 결과를 하나의 JSON 파일로 저장
    all_combinations_file = combination_base_dir / "all_combinations.json"
    all_combinations_output = {
        "test_date": datetime.now().isoformat(),
        "report_type": report_type,
        "retriever_name": f"bge-m3-rrf-reranker-k{top_k}",
        "retriever_display_name": f"BGE-M3 + RRF + Qwen3-Reranker (Top {top_k})",
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


def evaluate_single_query_with_reranker(
    question: str,
    ground_truth: str,
    context_page_id: str,
    item_metadata: Dict[str, Any],
    report_type: str,
    top_k: int,
    qdrant_locks: List[str],
    system_prompt: str,
    answer_generation_prompt: str,
    langfuse,
    idx: int,
    version_tag: str = "v1"
) -> Dict[str, Any]:
    """Reranker를 사용한 단일 쿼리 평가 (evaluate_report_types.py 스타일)

    Args:
        question: 질문
        ground_truth: 정답
        context_page_id: 정답 문서의 page_id
        item_metadata: 질문 메타데이터
        report_type: 보고서 타입
        top_k: Top-K 값
        qdrant_locks: Qdrant 락 파일 경로 리스트
        system_prompt: 시스템 프롬프트
        answer_generation_prompt: 답변 생성 프롬프트
        langfuse: Langfuse 클라이언트
        idx: 질문 인덱스
        version_tag: 버전 태그

    Returns:
        평가 결과 딕셔너리
    """
    import time

    start_time = time.time()

    # 문서 검색 및 재순위화
    docs = retrieve_and_rerank_documents(
        question=question,
        qdrant_locks=qdrant_locks,
        top_k=top_k,
        date_filter=None,  # 평가 데이터셋에는 날짜 필터 사용하지 않음
        langfuse=langfuse,
        question_id=idx
    )

    # 컨텍스트 텍스트 변환
    contexts = []
    context_metadata = []

    for result in docs:
        contexts.append(result.page_content)
        context_metadata.append({
            "page_title": result.metadata.get('page_title', 'Unknown'),
            "section_title": result.metadata.get('section_title', 'N/A'),
            "chunk_id": result.metadata.get('chunk_id', 'unknown'),
            "score": result.metadata.get('_combined_score') or result.metadata.get('_similarity_score')
        })

    if not contexts:
        print(f"  ⚠️ [{idx}] No contexts found for question!")
        contexts = ["검색 결과가 없습니다."]

    # LLM 답변 생성 (common_utils의 generate_llm_answer 사용)
    from utils.common_utils import generate_llm_answer
    answer = generate_llm_answer(
        question=question,
        contexts=contexts,
        system_prompt=system_prompt,
        answer_generation_prompt=answer_generation_prompt,
        num_contexts=min(5, len(contexts)),
        temperature=0.1,
        max_tokens=1000
    )

    if not answer or answer.startswith("답변 생성 실패") or answer.startswith("Azure OpenAI 설정"):
        print(f"  ⚠️ [{idx}] LLM answer generation failed!")
        if not answer:
            answer = "답변을 생성할 수 없습니다."

    total_time = time.time() - start_time

    # Langfuse Trace & Generation 생성
    retriever_name = f"bge-m3-rrf-reranker-k{top_k}"
    retriever_tags = [
        "bge-m3",
        "rrf_ensemble",
        "qwen3-reranker",
        f"top_k_{top_k}",
        report_type
    ]

    additional_metadata = {
        "context_page_id": context_page_id,
        "retriever_name": retriever_name,
        "display_name": f"BGE-M3 + RRF + Qwen3-Reranker (Top {top_k})",
        "report_type": report_type,
        "top_k": top_k,
        "embedding_preset": "bge-m3",
        "retriever_type": "rrf_ensemble_reranker",
        "reranker_model": "Qwen3-Reranker-4B"
    }

    trace_id = create_trace_and_generation(
        langfuse=langfuse,
        retriever_name=retriever_name,
        question=question,
        contexts=contexts,
        answer=answer,
        ground_truth=ground_truth,
        context_metadata=context_metadata,
        item_metadata=item_metadata,
        total_time=total_time,
        idx=idx,
        version_tag=version_tag,
        retriever_tags=retriever_tags,
        additional_metadata=additional_metadata
    )

    # 검색 품질 스코어 추가
    add_retrieval_quality_score(langfuse, trace_id, context_metadata)

    print(f"  [{idx}] {question[:50]}... ({len(contexts)}개 문서, {total_time*1000:.0f}ms)")

    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "num_contexts": len(contexts),
        "time": total_time,
        "trace_id": trace_id,
        "context_metadata": context_metadata
    }


def run_reranker_evaluation(
    report_type: str,
    dataset_path: str,
    top_k: int = 6,
    version: str = "v1",
    output_dir: str = None
) -> Dict[str, Any]:
    """Reranker를 사용한 평가 실행 (evaluate_report_types.py 스타일)

    Args:
        report_type: 보고서 타입 ('executive' 또는 'weekly')
        dataset_path: 평가 데이터셋 경로
        top_k: Top-K 값
        version: 버전 태그
        output_dir: 결과 저장 디렉토리

    Returns:
        평가 결과 딕셔너리
    """
    import yaml

    # 설정 로드
    config = get_report_config(report_type)

    # 프롬프트 로드 (common_utils 사용)
    from utils.common_utils import load_prompt as load_prompt_util

    report_type_suffix = "weekly_report" if report_type == "weekly" else "executive_report"
    system_prompt = load_prompt_util(f"prompts/templates/service/{report_type_suffix}/system_prompt.txt")
    answer_generation_prompt = load_prompt_util(f"prompts/templates/service/{report_type_suffix}/answer_generation_prompt.txt")

    # 평가 데이터셋 로드
    eval_data = load_evaluation_dataset(dataset_path)

    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()
    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        return {}

    report_type_display = "최종보고서" if report_type == "executive" else "주간보고서"

    print("\n" + "=" * 80)
    print(f"🔍 {report_type_display} BGE-M3 + RRF + Qwen3-Reranker 평가 (Langfuse 연동)")
    print("=" * 80)
    print(f"📋 데이터셋: {len(eval_data)} 개 샘플")
    print(f"📊 Top-K: {top_k}")
    print(f"🏷️  Version: {version}")
    print()

    stats = {
        "total_queries": len(eval_data),
        "total_time": 0,
        "evaluations": []
    }

    for idx, item in enumerate(eval_data, 1):
        try:
            eval_result = evaluate_single_query_with_reranker(
                question=item["question"],
                ground_truth=item["ground_truth"],
                context_page_id=item.get("context_page_id"),
                item_metadata=item.get("metadata", {}),
                report_type=report_type,
                top_k=top_k,
                qdrant_locks=config['qdrant_locks'],
                system_prompt=system_prompt,
                answer_generation_prompt=answer_generation_prompt,
                langfuse=langfuse,
                idx=idx,
                version_tag=version
            )

            stats["evaluations"].append(eval_result)
            stats["total_time"] += eval_result["time"]

            # 캐시 저장
            save_embedding_cache()

        except Exception as e:
            print(f"  ❌ [{idx}] 평가 실패: {e}")
            import traceback
            traceback.print_exc()

    stats["avg_time"] = stats["total_time"] / stats["total_queries"] if stats["total_queries"] > 0 else 0
    stats["avg_contexts"] = sum(e["num_contexts"] for e in stats["evaluations"]) / len(stats["evaluations"]) if stats["evaluations"] else 0

    # 결과 저장
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path("data/langfuse/evaluation_results") / f"{report_type}_report"

    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"bge-m3-rrf-reranker_k{top_k}_stats.json"
    save_result = {k: v for k, v in stats.items() if k != "evaluations"}
    save_result["num_evaluations"] = len(stats.get("evaluations", []))
    save_result["config"] = {
        "retriever_name": f"bge-m3-rrf-reranker-k{top_k}",
        "display_name": f"BGE-M3 + RRF + Qwen3-Reranker (Top {top_k})",
        "report_type": report_type,
        "embedding_preset": "bge-m3",
        "retriever_type": "rrf_ensemble_reranker",
        "reranker_model": "Qwen3-Reranker-4B",
        "top_k": top_k,
        "version": version,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 평가 완료!")
    print(f"   - 평균 시간: {stats['avg_time']*1000:.2f}ms")
    print(f"   - 평균 컨텍스트 수: {stats['avg_contexts']:.2f}")
    print(f"   - 결과 저장: {output_file}")

    # Langfuse flush
    print("\n⏳ Langfuse에 데이터 전송 중...")
    langfuse.flush()

    # 임베딩 캐시 저장
    print("\n💾 임베딩 캐시 저장 중...")
    save_embedding_cache()

    return stats


def run_reranker_test(
    questions: List[str],
    report_type: str,
    top_k_values: List[int] = [6, 8, 10],
    output_file: str = None,
    global_date_filter: tuple = None
):
    """BGE-M3 + RRF + Qwen3-Reranker 테스트 (여러 k 값)

    Args:
        questions: 질문 리스트
        report_type: 보고서 타입 ('executive' 또는 'weekly')
        top_k_values: 테스트할 k 값 리스트 (기본: [6, 8, 10])
        output_file: 결과 저장 경로
        global_date_filter: 전역 날짜 필터
    """

    config = get_report_config(report_type)
    llm_configs = config['llm_configs']
    langfuse = get_langfuse_client()

    report_type_display = "최종보고서용" if report_type == "executive" else "주간보고서용"

    print("\n" + "=" * 100)
    print(f"🧪 {report_type_display} BGE-M3 + RRF + Qwen3-Reranker 테스트 (Langfuse 연동)")
    print("=" * 100)
    print(f"\n🔍 리트리버: BGE-M3 + RRF Ensemble + Qwen3-Reranker-4B")
    print(f"📊 테스트할 k 값: {top_k_values}")
    if global_date_filter:
        print(f"📅 전역 날짜 필터: {global_date_filter[0][:10]} ~ {global_date_filter[1][:10]}")
    print("\n💬 평가 대상 LLM:")
    for i, llm_config in enumerate(llm_configs, 1):
        print(f"  {i}. {llm_config['display_name']} - {llm_config['description']}")
    print()

    # 각 k 값에 대해 테스트 수행
    for top_k in top_k_values:
        print(f"\n{'#' * 100}")
        print(f"🎯 Top-{top_k} 테스트 시작")
        print(f"{'#' * 100}\n")

        all_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combination_base_dir = Path("data/results/reranker_combinations") / f"k{top_k}_{timestamp}"
        combination_base_dir.mkdir(parents=True, exist_ok=True)

        for q_idx, question in enumerate(questions, 1):
            print(f"\n{'=' * 100}")
            print(f"📋 질문 {q_idx}/{len(questions)} (k={top_k})")
            print(f"{'=' * 100}")
            print(f"❓ {question}\n")

            # 날짜 필터 결정
            if global_date_filter:
                date_filter = global_date_filter
            else:
                date_filter = extract_date_filter_from_question(question)
                if date_filter:
                    print(f"📅 감지된 날짜 필터: {date_filter[0][:10]} ~ {date_filter[1][:10]}\n")

            # 문서 검색 및 재순위화
            docs = retrieve_and_rerank_documents(
                question,
                config['qdrant_locks'],
                top_k=top_k,
                date_filter=date_filter,
                langfuse=langfuse,
                question_id=q_idx
            )

            question_result = {
                "question_id": q_idx,
                "question": question,
                "date_filter": f"{date_filter[0][:10]} ~ {date_filter[1][:10]}" if date_filter else None,
                "retriever": f"bge-m3-rrf-reranker-k{top_k}",
                "num_docs": len(docs),
                "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in docs],
                "llms": []
            }

            # 각 LLM으로 병렬 테스트
            print(f"\n🚀 {len(llm_configs)}개 LLM으로 병렬 답변 생성 시작...\n")

            with ThreadPoolExecutor(max_workers=7) as executor:
                future_to_llm = {
                    executor.submit(test_single_llm, llm_config, question, docs, report_type, langfuse, q_idx): (rank, llm_config)
                    for rank, llm_config in enumerate(llm_configs, 1)
                }

                for future in as_completed(future_to_llm):
                    rank, llm_config = future_to_llm[future]
                    try:
                        result = future.result()
                        result["rank"] = rank
                        question_result["llms"].append(result)

                        # 중간 저장
                        temp_results = all_results + [question_result]
                        try:
                            save_combination_results(temp_results, top_k, llm_configs, timestamp, combination_base_dir, questions, report_type)
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

            question_result["llms"].sort(key=lambda x: x["rank"])
            all_results.append(question_result)

            # 질문 완료 시 최종 저장
            print(f"\n💾 질문 {q_idx} 완료 - 최종 결과 저장 중...")
            save_combination_results(all_results, top_k, llm_configs, timestamp, combination_base_dir, questions, report_type)

        # k 값별 summary 저장
        base_output_dir = Path("data/results/reranker_summary") / f"k{top_k}_{timestamp}"

        if output_file:
            output_path = Path(output_file)
        else:
            output_path = base_output_dir / "summary.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_result = {
            "test_date": datetime.now().isoformat(),
            "report_type": report_type,
            "top_k": top_k,
            "num_questions": len(questions),
            "retriever_config": {
                "name": f"bge-m3-rrf-reranker-k{top_k}",
                "display_name": f"BGE-M3 + RRF + Qwen3-Reranker (Top {top_k})",
                "description": f"BGE-M3 embedding with RRF ensemble and Qwen3-Reranker-4B reranking (final k={top_k})",
                "type": "rrf_ensemble_reranker",
                "embedding": "baai-bge-m3",
                "reranker": "Qwen3-Reranker-4B",
                "top_k": top_k
            },
            "llm_configs": llm_configs,
            "results": all_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 100}")
        print(f"✅ Top-{top_k} 테스트 완료!")
        print(f"📁 Summary 저장: {output_path}")
        print(f"📁 조합별 결과: {combination_base_dir}")
        print(f"{'=' * 100}\n")

    print(f"\n🔗 Langfuse에서 상세 추적 정보를 확인하세요")
    langfuse.flush()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="BGE-M3 + RRF + Qwen3-Reranker 보고서 생성 및 비교 (Langfuse 연동)"
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
        choices=["single", "multi", "interactive", "evaluate"],
        default="interactive",
        help="테스트 모드: single(단일 질문), multi(파일에서 여러 질문), interactive(대화형 입력), evaluate(평가 데이터셋 사용)"
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
        "--top-k",
        type=str,
        default="6,8,10",
        help="테스트할 k 값 (콤마로 구분, 기본: 6,8,10). evaluate 모드에서는 단일 값만 사용"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/work/rag/Project/rag-report-generator/data/evaluation/merged_qa_dataset.json",
        help="evaluate 모드에서 사용할 평가 데이터셋 경로 (기본: merged_qa_dataset.json)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="평가 버전 태그 (기본: v1)"
    )

    args = parser.parse_args()

    # k 값 파싱
    top_k_values = [int(k.strip()) for k in args.top_k.split(',')]

    # 날짜 필터 파싱
    date_filter = parse_date_range(
        date_input=args.date_range,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 보고서 타입에 맞는 설정 로드
    config = get_report_config(args.report_type)

    if args.mode == "evaluate":
        # 평가 데이터셋을 사용한 평가 모드
        # evaluate 모드에서는 top_k_values의 첫 번째 값만 사용
        top_k = top_k_values[0]

        print("\n" + "=" * 80)
        print(f"📊 평가 모드: 평가 데이터셋을 사용한 성능 평가")
        print("=" * 80)
        print(f"보고서 타입: {args.report_type}")
        print(f"데이터셋: {args.dataset}")
        print(f"Top-K: {top_k}")
        print(f"버전: {args.version}")
        print()

        run_reranker_evaluation(
            report_type=args.report_type,
            dataset_path=args.dataset,
            top_k=top_k,
            version=args.version,
            output_dir=args.output
        )

    elif args.mode == "interactive":
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
            questions = [q.strip() for q in user_input.split(';') if q.strip()]
            print(f"\n✅ 총 {len(questions)}개의 질문이 입력되었습니다.")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")

        run_reranker_test(questions, args.report_type, top_k_values, args.output, date_filter)

    elif args.mode == "single":
        if not args.question:
            print("❌ --question 인자가 필요합니다.")
            return
        run_reranker_test([args.question], args.report_type, top_k_values, args.output, date_filter)

    else:
        # 여러 질문 모드 (파일에서 로드)
        test_questions_data = load_test_questions(n=args.num_questions)
        questions = [q["question"] for q in test_questions_data]

        run_reranker_test(questions, args.report_type, top_k_values, args.output, date_filter)


if __name__ == "__main__":
    main()
