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
from FlagEmbedding import FlagReranker

from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT, EVALUATION_CONFIG
from utils.langfuse import get_langfuse_client
from utils.dates import parse_date_range, extract_date_filter_from_question
from utils.common import (
    load_prompt,
    load_evaluation_dataset,
    create_trace_and_generation,
    add_retrieval_quality_score,
    save_embedding_cache
)

# 리트리버 임포트
from utils.retriever_factory import create_retriever_from_config


# Reranker 모델 초기화 (전역 변수로 한 번만 로드)
RERANKER_MODEL = None
RERANKER_TYPE = None  # 현재 로드된 reranker 타입

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


def get_optimal_batch_size():
    """GPU 메모리에 따른 최적 배치 크기 계산"""
    import torch
    # VLLM과 공존하기 위해 배치 크기를 16으로 고정
    return 16


def get_reranker(reranker_type: str = "qwen3"):
    """Reranker 모델 로드 (캐싱)

    Args:
        reranker_type: 'qwen3', 'bge', 또는 'korean' 중 선택

    Returns:
        로드된 reranker 모델
    """
    global RERANKER_MODEL, RERANKER_TYPE

    # 이미 로드된 모델과 타입이 같으면 재사용
    if RERANKER_MODEL is not None and RERANKER_TYPE == reranker_type:
        return RERANKER_MODEL

    # 다른 타입의 모델을 로드해야 하면 기존 모델 제거
    if RERANKER_MODEL is not None and RERANKER_TYPE != reranker_type:
        print(f"🔄 기존 {RERANKER_TYPE} 모델 언로드 중...")
        del RERANKER_MODEL
        RERANKER_MODEL = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if reranker_type == "qwen3":
        print(f"🔄 Qwen3-Reranker-4B-seq-cls 모델 로딩 중... (device: {device})")
        RERANKER_MODEL = CrossEncoder(
            "tomaarsen/Qwen3-Reranker-4B-seq-cls",
            max_length=8192,
            device=device,
            trust_remote_code=True
        )
        print("✅ Qwen3-Reranker-4B-seq-cls 모델 로드 완료!")

    elif reranker_type == "bge":
        print(f"🔄 BGE-Reranker-v2-m3 모델 로딩 중... (device: {device})")
        RERANKER_MODEL = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=True,
            device=device
        )
        print("✅ BGE-Reranker-v2-m3 모델 로드 완료!")

    elif reranker_type == "korean":
        print(f"🔄 Korean-Reranker-8k 모델 로딩 중... (device: {device})")
        RERANKER_MODEL = FlagReranker(
            'upskyy/ko-reranker-8k',
            use_fp16=True,
            device=device
        )
        print("✅ Korean-Reranker-8k 모델 로드 완료!")

    else:
        raise ValueError(f"Unknown reranker_type: {reranker_type}. Choose 'qwen3', 'bge', or 'korean'")

    RERANKER_TYPE = reranker_type

    # 최적 배치 크기 출력
    optimal_bs = get_optimal_batch_size()
    print(f"💡 권장 배치 크기: {optimal_bs}")

    return RERANKER_MODEL


def rerank_documents(query: str, docs: list, top_k: int = 6, batch_size: int = None, reranker_type: str = "qwen3") -> list:
    """Reranker 모델로 문서 재순위화

    Args:
        query: 검색 쿼리
        docs: 검색된 문서 리스트 (langchain Document 객체)
        top_k: 최종 반환할 문서 수
        batch_size: 배치 처리 크기 (None이면 자동 계산)
        reranker_type: 'qwen3', 'bge', 또는 'korean' 중 선택

    Returns:
        재순위화된 상위 k개 문서
    """
    import torch
    reranker = get_reranker(reranker_type)

    # 배치 크기 자동 설정
    if batch_size is None:
        batch_size = get_optimal_batch_size()

    # Reranker 타입에 따라 다른 포맷 사용
    if reranker_type == "qwen3":
        # Qwen3-Reranker 포맷
        formatted_query = format_queries(query)
        pairs = [
            [formatted_query, format_document(doc.page_content)]
            for doc in docs
        ]

        # 배치 단위로 재순위화 점수 계산 (메모리 절약)
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = reranker.predict(batch_pairs)
            all_scores.extend(batch_scores)

            # 배치 처리 후 GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elif reranker_type in ["bge", "korean"]:
        # BGE/Korean Reranker 포맷 (FlagReranker)
        pairs = [[query, doc.page_content] for doc in docs]
        all_scores = reranker.compute_score(pairs)

    else:
        raise ValueError(f"Unknown reranker_type: {reranker_type}")

    # 점수와 문서를 함께 정렬
    doc_score_pairs = list(zip(docs, all_scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 상위 k개 문서만 반환
    reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

    reranker_name = {"qwen3": "Qwen3", "bge": "BGE", "korean": "Korean"}[reranker_type]
    print(f"\n🔄 {reranker_name} Reranking 완료: {len(docs)}개 → {len(reranked_docs)}개 (배치 크기: {batch_size})")
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
        'prompt_dir': f'{report_type}_report'
    }


def _load_report_prompt(prompt_file: str, report_type: str) -> str:
    """보고서용 프롬프트 파일 로드 (헬퍼 함수)"""
    prompt_path = f"prompts/templates/service/{report_type}_report/{prompt_file}"
    return load_prompt(prompt_path)


def load_test_questions(n: int = 5) -> List[Dict[str, Any]]:
    """테스트용 질문 로드 (레거시, interactive 모드용)"""
    qa_file = Path(__file__).parent.parent / "data" / "evaluation" / "llm_generated_qa_v2.json"

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    return qa_data[:n]


def count_tokens(text: str, model_id: str = "gpt-4") -> int:
    """텍스트의 토큰 수를 계산

    Args:
        text: 토큰을 계산할 텍스트
        model_id: 모델 ID (기본: gpt-4)

    Returns:
        토큰 수
    """
    try:
        import tiktoken

        # 모델에 따른 인코딩 선택
        model_lower = model_id.lower()

        if "claude" in model_lower or "anthropic" in model_lower:
            # Claude는 GPT-4와 유사한 토큰화 사용
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "gpt-4" in model_lower or "gpt-3.5" in model_lower or "gpt-35" in model_lower:
            # GPT-4, GPT-3.5 계열
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "deepseek" in model_lower:
            # DeepSeek는 cl100k_base 사용 (GPT-4와 호환)
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "qwen" in model_lower or "qwq" in model_lower:
            # Qwen 계열도 cl100k_base 사용
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "llama" in model_lower or "phi" in model_lower or "mistral" in model_lower:
            # Llama, Phi, Mistral 등 오픈소스 모델들
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # 기본값으로 cl100k_base 사용
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception as e:
        # tiktoken 실패 시 대략적인 계산 (1 토큰 ≈ 4 글자)
        print(f"⚠️ tiktoken 계산 실패, 근사값 사용: {e}")
        return len(text) // 4


def calculate_message_tokens(messages: List[Dict[str, str]], model_id: str = "gpt-4") -> int:
    """메시지 리스트의 총 토큰 수를 계산

    Args:
        messages: 메시지 리스트 [{"role": "system", "content": "..."}, ...]
        model_id: 모델 ID

    Returns:
        총 입력 토큰 수
    """
    total_tokens = 0

    # 메시지 구조 오버헤드 (role, formatting 등)
    # OpenAI 기준: 메시지당 약 3~4 토큰 오버헤드
    message_overhead = 3

    for message in messages:
        content = message.get("content", "")
        role = message.get("role", "")

        # role 토큰 계산
        total_tokens += count_tokens(role, model_id)
        # content 토큰 계산
        total_tokens += count_tokens(content, model_id)
        # 메시지 포맷 오버헤드
        total_tokens += message_overhead

    # 전체 대화 구조 오버헤드
    total_tokens += 3

    return total_tokens


def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> Dict[str, float]:
    """토큰 수를 기반으로 비용 계산

    Args:
        input_tokens: 입력 토큰 수
        output_tokens: 출력 토큰 수
        model_id: 모델 ID

    Returns:
        비용 정보 딕셔너리 {"input_cost": 0.001, "output_cost": 0.002, "total_cost": 0.003, "currency": "USD"}
    """
    # 모델별 가격 (1M 토큰당 USD)
    # 참고: 2024-2025년 기준 가격, 실제 가격은 공식 문서 확인 필요
    PRICING = {
        # OpenAI GPT-4
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},

        # Anthropic Claude
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3.5-haiku": {"input": 0.8, "output": 4.0},

        # DeepSeek (매우 저렴)
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-v3": {"input": 0.27, "output": 1.1},
        "deepseek-v3.1": {"input": 0.27, "output": 1.1},

        # Qwen
        "qwen-max": {"input": 0.4, "output": 1.2},
        "qwen-plus": {"input": 0.08, "output": 0.24},
        "qwen-turbo": {"input": 0.03, "output": 0.06},

        # Meta Llama
        "llama-3-70b": {"input": 0.9, "output": 0.9},
        "llama-3-8b": {"input": 0.2, "output": 0.2},

        # Microsoft Phi
        "phi-3-mini": {"input": 0.1, "output": 0.1},
        "phi-3-medium": {"input": 0.2, "output": 0.2},

        # Mistral
        "mistral-large": {"input": 3.0, "output": 9.0},
        "mistral-medium": {"input": 2.7, "output": 8.1},
        "mistral-small": {"input": 0.2, "output": 0.6},
    }

    model_lower = model_id.lower()

    # 모델 매칭
    pricing = None
    for model_key, price in PRICING.items():
        if model_key in model_lower:
            pricing = price
            break

    # 매칭되지 않으면 기본값 (GPT-4o 가격)
    if pricing is None:
        pricing = {"input": 2.5, "output": 10.0}
        print(f"⚠️ 모델 '{model_id}'의 가격 정보를 찾을 수 없어 기본값(GPT-4o) 사용")

    # 비용 계산 (1M 토큰당 가격 → 실제 토큰 수로 환산)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "currency": "USD"
    }


def generate_answer_with_llm(query: str, docs: list, llm_config: dict, report_type: str, langfuse=None, question_id: int = None, version: str = "v1") -> str:
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
    system_prompt = _load_report_prompt("system_prompt.txt", report_type)
    answer_generation_template = _load_report_prompt("answer_generation_prompt.txt", report_type)

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

                # 토큰 직접 계산 (추가)
                calculated_input_tokens = calculate_message_tokens(messages, llm_config['model_id'])
                calculated_output_tokens = count_tokens(answer, llm_config['model_id'])
                calculated_total_tokens = calculated_input_tokens + calculated_output_tokens

                # 비용 계산
                cost_info = calculate_cost(calculated_input_tokens, calculated_output_tokens, llm_config['model_id'])

                generation.update(
                    output={"answer": answer},
                    usage=usage_dict if usage_dict else None
                )

                langfuse.update_current_trace(
                    name=f"llm_comparison_{llm_config['name']}_q{question_id}",
                    tags=[llm_config['name'], "reranker_test", f"{report_type}_report", version],
                    input={"question": query},
                    output={"answer": answer},
                    metadata={
                        "llm_name": llm_config['name'],
                        "llm_display_name": llm_config['display_name'],
                        "llm_description": llm_config['description'],
                        "model_id": llm_config['model_id'],
                        "question_id": question_id,
                        "num_docs": len(docs),
                        "report_type": report_type,
                        "calculated_tokens": {
                            "input": calculated_input_tokens,
                            "output": calculated_output_tokens,
                            "total": calculated_total_tokens
                        },
                        "estimated_cost": cost_info
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

                # 토큰 직접 계산 (추가)
                calculated_input_tokens = calculate_message_tokens(messages, llm_config['model_id'])
                calculated_output_tokens = count_tokens(answer, llm_config['model_id'])
                calculated_total_tokens = calculated_input_tokens + calculated_output_tokens

                # 비용 계산
                cost_info = calculate_cost(calculated_input_tokens, calculated_output_tokens, llm_config['model_id'])

                generation.update(
                    output={"answer": answer},
                    usage=usage_dict if usage_dict else None
                )

                langfuse.update_current_trace(
                    name=f"llm_comparison_{llm_config['name']}_q{question_id}",
                    tags=[llm_config['name'], "reranker_test", f"{report_type}_report", version],
                    input={"question": query},
                    output={"answer": answer},
                    metadata={
                        "llm_name": llm_config['name'],
                        "llm_display_name": llm_config['display_name'],
                        "llm_description": llm_config['description'],
                        "model_id": llm_config['model_id'],
                        "question_id": question_id,
                        "num_docs": len(docs),
                        "report_type": report_type,
                        "calculated_tokens": {
                            "input": calculated_input_tokens,
                            "output": calculated_output_tokens,
                            "total": calculated_total_tokens
                        },
                        "estimated_cost": cost_info
                    }
                )
        else:
            response = model.invoke(messages)
            answer = response.content

    return answer


def retrieve_and_rerank_documents(
    question: str,
    report_type: str,
    top_k: int = 6,
    date_filter: tuple = None,
    langfuse=None,
    question_id: int = None,
    batch_size: int = None,
    retriever_type: str = "bge-m3",
    reranker_type: str = "qwen3"
):
    """RRF Ensemble로 검색 후 Reranker로 재순위화

    Args:
        question: 질문
        report_type: 보고서 타입 ('executive' 또는 'weekly')
        top_k: 최종 반환할 문서 수
        date_filter: 날짜 필터
        langfuse: Langfuse 클라이언트
        question_id: 질문 ID
        batch_size: Reranker 배치 크기 (None이면 자동)
        retriever_type: 리트리버 타입 ('bge-m3', 'openai', 'openai-rrf-multiquery' 등)
        reranker_type: Reranker 타입 ('qwen3', 'bge', 'korean')

    Returns:
        재순위화된 문서 리스트
    """
    import gc
    import torch

    # 배치 크기 자동 설정
    if batch_size is None:
        batch_size = get_optimal_batch_size()

    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Retriever 타입에 따른 설정
    if retriever_type == "openai-rrf-multiquery":
        retriever_config = {
            "name": "openai-rrf-multiquery",
            "display_name": "OpenAI + RRF MultiQuery",
            "description": "OpenAI embedding with RRF multiquery",
            "embedding_preset": "openai",
            "retriever_type": "rrf_multiquery",
            "bm25_weight": 0.5,
            "dense_weight": 0.5,
            "use_mmr": True,
            "lambda_mult": 0.5
        }
    elif retriever_type == "openai-large-rrf-multiquery":
        retriever_config = {
            "name": "openai-large-rrf-multiquery",
            "display_name": "OpenAI Large + RRF MultiQuery",
            "description": "OpenAI Large embedding with RRF multiquery",
            "embedding_preset": "openai-large",
            "retriever_type": "rrf_multiquery",
            "bm25_weight": 0.5,
            "dense_weight": 0.5,
            "use_mmr": True,
            "lambda_mult": 0.5
        }
    else:  # 기본값: bge-m3
        retriever_config = {
            "name": "bge-m3-rrf-ensemble",
            "display_name": "BGE-M3 + RRF Ensemble",
            "description": "BGE-M3 embedding with RRF ensemble",
            "embedding_preset": "bge-m3",
            "retriever_type": "rrf_ensemble",
            "bm25_weight": 0.5,
            "dense_weight": 0.5,
            "use_mmr": True,
            "lambda_mult": 0.5
        }

    # 재순위화를 위해 더 많은 문서 검색
    initial_k = max(20, top_k * 3)

    # Retriever 생성 (OpenRouter API 사용)
    retriever, _ = create_retriever_from_config(retriever_config, initial_k)

    print(f"🔍 {retriever_config['display_name']} 검색 중 (초기 k={initial_k})...")

    # Reranker 모델명 설정
    reranker_model_names = {
        "qwen3": "Qwen3-Reranker-4B",
        "bge": "BGE-Reranker-v2-m3",
        "korean": "Korean-Reranker-8k"
    }
    reranker_model_name = reranker_model_names.get(reranker_type, "Unknown")

    # Langfuse로 검색 과정 기록
    if langfuse and question_id:
        with langfuse.start_as_current_observation(
            as_type='span',
            name=f"retrieval_{retriever_config['name']}_q{question_id}",
            input={"question": question},
            metadata={
                "retriever_name": retriever_config['name'],
                "embedding_preset": retriever_config['embedding_preset'],
                "retrieval_strategy": retriever_config['retriever_type'],
                "initial_k": initial_k,
                "final_k": top_k,
                "reranker": reranker_model_name,
                "reranker_type": reranker_type,
                "batch_size": batch_size
            }
        ) as span:
            # 문서 검색
            initial_docs = retriever.invoke(question)
            print(f"📄 초기 검색 문서 수: {len(initial_docs)}")

            # Reranking
            reranked_docs = rerank_documents(question, initial_docs, top_k=top_k, batch_size=batch_size, reranker_type=reranker_type)

            span.update(output={
                "initial_num_docs": len(initial_docs),
                "final_num_docs": len(reranked_docs),
                "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in reranked_docs]
            })

            # 초기 문서 메모리 해제
            del initial_docs
    else:
        initial_docs = retriever.invoke(question)
        print(f"📄 초기 검색 문서 수: {len(initial_docs)}")
        reranked_docs = rerank_documents(question, initial_docs, top_k=top_k, batch_size=batch_size, reranker_type=reranker_type)

        # 초기 문서 메모리 해제
        del initial_docs

    print(f"\n📄 최종 문서 수: {len(reranked_docs)}")
    print("\n최종 문서 제목:")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"  {i}. {doc.metadata.get('page_title', 'Unknown')}")

    # Retriever 메모리 해제
    del retriever

    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    system_prompt: str,
    answer_generation_prompt: str,
    langfuse,
    idx: int,
    version_tag: str = "v1",
    batch_size: int = 16,
    retriever_type: str = "bge-m3",
    reranker_type: str = "qwen3"
) -> Dict[str, Any]:
    """Reranker를 사용한 단일 쿼리 평가 (evaluate_report_types.py 스타일)

    Args:
        question: 질문
        ground_truth: 정답
        context_page_id: 정답 문서의 page_id
        item_metadata: 질문 메타데이터
        report_type: 보고서 타입
        top_k: Top-K 값
        system_prompt: 시스템 프롬프트
        answer_generation_prompt: 답변 생성 프롬프트
        langfuse: Langfuse 클라이언트
        idx: 질문 인덱스
        version_tag: 버전 태그
        batch_size: Reranker 배치 크기
        retriever_type: 리트리버 타입
        reranker_type: Reranker 타입 ('qwen3', 'bge', 'korean')

    Returns:
        평가 결과 딕셔너리
    """
    import time

    start_time = time.time()

    # 문서 검색 및 재순위화
    docs = retrieve_and_rerank_documents(
        question=question,
        report_type=report_type,
        top_k=top_k,
        date_filter=None,  # 평가 데이터셋에는 날짜 필터 사용하지 않음
        langfuse=langfuse,
        question_id=idx,
        batch_size=batch_size,
        retriever_type=retriever_type,
        reranker_type=reranker_type
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
    from utils.common import generate_llm_answer
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

    # Reranker 모델명 설정
    reranker_model_names = {
        "qwen3": "Qwen3-Reranker-4B",
        "bge": "BGE-Reranker-v2-m3",
        "korean": "Korean-Reranker-8k"
    }
    reranker_model_name = reranker_model_names.get(reranker_type, "Unknown")
    reranker_tag = f"{reranker_type}-reranker"

    # Langfuse Trace & Generation 생성
    # retriever_type에 따라 메타데이터 동적 생성
    if retriever_type == "openai-rrf-multiquery":
        base_name = "openai"
        embedding_preset = "openai"
        retrieval_strategy = "rrf_multiquery"
        display_name = f"OpenAI + RRF MultiQuery + {reranker_model_name} (Top {top_k})"
        retriever_tags = ["openai", "rrf_multiquery", reranker_tag, f"top_k_{top_k}", report_type]
    elif retriever_type == "openai-large-rrf-multiquery":
        base_name = "openai-large"
        embedding_preset = "openai-large"
        retrieval_strategy = "rrf_multiquery"
        display_name = f"OpenAI Large + RRF MultiQuery + {reranker_model_name} (Top {top_k})"
        retriever_tags = ["openai-large", "rrf_multiquery", reranker_tag, f"top_k_{top_k}", report_type]
    else:  # bge-m3
        base_name = "bge-m3"
        embedding_preset = "bge-m3"
        retrieval_strategy = "rrf_ensemble"
        display_name = f"BGE-M3 + RRF + {reranker_model_name} (Top {top_k})"
        retriever_tags = ["bge-m3", "rrf_ensemble", reranker_tag, f"top_k_{top_k}", report_type]

    retriever_name = f"{base_name}-rrf-{reranker_type}-reranker-k{top_k}"

    additional_metadata = {
        "context_page_id": context_page_id,
        "retriever_name": retriever_name,
        "display_name": display_name,
        "report_type": report_type,
        "top_k": top_k,
        "embedding_preset": embedding_preset,
        "retriever_type": retrieval_strategy,
        "reranker_model": reranker_model_name,
        "reranker_type": reranker_type
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
    output_dir: str = None,
    batch_size: int = None,
    max_workers: int = 3,
    retriever_type: str = "bge-m3",
    reranker_type: str = "qwen3"
) -> Dict[str, Any]:
    """Reranker를 사용한 평가 실행 (evaluate_report_types.py 스타일)

    Args:
        report_type: 보고서 타입 ('executive' 또는 'weekly')
        dataset_path: 평가 데이터셋 경로
        top_k: Top-K 값
        version: 버전 태그
        output_dir: 결과 저장 디렉토리
        batch_size: Reranker 배치 크기 (None이면 자동)
        max_workers: 병렬 처리 워커 수 (기본: 3)
        retriever_type: 리트리버 타입 (기본: 'bge-m3')
        reranker_type: Reranker 타입 ('qwen3', 'bge', 'korean')

    Returns:
        평가 결과 딕셔너리
    """
    import yaml

    # 설정 로드
    config = get_report_config(report_type)

    # 프롬프트 로드
    report_type_suffix = "weekly_report" if report_type == "weekly" else "executive_report"
    system_prompt = load_prompt(f"prompts/templates/service/{report_type_suffix}/system_prompt.txt")
    answer_generation_prompt = load_prompt(f"prompts/templates/service/{report_type_suffix}/answer_generation_prompt.txt")

    # 평가 데이터셋 로드
    eval_data = load_evaluation_dataset(dataset_path)

    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()
    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        return {}

    report_type_display = "최종보고서" if report_type == "executive" else "주간보고서"

    # 배치 크기 자동 설정
    if batch_size is None:
        batch_size = get_optimal_batch_size()

    # Reranker 모델명 설정
    reranker_model_names = {
        "qwen3": "Qwen3-Reranker",
        "bge": "BGE-Reranker",
        "korean": "Korean-Reranker"
    }
    reranker_display = reranker_model_names.get(reranker_type, "Unknown")

    # Retriever 타입에 따른 디스플레이 이름 생성
    if retriever_type == "openai-rrf-multiquery":
        retriever_display = f"OpenAI + RRF MultiQuery + {reranker_display}"
    elif retriever_type == "openai-large-rrf-multiquery":
        retriever_display = f"OpenAI Large + RRF MultiQuery + {reranker_display}"
    else:
        retriever_display = f"BGE-M3 + RRF + {reranker_display}"

    print("\n" + "=" * 80)
    print(f"🔍 {report_type_display} {retriever_display} 평가 (Langfuse 연동)")
    print("=" * 80)
    print(f"📋 데이터셋: {len(eval_data)} 개 샘플")
    print(f"🔧 Retriever: {retriever_type}")
    print(f"🤖 Reranker: {reranker_type} ({reranker_model_names.get(reranker_type, 'Unknown')})")
    print(f"📊 Top-K: {top_k}")
    print(f"🏷️  Version: {version}")
    print(f"🔢 Batch Size: {batch_size} (자동 최적화)")
    print(f"⚡ 병렬 워커: {max_workers}")
    print()

    stats = {
        "total_queries": len(eval_data),
        "total_time": 0,
        "evaluations": []
    }

    # 병렬 처리로 평가 실행
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(
                evaluate_single_query_with_reranker,
                item["question"],
                item["ground_truth"],
                item.get("context_page_id"),
                item.get("metadata", {}),
                report_type,
                top_k,
                system_prompt,
                answer_generation_prompt,
                langfuse,
                idx,
                version,
                batch_size,
                retriever_type,
                reranker_type
            ): (idx, item)
            for idx, item in enumerate(eval_data, 1)
        }

        for future in as_completed(future_to_item):
            idx, item = future_to_item[future]
            try:
                eval_result = future.result()
                stats["evaluations"].append(eval_result)
                stats["total_time"] += eval_result["time"]

                # 캐시 저장 (주기적으로)
                if idx % 5 == 0:
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

    # retriever_type에 따른 파일명 및 설정 생성
    if retriever_type == "openai-rrf-multiquery":
        file_prefix = "openai-rrf-reranker"
        config_name = f"openai-rrf-reranker-k{top_k}"
        config_display = f"OpenAI + RRF MultiQuery + Qwen3-Reranker (Top {top_k})"
        config_embedding = "openai"
        config_retrieval = "rrf_multiquery_reranker"
    elif retriever_type == "openai-large-rrf-multiquery":
        file_prefix = "openai-large-rrf-reranker"
        config_name = f"openai-large-rrf-reranker-k{top_k}"
        config_display = f"OpenAI Large + RRF MultiQuery + Qwen3-Reranker (Top {top_k})"
        config_embedding = "openai-large"
        config_retrieval = "rrf_multiquery_reranker"
    else:
        file_prefix = "bge-m3-rrf-reranker"
        config_name = f"bge-m3-rrf-reranker-k{top_k}"
        config_display = f"BGE-M3 + RRF + Qwen3-Reranker (Top {top_k})"
        config_embedding = "bge-m3"
        config_retrieval = "rrf_ensemble_reranker"

    output_file = output_path / f"{file_prefix}_k{top_k}_stats.json"
    save_result = {k: v for k, v in stats.items() if k != "evaluations"}
    save_result["num_evaluations"] = len(stats.get("evaluations", []))
    save_result["config"] = {
        "retriever_name": config_name,
        "display_name": config_display,
        "report_type": report_type,
        "embedding_preset": config_embedding,
        "retriever_type": config_retrieval,
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
    global_date_filter: tuple = None,
    retriever_type: str = "bge-m3",
    reranker_type: str = "qwen3"
):
    """Reranker 테스트 (여러 k 값)

    Args:
        questions: 질문 리스트
        report_type: 보고서 타입 ('executive' 또는 'weekly')
        top_k_values: 테스트할 k 값 리스트 (기본: [6, 8, 10])
        output_file: 결과 저장 경로
        global_date_filter: 전역 날짜 필터
        retriever_type: 리트리버 타입 (기본: 'bge-m3')
        reranker_type: Reranker 타입 ('qwen3', 'bge', 'korean')
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
                question=question,
                report_type=report_type,
                top_k=top_k,
                date_filter=date_filter,
                langfuse=langfuse,
                question_id=q_idx,
                retriever_type=retriever_type,
                reranker_type=reranker_type
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Reranker 배치 크기 (기본: None=자동, 메모리 부족 시 줄이기)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="병렬 처리 워커 수 (기본: 3, 메모리 많으면 늘리기)"
    )
    parser.add_argument(
        "--retriever-type",
        type=str,
        default="bge-m3",
        choices=["bge-m3", "openai-rrf-multiquery", "openai-large-rrf-multiquery"],
        help="리트리버 타입 (기본: bge-m3, 선택: openai-rrf-multiquery, openai-large-rrf-multiquery)"
    )
    parser.add_argument(
        "--reranker-type",
        type=str,
        default="qwen3",
        choices=["qwen3", "bge", "korean"],
        help="Reranker 타입 (기본: qwen3, 선택: bge=BGE-Reranker-v2-m3, korean=Korean-Reranker-8k)"
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
            output_dir=args.output,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            retriever_type=args.retriever_type,
            reranker_type=args.reranker_type
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

        run_reranker_test(questions, args.report_type, top_k_values, args.output, date_filter, args.retriever_type, args.reranker_type)

    elif args.mode == "single":
        if not args.question:
            print("❌ --question 인자가 필요합니다.")
            return
        run_reranker_test([args.question], args.report_type, top_k_values, args.output, date_filter, args.retriever_type, args.reranker_type)

    else:
        # 여러 질문 모드 (파일에서 로드)
        test_questions_data = load_test_questions(n=args.num_questions)
        questions = [q["question"] for q in test_questions_data]

        run_reranker_test(questions, args.report_type, top_k_values, args.output, date_filter, args.retriever_type, args.reranker_type)


if __name__ == "__main__":
    main()
