#!/usr/bin/env python3
"""평가 스크립트 공통 유틸리티 함수"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.chat_models import init_chat_model

from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT


def load_prompt(prompt_file: str) -> str:
    """프롬프트 템플릿 로드

    Args:
        prompt_file: 프롬프트 파일 상대 경로 (예: "prompts/templates/evaluation/system_prompt.txt")

    Returns:
        프롬프트 텍스트
    """
    prompt_path = Path(__file__).parent.parent / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_evaluation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """평가용 데이터셋 로드

    Args:
        file_path: 데이터셋 파일 경로

    Returns:
        평가 데이터 리스트
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_llm_answer(
    question: str,
    contexts: List[str],
    system_prompt: str,
    answer_generation_prompt: str,
    num_contexts: int = 5,
    temperature: float = 0.1,
    max_tokens: int = 1000
) -> str:
    """LLM API를 호출하여 답변 생성

    Args:
        question: 질문
        contexts: 검색된 문서 컨텍스트 리스트
        system_prompt: 시스템 프롬프트
        answer_generation_prompt: 답변 생성 프롬프트 템플릿
        num_contexts: 사용할 컨텍스트 개수
        temperature: 생성 온도
        max_tokens: 최대 토큰 수

    Returns:
        생성된 답변
    """
    if not AZURE_AI_CREDENTIAL or not AZURE_AI_ENDPOINT:
        return "Azure OpenAI 설정이 올바르지 않습니다. .env 파일을 확인하세요."

    os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
    os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

    context_text = "\n\n".join(contexts[:num_contexts]) if contexts else "관련 문서를 찾을 수 없습니다."
    prompt = answer_generation_prompt.replace("{context}", context_text).replace("{question}", question)

    try:
        model = init_chat_model(
            "azure_ai:gpt-4.1",
            temperature=temperature,
            max_completion_tokens=max_tokens
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = model.invoke(messages)
        return response.content
    except Exception as e:
        error_msg = f"답변 생성 실패: {str(e)}"
        print(f"  ⚠️ LLM API 호출 실패: {e}")
        return error_msg


def create_trace_and_generation(
    langfuse,
    retriever_name: str,
    question: str,
    contexts: List[str],
    answer: str,
    ground_truth: str,
    context_metadata: List[Dict],
    item_metadata: Dict,
    total_time: float,
    idx: int,
    version_tag: str = "v1",
    retriever_tags: List[str] = None,
    additional_metadata: Dict = None
) -> str:
    """Langfuse Trace와 Generation 생성

    Args:
        langfuse: Langfuse 클라이언트
        retriever_name: 리트리버 이름
        question: 질문
        contexts: 검색된 컨텍스트
        answer: 생성된 답변
        ground_truth: 정답
        context_metadata: 컨텍스트 메타데이터
        item_metadata: 질문 메타데이터
        total_time: 전체 소요 시간 (초)
        idx: 질문 인덱스
        version_tag: 버전 태그
        retriever_tags: 리트리버 태그 리스트
        additional_metadata: 추가 메타데이터

    Returns:
        Trace ID
    """
    context_text = "\n\n---\n\n".join(contexts) if contexts else ""

    if retriever_tags is None:
        retriever_tags = []

    all_tags = [
        f"{retriever_name}_{version_tag}",
        version_tag,
        "evaluation"
    ] + retriever_tags

    with langfuse.start_as_current_observation(
        as_type='generation',
        name=f"generation_{retriever_name}_{version_tag}",
        model="gpt-4.1",
        input={
            "question": question,
            "context": context_text
        },
        output={
            "answer": answer
        },
        metadata={
            "ground_truth": ground_truth,
            "contexts": contexts,
            "context_metadata": context_metadata,
            "retriever_type": retriever_name,
            "version": version_tag,
            "retriever_tags": retriever_tags,
            **(additional_metadata or {})
        }
    ) as generation:
        trace_id = generation.trace_id

        langfuse.update_current_trace(
            name=f"eval_{retriever_name}_{version_tag}_q{idx}",
            tags=all_tags,
            input={
                "question": question,
                "context": context_text
            },
            output={
                "answer": answer
            },
            metadata={
                "retriever": retriever_name,
                "version": version_tag,
                "total_time_ms": total_time * 1000,
                "num_retrieved_contexts": len(contexts),
                "question_id": idx,
                "category": item_metadata.get("category", "unknown"),
                "difficulty": item_metadata.get("difficulty", "unknown"),
                "retriever_components": retriever_tags,
                **(additional_metadata or {})
            }
        )

    return trace_id


def add_retrieval_quality_score(langfuse, trace_id: str, context_metadata: List[Dict]):
    """검색 품질 스코어 추가

    Args:
        langfuse: Langfuse 클라이언트
        trace_id: Trace ID
        context_metadata: 컨텍스트 메타데이터 리스트
    """
    if not context_metadata:
        return

    scores = [m.get("score") for m in context_metadata if m.get("score") is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        langfuse.create_score(
            trace_id=trace_id,
            name="retrieval_quality",
            value=avg_score,
            comment=f"Average retrieval score from {len(scores)} contexts"
        )


def save_embedding_cache():
    """임베딩 캐시 저장"""
    try:
        from models.embeddings.factory import get_embedder
        embedder = get_embedder()
        if hasattr(embedder, 'use_cache') and embedder.use_cache and embedder.cache:
            embedder.cache.save()
    except:
        pass  # 캐시 저장 실패는 무시
