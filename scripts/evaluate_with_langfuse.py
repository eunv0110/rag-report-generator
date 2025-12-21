#!/usr/bin/env python3
"""Langfuse 자동 평가(RAGAS) 기반 RAG 성능 평가"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from langchain.chat_models import init_chat_model
from langfuse.types import TraceContext

from config.settings import (
    QDRANT_PATH,
    QDRANT_COLLECTION,
    AZURE_AI_CREDENTIAL,
    AZURE_AI_ENDPOINT
)
from retrievers import RetrieverFactory, BaseRetriever
from utils.langfuse_utils import get_langfuse_client

# 상수 정의
DEFAULT_DATASET_PATH = "/home/work/rag/Project/rag-report-generator/data/evaluation/llm_generated_qa_azure.json"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 500
SYSTEM_PROMPT = "당신은 주어진 문서를 바탕으로 정확하게 답변하는 AI 어시스턴트입니다."


def load_evaluation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """평가용 데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_llm_answer(question: str, contexts: List[str]) -> str:
    """
    LLM API를 호출하여 답변 생성

    Args:
        question: 사용자 질문
        contexts: 검색된 문서 리스트

    Returns:
        생성된 답변 또는 에러 메시지
    """
    if not AZURE_AI_CREDENTIAL or not AZURE_AI_ENDPOINT:
        return "Azure OpenAI 설정이 올바르지 않습니다. .env 파일을 확인하세요."

    # 환경변수 설정
    os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
    os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

    context_text = "\n\n".join(contexts[:3]) if contexts else "관련 문서를 찾을 수 없습니다."

    prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

문서:
{context_text}

질문: {question}

답변:"""

    try:
        # langchain의 init_chat_model 사용
        model = init_chat_model(
            "azure_ai:gpt-5.1",
            temperature=DEFAULT_TEMPERATURE,
            max_completion_tokens=DEFAULT_MAX_TOKENS
        )

        response = model.invoke(prompt)
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
    context_page_id: Optional[str] = None
) -> str:
    """Langfuse Trace와 Generation 생성
    
    Returns:
        trace_id
    """
    
    # ✅ contexts 리스트를 하나의 문자열로 변환
    context_text = "\n\n---\n\n".join(contexts) if contexts else ""
    
    # 1. 현재 컨텍스트로 Generation 시작 (최소한의 정보만)
    with langfuse.start_as_current_observation(
        as_type='generation',
        name=f"{retriever_name}_generation",
        model="gpt-5.1"
    ) as generation:
        
        # 2. Generation에서 trace_id 추출
        trace_id = generation.trace_id
        
        # ✅ 3. Generation 명시적 업데이트
        langfuse.update_current_generation(
            input={
                "question": question,
                "context": context_text  # ✅ 반드시 여기서 추가!
            },
            output={
                "answer": answer
            },
            metadata={
                "ground_truth": ground_truth,
                "contexts": contexts,
                "context_metadata": context_metadata
            }
        )
        
        # 4. 현재 trace 업데이트
        langfuse.update_current_trace(
            name=f"{retriever_name}_evaluation_{idx}",
            input={
                "question": question
            },
            output={
                "answer": answer
            },
            metadata={
                "retriever": retriever_name,
                "total_time_ms": total_time * 1000,
                "num_retrieved_contexts": len(contexts),
                "context_page_id": context_page_id,
                "question_id": idx,
                "category": item_metadata.get("category", "unknown"),
                "difficulty": item_metadata.get("difficulty", "unknown")
            }
        )
    
    # 디버깅
    print(f"\n[DEBUG] Trace {idx}:")
    print(f"  - ID: {trace_id}")
    print(f"  - Question: {question[:50]}...")
    print(f"  - Context length: {len(context_text)} chars")
    print(f"  - Answer length: {len(answer)} chars")
    
    return trace_id


def add_retrieval_quality_score(
    langfuse,
    trace_id: str,
    context_metadata: List[Dict]
):
    """검색 품질 스코어 추가"""
    if not context_metadata:
        return
    
    avg_score = sum(m["score"] for m in context_metadata) / len(context_metadata)
    langfuse.create_score(
        trace_id=trace_id,
        name="retrieval_quality",
        value=avg_score,
        comment=f"Average retrieval score from {len(context_metadata)} contexts"
    )


def evaluate_single_query(
    retriever: BaseRetriever,
    item: Dict[str, Any],
    langfuse,
    idx: int,
    top_k: int
) -> Dict[str, Any]:
    """단일 쿼리 평가"""
    question = item["question"]
    ground_truth = item["ground_truth"]
    context_page_id = item.get("context_page_id")
    item_metadata = item.get("metadata", {})
    
    start_time = time.time()
    
    # 1. 검색 수행
    search_results = retriever.search(question, top_k=top_k)
    contexts = [result.combined_text for result in search_results]
    
    # ✅ contexts가 비어있으면 경고
    if not contexts:
        print(f"  ⚠️ [{idx}] No contexts found for question!")
        contexts = ["검색 결과가 없습니다."]  # RAGAS를 위한 더미 컨텍스트
    
    context_metadata = [
        {
            "score": result.score,
            "page_title": result.page_title,
            "section_title": result.section_title,
            "chunk_id": result.chunk_id
        }
        for result in search_results
    ]
    
    # 2. LLM 답변 생성
    answer = generate_llm_answer(question, contexts)
    
    # ✅ answer가 비어있거나 에러 메시지면 경고
    if not answer or answer.startswith("답변 생성 실패") or answer.startswith("Azure OpenAI 설정"):
        print(f"  ⚠️ [{idx}] LLM answer generation failed!")
        if not answer:
            answer = "답변을 생성할 수 없습니다."
    
    total_time = time.time() - start_time
    
    # 3. Langfuse Trace & Generation
    trace_id = create_trace_and_generation(
        langfuse=langfuse,
        retriever_name=retriever.name,
        question=question,
        contexts=contexts,
        answer=answer,
        ground_truth=ground_truth,
        context_metadata=context_metadata,
        item_metadata=item_metadata,
        total_time=total_time,
        idx=idx,
        context_page_id=context_page_id
    )
    
    # 4. 검색 품질 스코어 추가
    add_retrieval_quality_score(langfuse, trace_id, context_metadata)
    
    # 진행 상황 출력
    print(f"  [{idx}] {question[:50]}... ({len(contexts)}개 문서, {total_time*1000:.0f}ms)")
    
    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "num_contexts": len(contexts),
        "time": total_time,
        "trace_id": trace_id
    }


def evaluate_rag_with_langfuse(
    retriever: BaseRetriever,
    eval_data: List[Dict[str, Any]],
    langfuse,
    qdrant_client: QdrantClient,
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    """Langfuse 자동 평가(RAGAS)로 RAG 전체 시스템 평가"""
    
    print(f"\n{'=' * 60}")
    print(f"🔍 {retriever.name} 평가 중...")
    print(f"{'=' * 60}")
    
    stats = {
        "total_queries": len(eval_data),
        "total_time": 0,
        "evaluations": []
    }
    
    for idx, item in enumerate(eval_data, 1):
        eval_result = evaluate_single_query(
            retriever=retriever,
            item=item,
            langfuse=langfuse,
            idx=idx,
            top_k=top_k
        )
        
        stats["evaluations"].append(eval_result)
        stats["total_time"] += eval_result["time"]
    
    stats["avg_time"] = stats["total_time"] / stats["total_queries"]
    stats["avg_contexts"] = sum(e["num_contexts"] for e in stats["evaluations"]) / stats["total_queries"]
    
    return stats


def initialize_retrievers(
    qdrant_client: QdrantClient,
    retriever_types: Optional[List[str]] = None
) -> List[BaseRetriever]:
    """리트리버 초기화"""
    if retriever_types is None:
        return RetrieverFactory.get_all_default_retrievers(qdrant_client)
    
    return [
        RetrieverFactory.create(ret_type, qdrant_client)
        for ret_type in retriever_types
    ]


def print_comparison_results(results: List[Dict[str, Any]]):
    """비교 결과 출력"""
    print("\n" + "=" * 60)
    print("📈 성능 비교 결과")
    print("=" * 60)
    
    print(f"\n{'Retriever':<20} {'Avg Contexts':<15} {'Avg Time (ms)':<15}")
    print("-" * 50)
    
    for result in results:
        print(
            f"{result['retriever']:<20} "
            f"{result['avg_contexts']:<15.2f} "
            f"{result['avg_time']*1000:<15.2f}"
        )


def print_next_steps():
    """다음 단계 안내 출력"""
    print("\n" + "=" * 60)
    print("✅ 평가 완료!")
    print("=" * 60)
    print(f"\n💡 다음 단계:")
    print(f"   1. 🌐 Langfuse 대시보드: https://cloud.langfuse.com")
    print(f"   2. 📊 Traces 탭에서 생성된 trace 확인")
    print(f"   3. 🔧 Settings → Evaluations → Context Recall 설정")
    print(f"   4. ⚙️  Evaluations 탭에서 자동 평가 결과 확인")
    print(f"\n⚠️  주의사항:")
    print(f"   • generate_llm_answer() 함수를 실제 LLM API 호출로 교체하세요")
    print(f"   • context_page_id가 실제 Qdrant의 page_id 필드와 일치해야 합니다")


def save_evaluation_results(results: List[Dict[str, Any]], dataset_path: str):
    """평가 결과를 JSON 파일로 저장"""
    output_file = Path(dataset_path).parent / "langfuse_rag_evaluation_stats.json"
    
    save_results = []
    for result in results:
        save_result = {k: v for k, v in result.items() if k != "evaluations"}
        save_result["num_evaluations"] = len(result.get("evaluations", []))
        save_results.append(save_result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 통계 저장: {output_file}")


def compare_retrievers_with_langfuse(
    dataset_path: str = DEFAULT_DATASET_PATH,
    top_k: int = DEFAULT_TOP_K,
    retriever_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Langfuse 자동 평가(RAGAS)로 여러 리트리버 성능 비교"""
    
    print("=" * 60)
    print("📊 리트리버 성능 비교 (Langfuse 자동 평가)")
    print("=" * 60)
    
    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()
    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        return []
    
    # Qdrant 클라이언트 초기화
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    
    # 평가 데이터셋 로드
    eval_data = load_evaluation_dataset(dataset_path)
    print(f"\n✅ 평가 데이터: {len(eval_data)}개 질문")
    
    # 리트리버 초기화
    print("\n📦 리트리버 초기화 중...")
    retrievers = initialize_retrievers(qdrant_client, retriever_types)
    
    print(f"✅ {len(retrievers)}개 리트리버 초기화 완료")
    for retriever in retrievers:
        print(f"   - {retriever.name}")
    
    # 각 리트리버 평가
    results = []
    for retriever in retrievers:
        stats = evaluate_rag_with_langfuse(
            retriever, eval_data, langfuse, qdrant_client, top_k
        )
        results.append({"retriever": retriever.name, **stats})
    
    # Langfuse flush
    print("\n⏳ Langfuse에 데이터 전송 중...")
    langfuse.flush()
    
    # 결과 출력 및 저장
    print_comparison_results(results)
    print_next_steps()
    save_evaluation_results(results, dataset_path)
    
    return results


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Langfuse 자동 평가(RAGAS) 기반 RAG 성능 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
사용 가능한 리트리버 타입:
  {', '.join(RetrieverFactory.list_available_types())}

예제:
  # 모든 기본 리트리버 평가
  python {__file__}

  # 특정 리트리버만 평가
  python {__file__} --retrievers bm25_korean dense

  # 데이터셋 지정
  python {__file__} --dataset data/evaluation/custom_qa.json
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="평가 데이터셋 경로"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="검색할 상위 k개 문서"
    )
    parser.add_argument(
        "--retrievers",
        type=str,
        nargs="+",
        default=None,
        help="평가할 리트리버 타입 (기본값: 모든 리트리버)"
    )
    parser.add_argument(
        "--list-retrievers",
        action="store_true",
        help="사용 가능한 리트리버 타입 목록 출력"
    )
    
    args = parser.parse_args()
    
    if args.list_retrievers:
        print("사용 가능한 리트리버 타입:")
        for ret_type in RetrieverFactory.list_available_types():
            print(f"  - {ret_type}")
        return
    
    compare_retrievers_with_langfuse(
        dataset_path=args.dataset,
        top_k=args.top_k,
        retriever_types=args.retrievers
    )


if __name__ == "__main__":
    main()