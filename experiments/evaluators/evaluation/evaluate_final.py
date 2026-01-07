#!/usr/bin/env python3
"""주간/최종보고서 최적 조합 재평가 스크립트

주간보고서: Qwen3-Reranker-4B + BGE-M3 + RRF Ensemble (Top 6)
최종보고서: OpenAI + RRF MultiQuery (Top 8)

답변 생성: GPT-4 Turbo (1106-preview)
추적: Langfuse

사용법:
    # 주간보고서 평가
    python re_evaluate_optimal.py --report-type weekly
    
    # 최종보고서 평가
    python re_evaluate_optimal.py --report-type executive
    
    # 둘 다 평가
    python re_evaluate_optimal.py --report-type both
"""

import sys
from pathlib import Path
# 프로젝트 루트와 app, experiments 디렉토리를 path에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))
sys.path.insert(0, str(project_root / "experiments"))

from dotenv import load_dotenv
load_dotenv()

import json
import time
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.langfuse import get_langfuse_client
from utils.common import (
    load_prompt,
    load_evaluation_dataset,
    create_trace_and_generation,
    add_retrieval_quality_score,
    save_embedding_cache
)
from utils.retriever_factory import create_retriever_from_config

# Reranker 관련 임포트
import torch
from sentence_transformers import CrossEncoder

# Reranker 모델 인스턴스 (전역 변수로 한 번만 로드)
QWEN3_RERANKER = None


def format_query(query: str, instruction: str = None) -> str:
    """Qwen3-Reranker를 위한 쿼리 포맷팅"""
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = "Given a query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document: str) -> str:
    """Qwen3-Reranker를 위한 문서 포맷팅"""
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


def get_optimal_batch_size() -> int:
    """GPU 메모리에 따른 최적 배치 크기 계산"""
    # VLLM과 공존하기 위해 배치 크기를 8으로 고정
    return 8


def get_qwen3_reranker() -> CrossEncoder:
    """Qwen3-Reranker-4B 모델 로드 (캐싱)"""
    global QWEN3_RERANKER

    # 이미 로드된 모델이 있으면 재사용
    if QWEN3_RERANKER is not None:
        return QWEN3_RERANKER

    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"🔄 Qwen3-Reranker-4B 모델 로딩 중... (device: {device})")
    QWEN3_RERANKER = CrossEncoder(
        "tomaarsen/Qwen3-Reranker-4B-seq-cls",
        max_length=8192,
        device=device,
        trust_remote_code=True
    )
    print("✅ Qwen3-Reranker-4B 모델 로드 완료!")

    # 최적 배치 크기 출력
    optimal_bs = get_optimal_batch_size()
    print(f"💡 권장 배치 크기: {optimal_bs}")

    return QWEN3_RERANKER


def rerank_documents(
    query: str,
    docs: list,
    top_k: int = 6,
    batch_size: int = None,
    initial_k: int = None
) -> list:
    """Qwen3-Reranker-4B 모델로 문서 재순위화"""
    # Reranker 모델 로드
    reranker = get_qwen3_reranker()

    # 배치 크기 자동 설정
    if batch_size is None:
        batch_size = get_optimal_batch_size()

    # 초기 문서 수 설정
    if initial_k is None:
        initial_k = len(docs)

    # Qwen3-Reranker 포맷으로 query-document 쌍 생성
    formatted_query = format_query(query)
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

    # 점수와 문서를 함께 정렬 (내림차순)
    doc_score_pairs = list(zip(docs, all_scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 상위 k개 문서만 반환
    reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

    print(f"\n🔄 Qwen3 Reranking 완료: {initial_k}개 → {len(reranked_docs)}개 (배치 크기: {batch_size})")
    print("Top 3 Reranked Scores:")
    for i, (doc, score) in enumerate(doc_score_pairs[:3], 1):
        print(f"  {i}. {doc.metadata.get('page_title', 'Unknown')}: {score:.4f}")

    return reranked_docs


def generate_answer(
    question: str,
    contexts: List[str],
    system_prompt: str,
    answer_generation_prompt: str,
    langfuse=None,
    trace_id: str = None
) -> str:
    """GPT-4 Turbo (1106-preview)로 답변 생성"""
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage

    # Context 구성
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        context_parts.append(f"[문서 {i}]\n{ctx}\n")

    context_text = "\n".join(context_parts)

    # 프롬프트 구성
    user_prompt = answer_generation_prompt.replace("{context}", context_text).replace("{question}", question)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # init_chat_model로 모델 초기화
    model = init_chat_model("azure_ai:gpt-4.1")

    # Langfuse 추적
    if langfuse and trace_id:
        with langfuse.start_as_current_observation(
            as_type='generation',
            name="final_answer_generation",
            model="gpt-4-1106-preview",
            input={"question": question, "num_contexts": len(contexts)},
            metadata={"model": "gpt-4.1"}
        ) as generation:
            response = model.invoke(messages)
            answer = response.content

            # 토큰 사용량 기록
            usage_dict = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_dict = {
                    "input": response.usage_metadata.get('input_tokens', 0),
                    "output": response.usage_metadata.get('output_tokens', 0),
                    "total": response.usage_metadata.get('total_tokens', 0)
                }

            generation.update(
                output={"answer": answer},
                usage=usage_dict
            )
    else:
        response = model.invoke(messages)
        answer = response.content

    return answer


def evaluate_weekly_report(
    item: Dict[str, Any],
    langfuse,
    idx: int,
    system_prompt: str,
    answer_generation_prompt: str,
    version_tag: str = "optimal_v1"
) -> Dict[str, Any]:
    """주간보고서 평가: BGE-M3 + RRF + Qwen3-Reranker (Top 6)"""
    question = item["question"]
    ground_truth = item["ground_truth"]
    context_page_id = item.get("context_page_id")
    item_metadata = item.get("metadata", {})
    
    start_time = time.time()
    
    # 1. BGE-M3 + RRF Ensemble로 초기 검색 (20개)
    retriever_config = {
        "name": "bge-m3-rrf-ensemble",
        "display_name": "BGE-M3 + RRF Ensemble",
        "description": "BGE-M3 embedding with RRF ensemble",
        "embedding_preset": "bge-m3",
        "retriever_type": "rrf_ensemble",
        "k": 20,
        "bm25_weight": 0.5,
        "dense_weight": 0.5,
        "use_mmr": True,
        "lambda_mult": 0.5
    }

    retriever, _ = create_retriever_from_config(retriever_config)
    
    print(f"  [{idx}] 🔍 BGE-M3 + RRF 검색 중...")
    initial_docs = retriever.invoke(question)
    print(f"  [{idx}] 📄 초기 검색: {len(initial_docs)}개 문서")
    
    # 2. Qwen3-Reranker로 재순위화 (Top 6)
    print(f"  [{idx}] 🔄 Qwen3 Reranking 중...")
    reranked_docs = rerank_documents(question, initial_docs, top_k=6, batch_size=16)
    
    # 컨텍스트 변환
    contexts = []
    context_metadata = []
    
    for result in reranked_docs:
        contexts.append(result.page_content)
        context_metadata.append({
            "page_title": result.metadata.get('page_title', 'Unknown'),
            "section_title": result.metadata.get('section_title', 'N/A'),
            "chunk_id": result.metadata.get('chunk_id', 'unknown'),
            "score": result.metadata.get('_combined_score') or result.metadata.get('_similarity_score')
        })
    
    if not contexts:
        print(f"  ⚠️ [{idx}] No contexts found!")
        contexts = ["검색 결과가 없습니다."]
    
    # 3. GPT-4로 답변 생성
    print(f"  [{idx}] 💬 GPT-4로 답변 생성 중...")
    
    # Langfuse Trace 생성
    retriever_name = "bge-m3-rrf-qwen3-reranker-k6"
    retriever_tags = ["bge-m3", "rrf_ensemble", "qwen3-reranker", "top_k_6", "weekly_report"]
    
    additional_metadata = {
        "context_page_id": context_page_id,
        "retriever_name": retriever_name,
        "display_name": "BGE-M3 + RRF + Qwen3-Reranker (Top 6)",
        "report_type": "weekly_report",
        "top_k": 6,
        "embedding_preset": "bge-m3",
        "retriever_type": "rrf_ensemble_reranker",
        "reranker_model": "Qwen3-Reranker-4B",
        "llm_model": "gpt-4-1106-preview"
    }
    
    trace_id = create_trace_and_generation(
        langfuse=langfuse,
        retriever_name=retriever_name,
        question=question,
        contexts=contexts,
        answer="",  # 임시
        ground_truth=ground_truth,
        context_metadata=context_metadata,
        item_metadata=item_metadata,
        total_time=0,  # 임시
        idx=idx,
        version_tag=version_tag,
        retriever_tags=retriever_tags,
        additional_metadata=additional_metadata
    )
    
    # GPT-4 Turbo 답변 생성
    answer = generate_answer(
        question=question,
        contexts=contexts,
        system_prompt=system_prompt,
        answer_generation_prompt=answer_generation_prompt,
        langfuse=langfuse,
        trace_id=trace_id
    )
    
    total_time = time.time() - start_time
    
    # Trace 업데이트
    if langfuse and trace_id:
        langfuse.update_current_trace(
            output={"answer": answer},
            metadata={**additional_metadata, "total_time": total_time}
        )
    
    # 검색 품질 스코어 추가
    add_retrieval_quality_score(langfuse, trace_id, context_metadata)
    
    print(f"  [{idx}] ✅ 완료 ({len(contexts)}개 문서, {total_time*1000:.0f}ms)")
    
    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "num_contexts": len(contexts),
        "time": total_time,
        "trace_id": trace_id,
        "context_metadata": context_metadata
    }


def evaluate_executive_report(
    item: Dict[str, Any],
    langfuse,
    idx: int,
    system_prompt: str,
    answer_generation_prompt: str,
    version_tag: str = "optimal_v1"
) -> Dict[str, Any]:
    """최종보고서 평가: OpenAI + RRF MultiQuery (Top 8)"""
    question = item["question"]
    ground_truth = item["ground_truth"]
    context_page_id = item.get("context_page_id")
    item_metadata = item.get("metadata", {})
    
    start_time = time.time()
    
    # OpenAI + RRF MultiQuery 설정
    retriever_config = {
        "name": "openai-rrf-multiquery",
        "display_name": "OpenAI + RRF MultiQuery",
        "description": "OpenAI embedding with RRF multiquery",
        "embedding_preset": "openai",
        "retriever_type": "rrf_multiquery",
        "k": 8,
        "bm25_weight": 0.5,
        "dense_weight": 0.5,
        "use_mmr": True,
        "lambda_mult": 0.5
    }

    retriever, _ = create_retriever_from_config(retriever_config)
    
    print(f"  [{idx}] 🔍 OpenAI + RRF MultiQuery 검색 중...")
    search_results = retriever.invoke(question)
    
    # 컨텍스트 변환
    contexts = []
    context_metadata = []
    
    for result in search_results[:8]:
        contexts.append(result.page_content)
        context_metadata.append({
            "page_title": result.metadata.get('page_title', 'Unknown'),
            "section_title": result.metadata.get('section_title', 'N/A'),
            "chunk_id": result.metadata.get('chunk_id', 'unknown'),
            "score": result.metadata.get('_combined_score') or result.metadata.get('_similarity_score')
        })
    
    if not contexts:
        print(f"  ⚠️ [{idx}] No contexts found!")
        contexts = ["검색 결과가 없습니다."]
    
    print(f"  [{idx}] 📄 검색 완료: {len(contexts)}개 문서")
    
    # GPT-4 Turbo로 답변 생성
    print(f"  [{idx}] 💬 GPT-4 Turbo 답변 생성 중...")
    
    # Langfuse Trace 생성
    retriever_name = "openai-rrf-multiquery-k8"
    retriever_tags = ["openai", "rrf_multiquery", "top_k_8", "executive_report"]
    
    additional_metadata = {
        "context_page_id": context_page_id,
        "retriever_name": retriever_name,
        "display_name": "OpenAI + RRF MultiQuery (Top 8)",
        "report_type": "executive_report",
        "top_k": 8,
        "embedding_preset": "openai",
        "retriever_type": "rrf_multiquery",
        "llm_model": "gpt-4-1106-preview"
    }
    
    trace_id = create_trace_and_generation(
        langfuse=langfuse,
        retriever_name=retriever_name,
        question=question,
        contexts=contexts,
        answer="",  # 임시
        ground_truth=ground_truth,
        context_metadata=context_metadata,
        item_metadata=item_metadata,
        total_time=0,  # 임시
        idx=idx,
        version_tag=version_tag,
        retriever_tags=retriever_tags,
        additional_metadata=additional_metadata
    )
    
    # GPT-4 Turbo 답변 생성
    answer = generate_answer(
        question=question,
        contexts=contexts,
        system_prompt=system_prompt,
        answer_generation_prompt=answer_generation_prompt,
        langfuse=langfuse,
        trace_id=trace_id
    )
    
    total_time = time.time() - start_time
    
    # Trace 업데이트
    if langfuse and trace_id:
        langfuse.update_current_trace(
            output={"answer": answer},
            metadata={**additional_metadata, "total_time": total_time}
        )
    
    # 검색 품질 스코어 추가
    add_retrieval_quality_score(langfuse, trace_id, context_metadata)
    
    print(f"  [{idx}] ✅ 완료 ({len(contexts)}개 문서, {total_time*1000:.0f}ms)")
    
    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "num_contexts": len(contexts),
        "time": total_time,
        "trace_id": trace_id,
        "context_metadata": context_metadata
    }


def run_evaluation(
    report_type: str,
    dataset_path: str,
    version: str = "optimal_v1",
    max_workers: int = 3
):
    """평가 실행"""
    
    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()
    if not langfuse:
        print("❌ Langfuse 클라이언트를 초기화할 수 없습니다.")
        return
    
    # 프롬프트 로드
    report_suffix = "weekly_report" if report_type == "weekly" else "executive_report_v3"
    system_prompt = load_prompt(f"prompts/templates/service/{report_suffix}/system_prompt.txt")
    answer_generation_prompt = load_prompt(f"prompts/templates/service/{report_suffix}/answer_generation_prompt.txt")
    
    # 데이터셋 로드
    eval_data = load_evaluation_dataset(dataset_path)
    
    report_display = "주간보고서" if report_type == "weekly" else "최종보고서"
    
    if report_type == "weekly":
        config_display = "BGE-M3 + RRF + Qwen3-Reranker (Top 6)"
    else:
        config_display = "OpenAI + RRF MultiQuery (Top 8)"
    
    print("\n" + "=" * 80)
    print(f"🎯 {report_display} 최적 조합 재평가")
    print("=" * 80)
    print(f"📊 설정: {config_display}")
    print(f"💬 LLM: GPT-4 Turbo (1106-preview)")
    print(f"📋 데이터셋: {len(eval_data)} 개 샘플")
    print(f"🏷️  Version: {version}")
    print(f"⚡ 병렬 워커: {max_workers}")
    print()
    
    stats = {
        "total_queries": len(eval_data),
        "total_time": 0,
        "evaluations": []
    }

    # 병렬 처리 전에 Reranker 모델 미리 로드 (race condition 방지)
    if report_type == "weekly":
        print("🔄 Reranker 모델 사전 로드 중...")
        get_qwen3_reranker()
        print("✅ Reranker 모델 사전 로드 완료!\n")

    # 병렬 처리로 평가 실행
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if report_type == "weekly":
            future_to_item = {
                executor.submit(
                    evaluate_weekly_report,
                    item,
                    langfuse,
                    idx,
                    system_prompt,
                    answer_generation_prompt,
                    version
                ): (idx, item)
                for idx, item in enumerate(eval_data, 1)
            }
        else:
            future_to_item = {
                executor.submit(
                    evaluate_executive_report,
                    item,
                    langfuse,
                    idx,
                    system_prompt,
                    answer_generation_prompt,
                    version
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
    output_dir = Path("data/langfuse/evaluation_results") / f"{report_type}_report_optimal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"optimal_{timestamp}_stats.json"
    
    save_result = {k: v for k, v in stats.items() if k != "evaluations"}
    save_result["num_evaluations"] = len(stats.get("evaluations", []))
    save_result["config"] = {
        "report_type": report_type,
        "retriever": config_display,
        "llm": "GPT-4 Turbo (1106-preview)",
        "version": version,
        "timestamp": timestamp
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="주간/최종보고서 최적 조합 재평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--report-type",
        type=str,
        choices=["weekly", "executive", "both"],
        default="both",
        help="평가할 보고서 타입"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/work/rag/Project/rag-report-generator/data/evaluation/merged_qa_dataset.json",
        help="평가 데이터셋 경로"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default="optimal_v1",
        help="버전 태그"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="병렬 처리 워커 수"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 주간/최종보고서 최적 조합 재평가 시스템")
    print("=" * 80)
    print(f"\n📊 평가 설정:")
    print(f"   - 보고서 타입: {args.report_type}")
    print(f"   - Dataset: {args.dataset}")
    print(f"   - Version: {args.version}")
    print(f"   - Max Workers: {args.max_workers}")
    
    if args.report_type in ["weekly", "both"]:
        print("\n" + "=" * 80)
        print("📊 주간보고서 평가")
        print("=" * 80)
        run_evaluation(
            report_type="weekly",
            dataset_path=args.dataset,
            version=args.version,
            max_workers=args.max_workers
        )
    
    if args.report_type in ["executive", "both"]:
        print("\n" + "=" * 80)
        print("📊 최종보고서 평가")
        print("=" * 80)
        run_evaluation(
            report_type="executive",
            dataset_path=args.dataset,
            version=args.version,
            max_workers=args.max_workers
        )
    
    print("\n" + "=" * 80)
    print("✅ 모든 평가 완료!")
    print("=" * 80)
    print("\n📊 Langfuse 대시보드에서 결과 확인: https://cloud.langfuse.com")


if __name__ == "__main__":
    main()
