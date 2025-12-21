#!/usr/bin/env python3
"""BM25 vs Dense Retrieval 성능 비교"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from config.settings import QDRANT_PATH
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from utils.langfuse_utils import get_langfuse_client
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain.chat_models import init_chat_model
import time
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


def load_evaluation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """평가용 데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_retriever(
    retriever,
    retriever_name: str,
    eval_data: List[Dict[str, Any]],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    특정 리트리버 평가

    Args:
        retriever: 평가할 리트리버
        retriever_name: 리트리버 이름
        eval_data: 평가 데이터
        top_k: 검색할 상위 k개 문서

    Returns:
        평가 결과 딕셔너리
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 {retriever_name} 평가 중...")
    print(f"{'=' * 60}")

    questions = []
    ground_truths = []
    contexts_list = []
    answers = []

    # 검색 시간 측정
    search_times = []

    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # 검색 수행 및 시간 측정
        start_time = time.time()
        search_results = retriever.search(question, top_k=top_k)
        search_time = time.time() - start_time
        search_times.append(search_time)

        # 검색 결과를 컨텍스트로 변환
        contexts = [result.combined_text for result in search_results]
        answer = contexts[0] if contexts else "검색 결과 없음"

        questions.append(question)
        ground_truths.append(ground_truth)
        contexts_list.append(contexts)
        answers.append(answer)

    # Azure AI LLM 초기화
    llm = init_chat_model("azure_ai:gpt-5.1")

    # Ragas 평가
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts_list,
        "answer": answers,
    })

    metrics = [context_precision, context_recall]

    try:
        results = evaluate(ragas_dataset, metrics=metrics, llm=llm)

        # 평균 검색 시간 추가
        avg_search_time = sum(search_times) / len(search_times) if search_times else 0

        return {
            "retriever": retriever_name,
            "context_precision": float(results['context_precision']),
            "context_recall": float(results['context_recall']),
            "avg_search_time": avg_search_time,
            "total_queries": len(questions)
        }

    except Exception as e:
        print(f"❌ {retriever_name} 평가 실패: {e}")
        return {
            "retriever": retriever_name,
            "error": str(e)
        }


def compare_retrievers(
    dataset_path: str = "data/evaluation/sample_qa.json",
    top_k: int = 5,
    use_korean_tokenizer: bool = True
):
    """
    BM25와 Dense 검색 성능 비교

    Args:
        dataset_path: 평가 데이터셋 경로
        top_k: 검색할 상위 k개 문서
        use_korean_tokenizer: BM25에서 한국어 토크나이저 사용 여부
    """
    print("=" * 60)
    print("📊 BM25 vs Dense Retrieval 성능 비교")
    print("=" * 60)

    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()

    # Qdrant 클라이언트 초기화
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    # 평가 데이터셋 로드
    eval_data = load_evaluation_dataset(dataset_path)
    print(f"\n✅ 평가 데이터: {len(eval_data)}개 질문")

    # 리트리버들 초기화
    print("\n📦 리트리버 초기화 중...")
    bm25_retriever = BM25Retriever(qdrant_client, use_korean_tokenizer=use_korean_tokenizer)
    dense_retriever = DenseRetriever(qdrant_client)

    # 각 리트리버 평가
    results = []

    # BM25 평가
    bm25_results = evaluate_retriever(
        bm25_retriever,
        "BM25",
        eval_data,
        top_k=top_k
    )
    results.append(bm25_results)

    # Dense 평가
    dense_results = evaluate_retriever(
        dense_retriever,
        "Dense (Vector)",
        eval_data,
        top_k=top_k
    )
    results.append(dense_results)

    # 비교 결과 출력
    print("\n" + "=" * 60)
    print("📈 종합 비교 결과")
    print("=" * 60)

    print(f"\n{'Retriever':<20} {'Precision':<12} {'Recall':<12} {'Avg Time (ms)':<15}")
    print("-" * 60)

    for result in results:
        if "error" not in result:
            print(
                f"{result['retriever']:<20} "
                f"{result['context_precision']:<12.4f} "
                f"{result['context_recall']:<12.4f} "
                f"{result['avg_search_time']*1000:<15.2f}"
            )
        else:
            print(f"{result['retriever']:<20} ERROR: {result['error']}")

    # 승자 판정
    if len(results) == 2 and "error" not in results[0] and "error" not in results[1]:
        print("\n" + "=" * 60)
        print("🏆 승자 판정")
        print("=" * 60)

        bm25_score = results[0]['context_precision'] + results[0]['context_recall']
        dense_score = results[1]['context_precision'] + results[1]['context_recall']

        if bm25_score > dense_score:
            winner = "BM25"
            diff = bm25_score - dense_score
        elif dense_score > bm25_score:
            winner = "Dense (Vector)"
            diff = dense_score - bm25_score
        else:
            winner = "무승부"
            diff = 0

        print(f"\n🥇 {winner}")
        if diff > 0:
            print(f"   점수 차이: {diff:.4f}")

        print(f"\n💡 분석:")
        if results[0]['context_precision'] > results[1]['context_precision']:
            print(f"   - BM25가 정밀도가 더 높음 (+{results[0]['context_precision'] - results[1]['context_precision']:.4f})")
        else:
            print(f"   - Dense가 정밀도가 더 높음 (+{results[1]['context_precision'] - results[0]['context_precision']:.4f})")

        if results[0]['context_recall'] > results[1]['context_recall']:
            print(f"   - BM25가 재현율이 더 높음 (+{results[0]['context_recall'] - results[1]['context_recall']:.4f})")
        else:
            print(f"   - Dense가 재현율이 더 높음 (+{results[1]['context_recall'] - results[0]['context_recall']:.4f})")

        if results[0]['avg_search_time'] < results[1]['avg_search_time']:
            print(f"   - BM25가 더 빠름 ({results[0]['avg_search_time']*1000:.2f}ms vs {results[1]['avg_search_time']*1000:.2f}ms)")
        else:
            print(f"   - Dense가 더 빠름 ({results[1]['avg_search_time']*1000:.2f}ms vs {results[0]['avg_search_time']*1000:.2f}ms)")

    # Langfuse에 결과 로깅
    if langfuse:
        print("\n📊 Langfuse에 결과 업로드 중...")

        for result in results:
            if "error" not in result:
                retriever_name = result['retriever'].lower().replace(" ", "_").replace("(", "").replace(")", "")

                langfuse.score(
                    name=f"{retriever_name}_context_precision",
                    value=result['context_precision'],
                    data_type="numeric",
                    comment=f"{result['retriever']} - Context Precision"
                )

                langfuse.score(
                    name=f"{retriever_name}_context_recall",
                    value=result['context_recall'],
                    data_type="numeric",
                    comment=f"{result['retriever']} - Context Recall"
                )

                langfuse.score(
                    name=f"{retriever_name}_avg_search_time",
                    value=result['avg_search_time'],
                    data_type="numeric",
                    comment=f"{result['retriever']} - Average Search Time (seconds)"
                )

        langfuse.flush()
        print("✅ Langfuse 업로드 완료")

    # 결과를 JSON 파일로 저장
    output_file = Path(dataset_path).parent / "comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 비교 결과 저장: {output_file}")

    return results


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="BM25 vs Dense 검색 성능 비교")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/sample_qa.json",
        help="평가 데이터셋 경로"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="검색할 상위 k개 문서"
    )
    parser.add_argument(
        "--no-korean-tokenizer",
        action="store_true",
        help="BM25에서 한국어 토크나이저 사용 안 함"
    )

    args = parser.parse_args()

    compare_retrievers(
        dataset_path=args.dataset,
        top_k=args.top_k,
        use_korean_tokenizer=not args.no_korean_tokenizer
    )


if __name__ == "__main__":
    main()
