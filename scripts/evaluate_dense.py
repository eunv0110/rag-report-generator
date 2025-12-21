#!/usr/bin/env python3
"""Dense Retrieval 성능 평가 with Langfuse + Ragas"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from config.settings import QDRANT_PATH
from retrievers.dense_retriever import DenseRetriever
from utils.langfuse_utils import get_langfuse_client
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)


def load_evaluation_dataset(file_path: str) -> List[Dict[str, Any]]:
    """평가용 데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_ragas_dataset(
    questions: List[str],
    ground_truths: List[str],
    contexts: List[List[str]],
    answers: List[str]
) -> Dataset:
    """
    Ragas 평가를 위한 데이터셋 준비

    Args:
        questions: 질문 리스트
        ground_truths: 정답 리스트
        contexts: 검색된 컨텍스트 리스트
        answers: 생성된 답변 리스트

    Returns:
        Ragas Dataset
    """
    data = {
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts,
        "answer": answers,
    }

    return Dataset.from_dict(data)


def evaluate_dense_retrieval(
    dataset_path: str = "data/evaluation/sample_qa.json",
    top_k: int = 5
):
    """
    Dense 검색 성능을 Ragas로 평가하고 Langfuse에 로깅

    Args:
        dataset_path: 평가 데이터셋 경로
        top_k: 검색할 상위 k개 문서
    """
    print("=" * 60)
    print("🔍 Dense Retrieval 평가 시작")
    print("=" * 60)

    # Langfuse 클라이언트 초기화
    langfuse = get_langfuse_client()

    # Qdrant 클라이언트 및 Dense Retriever 초기화
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    retriever = DenseRetriever(qdrant_client)

    # 평가 데이터셋 로드
    eval_data = load_evaluation_dataset(dataset_path)
    print(f"\n📊 평가 데이터: {len(eval_data)}개 질문")

    # 검색 수행
    questions = []
    ground_truths = []
    contexts_list = []
    answers = []

    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Dense 검색
        search_results = retriever.search(question, top_k=top_k)

        # 검색 결과를 컨텍스트로 변환
        contexts = [result.combined_text for result in search_results]

        # 검색 결과의 첫 번째를 답변으로 사용
        answer = contexts[0] if contexts else "검색 결과 없음"

        questions.append(question)
        ground_truths.append(ground_truth)
        contexts_list.append(contexts)
        answers.append(answer)

        print(f"\n질문: {question}")
        print(f"검색된 문서 수: {len(contexts)}")
        print(f"Top 1 Score: {search_results[0].score if search_results else 0:.4f}")

    # Ragas 평가 데이터셋 준비
    print("\n" + "=" * 60)
    print("📈 Ragas 평가 시작")
    print("=" * 60)

    ragas_dataset = prepare_ragas_dataset(
        questions=questions,
        ground_truths=ground_truths,
        contexts=contexts_list,
        answers=answers
    )

    # Ragas 평가 실행
    # Context-based metrics만 사용 (Dense는 검색만 수행하므로)
    metrics = [
        context_precision,
        context_recall,
    ]

    # LLM이 필요한 메트릭을 사용하려면 OpenAI API 키 등이 필요
    # faithfulness와 answer_relevancy는 LLM을 사용하므로 주석 처리
    # metrics.extend([faithfulness, answer_relevancy])

    try:
        results = evaluate(
            ragas_dataset,
            metrics=metrics,
        )

        print("\n" + "=" * 60)
        print("✅ 평가 결과")
        print("=" * 60)
        print(f"Context Precision: {results['context_precision']:.4f}")
        print(f"Context Recall: {results['context_recall']:.4f}")

        # Langfuse에 평가 결과 로깅
        if langfuse:
            print("\n📊 Langfuse에 결과 업로드 중...")

            # Score 로깅
            for metric_name, score in results.items():
                if isinstance(score, (int, float)):
                    langfuse.score(
                        name=f"dense_{metric_name}",
                        value=score,
                        data_type="numeric",
                        comment=f"Dense Retrieval - {metric_name}"
                    )

            langfuse.flush()
            print("✅ Langfuse 업로드 완료")

        return results

    except Exception as e:
        print(f"\n❌ 평가 중 오류 발생: {e}")
        print("\n💡 Tip: LLM 기반 메트릭(faithfulness, answer_relevancy)을 사용하려면")
        print("   OPENAI_API_KEY 등을 설정해야 합니다.")
        return None


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Dense 검색 성능 평가")
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

    args = parser.parse_args()

    evaluate_dense_retrieval(
        dataset_path=args.dataset,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
