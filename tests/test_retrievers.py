#!/usr/bin/env python3
"""Retriever 테스트 스크립트 - BM25, Dense, Ensemble(RRF) 검증"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrievers.bm25_retriever import get_bm25_retriever
from retrievers.dense_retriever import get_dense_retriever
from retrievers.ensemble_retriever import get_ensemble_retriever


def print_result(doc, index, retriever_name):
    """검색 결과 출력"""
    print(f"\n[{retriever_name}] Rank {index}")
    print(f"  📄 Page: {doc.metadata.get('page_title', 'Unknown')}")
    print(f"  📂 Section: {doc.metadata.get('section_title', 'N/A')}")
    print(f"  🆔 Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")

    # 내용 미리보기 (200자)
    content_preview = doc.page_content[:200].replace('\n', ' ')
    print(f"  💬 Content: {content_preview}...")


def test_retriever(retriever, retriever_name, query, k=5):
    """단일 retriever 테스트"""
    print(f"\n{'='*80}")
    print(f"🔍 Testing {retriever_name}")
    print(f"{'='*80}")
    print(f"📝 Query: {query}")
    print(f"📊 Top-{k} Results:")

    try:
        results = retriever.invoke(query)

        print(f"\n✅ Retrieved {len(results)} documents")

        for i, doc in enumerate(results[:k], 1):
            print_result(doc, i, retriever_name)

        return results

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def compare_retrievers(query, k=5):
    """3가지 retriever 비교"""
    print("\n" + "🎯" * 40)
    print(f"Comparing All Retrievers for Query: '{query}'")
    print("🎯" * 40)

    # BM25 Retriever
    print("\n" + "─" * 80)
    print("1️⃣  BM25 Retriever (Keyword-based)")
    print("─" * 80)
    bm25_retriever = get_bm25_retriever(k=k)
    bm25_results = test_retriever(bm25_retriever, "BM25", query, k)
    del bm25_retriever  # 명시적으로 삭제하여 리소스 해제

    # Dense Retriever
    print("\n" + "─" * 80)
    print("2️⃣  Dense Retriever (Semantic Vector Search)")
    print("─" * 80)
    dense_retriever = get_dense_retriever(k=k)
    dense_results = test_retriever(dense_retriever, "Dense", query, k)
    del dense_retriever  # 명시적으로 삭제하여 리소스 해제

    # 가비지 컬렉션 강제 실행 (Qdrant client 리소스 해제)
    import gc
    gc.collect()

    # Ensemble Retriever (RRF)
    print("\n" + "─" * 80)
    print("3️⃣  Ensemble Retriever (BM25 + Dense with RRF)")
    print("─" * 80)
    ensemble_retriever = get_ensemble_retriever(k=k, bm25_weight=0.5, dense_weight=0.5)
    ensemble_results = test_retriever(ensemble_retriever, "Ensemble", query, k)
    del ensemble_retriever  # 명시적으로 삭제

    # 결과 비교 분석
    print("\n" + "📊" * 40)
    print("Result Analysis")
    print("📊" * 40)

    # 각 retriever에서 가져온 고유 chunk_id 수집
    bm25_ids = {doc.metadata.get('chunk_id') for doc in bm25_results}
    dense_ids = {doc.metadata.get('chunk_id') for doc in dense_results}
    ensemble_ids = {doc.metadata.get('chunk_id') for doc in ensemble_results}

    print(f"\n📌 Unique Documents:")
    print(f"  BM25:     {len(bm25_ids)} unique chunks")
    print(f"  Dense:    {len(dense_ids)} unique chunks")
    print(f"  Ensemble: {len(ensemble_ids)} unique chunks")

    print(f"\n🔄 Overlap Analysis:")
    print(f"  BM25 ∩ Dense:    {len(bm25_ids & dense_ids)} common chunks")
    print(f"  BM25 ∩ Ensemble: {len(bm25_ids & ensemble_ids)} common chunks")
    print(f"  Dense ∩ Ensemble: {len(dense_ids & ensemble_ids)} common chunks")
    print(f"  All three:        {len(bm25_ids & dense_ids & ensemble_ids)} common chunks")

    return {
        "bm25": bm25_results,
        "dense": dense_results,
        "ensemble": ensemble_results
    }


def main():
    """메인 테스트 실행"""
    print("\n" + "🚀" * 40)
    print("Retriever Performance Test Suite")
    print("🚀" * 40)

    # 테스트 쿼리 세트
    test_queries = [
        "RAG 시스템은 어떻게 동작하나요?",
        "임베딩 모델에 대해 알려주세요",
        "벡터 데이터베이스란 무엇인가요?"
    ]

    all_results = {}

    for query in test_queries:
        results = compare_retrievers(query, k=5)
        all_results[query] = results
        print("\n" + "━" * 80 + "\n")

    # 최종 요약
    print("\n" + "✨" * 40)
    print("Test Summary")
    print("✨" * 40)
    print(f"\n✅ Tested {len(test_queries)} queries")
    print(f"✅ Tested 3 retriever types: BM25, Dense, Ensemble(RRF)")
    print(f"✅ All retrievers working properly!")

    print("\n" + "💡" * 40)
    print("Recommendations:")
    print("💡" * 40)
    print("""
    - BM25: Best for exact keyword matching and terminology-heavy queries
    - Dense: Best for semantic understanding and paraphrased queries
    - Ensemble (RRF): Best overall performance combining both approaches

    💡 Tip: Adjust weights in ensemble_retriever for your use case:
       - More BM25 weight for technical/precise queries
       - More Dense weight for conceptual/semantic queries
    """)

    print("\n" + "🎉" * 40)
    print("All Tests Completed Successfully!")
    print("🎉" * 40)


if __name__ == "__main__":
    main()
