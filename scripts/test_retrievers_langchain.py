#!/usr/bin/env python3
"""LangChain 호환 retriever 테스트 스크립트"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from pathlib import Path
from config.settings import DATA_DIR
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.ensemble_retriever import EnsembleRetriever

# qwen3-embedding-4b DB 사용 (데이터가 있는 DB)
QDRANT_PATH = str(DATA_DIR / "qdrant_data" / "qwen3-embedding-4b")


def test_retriever(retriever, query: str, name: str):
    """리트리버 테스트"""
    print(f"\n{'=' * 60}")
    print(f"🧪 {name} 테스트")
    print(f"{'=' * 60}")
    print(f"쿼리: {query}")

    # 1. 기존 search() 메서드 테스트
    print("\n1️⃣ search() 메서드 테스트:")
    results = retriever.search(query, top_k=3)
    print(f"   ✅ {len(results)}개 결과 반환")
    for i, result in enumerate(results, 1):
        print(f"   [{i}] {result.page_title} (score: {result.score:.4f})")
        print(f"       {result.text[:100]}...")

    # 2. LangChain invoke() 메서드 테스트
    print("\n2️⃣ LangChain invoke() 메서드 테스트:")
    documents = retriever.invoke(query)
    print(f"   ✅ {len(documents)}개 Document 반환")
    for i, doc in enumerate(documents, 1):
        print(f"   [{i}] {doc.metadata.get('page_title')} (score: {doc.metadata.get('score'):.4f})")
        print(f"       {doc.page_content[:100]}...")

    # 3. LangChain get_relevant_documents() 메서드 테스트
    print("\n3️⃣ LangChain get_relevant_documents() 메서드 테스트:")
    docs = retriever.get_relevant_documents(query)
    print(f"   ✅ {len(docs)}개 Document 반환")

    # 4. 리트리버 정보 출력
    print("\n4️⃣ 리트리버 정보:")
    info = retriever.get_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print(f"\n✅ {name} 테스트 완료!")


def main():
    """메인 함수"""
    print("=" * 60)
    print("🔍 LangChain 호환 Retriever 테스트")
    print("=" * 60)

    # Qdrant 클라이언트 초기화
    print("\n📦 Qdrant 클라이언트 초기화 중...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    print("✅ Qdrant 초기화 완료")

    # 테스트 쿼리
    test_query = "Notion API를 사용하는 방법"

    # 1. BM25 Retriever 테스트
    print("\n" + "=" * 60)
    print("1. BM25 Retriever")
    print("=" * 60)
    bm25 = BM25Retriever(qdrant_client, use_korean_tokenizer=True, top_k=3)
    test_retriever(bm25, test_query, "BM25 Retriever")

    # 2. Dense Retriever 테스트
    print("\n" + "=" * 60)
    print("2. Dense Retriever")
    print("=" * 60)
    dense = DenseRetriever(qdrant_client, top_k=3)
    test_retriever(dense, test_query, "Dense Retriever")

    # 3. Ensemble (RRF) Retriever 테스트
    print("\n" + "=" * 60)
    print("3. Ensemble RRF Retriever")
    print("=" * 60)
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5],
        k=60,
        top_k=3
    )
    test_retriever(ensemble, test_query, "Ensemble RRF Retriever")

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)
    print("\n💡 결과:")
    print("   - 모든 retriever가 LangChain과 호환됩니다")
    print("   - search(), invoke(), get_relevant_documents() 모두 정상 작동")
    print("   - RRF 앙상블도 정상 작동")


if __name__ == "__main__":
    main()
