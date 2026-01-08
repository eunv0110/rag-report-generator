#!/usr/bin/env python3
"""MultiQuery Retriever - LangChain 기반 쿼리 확장을 통한 검색 개선"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chat_models import init_chat_model
from typing import List, Optional, Dict, Any
import os
from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT
from retrievers.dense_retriever import get_dense_retriever
from utils.common import load_prompt


class MultiQueryRetriever(BaseRetriever):
    """
    멀티 쿼리 리트리버 - LLM을 사용하여 쿼리 확장

    하나의 질문을 여러 관점으로 재구성하여 검색 품질을 향상시킵니다.
    LangChain의 BaseRetriever를 상속하여 구현되었습니다.
    """

    base_retriever: BaseRetriever
    """기본 리트리버 (Dense, BM25, Ensemble 등)"""

    num_queries: int = 3
    """생성할 쿼리 수 (원본 포함)"""

    temperature: float = 0.7
    """LLM temperature 설정"""

    llm_model: str = "azure_ai:gpt-4.1"
    """사용할 LLM 모델"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        num_queries: int = 3,
        temperature: float = 0.7,
        llm_model: str = "azure_ai:gpt-4.1",
        **kwargs
    ):
        """
        Args:
            base_retriever: 기본 리트리버 (Dense, BM25, Ensemble 등)
            num_queries: 생성할 쿼리 수 (기본값: 3)
            temperature: LLM temperature (기본값: 0.7)
            llm_model: 사용할 LLM 모델 (기본값: azure_ai:gpt-4.1)
        """
        super().__init__(
            base_retriever=base_retriever,
            num_queries=num_queries,
            temperature=temperature,
            llm_model=llm_model,
            **kwargs
        )

        # Azure OpenAI 설정
        if AZURE_AI_CREDENTIAL and AZURE_AI_ENDPOINT:
            os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
            os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

    def generate_queries(self, query: str) -> List[str]:
        """
        LLM을 사용하여 다양한 관점의 쿼리 생성

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리 리스트 (원본 포함)
        """
        # 프롬프트 템플릿 로드
        prompt_template = load_prompt("prompts/templates/service/multiquery.txt")
        prompt = prompt_template.format(
            num_queries=self.num_queries - 1,
            query=query
        )

        try:
            model = init_chat_model(
                self.llm_model,
                temperature=self.temperature,
                max_completion_tokens=300
            )
            response = model.invoke(prompt)

            # 응답에서 질문 추출
            generated_queries = [
                line.strip()
                for line in response.content.strip().split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

            # 원본 쿼리 + 생성된 쿼리
            all_queries = [query] + generated_queries[:self.num_queries - 1]

            print(f"\n[MultiQuery] 생성된 쿼리:")
            for i, q in enumerate(all_queries, 1):
                print(f"  {i}. {q}")

            return all_queries

        except Exception as e:
            print(f"  ⚠️ 쿼리 생성 실패: {e}")
            print(f"  → 원본 쿼리만 사용합니다.")
            return [query]

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        LangChain 호환 메서드: 관련 문서 검색

        Args:
            query: 검색 쿼리
            run_manager: 콜백 매니저

        Returns:
            LangChain Document 리스트
        """
        # 쿼리 확장
        queries = self.generate_queries(query)

        # 각 쿼리로 검색 수행
        all_results: Dict[str, Document] = {}

        for q in queries:
            # base_retriever는 invoke() 메서드를 사용
            results = self.base_retriever.invoke(q)

            for doc in results:
                # chunk_id를 키로 사용 (중복 제거)
                chunk_id = doc.metadata.get("chunk_id", id(doc))

                # 이미 있는 경우, 더 높은 스코어로 업데이트
                # (여러 쿼리에서 나온 문서는 더 관련성이 높다고 판단)
                if chunk_id not in all_results:
                    all_results[chunk_id] = doc
                # 같은 문서가 여러 쿼리에서 나왔다면 중요도 증가를 반영할 수 있음
                # (현재는 단순히 첫 번째 발견된 것을 유지)

        # 원래 순서대로 반환 (나중에 점수 기반 정렬도 가능)
        sorted_results = list(all_results.values())

        print(f"  → 총 {len(all_results)}개 고유 문서 반환")

        return sorted_results

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        return {
            "name": "MultiQueryRetriever",
            "type": self.__class__.__name__,
            "base_retriever": type(self.base_retriever).__name__,
            "num_queries": self.num_queries,
            "temperature": self.temperature,
            "llm_model": self.llm_model
        }


def get_multiquery_retriever(
    base_retriever: Optional[BaseRetriever] = None,
    num_queries: int = 3,
    temperature: float = 0.7,
    k: int = 5
) -> MultiQueryRetriever:
    """
    MultiQuery Retriever 생성

    Args:
        base_retriever: 기본 리트리버 (None이면 Dense Retriever 사용)
        num_queries: 생성할 쿼리 수
        temperature: LLM temperature
        k: 기본 리트리버에서 반환할 문서 수

    Returns:
        MultiQueryRetriever 인스턴스
    """
    # 기본 리트리버가 없으면 Dense Retriever 사용
    if base_retriever is None:
        base_retriever = get_dense_retriever(k=k)

    # MultiQuery Retriever 생성
    retriever = MultiQueryRetriever(
        base_retriever=base_retriever,
        num_queries=num_queries,
        temperature=temperature
    )

    return retriever


if __name__ == "__main__":
    # 테스트
    print("🔍 MultiQuery Retriever 테스트")
    print("=" * 60)

    # Retriever 생성 (Dense Retriever 기반)
    retriever = get_multiquery_retriever(num_queries=3, k=5)

    # 테스트 쿼리
    test_queries = [
        "RAG 시스템은 어떻게 동작하나요?",
        "임베딩 모델에 대해 알려주세요",
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)

        # invoke() 메서드 사용
        results = retriever.invoke(query)

        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.metadata.get('page_title', 'Unknown')}")
            print(f"    Section: {doc.metadata.get('section_title', 'N/A')}")
            print(f"    Content: {doc.page_content[:200]}...")

    print("\n" + "=" * 60)
    print("✅ MultiQuery Retriever 테스트 완료")
    print(f"\n리트리버 정보: {retriever.get_info()}")
