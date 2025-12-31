#!/usr/bin/env python3
"""Query Rewrite Retriever - LangChain 기반 쿼리 재작성을 통한 검색 개선"""

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


class QueryRewriteRetriever(BaseRetriever):
    """
    쿼리 재작성 리트리버 - LLM을 사용하여 쿼리 개선

    사용자의 모호하거나 불완전한 질문을 명확하고 구체적인 질문으로 재작성하여
    검색 품질을 향상시킵니다.
    """

    base_retriever: BaseRetriever
    """기본 리트리버 (Dense, BM25, Ensemble 등)"""

    temperature: float = 0.3
    """LLM temperature 설정 (낮을수록 더 결정적)"""

    llm_model: str = "azure_ai:gpt-4.1"
    """사용할 LLM 모델"""

    use_original_on_failure: bool = True
    """재작성 실패 시 원본 쿼리 사용 여부"""

    rewrite_template: Optional[str] = None
    """커스텀 재작성 프롬프트 템플릿"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        temperature: float = 0.3,
        llm_model: str = "azure_ai:gpt-4.1",
        use_original_on_failure: bool = True,
        rewrite_template: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            base_retriever: 기본 리트리버 (Dense, BM25, Ensemble 등)
            temperature: LLM temperature (기본값: 0.3, 낮을수록 일관성 있음)
            llm_model: 사용할 LLM 모델 (기본값: azure_ai:gpt-4.1)
            use_original_on_failure: 재작성 실패 시 원본 쿼리 사용 (기본값: True)
            rewrite_template: 커스텀 재작성 프롬프트 (None이면 기본 템플릿 사용)
        """
        super().__init__(
            base_retriever=base_retriever,
            temperature=temperature,
            llm_model=llm_model,
            use_original_on_failure=use_original_on_failure,
            rewrite_template=rewrite_template,
            **kwargs
        )

        # Azure OpenAI 설정
        if AZURE_AI_CREDENTIAL and AZURE_AI_ENDPOINT:
            os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
            os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

        print(f"✅ Query Rewrite Retriever 초기화 완료 (temperature={temperature})")

    def _get_default_template(self) -> str:
        """기본 재작성 프롬프트 템플릿"""
        return """당신은 검색 쿼리 최적화 전문가입니다. 사용자의 질문을 분석하여 더 효과적인 검색 결과를 얻을 수 있도록 질문을 재작성하세요.

원본 질문: {query}

재작성 지침:
1. 질문의 핵심 의도를 파악하고 유지하세요
2. 모호한 표현을 구체적으로 명확하게 만드세요
3. 불필요한 단어는 제거하고 핵심 키워드를 강조하세요
4. 검색에 도움이 되는 관련 용어나 동의어를 포함하세요
5. 한 문장으로 간결하게 작성하세요
6. 질문 형식을 유지하세요

재작성된 질문 (한 줄로만 작성):"""

    def rewrite_query(self, query: str) -> str:
        """
        LLM을 사용하여 쿼리 재작성

        Args:
            query: 원본 쿼리

        Returns:
            재작성된 쿼리 (실패 시 원본 쿼리 또는 빈 문자열)
        """
        # 프롬프트 템플릿 선택
        template = self.rewrite_template or self._get_default_template()
        prompt = template.format(query=query)

        try:
            model = init_chat_model(
                self.llm_model,
                temperature=self.temperature,
                max_completion_tokens=200
            )
            response = model.invoke(prompt)

            # 응답에서 재작성된 질문 추출
            rewritten = response.content.strip()

            # 여러 줄인 경우 첫 번째 줄만 사용
            if '\n' in rewritten:
                lines = [line.strip() for line in rewritten.split('\n') if line.strip()]
                rewritten = lines[0] if lines else query

            # 빈 응답이면 원본 사용
            if not rewritten:
                rewritten = query

            print(f"\n[Query Rewrite]")
            print(f"  원본: {query}")
            print(f"  재작성: {rewritten}")

            return rewritten

        except Exception as e:
            print(f"  ⚠️ 쿼리 재작성 실패: {e}")
            if self.use_original_on_failure:
                print(f"  → 원본 쿼리를 사용합니다.")
                return query
            else:
                print(f"  → 빈 쿼리를 반환합니다.")
                return ""

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        LangChain 호환 메서드: 쿼리 재작성 후 관련 문서 검색

        Args:
            query: 검색 쿼리
            run_manager: 콜백 매니저

        Returns:
            LangChain Document 리스트
        """
        # 쿼리 재작성
        rewritten_query = self.rewrite_query(query)

        # 빈 쿼리면 빈 결과 반환
        if not rewritten_query:
            return []

        # 재작성된 쿼리로 검색
        results = self.base_retriever.invoke(rewritten_query)

        return results

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        return {
            "name": "QueryRewriteRetriever",
            "type": self.__class__.__name__,
            "base_retriever": type(self.base_retriever).__name__,
            "temperature": self.temperature,
            "llm_model": self.llm_model,
            "use_original_on_failure": self.use_original_on_failure
        }


class HyDERetriever(BaseRetriever):
    """
    HyDE (Hypothetical Document Embeddings) Retriever

    쿼리에 대한 가상의 답변(문서)을 생성한 후, 그 답변을 임베딩하여 검색합니다.
    질문보다 답변이 실제 문서와 더 유사할 것이라는 가정에 기반합니다.
    """

    base_retriever: BaseRetriever
    """기본 리트리버"""

    temperature: float = 0.7
    """LLM temperature 설정"""

    llm_model: str = "azure_ai:gpt-4.1"
    """사용할 LLM 모델"""

    use_original_on_failure: bool = True
    """생성 실패 시 원본 쿼리 사용 여부"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        temperature: float = 0.7,
        llm_model: str = "azure_ai:gpt-4.1",
        use_original_on_failure: bool = True,
        **kwargs
    ):
        """
        Args:
            base_retriever: 기본 리트리버
            temperature: LLM temperature (기본값: 0.7)
            llm_model: 사용할 LLM 모델
            use_original_on_failure: 실패 시 원본 쿼리 사용
        """
        super().__init__(
            base_retriever=base_retriever,
            temperature=temperature,
            llm_model=llm_model,
            use_original_on_failure=use_original_on_failure,
            **kwargs
        )

        # Azure OpenAI 설정
        if AZURE_AI_CREDENTIAL and AZURE_AI_ENDPOINT:
            os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
            os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

        print(f"✅ HyDE Retriever 초기화 완료")

    def generate_hypothetical_document(self, query: str) -> str:
        """
        질문에 대한 가상의 답변 생성

        Args:
            query: 검색 쿼리

        Returns:
            생성된 가상 문서 (실패 시 원본 쿼리)
        """
        prompt = f"""다음 질문에 대해 간결하고 정확한 답변을 작성하세요. 실제 문서에서 발견될 수 있는 형태로 작성하세요.

질문: {query}

답변 (2-3문장으로 작성):"""

        try:
            model = init_chat_model(
                self.llm_model,
                temperature=self.temperature,
                max_completion_tokens=300
            )
            response = model.invoke(prompt)

            hypothetical_doc = response.content.strip()

            print(f"\n[HyDE - Hypothetical Document]")
            print(f"  질문: {query}")
            print(f"  가상 답변: {hypothetical_doc[:200]}...")

            return hypothetical_doc

        except Exception as e:
            print(f"  ⚠️ 가상 문서 생성 실패: {e}")
            if self.use_original_on_failure:
                print(f"  → 원본 쿼리를 사용합니다.")
                return query
            else:
                return ""

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        LangChain 호환 메서드: HyDE로 문서 검색

        Args:
            query: 검색 쿼리
            run_manager: 콜백 매니저

        Returns:
            LangChain Document 리스트
        """
        # 가상 문서 생성
        hypothetical_doc = self.generate_hypothetical_document(query)

        if not hypothetical_doc:
            return []

        # 가상 문서로 검색
        results = self.base_retriever.invoke(hypothetical_doc)

        return results

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        return {
            "name": "HyDERetriever",
            "type": self.__class__.__name__,
            "base_retriever": type(self.base_retriever).__name__,
            "temperature": self.temperature,
            "llm_model": self.llm_model
        }


def get_query_rewrite_retriever(
    base_retriever: Optional[BaseRetriever] = None,
    temperature: float = 0.3,
    k: int = 5,
    rewrite_template: Optional[str] = None
) -> QueryRewriteRetriever:
    """
    Query Rewrite Retriever 생성

    Args:
        base_retriever: 기본 리트리버 (None이면 Dense Retriever 사용)
        temperature: LLM temperature
        k: 기본 리트리버에서 반환할 문서 수
        rewrite_template: 커스텀 재작성 프롬프트

    Returns:
        QueryRewriteRetriever 인스턴스
    """
    if base_retriever is None:
        base_retriever = get_dense_retriever(k=k)

    retriever = QueryRewriteRetriever(
        base_retriever=base_retriever,
        temperature=temperature,
        rewrite_template=rewrite_template
    )

    return retriever


def get_hyde_retriever(
    base_retriever: Optional[BaseRetriever] = None,
    temperature: float = 0.7,
    k: int = 5
) -> HyDERetriever:
    """
    HyDE Retriever 생성

    Args:
        base_retriever: 기본 리트리버 (None이면 Dense Retriever 사용)
        temperature: LLM temperature
        k: 기본 리트리버에서 반환할 문서 수

    Returns:
        HyDERetriever 인스턴스
    """
    if base_retriever is None:
        base_retriever = get_dense_retriever(k=k)

    retriever = HyDERetriever(
        base_retriever=base_retriever,
        temperature=temperature
    )

    return retriever


if __name__ == "__main__":
    print("🔍 Query Rewrite Retriever 테스트")
    print("=" * 60)

    # 테스트 쿼리 (의도적으로 모호하거나 불완전한 질문)
    test_queries = [
        "RAG가 뭐야?",
        "임베딩 설명해줘",
    ]

    # Query Rewrite Retriever 테스트
    print("\n" + "=" * 60)
    print("📝 Query Rewrite Retriever")
    print("=" * 60)

    retriever = get_query_rewrite_retriever(k=5)

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-" * 60)

        results = retriever.invoke(query)

        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.metadata.get('page_title', 'Unknown')}")
            print(f"    Section: {doc.metadata.get('section_title', 'N/A')}")
            print(f"    Content: {doc.page_content[:150]}...")

    print("\n" + "=" * 60)
    print("✅ Query Rewrite Retriever 테스트 완료")
    print(f"\n리트리버 정보: {retriever.get_info()}")
