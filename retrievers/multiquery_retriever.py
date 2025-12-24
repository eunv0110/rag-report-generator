"""MultiQuery Retriever - 쿼리 확장을 통한 검색 개선"""

from typing import List, Dict, Any, Optional
import os
from langchain.chat_models import init_chat_model
from .base_retriever import BaseRetriever, SearchResult
from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT


class MultiQueryRetriever(BaseRetriever):
    """
    멀티 쿼리 리트리버 - LLM을 사용하여 쿼리 확장

    하나의 질문을 여러 관점으로 재구성하여 검색 품질을 향상시킵니다.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        num_queries: int = 3,
        name: str = None,
        temperature: float = 0.7
    ):
        """
        Args:
            base_retriever: 기본 리트리버
            num_queries: 생성할 쿼리 수 (기본값: 3)
            name: 리트리버 이름
            temperature: LLM temperature
        """
        self.base_retriever = base_retriever
        self.num_queries = num_queries
        self.temperature = temperature
        self._name = name or f"multiquery_{base_retriever.name}"

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
        prompt = f"""당신은 정보 검색 전문가입니다. 주어진 질문에 대해 다양한 관점에서 {self.num_queries - 1}개의 유사한 질문을 생성하세요.

원본 질문: {query}

요구사항:
1. 각 질문은 원본 질문과 같은 의도를 가져야 합니다
2. 서로 다른 표현, 키워드, 관점을 사용하세요
3. 각 질문은 한 줄로 작성하세요
4. 번호나 특수 기호 없이 질문만 작성하세요

생성된 질문들 (각 줄에 하나씩):"""

        try:
            model = init_chat_model(
                "azure_ai:gpt-4.1",
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

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        멀티 쿼리 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수

        Returns:
            검색 결과 리스트
        """
        # 쿼리 확장
        queries = self.generate_queries(query)

        # 각 쿼리로 검색 수행
        all_results: Dict[str, SearchResult] = {}

        for q in queries:
            results = self.base_retriever.search(q, top_k=top_k * 2)

            for result in results:
                chunk_id = result.chunk_id

                # 이미 있는 경우, 더 높은 스코어로 업데이트
                if chunk_id not in all_results or result.score > all_results[chunk_id].score:
                    all_results[chunk_id] = result

        # 스코어 기준 정렬
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]

        print(f"  → 총 {len(all_results)}개 고유 문서, 상위 {len(sorted_results)}개 반환")

        return sorted_results

    @property
    def name(self) -> str:
        """리트리버 이름 반환"""
        return self._name

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "base_retriever": self.base_retriever.name,
            "num_queries": self.num_queries,
            "temperature": self.temperature
        }
