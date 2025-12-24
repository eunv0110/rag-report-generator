"""Retriever 추상 베이스 클래스"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    chunk_id: str
    page_id: str
    text: str
    combined_text: str
    page_title: str
    section_title: Optional[str]
    section_path: Optional[str]
    score: float
    has_image: bool = False
    image_descriptions: List[str] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        """초기화 후 처리"""
        if self.image_descriptions is None:
            self.image_descriptions = []
        if self.properties is None:
            self.properties = {}


class BaseRetriever(ABC):
    """모든 리트리버가 구현해야 하는 추상 베이스 클래스"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        단일 쿼리 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수

        Returns:
            검색 결과 리스트
        """
        pass

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        여러 쿼리를 배치로 검색

        기본 구현은 단순히 각 쿼리를 순회하면서 search를 호출
        서브클래스에서 최적화된 배치 처리를 구현할 수 있음

        Args:
            queries: 검색 쿼리 리스트
            top_k: 각 쿼리당 반환할 상위 결과 개수

        Returns:
            각 쿼리별 검색 결과 리스트
        """
        return [self.search(query, top_k) for query in queries]

    @property
    @abstractmethod
    def name(self) -> str:
        """리트리버 이름을 반환"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        리트리버 정보를 반환

        Returns:
            리트리버 정보 딕셔너리
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }
