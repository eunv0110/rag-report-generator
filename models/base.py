from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 임베딩으로 변환"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """단일 쿼리를 임베딩으로 변환"""
        pass

class BaseVisionModel(ABC):
    @abstractmethod
    def describe_image(self, image_path: str, context: Dict[str, Any]) -> str:
        """이미지 설명 생성"""
        pass