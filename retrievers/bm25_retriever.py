"""BM25 기반 검색 리트리버"""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from config.settings import QDRANT_COLLECTION
from .base_retriever import BaseRetriever, SearchResult
import jieba


class BM25Retriever(BaseRetriever):
    """BM25 기반 검색 엔진"""

    def __init__(self, qdrant_client: QdrantClient, use_korean_tokenizer: bool = True):
        """
        Args:
            qdrant_client: Qdrant 클라이언트
            use_korean_tokenizer: 한국어 토크나이저 사용 여부 (jieba 사용)
        """
        self.client = qdrant_client
        self.use_korean_tokenizer = use_korean_tokenizer
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.metadata = []

        self._load_corpus()

    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토크나이징"""
        if self.use_korean_tokenizer:
            # jieba를 사용한 한국어/중국어 토크나이징
            return list(jieba.cut(text.lower()))
        else:
            # 기본 공백 기반 토크나이징
            return text.lower().split()

    def _load_corpus(self):
        """Qdrant에서 전체 문서 로드 및 BM25 인덱스 구축"""
        print("📚 BM25 인덱스 구축 중...")

        # Qdrant에서 모든 문서 가져오기
        scroll_result = self.client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=True,
            with_vectors=False  # 벡터는 필요 없음
        )

        points = scroll_result[0]

        for point in points:
            payload = point.payload

            # combined_text를 corpus로 사용
            text = payload.get("combined_text", "")
            self.corpus.append(text)
            self.tokenized_corpus.append(self._tokenize(text))

            # 메타데이터 저장
            self.metadata.append({
                "chunk_id": payload.get("chunk_id"),
                "page_id": payload.get("page_id"),
                "text": payload.get("text"),
                "combined_text": text,
                "page_title": payload.get("page_title"),
                "section_title": payload.get("section_title"),
                "section_path": payload.get("section_path"),
                "has_image": payload.get("has_image", False),
                "image_descriptions": payload.get("image_descriptions", []),
                "properties": payload.get("properties", {})
            })

        # BM25 인덱스 구축
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"✅ BM25 인덱스 구축 완료: {len(self.corpus)}개 문서")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        BM25로 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수

        Returns:
            검색 결과 리스트
        """
        if not self.bm25:
            raise ValueError("BM25 인덱스가 구축되지 않았습니다")

        # 쿼리 토크나이징
        tokenized_query = self._tokenize(query)

        # BM25 스코어 계산
        scores = self.bm25.get_scores(tokenized_query)

        # 상위 k개 선택
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # 결과 구성
        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                page_id=meta["page_id"],
                text=meta["text"],
                combined_text=meta["combined_text"],
                page_title=meta["page_title"],
                section_title=meta["section_title"],
                section_path=meta["section_path"],
                score=float(scores[idx]),
                has_image=meta["has_image"],
                image_descriptions=meta["image_descriptions"],
                properties=meta["properties"]
            ))

        return results

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        여러 쿼리를 배치로 검색

        Args:
            queries: 검색 쿼리 리스트
            top_k: 각 쿼리당 반환할 상위 결과 개수

        Returns:
            각 쿼리별 검색 결과 리스트
        """
        return [self.search(query, top_k) for query in queries]

    @property
    def name(self) -> str:
        """리트리버 이름 반환"""
        tokenizer_name = "Korean" if self.use_korean_tokenizer else "Basic"
        return f"BM25_{tokenizer_name}"

    def get_info(self) -> Dict[str, Any]:
        """리트리버 정보 반환"""
        info = super().get_info()
        info.update({
            "use_korean_tokenizer": self.use_korean_tokenizer,
            "corpus_size": len(self.corpus)
        })
        return info
