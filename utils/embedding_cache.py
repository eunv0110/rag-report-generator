"""임베딩 캐시 유틸리티

평가 시 동일한 질문의 임베딩을 재사용하여 API 비용과 시간을 절약합니다.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime


class EmbeddingCache:
    """임베딩 결과를 파일 기반으로 캐싱하는 클래스"""

    def __init__(self, cache_dir: str = "data/evaluation/embedding_cache"):
        """
        Args:
            cache_dir: 캐시 파일을 저장할 디렉토리
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings.json"
        self.metadata_file = self.cache_dir / "metadata.json"

        # 캐시 로드
        self.cache: Dict[str, List[float]] = self._load_cache()
        self.metadata: Dict[str, Any] = self._load_metadata()

        # 통계
        self.hits = 0
        self.misses = 0

    def _load_cache(self) -> Dict[str, List[float]]:
        """캐시 파일 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 캐시 로드 실패 (새로 생성): {e}")
                return {}
        return {}

    def _load_metadata(self) -> Dict[str, Any]:
        """메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 메타데이터 로드 실패: {e}")
                return {}
        return {}

    def _generate_key(self, text: str, model: str = "default") -> str:
        """
        텍스트와 모델로부터 캐시 키 생성

        Args:
            text: 임베딩할 텍스트
            model: 임베딩 모델명

        Returns:
            SHA256 해시 기반 캐시 키
        """
        # 모델명 + 텍스트로 고유 키 생성
        key_input = f"{model}:{text}"
        return hashlib.sha256(key_input.encode('utf-8')).hexdigest()

    def get(self, text: str, model: str = "default") -> Optional[List[float]]:
        """
        캐시에서 임베딩 조회

        Args:
            text: 임베딩할 텍스트
            model: 임베딩 모델명

        Returns:
            캐시된 임베딩 벡터 또는 None
        """
        key = self._generate_key(text, model)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def set(self, text: str, embedding: List[float], model: str = "default"):
        """
        임베딩을 캐시에 저장

        Args:
            text: 임베딩할 텍스트
            embedding: 임베딩 벡터
            model: 임베딩 모델명
        """
        key = self._generate_key(text, model)
        self.cache[key] = embedding

        # 메타데이터 업데이트
        if key not in self.metadata:
            self.metadata[key] = {
                "text_preview": text[:100],  # 디버깅용
                "model": model,
                "created_at": datetime.now().isoformat(),
                "dimension": len(embedding)
            }

    def get_batch(
        self,
        texts: List[str],
        model: str = "default"
    ) -> tuple[List[Optional[List[float]]], List[int]]:
        """
        배치로 캐시 조회

        Args:
            texts: 임베딩할 텍스트 리스트
            model: 임베딩 모델명

        Returns:
            (임베딩 리스트, 캐시 미스 인덱스 리스트)
            - 임베딩 리스트: 캐시된 임베딩 또는 None
            - 미스 인덱스: 캐시에 없는 텍스트의 인덱스
        """
        embeddings = []
        miss_indices = []

        for i, text in enumerate(texts):
            embedding = self.get(text, model)
            embeddings.append(embedding)
            if embedding is None:
                miss_indices.append(i)

        return embeddings, miss_indices

    def set_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        model: str = "default"
    ):
        """
        배치로 캐시에 저장

        Args:
            texts: 임베딩할 텍스트 리스트
            embeddings: 임베딩 벡터 리스트
            model: 임베딩 모델명
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, model)

    def save(self):
        """캐시를 파일에 저장"""
        try:
            # 캐시 저장
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False)

            # 메타데이터 저장
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["total_entries"] = len(self.cache)
            self.metadata["stats"] = {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            }

            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"❌ 캐시 저장 실패: {e}")
            return False

    def clear(self):
        """캐시 초기화"""
        self.cache = {}
        self.metadata = {}
        self.hits = 0
        self.misses = 0

        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "total_entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size_mb": self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0
        }

    def print_stats(self):
        """캐시 통계 출력"""
        stats = self.get_stats()
        print("\n📊 임베딩 캐시 통계:")
        print(f"  - 총 캐시 항목: {stats['total_entries']}")
        print(f"  - 캐시 히트: {stats['hits']}")
        print(f"  - 캐시 미스: {stats['misses']}")
        print(f"  - 히트율: {stats['hit_rate']*100:.1f}%")
        print(f"  - 캐시 크기: {stats['cache_size_mb']:.2f} MB")


class CachedEmbedder:
    """임베딩 모델을 래핑하여 캐시 기능을 제공하는 클래스"""

    def __init__(self, embedder, cache: Optional[EmbeddingCache] = None, model_name: str = "default"):
        """
        Args:
            embedder: 원본 임베딩 모델 (embed_texts 메서드 필요)
            cache: 임베딩 캐시 객체 (None이면 새로 생성)
            model_name: 모델 이름 (캐시 키 생성에 사용)
        """
        self.embedder = embedder
        self.cache = cache or EmbeddingCache()
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        캐시를 사용하여 텍스트 임베딩 생성

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        # 1. 캐시에서 조회
        embeddings, miss_indices = self.cache.get_batch(texts, self.model_name)

        # 2. 캐시 미스 항목만 임베딩 생성
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            print(f"  💾 캐시 미스 {len(miss_indices)}개 항목 임베딩 생성 중...")

            new_embeddings = self.embedder.embed_texts(miss_texts)

            # 3. 새로 생성한 임베딩을 캐시에 저장
            self.cache.set_batch(miss_texts, new_embeddings, self.model_name)

            # 4. 결과 병합
            for i, embedding in zip(miss_indices, new_embeddings):
                embeddings[i] = embedding
        else:
            print(f"  ✅ 모든 항목({len(texts)}개)이 캐시에서 조회됨")

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        단일 쿼리 임베딩 (캐시 사용)

        Args:
            query: 임베딩할 쿼리

        Returns:
            임베딩 벡터
        """
        # 캐시 조회
        cached_embedding = self.cache.get(query, self.model_name)
        if cached_embedding is not None:
            return cached_embedding

        # 캐시 미스 - 새로 생성
        if hasattr(self.embedder, 'embed_query'):
            embedding = self.embedder.embed_query(query)
        else:
            embedding = self.embedder.embed_texts([query])[0]

        # 캐시에 저장
        self.cache.set(query, embedding, self.model_name)

        return embedding

    def save_cache(self):
        """캐시를 파일에 저장"""
        return self.cache.save()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return self.cache.get_stats()

    def print_stats(self):
        """캐시 통계 출력"""
        self.cache.print_stats()
