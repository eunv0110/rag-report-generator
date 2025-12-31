from typing import List
import time
from openai import OpenAI
from langchain_core.embeddings import Embeddings

class UpstageEmbedder(Embeddings):
    """
    Upstage API를 사용한 임베딩 생성기
    Upstage는 OpenAI 호환 API를 제공하므로 OpenAI 클라이언트를 사용합니다.
    """

    def __init__(
        self,
        model: str = "solar-embedding-1-large-passage",
        query_model: str = None,
        api_key: str = None,
        base_url: str = "https://api.upstage.ai/v1/solar",
        batch_size: int = 100,
        use_cache: bool = True,
        cache_dir: str = "data/evaluation/embedding_cache"
    ):
        """
        Args:
            model: Upstage 임베딩 모델명 (문서용)
                - solar-embedding-1-large-passage: 문서용 (1024 dim)
            query_model: Upstage 쿼리 임베딩 모델명 (검색 시 사용)
                - solar-embedding-1-large-query: 쿼리용 (1024 dim)
                - None인 경우 model과 동일한 모델 사용
            api_key: Upstage API 키
            base_url: Upstage API 엔드포인트
            batch_size: 배치 크기
            use_cache: 캐시 사용 여부
            cache_dir: 캐시 저장 경로
        """
        self.model = model  # 문서 임베딩용
        self.query_model = query_model if query_model else model  # 쿼리 임베딩용
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        print(f"📄 문서 임베딩 모델: {self.model}")
        print(f"🔍 쿼리 임베딩 모델: {self.query_model}")

        # 캐시 설정
        self.use_cache = use_cache
        self.cache = None
        if use_cache:
            try:
                from utils.embedding_cache import EmbeddingCache
                self.cache = EmbeddingCache(cache_dir=cache_dir)
                print(f"✅ 임베딩 캐시 활성화 (캐시 항목: {len(self.cache.cache)}개)")
            except Exception as e:
                print(f"⚠️ 캐시 초기화 실패, 캐시 없이 진행: {e}")
                self.use_cache = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        배치 임베딩 생성 (문서용)

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # 빈 텍스트 필터링
            batch = [text.strip() if text else " " for text in batch]

            # 재시도 로직 (최대 3번)
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )

                    # 응답 검증
                    if not response.data:
                        raise ValueError("빈 응답 받음")

                    embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(embeddings)
                    print(f"  → {min(i + self.batch_size, len(texts))}/{len(texts)} 임베딩 생성")

                    # API 레이트 리밋 방지
                    time.sleep(0.5)
                    break

                except Exception as e:
                    if attempt < 2:
                        wait_time = (attempt + 1) * 2
                        print(f"  ⚠️ 재시도 {attempt + 1}/3 (대기: {wait_time}초): {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ 배치 {i}~{i+len(batch)} 실패: {e}")
                        # 개별 텍스트로 재시도
                        print(f"  🔄 개별 텍스트로 재시도 중...")
                        batch_embeddings = self._embed_one_by_one(batch)
                        all_embeddings.extend(batch_embeddings)
                        break

        return all_embeddings

    def _embed_one_by_one(self, texts: List[str]) -> List[List[float]]:
        """
        개별 텍스트 임베딩 (폴백)

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text.strip() if text else " "]
                )
                embeddings.append(response.data[0].embedding)
                time.sleep(0.3)
            except Exception as e:
                print(f"    ⚠️ 텍스트 임베딩 실패: {text[:50]}... - {e}")
                # 실패한 텍스트는 제로 벡터로 대체 (1024 dim)
                embeddings.append([0.0] * 1024)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리 임베딩 (캐시 지원)
        쿼리 전용 모델(query_model)을 사용합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터
        """
        # 캐시에서 확인 (쿼리 모델 기준)
        if self.use_cache and self.cache:
            cached = self.cache.get(text, model=self.query_model)
            if cached is not None:
                return cached

        # 캐시 미스 - API 호출 (쿼리 모델 사용)
        try:
            response = self.client.embeddings.create(
                model=self.query_model,  # 쿼리 전용 모델 사용
                input=[text]
            )
            embedding = response.data[0].embedding

            # 캐시에 저장 (쿼리 모델 기준)
            if self.use_cache and self.cache:
                self.cache.set(text, embedding, model=self.query_model)

            return embedding
        except Exception as e:
            print(f"❌ 쿼리 임베딩 실패: {e}")
            raise

    def save_cache(self):
        """캐시를 파일에 저장"""
        if self.use_cache and self.cache:
            success = self.cache.save()
            if success:
                self.cache.print_stats()
            return success
        return False
