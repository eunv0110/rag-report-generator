from typing import List
import time
from openai import OpenAI
from langchain_core.embeddings import Embeddings

class OpenAIEmbedder(Embeddings):
    def __init__(self, model: str, api_key: str, base_url: str, batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 생성 (에러 핸들링 + 재시도)"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # ✅ 빈 텍스트 필터링
            batch = [text.strip() if text else " " for text in batch]
            
            # ✅ 재시도 로직 (최대 3번)
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    
                    # ✅ 응답 검증
                    if not response.data:
                        raise ValueError("빈 응답 받음")
                    
                    embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(embeddings)
                    print(f"  → {min(i + self.batch_size, len(texts))}/{len(texts)} 임베딩 생성")
                    
                    # ✅ API 레이트 리밋 방지
                    time.sleep(0.5)
                    break
                    
                except Exception as e:
                    if attempt < 2:
                        wait_time = (attempt + 1) * 2
                        print(f"  ⚠️ 재시도 {attempt + 1}/3 (대기: {wait_time}초): {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ 배치 {i}~{i+len(batch)} 실패: {e}")
                        # ✅ 개별 텍스트로 재시도
                        print(f"  🔄 개별 텍스트로 재시도 중...")
                        batch_embeddings = self._embed_one_by_one(batch)
                        all_embeddings.extend(batch_embeddings)
                        break
        
        return all_embeddings
    
    def _embed_one_by_one(self, texts: List[str]) -> List[List[float]]:
        """개별 텍스트 임베딩 (폴백)"""
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
                # ✅ 실패한 텍스트는 제로 벡터로 대체
                embeddings.append([0.0] * 3072)  # dimension 맞춰야 함
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 쿼리 임베딩 실패: {e}")
            raise