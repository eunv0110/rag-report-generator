from typing import List
from openai import OpenAI
from models.base import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str, api_key: str, base_url: str, batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 생성"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            all_embeddings.extend([d.embedding for d in response.data])
            print(f"  → {min(i + self.batch_size, len(texts))}/{len(texts)} 임베딩 생성")
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """단일 쿼리 임베딩"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return response.data[0].embedding