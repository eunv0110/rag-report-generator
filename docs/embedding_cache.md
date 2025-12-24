# 임베딩 캐시 (Embedding Cache)

## 개요

임베딩 캐시는 동일한 텍스트에 대해 반복적으로 임베딩 API를 호출하는 것을 방지하여 **비용과 시간을 절약**하는 기능입니다.

## 주요 특징

### 1. 자동 캐싱
- 한 번 생성된 임베딩은 자동으로 캐시에 저장됩니다.
- 같은 텍스트에 대해 두 번째 요청부터는 캐시에서 즉시 반환됩니다.

### 2. 파일 기반 영속성
- 캐시는 JSON 파일로 저장되어 프로그램 재시작 후에도 유지됩니다.
- 기본 저장 위치: `data/evaluation/embedding_cache/`

### 3. 모델별 관리
- 서로 다른 임베딩 모델의 결과를 구분하여 캐싱합니다.
- SHA256 해시를 사용하여 고유 키를 생성합니다.

### 4. 통계 추적
- 캐시 히트/미스 비율을 추적하여 효율성을 측정할 수 있습니다.

## 사용 방법

### 평가 스크립트에서 사용

기본적으로 `evaluate_with_langfuse.py`에서 자동으로 활성화됩니다:

```bash
# 캐시 활성화 (기본값)
python scripts/evaluate_with_langfuse.py --retrievers dense

# 캐시 비활성화
python scripts/evaluate_with_langfuse.py --retrievers dense --no-cache
```

### 프로그래밍 방식 사용

#### 1. 기본 캐시 사용

```python
from utils.embedding_cache import EmbeddingCache

# 캐시 생성
cache = EmbeddingCache(cache_dir="data/evaluation/embedding_cache")

# 임베딩 저장
text = "안녕하세요"
embedding = [0.1, 0.2, 0.3, ...]
cache.set(text, embedding, model="text-embedding-3-large")

# 임베딩 조회
cached_embedding = cache.get(text, model="text-embedding-3-large")

# 캐시 저장
cache.save()

# 통계 출력
cache.print_stats()
```

#### 2. CachedEmbedder 사용 (권장)

```python
from utils.embedding_cache import CachedEmbedder, EmbeddingCache
from models.embeddings.factory import get_embedder

# 원본 임베더
base_embedder = get_embedder()

# 캐시와 함께 래핑
cache = EmbeddingCache()
cached_embedder = CachedEmbedder(
    embedder=base_embedder,
    cache=cache,
    model_name="text-embedding-3-large"
)

# 일반적인 임베더처럼 사용
texts = ["텍스트 1", "텍스트 2", "텍스트 3"]
embeddings = cached_embedder.embed_texts(texts)  # 첫 호출: API 호출
embeddings2 = cached_embedder.embed_texts(texts)  # 두 번째: 캐시에서 조회

# 캐시 저장
cached_embedder.save_cache()

# 통계 출력
cached_embedder.print_stats()
```

## 캐시 파일 구조

### embeddings.json
```json
{
  "hash_key_1": [0.1, 0.2, 0.3, ...],
  "hash_key_2": [0.4, 0.5, 0.6, ...],
  ...
}
```

### metadata.json
```json
{
  "hash_key_1": {
    "text_preview": "텍스트 일부...",
    "model": "text-embedding-3-large",
    "created_at": "2025-12-23T10:30:00",
    "dimension": 3072
  },
  "last_updated": "2025-12-23T10:30:00",
  "total_entries": 150,
  "stats": {
    "hits": 450,
    "misses": 150,
    "hit_rate": 0.75
  }
}
```

## 비용 절감 효과

### 예시: 100개 질문으로 3개 리트리버 평가

**캐시 없이:**
- Dense 리트리버: 100회 임베딩 호출
- RRF Ensemble: 100회 임베딩 호출 (Dense 포함)
- **총 200회 API 호출**

**캐시 사용:**
- 첫 리트리버 (Dense): 100회 호출 (캐시에 저장)
- 두 번째 리트리버 (RRF): 0회 호출 (캐시에서 조회)
- **총 100회 API 호출 (50% 절감)**

### 실제 비용 계산

OpenAI text-embedding-3-large 기준:
- $0.00013 / 1K tokens
- 평균 질문 길이: 50 tokens

```
캐시 없이: 200 호출 × 50 tokens × $0.00013/1K = $0.0013
캐시 사용: 100 호출 × 50 tokens × $0.00013/1K = $0.00065
절감액: $0.00065 (50%)
```

반복 평가 시 절감률은 더욱 높아집니다:
- 2회 평가: 75% 절감
- 3회 평가: 83% 절감
- 10회 평가: 95% 절감

## 캐시 관리

### 캐시 초기화
```python
cache = EmbeddingCache()
cache.clear()  # 모든 캐시 삭제
```

### 캐시 통계 확인
```bash
# 평가 실행 후 자동으로 출력됨
📊 임베딩 캐시 통계:
  - 총 캐시 항목: 100
  - 캐시 히트: 200
  - 캐시 미스: 100
  - 히트율: 66.7%
  - 캐시 크기: 2.34 MB
```

## 주의사항

1. **임베딩 모델 변경 시**: 모델이 변경되면 새로운 캐시 키가 생성되므로 기존 캐시가 사용되지 않습니다.

2. **디스크 공간**: 캐시 파일은 임베딩 차원에 따라 크기가 증가합니다.
   - text-embedding-3-large (3072차원): 약 25KB per 1000 embeddings

3. **캐시 무효화**: 텍스트가 조금이라도 변경되면 다른 해시가 생성되어 새로운 캐시 항목이 생성됩니다.

## 테스트

캐시 기능 테스트:
```bash
python scripts/test_embedding_cache.py
```

## 관련 파일

- 구현: [utils/embedding_cache.py](../utils/embedding_cache.py)
- 테스트: [scripts/test_embedding_cache.py](../scripts/test_embedding_cache.py)
- 사용 예시: [scripts/evaluate_with_langfuse.py](../scripts/evaluate_with_langfuse.py)
