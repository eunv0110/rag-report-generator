# BM25 검색 성능 평가 가이드

## 개요

이 가이드는 BM25 알고리즘을 사용한 검색 성능을 Ragas와 Langfuse로 평가하는 방법을 설명합니다.

## 구성 요소

### 1. BM25 Retriever
- **파일**: `retrievers/bm25_retriever.py`
- **기능**: Qdrant에 저장된 문서를 BM25 알고리즘으로 검색
- **특징**:
  - 한국어 토크나이저 지원 (jieba)
  - 배치 검색 지원
  - 스코어 기반 순위 정렬

### 2. 평가 데이터셋
- **파일**: `data/evaluation/sample_qa.json`
- **형식**:
```json
[
  {
    "question": "질문 내용",
    "ground_truth": "정답 내용",
    "metadata": {
      "category": "카테고리",
      "difficulty": "난이도"
    }
  }
]
```

### 3. 평가 스크립트
- **파일**: `scripts/evaluate_bm25.py`
- **기능**:
  - BM25로 문서 검색
  - Ragas로 성능 평가
  - Langfuse에 결과 로깅

## 설치

```bash
# 의존성 설치
cd /home/work/rag/Project/rag-report-generator
uv pip install -e .
```

## 사용 방법

### 1. 평가 데이터셋 준비

`data/evaluation/sample_qa.json` 파일을 수정하여 실제 평가 질문을 추가하세요.

```json
[
  {
    "question": "RAG 시스템이란 무엇인가요?",
    "ground_truth": "RAG는 Retrieval-Augmented Generation의 약자로, 검색을 통해 얻은 정보를 바탕으로 답변을 생성하는 시스템입니다."
  }
]
```

### 2. 벡터 DB 구축 (필요시)

```bash
python scripts/build_vectordb.py
```

### 3. BM25 평가 실행

```bash
# 기본 실행
python scripts/evaluate_bm25.py

# 옵션 지정
python scripts/evaluate_bm25.py \
  --dataset data/evaluation/sample_qa.json \
  --top-k 5 \
  --no-korean-tokenizer  # 한국어 토크나이저 비활성화
```

### 4. Langfuse에서 결과 확인

1. Langfuse 대시보드 접속 (https://cloud.langfuse.com)
2. Scores 탭에서 평가 결과 확인
3. Metrics:
   - `bm25_context_precision`: 검색된 문서의 정확도
   - `bm25_context_recall`: 관련 문서 검색 재현율

## Ragas 메트릭 설명

### Context Precision (컨텍스트 정밀도)
- 검색된 문서 중 실제로 관련 있는 문서의 비율
- 높을수록 불필요한 문서가 적음

### Context Recall (컨텍스트 재현율)
- 관련 있는 문서를 얼마나 잘 찾아냈는지
- 높을수록 필요한 정보를 놓치지 않음

### Faithfulness (충실도) - LLM 필요
- 생성된 답변이 컨텍스트에 근거하는지
- OpenAI API 키 필요

### Answer Relevancy (답변 관련성) - LLM 필요
- 생성된 답변이 질문과 얼마나 관련 있는지
- OpenAI API 키 필요

## 고급 설정

### LLM 기반 메트릭 추가

LLM이 필요한 메트릭을 사용하려면:

1. `.env` 파일에 OpenAI API 키 추가:
```bash
OPENAI_API_KEY=sk-...
```

2. `scripts/evaluate_bm25.py` 수정:
```python
metrics = [
    context_precision,
    context_recall,
    faithfulness,      # 추가
    answer_relevancy,  # 추가
]
```

### 한국어 토크나이저 커스터마이징

`retrievers/bm25_retriever.py`의 `_tokenize` 메서드를 수정하여 다른 토크나이저 사용:

```python
def _tokenize(self, text: str) -> List[str]:
    # KoNLPy, mecab 등 다른 한국어 토크나이저 사용
    from konlpy.tag import Okt
    okt = Okt()
    return okt.morphs(text.lower())
```

## 워크플로우

```
1. 평가 데이터 준비
   ↓
2. BM25 인덱스 구축 (Qdrant에서 자동 로드)
   ↓
3. 검색 수행
   ↓
4. Ragas 평가
   ↓
5. Langfuse 로깅
   ↓
6. 대시보드에서 결과 확인
```

## 트러블슈팅

### BM25 인덱스 구축 실패
- Qdrant 데이터가 있는지 확인: `ls qdrant_data/`
- Vector DB 재구축: `python scripts/build_vectordb.py --force`

### Ragas 평가 오류
- LLM 기반 메트릭 사용 시 API 키 확인
- 데이터셋 형식 확인 (question, ground_truth 필수)

### Langfuse 업로드 실패
- `.env` 파일의 Langfuse 키 확인
- 네트워크 연결 확인

## 다음 단계

1. **하이브리드 검색**: BM25 + 벡터 검색 결합
2. **리랭킹**: Cross-encoder로 재정렬
3. **A/B 테스팅**: 다양한 검색 전략 비교
4. **자동화**: CI/CD 파이프라인에 평가 추가

## 참고 자료

- [Ragas 문서](https://docs.ragas.io/)
- [Langfuse 문서](https://langfuse.com/docs)
- [BM25 알고리즘](https://en.wikipedia.org/wiki/Okapi_BM25)
