# 검색 성능 평가 빠른 시작 가이드

## 1분 안에 시작하기

### 1. 의존성 설치

```bash
cd /home/work/rag/Project/rag-report-generator
source .venv/bin/activate
uv pip install -e .
```

### 2. 평가 데이터 생성

#### 방법 A: 자동 생성 (가장 빠름, API 불필요)

```bash
python scripts/generate_eval_dataset.py --method auto --num-samples 20
```

#### 방법 B: LLM 생성 (고품질, Azure AI 사용)

```bash
# .env 파일에 Azure AI 키가 있어야 함
python scripts/generate_eval_dataset.py --method llm --num-samples 10 --llm-provider azure
```

생성된 파일:
- 자동: `data/evaluation/auto_qa_from_headings.json`
- LLM: `data/evaluation/llm_generated_qa_azure.json`

### 3. 데이터셋 검증

```bash
python scripts/validate_eval_dataset.py data/evaluation/auto_qa_from_headings.json
```

### 4. 검색 성능 평가

#### 옵션 A: Langfuse 기반 평가 (추천) 🏆

**LLM 자동 평가 + 시각화 대시보드 제공**

```bash
python scripts/evaluate_with_langfuse.py --dataset data/evaluation/llm_generated_qa_azure.json
```

**특징**:
- ✅ Langfuse 내장 RAGAS 프롬프트 활용
- ✅ Context Precision, Context Recall 자동 측정
- ✅ Azure AI / OpenRouter 지원
- ✅ 실시간 대시보드 확인

**결과**: https://cloud.langfuse.com 에서 즉시 확인

---

#### 옵션 B: 기존 RAGAS 직접 평가 (연구용)

**BM25만 평가**:
```bash
python scripts/evaluate_bm25.py --dataset data/evaluation/auto_qa_from_headings.json
```

**Dense (벡터 검색)만 평가**:
```bash
python scripts/evaluate_dense.py --dataset data/evaluation/auto_qa_from_headings.json
```

**BM25 vs Dense 비교**:
```bash
python scripts/compare_retrievers.py --dataset data/evaluation/auto_qa_from_headings.json
```

⚠️ **주의**: RAGAS 직접 사용 시 OpenAI API 키 필요

### 5. Langfuse에서 결과 확인

https://cloud.langfuse.com 접속하여 평가 결과 확인

**LLM 평가 설정**:
1. Settings → Evaluators → Add Evaluator
2. Azure AI 또는 OpenRouter 설정
3. Evaluations → Run Evaluation
4. Context Precision, Context Recall 자동 측정

---

## 주요 옵션

### 평가 데이터 생성

```bash
# 자동 생성 (50개)
python scripts/generate_eval_dataset.py --method auto --num-samples 50

# LLM 생성 (Azure, 10개)
python scripts/generate_eval_dataset.py --method llm --num-samples 10 --llm-provider azure

# 수동 템플릿
python scripts/generate_eval_dataset.py --method manual --num-samples 15
```

### 검색 성능 평가

#### BM25 평가

```bash
# 기본 평가
python scripts/evaluate_bm25.py --dataset data/evaluation/auto_qa_from_headings.json

# Top-K 조정
python scripts/evaluate_bm25.py --dataset data/evaluation/llm_generated_qa_azure.json --top-k 10

# 한국어 토크나이저 비활성화
python scripts/evaluate_bm25.py --dataset data/evaluation/auto_qa_from_headings.json --no-korean-tokenizer
```

#### Dense 평가

```bash
# 기본 평가
python scripts/evaluate_dense.py --dataset data/evaluation/auto_qa_from_headings.json

# Top-K 조정
python scripts/evaluate_dense.py --dataset data/evaluation/llm_generated_qa_azure.json --top-k 10
```

#### BM25 vs Dense 비교

```bash
# 기본 비교
python scripts/compare_retrievers.py --dataset data/evaluation/auto_qa_from_headings.json

# Top-K 조정
python scripts/compare_retrievers.py --dataset data/evaluation/llm_generated_qa_azure.json --top-k 10
```

---

## 생성된 QA 예시

### 자동 생성
```json
{
  "question": "Notion API에 대해 설명해주세요.",
  "ground_truth": "Notion API는...",
  "metadata": {
    "category": "page_summary",
    "difficulty": "easy"
  }
}
```

### LLM 생성 (고품질)
```json
{
  "question": "이 테니스 모멘텀 프로젝트에서 HMM과 EMA는 각각 어떤 역할을 하며...",
  "ground_truth": "이 프로젝트에서는 논문에서 제안된 HMM + EMA 기반...",
  "metadata": {
    "category": "llm_generated",
    "difficulty": "medium"
  }
}
```

---

## 트러블슈팅

### Notion 데이터 없음
```bash
python scripts/build_vectordb.py
```

### Azure AI API 에러
- `.env` 파일의 `AZURE_AI_CREDENTIAL` 확인
- 모델명 확인: `gpt-5.1` 사용 중

### 한국어 토크나이저 오류
```bash
uv pip install jieba
```

---

## 다음 단계

1. **자동 생성으로 대량 데이터 생성** (50개)
2. **LLM으로 고품질 데이터 추가** (10개)
3. **두 파일 병합** → `combined_qa.json`
4. **평가 실행 및 Langfuse 확인**
5. **성능 분석 및 개선**

---

## 상세 문서

- [평가 데이터셋 생성 가이드](EVALUATION_DATASET_GUIDE.md)
- [BM25 평가 가이드](BM25_EVALUATION_GUIDE.md)
