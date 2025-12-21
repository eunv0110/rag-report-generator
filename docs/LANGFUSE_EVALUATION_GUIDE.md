# Langfuse 기반 검색 성능 평가 가이드

## 개요

Langfuse는 내부에 RAGAS 프롬프트를 통합하고 있어, 별도로 RAGAS 파이프라인을 구성하지 않아도 LLM 기반 평가가 가능합니다. 이 가이드는 Langfuse 트레이싱을 활용한 간단하고 효율적인 검색 성능 평가 방법을 안내합니다.

## 주요 특징

- ✅ **RAGAS 파이프라인 불필요**: Langfuse 내장 평가 기능 활용
- ✅ **LLM 기반 자동 평가**: Context Precision, Context Recall 등 자동 측정
- ✅ **Azure AI / OpenRouter 지원**: OpenAI API 없이도 사용 가능
- ✅ **비교 분석**: BM25 vs Dense(Vector) 검색 성능 비교
- ✅ **실시간 모니터링**: Langfuse 대시보드에서 즉시 확인

## 빠른 시작

### 1. 평가 데이터 준비

LLM 기반 고품질 QA 데이터 생성:

```bash
python scripts/generate_eval_dataset.py --method llm --num-samples 10 --llm-provider azure
```

### 2. 검색 성능 평가 실행

```bash
python scripts/evaluate_with_langfuse.py --dataset data/evaluation/llm_generated_qa_azure.json --top-k 5
```

### 3. Langfuse 대시보드 확인

https://cloud.langfuse.com 접속하여 결과 확인

## 평가 결과 예시

### 성능 비교 (10개 질문 기준)

| Retriever    | Avg Results | Avg Time (ms) |
|--------------|-------------|---------------|
| BM25         | 5.00        | **10.43**     |
| Dense_Vector | 5.00        | 1120.42       |

### 주요 인사이트

1. **속도**: BM25가 Dense보다 **100배 이상 빠름** (10ms vs 1120ms)
2. **정확도**: Langfuse UI에서 Context Precision/Recall 자동 평가 가능
3. **활용**:
   - 실시간 응답이 중요한 경우 → BM25 추천
   - 의미적 유사도가 중요한 경우 → Dense 추천
   - 최고 성능 → 하이브리드(BM25 + Dense 앙상블)

## Langfuse에서 LLM 기반 평가 설정

### Step 1: Langfuse 대시보드 접속

1. https://cloud.langfuse.com 로그인
2. 프로젝트 선택

### Step 2: Evaluation LLM 설정

#### 옵션 A: Azure AI 사용

1. **Settings** → **Evaluators** 이동
2. **Add Evaluator** 클릭
3. 다음 정보 입력:
   ```
   Provider: Azure OpenAI
   Deployment Name: gpt-5.1
   API Key: [.env의 AZURE_AI_CREDENTIAL]
   Endpoint: https://ddokai-resource.services.ai.azure.com/models/
   ```

#### 옵션 B: OpenRouter 사용

1. **Settings** → **Evaluators** 이동
2. **Add Evaluator** 클릭
3. 다음 정보 입력:
   ```
   Provider: OpenAI (Compatible)
   Model: openai/gpt-4o-mini
   API Key: [.env의 OPENROUTER_API_KEY]
   Base URL: https://openrouter.ai/api/v1
   ```

### Step 3: 자동 평가 활성화

1. **Evaluations** 탭 이동
2. 평가 메트릭 선택:
   - ✅ Context Precision (검색된 컨텍스트의 정밀도)
   - ✅ Context Recall (ground truth 대비 재현율)
   - ✅ Faithfulness (답변의 충실도)
   - ✅ Answer Relevancy (답변의 관련성)

3. **Run Evaluation** 클릭

### Step 4: 결과 확인

1. **Traces** 탭에서 각 검색 결과 확인
2. **Scores** 탭에서 평가 점수 확인
3. **Analytics**에서 통계 및 트렌드 분석

## 고급 사용법

### 커스텀 평가 메트릭 추가

스크립트 수정하여 추가 점수 기록:

```python
# scripts/evaluate_with_langfuse.py 내부
langfuse.create_score(
    trace_id=event.trace_id,
    name="custom_relevance",
    value=0.95,
    comment="Custom relevance score"
)
```

### 배치 평가

여러 데이터셋을 한 번에 평가:

```bash
for dataset in data/evaluation/*.json; do
    python scripts/evaluate_with_langfuse.py --dataset "$dataset"
done
```

### Top-K 튜닝

다양한 Top-K 값으로 실험:

```bash
for k in 3 5 10; do
    python scripts/evaluate_with_langfuse.py --dataset data/evaluation/llm_generated_qa_azure.json --top-k $k
done
```

## 주요 명령어 정리

### 평가 데이터 생성

```bash
# 자동 생성 (빠름, API 불필요)
python scripts/generate_eval_dataset.py --method auto --num-samples 20

# LLM 생성 (고품질, Azure AI)
python scripts/generate_eval_dataset.py --method llm --num-samples 10 --llm-provider azure
```

### 검색 성능 평가

```bash
# 기본 평가
python scripts/evaluate_with_langfuse.py --dataset data/evaluation/llm_generated_qa_azure.json

# Top-K 조정
python scripts/evaluate_with_langfuse.py --dataset data/evaluation/llm_generated_qa_azure.json --top-k 10

# BM25 한국어 토크나이저 비활성화
python scripts/evaluate_with_langfuse.py --dataset data/evaluation/llm_generated_qa_azure.json --no-korean-tokenizer
```

## 트러블슈팅

### 1. Langfuse 연결 실패

**증상**: `Authentication error: Langfuse client initialized without public_key`

**해결**:
```bash
# .env 파일 확인
cat .env | grep LANGFUSE

# 환경변수 설정 확인
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY
```

### 2. 평가 데이터 없음

**증상**: `FileNotFoundError: data/evaluation/llm_generated_qa_azure.json`

**해결**:
```bash
# 평가 데이터 생성
python scripts/generate_eval_dataset.py --method llm --num-samples 10 --llm-provider azure
```

### 3. Azure AI API 오류

**증상**: `Unavailable model: gpt-4o-mini`

**해결**:
- `.env` 파일에서 모델명을 `gpt-5.1`로 변경
- Azure AI 크레덴셜 확인

## Langfuse vs RAGAS 직접 사용 비교

| 항목 | Langfuse 통합 | RAGAS 직접 사용 |
|------|---------------|-----------------|
| **설정 복잡도** | ⭐ 간단 (UI 설정) | ⭐⭐⭐ 복잡 (코드 작성) |
| **LLM 제공자** | Azure AI, OpenRouter 등 | OpenAI API 필수 |
| **평가 자동화** | ✅ 자동 | ❌ 수동 구현 필요 |
| **시각화** | ✅ 대시보드 제공 | ❌ 별도 구현 필요 |
| **추천 상황** | 프로덕션, 지속적 모니터링 | 연구, 일회성 실험 |

## 다음 단계

1. ✅ **평가 데이터 생성 완료** (10개 LLM 기반 QA)
2. ✅ **검색 성능 평가 완료** (BM25 vs Dense)
3. 🔄 **Langfuse 대시보드에서 LLM 평가 설정**
4. 📊 **Context Precision/Recall 자동 평가 확인**
5. 🚀 **하이브리드 리트리버 성능 평가**

## 참고 자료

- [Langfuse 공식 문서](https://langfuse.com/docs)
- [RAGAS 평가 메트릭](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
- [빠른 시작 가이드](QUICK_START.md)
- [BM25 평가 가이드](BM25_EVALUATION_GUIDE.md)
