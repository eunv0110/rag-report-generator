# 평가 데이터셋 생성 가이드

## 개요

RAG 시스템의 성능을 평가하기 위한 QA(Question-Answer) 데이터셋을 생성하는 방법을 설명합니다.

## 3가지 생성 방법

### 1. 자동 생성 (추천) ⚡

Notion 문서의 제목과 헤딩을 기반으로 자동으로 QA를 생성합니다.

```bash
# 기본 실행 (20개 생성)
python scripts/generate_eval_dataset.py --method auto

# 샘플 수 지정
python scripts/generate_eval_dataset.py --method auto --num-samples 50

# 출력 파일 지정
python scripts/generate_eval_dataset.py --method auto \
  --num-samples 30 \
  --output data/evaluation/my_qa.json
```

**장점:**
- 빠르고 쉬움
- API 키 불필요
- 대량 생성 가능

**단점:**
- 질문의 품질이 낮을 수 있음
- 단순한 패턴의 질문

**생성되는 질문 예시:**
```json
{
  "question": "Notion API에 대해 설명해주세요.",
  "ground_truth": "Notion API는 프로그래밍 방식으로 Notion 데이터베이스와 페이지에 접근할 수 있게 해주는 RESTful API입니다...",
  "metadata": {
    "category": "page_summary",
    "difficulty": "easy",
    "source": "auto_title"
  }
}
```

### 2. LLM 기반 생성 🤖

GPT-4 등의 LLM을 사용하여 고품질 QA를 생성합니다.

```bash
# Azure OpenAI 사용 (기본)
python scripts/generate_eval_dataset.py --method llm --num-samples 10

# OpenAI 사용
python scripts/generate_eval_dataset.py \
  --method llm \
  --llm-provider openai \
  --num-samples 10
```

**사전 준비:**
```bash
# .env 파일에 API 키 설정
OPENAI_API_KEY=sk-...
# 또는
AZURE_AI_CREDENTIAL=...
AZURE_AI_ENDPOINT=...
```

**장점:**
- 고품질 질문/답변
- 다양한 패턴
- 컨텍스트를 잘 이해한 질문

**단점:**
- API 비용 발생
- 생성 속도가 느림

**생성되는 질문 예시:**
```json
{
  "question": "Notion API를 사용할 때 rate limit은 어떻게 처리해야 하나요?",
  "ground_truth": "Notion API는 초당 3개의 요청 제한이 있습니다. 이를 처리하기 위해서는 retry 로직과 exponential backoff를 구현해야 합니다...",
  "metadata": {
    "category": "llm_generated",
    "difficulty": "medium",
    "source": "llm_azure"
  }
}
```

### 3. 수동 작성 템플릿 ✍️

직접 작성할 수 있는 템플릿을 생성합니다.

```bash
# 템플릿 생성
python scripts/generate_eval_dataset.py --method manual --num-samples 10
```

**생성된 파일:** `data/evaluation/manual_qa_template.json`

```json
[
  {
    "id": "qa_1",
    "question": "[TODO: Notion API에 대한 질문을 작성하세요]",
    "ground_truth": "[TODO: 정답을 작성하세요]",
    "context_page_id": "abc123",
    "context_page_title": "Notion API 가이드",
    "content_preview": "Notion API는...",
    "metadata": {
      "category": "[TODO: 카테고리]",
      "difficulty": "medium",
      "source": "manual"
    }
  }
]
```

**작성 방법:**
1. 템플릿 파일 열기
2. `[TODO]` 부분을 실제 내용으로 수정
3. `content_preview`를 참고하여 작성
4. 저장

**장점:**
- 가장 정확한 QA
- 도메인 전문 지식 반영
- 평가 목적에 맞게 커스터마이징

**단점:**
- 시간이 많이 소요
- 수작업 필요

## 데이터셋 검증

생성한 데이터셋이 올바른지 검증합니다.

```bash
# 기본 검증
python scripts/validate_eval_dataset.py data/evaluation/auto_qa_from_headings.json

# 엄격 모드 (경고도 에러로 처리)
python scripts/validate_eval_dataset.py \
  data/evaluation/manual_qa_template.json \
  --strict
```

**검증 항목:**
- 필수 필드 존재 여부 (`question`, `ground_truth`)
- TODO 마커 잔여 확인
- 질문/답변 길이 체크
- 메타데이터 분석

**출력 예시:**
```
📊 검증 결과
============================================================

총 항목 수: 20
유효한 항목: 18
메타데이터 포함: 20

카테고리: page_summary, section_explanation
난이도: easy, medium

⚠️  경고 (2개):
  - qa_5: 답변이 너무 짧음 (15자)
  - qa_8: 질문에 TODO 마커가 남아있음

✅ 데이터셋이 유효합니다!
```

## 평가 실행

생성한 데이터셋으로 BM25 성능을 평가합니다.

```bash
# 평가 실행
python scripts/evaluate_bm25.py \
  --dataset data/evaluation/auto_qa_from_headings.json \
  --top-k 5
```

## 추천 워크플로우

### 빠른 프로토타입
```bash
# 1. 자동 생성 (50개)
python scripts/generate_eval_dataset.py --method auto --num-samples 50

# 2. 검증
python scripts/validate_eval_dataset.py data/evaluation/auto_qa_from_headings.json

# 3. 평가
python scripts/evaluate_bm25.py --dataset data/evaluation/auto_qa_from_headings.json
```

### 고품질 평가 (API 비용 발생)
```bash
# 1. LLM으로 생성 (10개)
python scripts/generate_eval_dataset.py --method llm --num-samples 10

# 2. 검증
python scripts/validate_eval_dataset.py data/evaluation/llm_generated_qa_azure.json

# 3. 평가
python scripts/evaluate_bm25.py --dataset data/evaluation/llm_generated_qa_azure.json
```

### 정밀 평가
```bash
# 1. 수동 템플릿 생성
python scripts/generate_eval_dataset.py --method manual --num-samples 20

# 2. 수동 작성 (에디터로 파일 열어서 [TODO] 부분 수정)
# data/evaluation/manual_qa_template.json 편집

# 3. 검증 (엄격 모드)
python scripts/validate_eval_dataset.py \
  data/evaluation/manual_qa_template.json \
  --strict

# 4. 평가
python scripts/evaluate_bm25.py --dataset data/evaluation/manual_qa_template.json
```

### 하이브리드 (추천)
```bash
# 1. 자동으로 많이 생성 + LLM으로 소수 생성
python scripts/generate_eval_dataset.py --method auto --num-samples 40
python scripts/generate_eval_dataset.py --method llm --num-samples 10

# 2. 두 파일 병합 (JSON 편집기 사용)
# auto_qa_from_headings.json + llm_generated_qa_azure.json
# → combined_qa.json

# 3. 평가
python scripts/evaluate_bm25.py --dataset data/evaluation/combined_qa.json
```

## 데이터셋 형식

### 최소 형식
```json
[
  {
    "question": "질문 내용",
    "ground_truth": "정답 내용"
  }
]
```

### 권장 형식
```json
[
  {
    "id": "qa_1",
    "question": "질문 내용",
    "ground_truth": "정답 내용",
    "context_page_id": "notion_page_id",
    "metadata": {
      "category": "카테고리",
      "difficulty": "easy|medium|hard",
      "source": "생성 방법"
    }
  }
]
```

## 팁

### 1. 다양한 난이도 포함
```python
# 쉬운 질문
"Notion API란 무엇인가요?"

# 중간 질문
"Notion API로 데이터베이스 쿼리를 어떻게 필터링하나요?"

# 어려운 질문
"Notion API의 rate limit을 고려한 대량 데이터 동기화 전략은?"
```

### 2. 다양한 카테고리
- `factual`: 사실 확인 질문
- `how_to`: 방법론 질문
- `concept`: 개념 설명 질문
- `comparison`: 비교 질문
- `troubleshooting`: 문제 해결 질문

### 3. 평가 목적별 데이터셋
```bash
# 검색 정확도 평가
# → 명확한 정답이 있는 팩트 중심 질문

# 컨텍스트 재현율 평가
# → 여러 문서에 걸쳐 있는 정보가 필요한 질문

# 답변 품질 평가
# → 복잡한 설명이 필요한 개념 질문
```

## 문제 해결

### Notion 데이터가 없음
```bash
# Vector DB 먼저 구축
python scripts/build_vectordb.py
```

### LLM API 오류
```bash
# API 키 확인
echo $OPENAI_API_KEY
# 또는
echo $AZURE_AI_CREDENTIAL

# .env 파일 확인
cat .env | grep API
```

### 생성된 질문 품질이 낮음
1. LLM 방법 사용 (고품질)
2. 자동 생성 후 수동 편집
3. 템플릿 방법으로 직접 작성

## 다음 단계

1. **A/B 테스팅**: 여러 검색 전략 비교
2. **메트릭 분석**: 카테고리/난이도별 성능 분석
3. **지속적 개선**: 실패한 케이스를 데이터셋에 추가
4. **자동화**: CI/CD에 평가 파이프라인 통합
