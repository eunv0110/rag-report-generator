# RAG 평가 가이드

## 📋 목차
1. [평가 리트리버 종류](#평가-리트리버-종류)
2. [평가 실행 방법](#평가-실행-방법)
3. [결과 비교 방법](#결과-비교-방법)
4. [태그 시스템](#태그-시스템)

---

## 평가 리트리버 종류

### 🔹 1. RRF + MultiQuery (기본)
**파일:** `scripts/evaluate_rrf_multiquery.py`

**특징:**
- MultiQuery로 쿼리를 3개로 확장
- BM25 + Dense를 RRF로 결합
- 다양한 표현으로 검색 성능 향상

**실행:**
```bash
python scripts/evaluate_rrf_multiquery.py \
    --version v1 \
    --num-queries 3 \
    --top-k 10
```

**태그:** `multiquery`, `rrf`, `bm25`, `dense`

---

### 🔹 2. RRF + LongContext + MultiQuery
**파일:** `scripts/evaluate_rrf_multiquery.py --use-longcontext`

**특징:**
- MultiQuery로 쿼리 확장
- BM25 + Dense를 RRF로 결합
- LongContextReorder로 "Lost in the Middle" 완화
- 중요 문서를 처음/끝에 배치

**실행:**
```bash
python scripts/evaluate_rrf_multiquery.py \
    --version v1 \
    --num-queries 3 \
    --top-k 10 \
    --use-longcontext
```

**태그:** `multiquery`, `rrf`, `bm25`, `dense`, `longcontext`

---

### 🔹 3. RRF + LongContext (기존)
**파일:** `scripts/evaluate_rrf_longcontext.py`

**특징:**
- BM25 + Dense를 RRF로 결합
- LongContextReorder 적용
- MultiQuery 없음 (기존 방식)

**실행:**
```bash
python scripts/evaluate_rrf_longcontext.py \
    --version v5 \
    --top-k 10
```

**태그:** 개선 필요 (현재는 기본 태그만)

---

### 🔹 4. Ensemble RRF (비교 대상)
**파일:** `scripts/evaluate_ensemble.py`

**특징:**
- 기본 RRF 앙상블
- LongContext 없음
- MultiQuery 없음

**실행:**
```bash
python scripts/evaluate_ensemble.py \
    --version v2 \
    --top-k 10
```

---

## 평가 실행 방법

### 전체 평가 파이프라인

```bash
cd /home/work/rag/Project/rag-report-generator

# 1. RRF + MultiQuery 평가
python scripts/evaluate_rrf_multiquery.py --version v1

# 2. RRF + LongContext + MultiQuery 평가
python scripts/evaluate_rrf_multiquery.py --version v1 --use-longcontext

# 3. (선택) 기존 방식 비교
python scripts/evaluate_rrf_longcontext.py --version v5
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dataset` | 평가 데이터셋 경로 | `merged_qa_dataset.json` |
| `--top-k` | 검색할 문서 개수 | 10 |
| `--num-queries` | MultiQuery 생성 쿼리 수 | 3 |
| `--version` | 버전 태그 | v1 |
| `--no-cache` | 임베딩 캐시 비활성화 | False |
| `--use-longcontext` | LongContextReorder 사용 | False |

---

## 결과 비교 방법

### 1. Langfuse UI에서 비교

1. **Langfuse 접속**
   ```
   https://cloud.langfuse.com
   ```

2. **Traces 탭에서 필터링**
   - 태그로 필터: `multiquery`, `longcontext`, `rrf` 등
   - 버전으로 필터: `20251223_v1`

3. **Evaluations 탭에서 메트릭 확인**
   - Answer Relevance
   - Context Precision
   - Context Recall
   - Faithfulness

### 2. CSV Export 및 비교

#### 2.1 CSV 다운로드
1. Langfuse → Evaluations 탭
2. 필터 적용 (예: 특정 태그)
3. Export → CSV

#### 2.2 CSV 비교 스크립트
```bash
python scripts/compare_csv_results.py \
    data/langfuse/longcontext_1223_v1.csv \
    data/langfuse/multiquery_rrf_1223_v1.csv
```

**출력 예시:**
```
================================================================================
🏆 두 파일 비교 결과
================================================================================

[Context Recall]
통계                   longcontext_1223_v1.csv   multiquery_rrf_1223_v1.csv  차이
-------------------------------------------------------------------------------------
평균                   0.6433                    0.7102                      +0.0669 (+10.40%)
중앙값                  0.8000                    0.9000                      +0.1000

[Faithfulness]
통계                   longcontext_1223_v1.csv   multiquery_rrf_1223_v1.csv  차이
-------------------------------------------------------------------------------------
평균                   0.9089                    0.9521                      +0.0432 (+4.75%)
```

---

## 태그 시스템

### 태그 구조
```
[{retriever_name}_{date}_{version}, {date}_{version}, "evaluation", ...component_tags]
```

### 컴포넌트 태그

| 태그 | 의미 |
|------|------|
| `multiquery` | MultiQuery 사용 (쿼리 확장) |
| `rrf` | Reciprocal Rank Fusion |
| `bm25` | BM25 리트리버 |
| `dense` | Dense Vector 리트리버 |
| `longcontext` | LongContextReorder 적용 |

### 태그 검색 예시

**MultiQuery 효과 비교:**
```
비교 1: Tag = "rrf" AND NOT "multiquery"
비교 2: Tag = "rrf" AND "multiquery"
```

**LongContext 효과 비교:**
```
비교 1: Tag = "multiquery" AND "rrf" AND NOT "longcontext"
비교 2: Tag = "multiquery" AND "rrf" AND "longcontext"
```

**전체 조합 비교:**
```
1. 기본: rrf, bm25, dense
2. +LongContext: rrf, bm25, dense, longcontext
3. +MultiQuery: multiquery, rrf, bm25, dense
4. 전체: multiquery, rrf, bm25, dense, longcontext
```

---

## 📊 평가 메트릭

### RAGAS 메트릭

1. **Answer Relevance** (0-1)
   - 답변이 질문과 얼마나 관련있는가
   - 높을수록 좋음

2. **Context Precision** (0-1)
   - 검색된 컨텍스트가 얼마나 정확한가
   - 높을수록 좋음

3. **Context Recall** (0-1)
   - Ground truth를 찾는데 필요한 정보가 컨텍스트에 포함되었는가
   - 높을수록 좋음

4. **Faithfulness** (0-1)
   - 답변이 주어진 컨텍스트에 기반했는가 (환각 방지)
   - 높을수록 좋음

### 추가 메트릭

- **Retrieval Quality**: 검색 스코어 평균
- **Total Time (ms)**: 검색 + 답변 생성 시간
- **Num Contexts**: 검색된 문서 개수

---

## 🔍 문제 해결

### Q1. 임베딩 캐시 오류
```bash
# 캐시 비활성화
python scripts/evaluate_rrf_multiquery.py --no-cache
```

### Q2. Azure OpenAI 오류
```bash
# .env 파일 확인
cat .env | grep AZURE_AI
```

### Q3. Langfuse 연결 오류
```bash
# .env 파일 확인
cat .env | grep LANGFUSE
```

### Q4. 평가 데이터셋 없음
```bash
# 데이터셋 경로 확인
ls -la data/evaluation/merged_qa_dataset.json
```

---

## 📈 결과 해석

### 성능 향상 기대치

**MultiQuery 추가 시:**
- Context Recall: +5~15%
- Answer Relevance: +3~10%
- 시간: +20~50% (쿼리 생성 오버헤드)

**LongContext 추가 시:**
- Faithfulness: +2~5%
- Context Precision: 변화 적음
- 시간: +5% 미만 (재정렬만)

**전체 조합:**
- 최고 품질, 최대 시간
- Context Recall 중점 개선

---

## 💡 Best Practices

1. **버전 관리**
   - 같은 날 여러 실험: v1, v2, v3
   - 날짜 자동 추가: `20251223_v1`

2. **비교 분석**
   - 최소 2개 이상 비교
   - 동일 데이터셋 사용
   - 동일 top-k 설정

3. **태그 활용**
   - 컴포넌트별 태그 추가
   - 교차 분석 가능하게

4. **결과 저장**
   - CSV 다운로드 및 백업
   - 통계 JSON 파일 보관

---

## 📚 참고 자료

- [RETRIEVER_TAGS.md](RETRIEVER_TAGS.md) - 상세 태그 시스템
- [Langfuse 문서](https://langfuse.com/docs)
- [RAGAS 문서](https://docs.ragas.io/)
- [Lost in the Middle 논문](https://arxiv.org/abs/2307.03172)
