# Retriever Tagging System

각 리트리버의 Langfuse 태그 구조를 명확하게 정의합니다.

## 📌 태그 구조

모든 평가 trace는 다음과 같은 태그 구조를 따릅니다:

```
[{retriever_name}_{version_tag}, {version_tag}, "evaluation", ...component_tags]
```

## 🔖 리트리버별 태그

### 1. **RRF + MultiQuery** (기본)
```bash
python scripts/evaluate_rrf_multiquery.py --version v1
```

**태그:**
- `multiquery_ensemble_rrf_YYYYMMDD_v1`
- `YYYYMMDD_v1`
- `evaluation`
- `multiquery` ← MultiQuery 사용
- `rrf` ← RRF 앙상블
- `bm25` ← BM25 리트리버
- `dense` ← Dense 리트리버

**설명:** MultiQuery로 쿼리 확장 → RRF로 BM25 + Dense 결합

---

### 2. **RRF + LongContext + MultiQuery**
```bash
python scripts/evaluate_rrf_multiquery.py --version v1 --use-longcontext
```

**태그:**
- `multiquery_ensemble_rrf_longcontext_YYYYMMDD_v1`
- `YYYYMMDD_v1`
- `evaluation`
- `multiquery` ← MultiQuery 사용
- `rrf` ← RRF 앙상블
- `bm25` ← BM25 리트리버
- `dense` ← Dense 리트리버
- `longcontext` ← LongContextReorder 적용

**설명:** MultiQuery로 쿼리 확장 → RRF로 BM25 + Dense 결합 → LongContextReorder로 재정렬

---

### 3. **RRF + LongContext** (기존)
```bash
python scripts/evaluate_rrf_longcontext.py --version v5
```

**태그:**
- `ensemble_rrf_longcontext_YYYYMMDD_v5`
- `YYYYMMDD_v5`
- `evaluation`

**권장 개선:**
추가 컴포넌트 태그를 포함하도록 업데이트:
- `rrf`
- `bm25`
- `dense`
- `longcontext`

---

### 4. **Ensemble (RRF 기본)** (비교 대상)
```bash
python scripts/evaluate_ensemble.py --version v2
```

**태그:**
- `ensemble_bm25_dense_rrf_YYYYMMDD_v2`
- `YYYYMMDD_v2`
- `evaluation`

**권장 개선:**
추가 컴포넌트 태그:
- `rrf`
- `bm25`
- `dense`

---

## 🎯 태그 검색 예시

### Langfuse UI에서 태그로 필터링

1. **MultiQuery를 사용한 모든 평가**
   ```
   Tag: "multiquery"
   ```

2. **LongContext를 사용한 모든 평가**
   ```
   Tag: "longcontext"
   ```

3. **RRF 앙상블을 사용한 모든 평가**
   ```
   Tag: "rrf"
   ```

4. **특정 버전의 MultiQuery + RRF + LongContext**
   ```
   Tag: "multiquery_ensemble_rrf_longcontext_20251223_v1"
   ```

---

## 📊 비교 분석 시나리오

### 시나리오 1: MultiQuery 효과 분석
```
비교 대상:
- Tag: "ensemble_rrf_longcontext" AND NOT "multiquery"
- Tag: "multiquery" AND "ensemble_rrf_longcontext"
```

### 시나리오 2: LongContext 효과 분석
```
비교 대상:
- Tag: "multiquery" AND "rrf" AND NOT "longcontext"
- Tag: "multiquery" AND "rrf" AND "longcontext"
```

### 시나리오 3: 전체 파이프라인 비교
```
비교 대상:
- 기본 RRF (rrf, bm25, dense)
- RRF + LongContext (rrf, bm25, dense, longcontext)
- RRF + MultiQuery (multiquery, rrf, bm25, dense)
- 전체 조합 (multiquery, rrf, bm25, dense, longcontext)
```

---

## 🔧 CSV Export 후 비교

평가 완료 후 Langfuse에서 CSV를 다운로드하여 비교:

```bash
# CSV 파일 비교
python scripts/compare_csv_results.py \
    data/langfuse/rrf_basic_1223_v1.csv \
    data/langfuse/rrf_multiquery_1223_v1.csv

python scripts/compare_csv_results.py \
    data/langfuse/rrf_longcontext_1223_v1.csv \
    data/langfuse/rrf_multiquery_longcontext_1223_v1.csv
```

---

## 📝 메타데이터 구조

각 trace의 metadata에는 다음 정보가 포함됩니다:

```json
{
  "retriever": "multiquery_ensemble_rrf_longcontext",
  "version": "20251223_v1",
  "retriever_components": ["multiquery", "rrf", "bm25", "dense", "longcontext"],
  "total_time_ms": 1234.56,
  "num_retrieved_contexts": 10,
  "question_id": 42,
  "category": "technical",
  "difficulty": "hard"
}
```

---

## 🚀 빠른 시작

### 1. 평가 실행
```bash
# RRF + MultiQuery
python scripts/evaluate_rrf_multiquery.py --version v1

# RRF + LongContext + MultiQuery
python scripts/evaluate_rrf_multiquery.py --version v1 --use-longcontext
```

### 2. Langfuse에서 결과 확인
- URL: https://cloud.langfuse.com
- Traces 탭 → 태그로 필터링

### 3. CSV 다운로드 및 비교
- Evaluations 탭 → Export → CSV
- `compare_csv_results.py`로 비교

---

## 💡 Best Practices

1. **버전 관리**
   - 날짜별로 버전 자동 생성 (YYYYMMDD)
   - 같은 날 여러 실험: v1, v2, v3 사용

2. **명확한 네이밍**
   - 리트리버 이름에 컴포넌트 포함
   - 예: `multiquery_ensemble_rrf_longcontext`

3. **컴포넌트 태그 활용**
   - 각 기법별로 태그 추가
   - 교차 분석 가능

4. **메타데이터 활용**
   - 카테고리, 난이도별 성능 분석
   - 응답 시간, 컨텍스트 수 추적
