# Bad Case 심층 분석 리포트

## 개요

이 리포트는 RAG 시스템 평가에서 성능이 낮은 케이스들을 분석하여,
그 원인이 모델의 한계인지, 프롬프트 개선으로 해결 가능한지 파악하기 위해 작성되었습니다.

**분석 대상:**
- Dataset 1: OpenAI RRF MultiQuery (Top 8)
- Dataset 2: BGE-M3 RRF Reranker (k=6)

**Bad Case 정의:** Score ≤ 0.5인 케이스

---

## 1. OpenAI RRF MultiQuery (Top 8)

### 기본 통계
- **전체 케이스 수**: 210

### 점수 분포

#### Context Precision
- **평균**: 0.986 (±0.120)
- **범위**: [0.0, 1.0]
- **중앙값**: 1.000
- **점수 분포**:
  - Score 0.0: 1 (1.4%)
  - Score ≤0.5 (but >0): 0 (0.0%)
  - Score 1.0: 69 (98.6%)
  - **Total Bad Cases**: 1 (1.4%)

#### Context Recall
- **평균**: 0.986 (±0.120)
- **범위**: [0.0, 1.0]
- **중앙값**: 1.000
- **점수 분포**:
  - Score 0.0: 1 (1.4%)
  - Score ≤0.5 (but >0): 0 (0.0%)
  - Score 1.0: 69 (98.6%)
  - **Total Bad Cases**: 1 (1.4%)

#### Faithfulness
- **평균**: 0.799 (±0.231)
- **범위**: [0.0, 1.0]
- **중앙값**: 0.800
- **점수 분포**:
  - Score 0.0: 1 (1.4%)
  - Score ≤0.5 (but >0): 9 (12.9%)
  - Score 1.0: 29 (41.4%)
  - **Total Bad Cases**: 10 (14.3%)

### Bad Cases 상세 (Score ≤ 0.5)

#### Context Precision
- **Bad case 수**: 1 (1.4%)
- **평균 점수**: 0.000

#### Context Recall
- **Bad case 수**: 1 (1.4%)
- **평균 점수**: 0.000

#### Faithfulness
- **Bad case 수**: 10 (14.3%)
- **평균 점수**: 0.360

### Bad Case 패턴 분석

#### Query 특성
- **평균 길이**: 159.1 문자
- **길이 범위**: [43, 356]

**샘플 쿼리 (Bad Cases)**:

1. `The answer is an error about content filtering and does not address the question; the context about detection methods and results is not used in the answer.`

2. `"ì² ë²_§¸ ë¬¸ì_  ê°ì§ __ ê¸°ë_ ____ __ ¬í___ __,  ë¬¸ì_  _ê° ë³µí___ë¡ ____ __. '__ ê¸°ë_ ì¶__' °ì_ ê²°í_ ë°©ì_ ë¶_¦¬ ¤ë_´ì_ __, ê³ ì¶__ YOLO ê¸°ì_ __ ë° ëª©ì_ ê°__ ë³__ ëª__ë¡ _´ì_ __."`

3. `"Ragas ê¸°ë_¼ë_ __¸ì_ 46ê°_ ____ _ë¥ ì§___¤ë_ ë¬¸ì_  ê°ì§ _³´(__¸ì_ __, _ ì§__)ë¥ ¬í__ë¡ ë³µí___."`

4. `The first sentence describes a complex process (algorithmic selection of content based on user-item interactions and matrix factorization).`

5. `"ë¬¸ì_ 1  ë¬¸ì___  ê°ì§ ì£¼ì_ë¥ ¬í__ë§, ¼ë¦¬__ë¡ °ê²°__ __ ë³µì_±ì ì¤__ _´ë_. HMMê³ EMA ê¸°ë_ __ ëª¨ë_ _¸¡ ëª¨ë_ ê°__ ____ __ ì£¼ì_ ____ê° ±ê³µ__ë¡ __ ì£¼ì_ê° __ __."`


---

## 2. BGE-M3 RRF Reranker (k=6)

### 기본 통계
- **전체 케이스 수**: 219

### 점수 분포

#### Context Precision
- **평균**: 1.000 (±0.000)
- **범위**: [1.0, 1.0]
- **중앙값**: 1.000
- **점수 분포**:
  - Score 0.0: 0 (0.0%)
  - Score ≤0.5 (but >0): 0 (0.0%)
  - Score 1.0: 73 (100.0%)
  - **Total Bad Cases**: 0 (0.0%)

#### Context Recall
- **평균**: 1.000 (±0.000)
- **범위**: [1.0, 1.0]
- **중앙값**: 1.000
- **점수 분포**:
  - Score 0.0: 0 (0.0%)
  - Score ≤0.5 (but >0): 0 (0.0%)
  - Score 1.0: 73 (100.0%)
  - **Total Bad Cases**: 0 (0.0%)

#### Faithfulness
- **평균**: 0.813 (±0.243)
- **범위**: [0.1, 1.0]
- **중앙값**: 0.900
- **점수 분포**:
  - Score 0.0: 0 (0.0%)
  - Score ≤0.5 (but >0): 10 (13.7%)
  - Score 1.0: 35 (47.9%)
  - **Total Bad Cases**: 10 (13.7%)

### Bad Cases 상세 (Score ≤ 0.5)

#### Faithfulness
- **Bad case 수**: 10 (13.7%)
- **평균 점수**: 0.320

### Bad Case 패턴 분석

#### Query 특성
- **평균 길이**: 160.2 문자
- **길이 범위**: [69, 452]

**샘플 쿼리 (Bad Cases)**:

1. `" ë¬¸ì_ ____ ëª©ì_ê³ ì¶__ ë°°ê²½ ¤ë__©°, ¨ì_ ê°__ ì¤__¼ë_ ____ ë¬¸ì_ ë³µí__ _."`

2. `This sentence introduces the title of the report but does not convey informational content needing simplification. The complexity is minimal.`

3. `ë³ ____ ê³µê³µAX °ì_°ì_ __ ¥ì_ê³ __ __ë¥ ëª©ì_¼ë_ ì¶____.  ë¬¸ì_  ê°__ ì£¼ì_  ê°__ __ë¡ êµ¬ì_ ¨ì_ ë¬¸ì_´ë_. ë³µì__ __.`

4. `"The first sentence introduces the report title, which is a simple declarative statement."`

5. `"ë³ ____ ¤ì_ __ __ AI ê¸°ë_¼ë_ ë¶____, __  ì£¼ì_ ê°_²´ ë° ë§¥ë_ _³´ë¥ __ ì¶____, ¼ì_ ê¶¤ë_Â·ê³ ë°©í_Â·ê³ __  ¤í_ì¸ ê²½ê¸° __ ¸ë __ ____ë¡ ë¶____ ê²__ ëª©ì_¼ë_ ì¶____µë_. ->  ë¬¸ì_ ê¸¸ê_ ë³µí___, ¤ì_ê³...`


---

## 3. 데이터셋 비교

### 기본 정보
- **Dataset 1 크기**: 210
- **Dataset 2 크기**: 219

### 메트릭 비교

| Metric | Dataset 1 평균 | Dataset 2 평균 | Dataset 1 Bad Cases | Dataset 2 Bad Cases | 개선율 |
|--------|----------------|----------------|---------------------|---------------------|--------|
| Faithfulness | 0.799 | 0.813 | 10 | 10 | +1.7% |
| Context Recall | 0.986 | 1.000 | 1 | 0 | +1.4% |
| Context Precision | 0.986 | 1.000 | 1 | 0 | +1.4% |


---

## 4. 주요 발견사항

### 4.1 Bad Case 특성 분석


#### Dataset 1 상위 Bad Cases

다음은 점수가 가장 낮은 케이스들입니다:


**Case 1**: Context Precision = 0.00
- Query: `The answer is an error about content filtering and does not address the question; the context about detection methods and results is not used in the answer.`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q60`

**Case 2**: Context Recall = 0.00
- Query: `" ë¬¸ì_ Azure OpenAI content management policy °ë_ __ __ë§____ __ ë° ê´ __ ë¬¸ì_ __ë¥ ¬í_ __ ë©__ì§ë¡, ë³¸ë¬¸ ì£¼ì_ì§ ë§¥ë_(ì¶__ __, ê°_²´ _, ë¬¼ë¦¬ AI ±ê³¼ ê´¨ë_ ´ì_)ê³ ì§____ ê´¨ì_ __."`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q60`

**Case 3**: Faithfulness = 0.00
- Query: `Sentence is a headline and not a statement.`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q16`

**Case 4**: Faithfulness = 0.10
- Query: `"The first sentence is a content filter error message and does not contain a complex technical description. Its complexity is very low, and its main meaning can be stated simply."`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q60`

**Case 5**: Faithfulness = 0.30
- Query: `"The sentence contains technical terms and conveys multiple pieces of information in a complex way, so it must be broken down into simple statements without pronouns."`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q9`

**Case 6**: Faithfulness = 0.30
- Query: `" ë¬¸ì_ __ RAG ì±__ ëª©í_ êµ¬ì_ ê¸°ë_ __ë¥ ___©°, ì£¼ì_ _³´ê° ëª¨ë_ ëª____ë¡ ¸ê___ __ ë³µì__ _. __´ì_ 'ê°____ __ë©' 'ê¸°ë_¼ë_ êµ¬ì___µë_'¼ë_ êµ¬ë¬¸ ì§___ë¡ ´í_ ê°¥í_."`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q34`

**Case 7**: Faithfulness = 0.40
- Query: `"ì² ë²_§¸ ë¬¸ì_  ê°ì§ __ ê¸°ë_ ____ __ ¬í___ __,  ë¬¸ì_  _ê° ë³µí___ë¡ ____ __. '__ ê¸°ë_ ì¶__' °ì_ ê²°í_ ë°©ì_ ë¶_¦¬ ¤ë_´ì_ __, ê³ ì¶__ YOLO ê¸°ì_ __ ë° ëª©ì_ ê°__ ë³__ ëª__ë¡ _´ì_ __."`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q63`

**Case 8**: Faithfulness = 0.50
- Query: `The first sentence describes a complex process (algorithmic selection of content based on user-item interactions and matrix factorization).`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q64`

**Case 9**: Faithfulness = 0.50
- Query: `"This sentence provides an overview of the two main content recommendation system types as well as the application of advancement methods based on the system type. The sentence is moderately complex a...`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q46`

**Case 10**: Faithfulness = 0.50
- Query: `"Ragas ê¸°ë_¼ë_ __¸ì_ 46ê°_ ____ _ë¥ ì§___¤ë_ ë¬¸ì_  ê°ì§ _³´(__¸ì_ __, _ ì§__)ë¥ ¬í__ë¡ ë³µí___."`
- Trace: `eval_openai_rrf_multiquery_executive_report_v1_q54`


#### Dataset 2 상위 Bad Cases

다음은 점수가 가장 낮은 케이스들입니다:


**Case 1**: Faithfulness = 0.10
- Query: `This sentence introduces the title of the report but does not convey informational content needing simplification. The complexity is minimal.`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q18`

**Case 2**: Faithfulness = 0.10
- Query: `"The first sentence introduces the report title, which is a simple declarative statement."`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q16`

**Case 3**: Faithfulness = 0.20
- Query: `ë³ ____ ê³µê³µAX °ì_°ì_ __ ¥ì_ê³ __ __ë¥ ëª©ì_¼ë_ ì¶____.  ë¬¸ì_  ê°__ ì£¼ì_  ê°__ __ë¡ êµ¬ì_ ¨ì_ ë¬¸ì_´ë_. ë³µì__ __.`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q32`

**Case 4**: Faithfulness = 0.30
- Query: `"ë³ ____ ¤ì_ __ __ AI ê¸°ë_¼ë_ ë¶____, __  ì£¼ì_ ê°_²´ ë° ë§¥ë_ _³´ë¥ __ ì¶____, ¼ì_ ê¶¤ë_Â·ê³ ë°©í_Â·ê³ __  ¤í_ì¸ ê²½ê¸° __ ¸ë __ ____ë¡ ë¶____ ê²__ ëª©ì_¼ë_ ì¶____µë_. ->  ë¬¸ì_ ê¸¸ê_ ë³µí___, ¤ì_ê³...`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q19`

**Case 5**: Faithfulness = 0.30
- Query: `" ë¬¸ì_ ____ ëª©í_ ëª©ì_  ë¬¸ì_¼ë_ ¤ë__©°, ê¸°ì_ ¸ë¬í_ __ ë³µì__ __."`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q57`

**Case 6**: Faithfulness = 0.40
- Query: `" ë¬¸ì_ ____ ëª©ì_ê³ ì¶__ ë°°ê²½ ¤ë__©°, ¨ì_ ê°__ ì¤__¼ë_ ____ ë¬¸ì_ ë³µí__ _."`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q38`

**Case 7**: Faithfulness = 0.40
- Query: `Sentence: 'ë³ ____ ëª©ì_ __ ê´ ì§__  ì¦____ê³ __ µë _³µ__ RAG(Retrieval-Augmented Generation) ê¸°ë_ ì±__ ê°____ ê²____.' contains one main idea and is straightforward after rephrasing without pronouns...`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q34`

**Case 8**: Faithfulness = 0.40
- Query: `"ì² ë²_§¸ ë¬¸ì_ ____ , ëª©ì_, ë°©ì_ ë³µí___ë¡ ¬í___ __ë©,  ê°ì§ ì£¼ì_ _³´(____ ì¶__ ë°°ê²½, ëª©ì_, ë¹__ ëª¨ë_, ë¶__ ë°©ì_)  ê°__ ´ì_ ¨ì_ __."`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q44`

**Case 9**: Faithfulness = 0.50
- Query: `"ë³ ____ ëª©ì_ _²´ì½__ì¸(__ )  ë©_°ì_ ì¶__ ê°´ë_ë¥ ì²´ê___ë¡ ____, ì½__ì¸ ê²_·ì_ì²_·êë¦¬ì_ __  __ ___ °ì_ ì¶__ ë°  __¸ì_ë¥ _¦½__ ê²____.   ë¬¸ì_ ë³µí_ ëª©í_ë¥ ´ê_ __ë¯ë¡ ¬ë_ __ë¡ ë¶__  __."`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q10`

**Case 10**: Faithfulness = 0.50
- Query: `The sentence introduces the objective of the project but consists of multiple concepts and can be broken down for clarity.`
- Trace: `eval_bge-m3-rrf-reranker-k6_v1_q4`


---

## 5. 개선 방안

### 5.1 모델 한계 vs 프롬프트 개선 가능성

Bad case 분석을 통해 다음을 확인할 수 있습니다:

#### Score 0.0 케이스 (완전 실패)
- **특징**: 모델이 전혀 올바른 답변을 생성하지 못한 경우
- **가능한 원인**:
  - Retrieval 실패: 관련 문서를 찾지 못함
  - Context 부족: 검색된 문서에 필요한 정보가 없음
  - 생성 실패: 모델이 컨텍스트를 이해하지 못함
- **개선 방향**:
  - Retrieval 전략 개선 (Query 확장, 하이브리드 검색)
  - Knowledge Base 확장
  - 더 강력한 생성 모델 사용

#### Score 0.1~0.5 케이스 (부분 실패)
- **특징**: 부분적으로 올바른 답변을 생성
- **프롬프트 개선 가능성 높음**:
  - 더 명확한 지시사항
  - Few-shot 예시 추가
  - 출력 형식 명확화
  - Chain-of-Thought 프롬프팅

### 5.2 구체적 개선 전략

#### 1. Retrieval 개선
```
- Query 확장 (동의어, 관련어)
- 하이브리드 검색 (Dense + Sparse)
- Reranker 성능 개선
- Chunk 크기 및 Overlap 최적화
```

#### 2. 프롬프트 엔지니어링
```
- System prompt 개선
- Few-shot 예시 추가
- Role-based prompting
- 구조화된 출력 요구
```

#### 3. 평가 메트릭 검증
```
- 현재 메트릭이 실제 사용자 만족도를 반영하는가?
- 정성적 평가 병행 필요
- 도메인 전문가 리뷰
```

#### 4. 시스템 아키텍처
```
- Multi-hop reasoning 도입
- Self-correction 메커니즘
- Confidence score 기반 fallback
```

---

## 6. 다음 단계

### 단계별 액션 아이템

1. **Bad Cases 정성 분석** (우선순위: 높음)
   - [ ] `bad_cases_dataset1.csv` 파일의 상위 20개 케이스 수동 검토
   - [ ] `bad_cases_dataset2.csv` 파일의 상위 20개 케이스 수동 검토
   - [ ] 각 케이스를 다음으로 분류:
     - Retrieval 실패
     - Generation 실패
     - 평가 메트릭 문제

2. **원인별 개선 실험** (우선순위: 높음)
   - [ ] Retrieval 실패 케이스: Query 확장 테스트
   - [ ] Generation 실패 케이스: 프롬프트 A/B 테스트
   - [ ] 복합 원인 케이스: 하이브리드 접근

3. **재평가 및 검증** (우선순위: 중간)
   - [ ] 개선된 시스템으로 동일 케이스 재평가
   - [ ] 개선율 측정 및 문서화
   - [ ] 새로운 Bad Cases 분석

4. **시스템 개선 반영** (우선순위: 중간)
   - [ ] 검증된 개선사항 프로덕션 반영
   - [ ] 모니터링 대시보드 구축
   - [ ] 지속적 평가 파이프라인 구축

---

## 참고 자료

- **Bad Cases CSV**: 
  - Dataset 1: `analysis/bad_cases/bad_cases_dataset1.csv`
  - Dataset 2: `analysis/bad_cases/bad_cases_dataset2.csv`
- **상세 분석 JSON**: `analysis/bad_cases/bad_case_analysis.json`

---

*분석 일시: 2026-01-06*
*스크립트: `scripts/analyze_bad_cases.py`*
