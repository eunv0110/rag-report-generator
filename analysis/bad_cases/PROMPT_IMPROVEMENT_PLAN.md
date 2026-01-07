# 프롬프트 개선 계획

## Bad Case 분석 결과 요약

### 주요 발견사항

1. **Faithfulness 점수가 가장 낮음**
   - Dataset 1 (OpenAI RRF MultiQuery): 14.3% Bad Cases (평균 0.36점)
   - Dataset 2 (BGE-M3 RRF Reranker): 13.7% Bad Cases (평균 0.32점)

2. **Context Precision/Recall은 거의 완벽**
   - 문서 검색 자체는 매우 잘 작동 (98.6~100%)
   - 문제는 생성 단계에서 발생

3. **Bad Case 패턴**
   - Content filtering 에러
   - 복잡한 기술 설명 처리 실패
   - Context를 제대로 활용하지 못함

## 프롬프트 개선 전략

### 1. Faithfulness 강화 전략

#### 현재 문제점
- 시스템 프롬프트에서 faithfulness 강조가 약함
- 답변 생성 프롬프트에 context 기반 답변 지시가 명확하지 않음

#### 개선 방안
1. **시스템 프롬프트 강화**
   - Context 충실성을 최우선 원칙으로 명시
   - "검색된 문서에 없는 내용은 절대 추가하지 마세요" 강조
   - Hallucination 방지 지침 추가

2. **답변 생성 프롬프트 개선**
   - 각 정보의 출처를 문서 번호로 명시하도록 유도
   - "검색된 문서에서 직접 인용" 지시 추가
   - 불확실한 정보는 작성하지 않도록 명확히 지시

3. **구조화된 출력 형식**
   - 각 섹션마다 "검색된 문서 기반" 확인
   - 정보 누락보다 정확성을 우선

### 2. Weekly Report vs Executive Report 차별화

#### Weekly Report (운영팀)
- **목적**: 상세한 운영 현황 파악
- **강조점**: 완전성(Completeness), 포괄성(Coverage)
- **개선 방향**: 
  - 더 상세한 정보 포함 유도
  - 모든 관련 데이터 누락 없이 포함

#### Executive Report (의사결정)
- **목적**: 전략적 의사결정
- **강조점**: 정확성(Accuracy), 신뢰성(Reliability), 간결성
- **개선 방향**:
  - 검증된 정보만 사용하도록 더 강화
  - 불확실한 정보 제외 명확화
  - 핵심 인사이트 중심으로 재구성

## 개선된 프롬프트 버전

### Version: v2 (Faithfulness Enhanced)

#### 주요 변경사항
1. Faithfulness 원칙을 시스템 프롬프트 최상단에 배치
2. Context 기반 답변 지시를 더 명확하고 강하게 표현
3. 예시 추가로 원하는 답변 형식 명확화
4. Hallucination 방지 장치 강화

#### 측정 메트릭
- **주요 지표**: Faithfulness 점수 개선 (목표: Bad Cases 14% → 7% 이하)
- **부수 효과**: Context Recall/Precision 유지 또는 소폭 개선
- **Trade-off**: Answer Completeness가 약간 감소할 수 있음 (허용)

## 평가 계획

### 1. A/B 테스트 구조
- **Baseline (v1)**: 현재 프롬프트
- **Improved (v2)**: Faithfulness 강화 프롬프트

### 2. 평가 데이터셋
- `merged_qa_dataset.json` 사용
- 동일한 retriever 조합으로 테스트

### 3. 비교 지표
- Faithfulness Score (주요)
- Context Recall
- Context Precision
- Answer Relevance
- Bad Case 수 및 비율

### 4. 실험 설정
```bash
# Baseline (v1)
python evaluate_reranker.py --mode evaluate --report-type executive --version v1

# Improved (v2) 
python evaluate_reranker.py --mode evaluate --report-type executive --version v2_faithfulness_enhanced
```

## 성공 기준

### 필수 기준 (Must-have)
1. **Faithfulness Bad Cases < 7%** (현재 14% → 50% 감소)
2. **Faithfulness Mean Score > 0.85** (현재 0.80)
3. **Context Recall/Precision 유지** (>0.95)

### 우수 기준 (Nice-to-have)
1. Faithfulness Bad Cases < 5%
2. Faithfulness Mean Score > 0.90
3. Overall Performance 향상

## 다음 단계

1. ✅ Bad Case 분석 완료
2. ⏳ 프롬프트 v2 작성
3. ⏳ Weekly Report 프롬프트 개선
4. ⏳ Executive Report 프롬프트 개선
5. ⏳ 재평가 스크립트 작성
6. ⏳ A/B 테스트 실행
7. ⏳ 결과 비교 및 분석

---

*작성일: 2026-01-06*
*기반 문서: analysis/bad_cases/BAD_CASE_ANALYSIS.md*
