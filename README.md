# Multimodal RAG-based Enterprise Report Automation System

기업 보고서 자동화를 위한 멀티모달 RAG 시스템 최적화 연구

## 목차

- [프로젝트 개요](#-프로젝트-개요)
- [핵심 성과](#-핵심-성과)
- [시스템 아키텍처](#-시스템-아키텍처)
- [실험 방법론](#-실험-방법론)
- [최종 선정 결과](#-최종-선정-결과)
- [주요 발견사항](#-주요-발견사항)
- [성능 비교](#-성능-비교)
- [기술 스택](#-기술-스택)

## 프로젝트 개요

Notion 문서를 기반으로 텍스트와 이미지를 모두 처리하는 멀티모달 RAG 시스템을 구축하고, 보고서 유형별로 최적의 구성을 실험적으로 검증한 프로젝트입니다.

### 타겟 보고서 유형

| 보고서 유형 | 핵심 목표 | 우선순위 지표 |
|------------|----------|--------------|
| **주간 보고서** | 전체 상황 파악 (정보 누락 최소화) | Context Recall > Faithfulness |
| **보고용 보고서** | 신뢰 가능한 요약 (정확성 확보) | Faithfulness > Context Precision |

### 데이터셋

- **총 문서**: 89개 (평균 710자/청크, 783개 청크)
- **평가 데이터**: 70개
- **평가 프레임워크**: RAGAS + Langfuse

## 핵심 성과

### 주간 보고서 최적화
- **Context Recall**: 1.000 (완벽한 정보 검색)
- **Context Precision**: 1.000 (완벽한 정확도)
- **Faithfulness**: 0.813 (+14.8% 향상)
- **비용 효율**: GPT-4.1 대비 94.9% 성능 유지하면서 비용 1/22 절감

### 보고용 보고서 최적화
- **Context Precision**: 0.986
- **Context Recall**: 0.983
- **Faithfulness**: 0.829
- **품질**: 육안 평가 및 LLM-as-Judge 모두 상위권

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Notion Documents                        │
│              (Text + Images + Metadata)                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Multimodal Processing                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │ 
│  │ Auto Image   │  │ GPT-4 Image  │  │  Recursive   │       │
│  │   Download   │  │  Captioning  │  │Text Splitter │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vector Database (Qdrant)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Embeddings: BGE-M3 / OpenAI / Upstage / Qwen      │     │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Advanced Retrieval Strategies                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ RRF Ensemble │  │  MultiQuery  │  │  Reranker    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Generation                           │
│  GPT-4.1 / DeepSeek-V3.1 / Claude / Gemini                  │
└─────────────────────────────────────────────────────────────┘
```

## 실험 방법론

### 1단계: 기본 Retriever 선정

14종의 Retriever 전략 비교 평가:
- Dense Retrieval, BM25, RRF Ensemble
- MultiQuery, LongContext, TimeWeighted
- RAPTOR, Summary Retrieval 등

**결과**: RRF Ensemble이 일관되고 안정적인 성능으로 Base Retriever 확정

### 2단계: 임베딩 모델 비교

5개 임베딩 모델 테스트:
- OpenAI text-embedding-3-large
- Gemini embedding-001
- Qwen3-embedding-4b
- Upstage solar-embedding-1-large
- BAAI/BGE-M3

### 3단계: Retriever 조합 최적화

4개 후보 선정 후 K값(6, 8, 10, 12) 최적화 실험

### 4단계: Reranker 도입 효과 검증

3종 Reranker 테스트:
- Qwen3-Reranker-4B ⭐ (주간 보고서용 최종 선정)
- bge-reranker-v2-m3
- ko-reranker-8k

### 5단계: LLM 선정

7개 LLM 비교 평가:
- 비용/성능 분석
- 육안 평가 (실무 관점)
- LLM-as-Judge (3개 Judge 모델)

## 📊 최종 선정 결과

### 주간 보고서
```
Qwen3-Reranker-4B + BGE-M3 + RRF Ensemble (Top 6) + DeepSeek-V3.1
```

**선정 근거**:
- GPT-4.1 대비 94.9% 성능 유지
- 비용 1/22 절감 (주간 작성 빈도 고려)
- Context Precision/Recall 모두 1.000 달성
- Faithfulness 0.813 (Reranker 적용 후 +14.8% 향상)

### 보고용 보고서
```
OpenAI + RRF MultiQuery (Top 8) + GPT-4.1
```

**선정 근거**:
- 완성도와 설명력 중시 (임원 보고서 특성)
- Executive Summary → 주요 현황 → 이슈 → 권고사항 구조 최적화
- 육안 평가 87.5/100, LLM-as-Judge 상위권
- 비용과 응답 속도(10.8s)의 균형

## 💡 주요 발견사항

### 1. Reranker의 차별적 효과

- **주간 보고서**: Reranker 도입 시 **+14.8% 성능 향상**
  - 명확하고 단순한 정보 검색 중심 작업에 효과적
  - 불필요한 문서 제거, 정확한 근거 문맥 상위 배치

- **보고용 보고서**: Reranker 도입 시 오히려 **-11.5% 성능 저하**
  - 복합적 서술과 맥락 연결이 요구되는 작업
  - 개별 문서 단위 재정렬로 보조 정보 제거됨

### 2. Faithfulness 저하 원인 분석

**관찰된 문제**:
- 모델이 검색 문서보다 사전 학습 지식(parametric knowledge)에 의존
- 프롬프트 최적화(CoT, 규칙 기반 제약) 3차 시도 → 유의미한 개선 없음
- 주간: 0.83 → 0.76, 보고용: 0.82 → 0.65 오히려 저하

**원인 분석** (선행 연구 3편 분석):
1. 검색 문서와 모델 지식 충돌 시 내부 지식 우선 활용
2. 장문 문서(평균 710자, 783개 청크) 환경에서 attention 분산
3. 검색 전략보다는 **모델의 구조적 한계**로 판단

### 3. 평가 방식 간 차이

**주간 보고서**: 육안 평가 ≠ LLM-as-Judge
- 육안: 사용 맥락(주기, 독자, 목적) 고려
- LLM-as-Judge: 문서 자체의 절대적 품질 중심

**보고용 보고서**: 육안 평가 ≈ LLM-as-Judge
- 완성도와 정확성이 공통 핵심 기준

## 성능 비교

### LLM 비용/성능 비교 (주간 보고서)

| 모델 | Latency | 비용 (USD) | 육안 평가 | 특징 |
|------|---------|-----------|----------|------|
| **DeepSeek-V3.1** | 7.8s | **$0.0029** | 90/100 | **최종 선정** (가성비) |
| **GPT-4.1** | **5.6s** | $0.0558 | **91/100** | 속도/품질 최고 |
| GPT-5.1 | 13.9s | $0.1529 | 81/100 | Output 과다 |
| Claude Sonnet 4.5 | 15.9s | $0.0946 | 88/100 | 안정적 |
| Phi-4 | ~75s | $0.000004 | 82/100 | 실무 부적합 |

### LLM 비용/성능 비교 (보고용 보고서)

| 모델 | Latency | 비용 (USD) | 육안 평가 | 특징 |
|------|---------|-----------|----------|------|
| **GPT-4.1** | **10.8s** | **$0.115** | **87.5/100** | **최종 선정** |
| DeepSeek-V3.1 | 11.0s | $0.0053 | 79.0/100 | 간략함 |
| GPT-5.1 | 22.2s | $0.263 | 85.5/100 | 과도한 상세 |
| Claude Opus 4.5 | 12.5s | $0.219 | 64.0/100 | 보수적 |
| Phi-4 | ~123s | $0.000006 | 54.5/100 | 실무 부적합 |

### Retriever 성능 비교 (상위 5개)

| 기법 | Precision | Recall | Faithfulness | 특징 |
|------|-----------|--------|--------------|------|
| **RRF + MultiQuery** | 0.95 | **0.87** | **0.96** | 가장 균형잡힌 성능 ⭐ |
| MultiQuery + LongContext | **1.00** | **0.99** | 0.70 | Precision 최고 |
| RRF + TimeWeighted | 0.99 | **0.95** | 0.76 | Recall 강력 |
| RAPTOR + RRF | 0.96 | **0.97** | 0.81 | 안정적 |
| RRF Ensemble | 0.94 | 0.64 | 0.94 | 기준선 |

## 기술 스택

### Core Framework
- **Vector Database**: Qdrant
- **Evaluation**: RAGAS, Langfuse
- **Text Processing**: RecursiveTextSplitter

### Embeddings
- OpenAI text-embedding-3-large
- BAAI/BGE-M3
- Upstage solar-embedding-1-large
- Qwen3-embedding-4b
- Gemini embedding-001

### Rerankers
- Qwen3-Reranker-4B
- bge-reranker-v2-m3
- ko-reranker-8k

### LLM Providers
- Azure AI Foundry
- OpenRouter
- Direct API (OpenAI, Anthropic, DeepSeek)

### Retrieval Strategies
- RRF Ensemble
- MultiQuery Retrieval
- LongContext Retrieval
- TimeWeighted Retrieval
- RAPTOR

## 참고자료

### 평가 지표
- **Context Precision**: 검색된 문서의 정확도
- **Context Recall**: 필요한 정보의 검색 완성도
- **Faithfulness**: 생성 답변의 문서 근거 충실도

### 실험 규모
- 총 실험 조합: **140+ 구성**
- 평가 질문: **70개**
- LLM 비교: **7개 모델**
- Judge 모델: **3개** (GPT-5.1, Claude Opus 4.5, DeepSeek-V3.1)

## 결론

본 프로젝트는 보고서 유형별 특성을 고려한 RAG 시스템 최적화의 중요성을 실증적으로 검증했습니다.

**핵심 인사이트**:
1. **작업 특성에 따른 Reranker 효과 차이**: 단순 정보 검색 vs 복합 서술
2. **비용-품질 트레이드오프**: 작성 빈도를 고려한 LLM 선정 전략
3. **평가 방식의 다양성**: 실무 관점(육안)과 객관적 지표(LLM-as-Judge) 병행 필요
4. **모델 구조적 한계**: 장문 문서 환경에서의 Faithfulness 개선 한계
