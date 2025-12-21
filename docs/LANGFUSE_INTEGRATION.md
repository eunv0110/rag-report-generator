# Langfuse 통합 가이드

이 문서는 RAG Report Generator에 Langfuse를 통합하는 방법을 설명합니다.

## 설정

### 1. 환경 변수 설정

`.env` 파일에 다음 변수를 추가하세요:

```bash
# Langfuse 설정
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com  # 또는 self-hosted URL
```

### 2. Langfuse 계정 생성

1. [Langfuse Cloud](https://cloud.langfuse.com)에 가입
2. 새 프로젝트 생성
3. Settings > API Keys에서 Public Key와 Secret Key 복사
4. `.env` 파일에 키 추가

## 사용법

### Vector DB 구축 시 자동 트레이싱

```bash
# 일반 실행 (Langfuse 트레이싱 포함)
python scripts/build_vectordb.py

# 전체 재생성
python scripts/build_vectordb.py --force

# 제한된 페이지만 처리
python scripts/build_vectordb.py --limit 10
```

### 커스텀 트레이싱

다른 스크립트에서 Langfuse를 사용하려면:

```python
from utils.langfuse_utils import trace_operation

# 컨텍스트 매니저 사용
with trace_operation(
    name="my_operation",
    metadata={"custom_field": "value"}
) as trace:
    # 작업 수행
    result = do_something()

    # 하위 스팬 추가
    if trace:
        span = trace.span(name="sub_operation")
        sub_result = do_sub_task()
        span.end(metadata={"result": sub_result})
```

## 트레이싱되는 정보

### Vector DB 구축 프로세스

`build_vectordb.py` 실행 시 다음 정보가 트레이싱됩니다:

#### 메인 트레이스: `vectordb_build`
- **메타데이터**:
  - `force_recreate`: 전체 재생성 여부
  - `check_updates`: 업데이트 체크 여부
  - `limit`: 페이지 수 제한
  - `db_name`: 데이터베이스 이름

#### 스팬들:

1. **model_initialization**
   - 임베딩 모델 및 비전 모델 로딩

2. **data_collection**
   - Notion 데이터 수집
   - 메타데이터: `mode` (force_recreate/incremental)
   - 출력: `total_pages_to_index`

3. **chunking** (전체 재생성 시)
   - 페이지를 청크로 분할
   - 출력: `total_chunks`

4. **embedding_generation** (전체 재생성 시)
   - 임베딩 벡터 생성
   - 출력: `num_embeddings`, `embedding_dimension`

5. **qdrant_storage** (전체 재생성 시)
   - Qdrant에 저장

6. **incremental_update** (증분 업데이트 시)
   - 변경된 페이지만 업데이트
   - 출력: `pages_updated`, `total_chunks`

## Langfuse 대시보드에서 확인하기

1. [Langfuse Cloud](https://cloud.langfuse.com) 로그인
2. 프로젝트 선택
3. **Traces** 탭에서 실행 기록 확인
4. 각 트레이스를 클릭하여 상세 정보 확인:
   - 실행 시간
   - 각 단계별 소요 시간
   - 메타데이터 (처리된 페이지 수, 생성된 청크 수 등)
   - 에러 발생 시 스택 트레이스

## 트레이싱 비활성화

Langfuse 키를 설정하지 않으면 자동으로 비활성화됩니다:

```bash
# .env에서 Langfuse 키를 제거하거나 주석 처리
# LANGFUSE_PUBLIC_KEY=...
# LANGFUSE_SECRET_KEY=...
```

실행 시 다음 메시지가 표시됩니다:
```
⚠️  Langfuse 키가 설정되지 않음 (트레이싱 비활성화)
```

## 고급 사용법

### 사용자 ID 추적

```python
with trace_operation(
    name="user_query",
    metadata={"query": "검색어"},
    user_id="user@example.com"
) as trace:
    results = search(query)
```

### 에러 추적

Langfuse는 자동으로 예외를 캡처하지만, 명시적으로 로깅할 수도 있습니다:

```python
with trace_operation(name="risky_operation") as trace:
    try:
        risky_function()
    except Exception as e:
        if trace:
            trace.event(
                name="error",
                metadata={"error_type": type(e).__name__, "message": str(e)}
            )
        raise
```

## 문제 해결

### 트레이스가 표시되지 않는 경우

1. API 키가 올바른지 확인
2. 네트워크 연결 확인 (특히 self-hosted인 경우)
3. `langfuse_client.flush()`가 호출되는지 확인 (trace_operation이 자동으로 처리)

### 성능 영향

Langfuse는 비동기적으로 데이터를 전송하므로 성능 영향이 최소화됩니다. 하지만 프로덕션에서는:
- 샘플링 사용 고려
- 필요한 메타데이터만 전송
- 네트워크 대역폭 모니터링

## 참고 자료

- [Langfuse 공식 문서](https://langfuse.com/docs)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
- [트레이싱 모범 사례](https://langfuse.com/docs/tracing)
