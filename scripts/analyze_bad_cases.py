#!/usr/bin/env python3
"""
Bad Case 심층 분석 스크립트
성능 평가에서 점수가 낮은 케이스들을 분석하여 원인을 파악
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter, defaultdict
import ast

# 파일 경로
FILE1 = "/home/work/rag/Project/rag-report-generator/data/langfuse/final2/eval_openai_rrf_multiquery_executive_report_v1.csv"
FILE2 = "/home/work/rag/Project/rag-report-generator/data/langfuse/final/eval_openai_rrf_multiquery_executive_report_v1_top8.csv"
OUTPUT_DIR = Path("/home/work/rag/Project/rag-report-generator/analysis/bad_cases_v2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_analyze_csv(file_path, dataset_name):
    """CSV 파일 로드 및 기본 분석 (Long format)"""
    print(f"\n{'='*80}")
    print(f"분석 중: {dataset_name}")
    print(f"{'='*80}")

    # 인코딩 자동 감지
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='iso-8859-1')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')

    # 컬럼 정보 출력
    print(f"\n총 데이터 수: {len(df)}")
    print(f"컬럼: {df.columns.tolist()}")

    # Metric별 데이터 확인
    if 'name' in df.columns:
        metrics = df['name'].unique()
        print(f"\nMetrics: {metrics}")
        for metric in metrics:
            count = len(df[df['name'] == metric])
            print(f"  - {metric}: {count} cases")

    return df

def pivot_data(df):
    """Long format을 Wide format으로 변환"""
    # traceId별로 pivot
    if 'traceId' not in df.columns or 'name' not in df.columns or 'value' not in df.columns:
        print("필요한 컬럼이 없습니다.")
        return None

    # Pivot: traceId를 인덱스로, name을 컬럼으로, value를 값으로
    pivot_df = df.pivot_table(
        index=['traceId', 'traceName', 'comment'],
        columns='name',
        values='value',
        aggfunc='first'
    ).reset_index()

    return pivot_df

def identify_bad_cases(pivot_df, threshold=0.5):
    """Bad Cases 식별"""
    bad_cases = {}

    # Metric 컬럼 찾기 (traceId, traceName, comment 제외)
    metric_cols = [col for col in pivot_df.columns if col not in ['traceId', 'traceName', 'comment']]

    for metric in metric_cols:
        if metric in pivot_df.columns:
            # NaN이 아닌 값만 필터링
            valid_df = pivot_df[pivot_df[metric].notna()].copy()

            if len(valid_df) > 0:
                # threshold 이하인 케이스
                bad_df = valid_df[valid_df[metric] <= threshold]

                if len(bad_df) > 0:
                    bad_cases[metric] = {
                        'count': len(bad_df),
                        'percentage': len(bad_df) / len(valid_df) * 100,
                        'mean_score': bad_df[metric].mean(),
                        'cases': bad_df
                    }

    return bad_cases

def analyze_score_distribution(pivot_df):
    """점수 분포 분석"""
    distribution = {}

    # Metric 컬럼 찾기
    metric_cols = [col for col in pivot_df.columns if col not in ['traceId', 'traceName', 'comment']]

    for metric in metric_cols:
        if metric in pivot_df.columns:
            valid_scores = pivot_df[pivot_df[metric].notna()][metric]

            if len(valid_scores) > 0:
                distribution[metric] = {
                    'count': len(valid_scores),
                    'mean': float(valid_scores.mean()),
                    'std': float(valid_scores.std()),
                    'min': float(valid_scores.min()),
                    'max': float(valid_scores.max()),
                    'q25': float(valid_scores.quantile(0.25)),
                    'median': float(valid_scores.median()),
                    'q75': float(valid_scores.quantile(0.75)),
                    'score_0': int((valid_scores == 0).sum()),
                    'score_0.5': int(((valid_scores > 0) & (valid_scores <= 0.5)).sum()),
                    'score_1.0': int((valid_scores == 1.0).sum()),
                }

    return distribution

def analyze_bad_case_patterns(bad_cases_df, dataset_name):
    """Bad Case 패턴 분석"""
    patterns = {
        'dataset': dataset_name,
        'total_bad_cases': len(bad_cases_df)
    }

    # Comment 분석 (query가 여기에 있음)
    if 'comment' in bad_cases_df.columns:
        comments = bad_cases_df['comment'].dropna()
        if len(comments) > 0:
            patterns['query_analysis'] = {
                'avg_length': float(comments.astype(str).str.len().mean()),
                'min_length': int(comments.astype(str).str.len().min()),
                'max_length': int(comments.astype(str).str.len().max()),
                'sample_queries': comments.head(5).tolist()
            }

    return patterns

def create_detailed_bad_case_report(pivot_df, bad_cases, dataset_name, top_n=20):
    """상세 Bad Case 리포트 생성"""
    detailed_cases = []

    for metric, bc_info in bad_cases.items():
        # Score가 낮은 순으로 정렬
        bad_df = bc_info['cases'].nsmallest(min(top_n, len(bc_info['cases'])), metric)

        for idx, row in bad_df.iterrows():
            case = {
                'dataset': dataset_name,
                'metric': metric,
                'score': float(row[metric]),
                'traceId': str(row['traceId']) if 'traceId' in row.index else None,
                'traceName': str(row['traceName']) if 'traceName' in row.index else None,
            }

            # Comment (query) 추출
            if 'comment' in row.index and pd.notna(row['comment']):
                comment = str(row['comment'])
                if len(comment) > 500:
                    case['query'] = comment[:500] + "..."
                else:
                    case['query'] = comment

            # 다른 메트릭 점수도 추가
            for col in pivot_df.columns:
                if col not in ['traceId', 'traceName', 'comment', metric] and col in row.index:
                    if pd.notna(row[col]):
                        case[f'other_{col}'] = float(row[col])

            detailed_cases.append(case)

    return detailed_cases

def compare_datasets(pivot_df1, pivot_df2):
    """두 데이터셋 비교 분석"""
    comparison = {
        'dataset1_size': len(pivot_df1),
        'dataset2_size': len(pivot_df2),
    }

    # 공통 metric 찾기
    metrics1 = set([col for col in pivot_df1.columns if col not in ['traceId', 'traceName', 'comment']])
    metrics2 = set([col for col in pivot_df2.columns if col not in ['traceId', 'traceName', 'comment']])
    common_metrics = metrics1 & metrics2

    if common_metrics:
        comparison['common_metrics'] = list(common_metrics)
        comparison['metric_comparison'] = {}

        for metric in common_metrics:
            valid1 = pivot_df1[pivot_df1[metric].notna()][metric]
            valid2 = pivot_df2[pivot_df2[metric].notna()][metric]

            comparison['metric_comparison'][metric] = {
                'dataset1_mean': float(valid1.mean()) if len(valid1) > 0 else None,
                'dataset2_mean': float(valid2.mean()) if len(valid2) > 0 else None,
                'dataset1_bad_cases': int((valid1 <= 0.5).sum()) if len(valid1) > 0 else 0,
                'dataset2_bad_cases': int((valid2 <= 0.5).sum()) if len(valid2) > 0 else 0,
            }

    return comparison

def main():
    """메인 분석 실행"""

    # Dataset 1 로드 및 변환
    df1_raw = load_and_analyze_csv(FILE1, "Executive Report V1 (final2)")
    pivot_df1 = pivot_data(df1_raw)

    if pivot_df1 is None or len(pivot_df1) == 0:
        print("Dataset 1 pivot 실패")
        return

    print(f"\nPivot 결과: {len(pivot_df1)} traces")
    print(f"Metrics: {[col for col in pivot_df1.columns if col not in ['traceId', 'traceName', 'comment']]}")

    dist1 = analyze_score_distribution(pivot_df1)
    bad_cases1 = identify_bad_cases(pivot_df1, threshold=0.5)

    print(f"\n[점수 분포 분석]")
    for metric, stats in dist1.items():
        print(f"\n{metric}:")
        print(f"  - Mean: {stats['mean']:.3f} (±{stats['std']:.3f})")
        print(f"  - Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
        print(f"  - Score 0: {stats['score_0']} | Score ≤0.5: {stats['score_0.5']} | Score 1.0: {stats['score_1.0']}")

    print(f"\n[Bad Cases (Score ≤ 0.5)]")
    for metric, bc_info in bad_cases1.items():
        print(f"\n{metric}:")
        print(f"  - Bad cases: {bc_info['count']} / {dist1[metric]['count']} ({bc_info['percentage']:.1f}%)")
        print(f"  - Mean score: {bc_info['mean_score']:.3f}")

    # Dataset 2 로드 및 변환
    df2_raw = load_and_analyze_csv(FILE2, "Executive Report V1 (final - reference)")
    pivot_df2 = pivot_data(df2_raw)

    if pivot_df2 is None or len(pivot_df2) == 0:
        print("Dataset 2 pivot 실패")
        return

    print(f"\nPivot 결과: {len(pivot_df2)} traces")
    print(f"Metrics: {[col for col in pivot_df2.columns if col not in ['traceId', 'traceName', 'comment']]}")

    dist2 = analyze_score_distribution(pivot_df2)
    bad_cases2 = identify_bad_cases(pivot_df2, threshold=0.5)

    print(f"\n[점수 분포 분석]")
    for metric, stats in dist2.items():
        print(f"\n{metric}:")
        print(f"  - Mean: {stats['mean']:.3f} (±{stats['std']:.3f})")
        print(f"  - Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
        print(f"  - Score 0: {stats['score_0']} | Score ≤0.5: {stats['score_0.5']} | Score 1.0: {stats['score_1.0']}")

    print(f"\n[Bad Cases (Score ≤ 0.5)]")
    for metric, bc_info in bad_cases2.items():
        print(f"\n{metric}:")
        print(f"  - Bad cases: {bc_info['count']} / {dist2[metric]['count']} ({bc_info['percentage']:.1f}%)")
        print(f"  - Mean score: {bc_info['mean_score']:.3f}")

    # 데이터셋 비교
    comparison = compare_datasets(pivot_df1, pivot_df2)

    print(f"\n{'='*80}")
    print("두 데이터셋 비교")
    print(f"{'='*80}")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))

    # 상세 Bad Case 리포트 생성
    print(f"\n상세 Bad Case 리포트 생성 중...")
    detailed1 = create_detailed_bad_case_report(pivot_df1, bad_cases1, "Dataset1", top_n=20)
    detailed2 = create_detailed_bad_case_report(pivot_df2, bad_cases2, "Dataset2", top_n=20)

    # Bad Case 패턴 분석
    all_bad_cases1 = pd.concat([bc_info['cases'] for bc_info in bad_cases1.values()]) if bad_cases1 else pd.DataFrame()
    all_bad_cases2 = pd.concat([bc_info['cases'] for bc_info in bad_cases2.values()]) if bad_cases2 else pd.DataFrame()

    # 중복 제거
    if len(all_bad_cases1) > 0:
        all_bad_cases1 = all_bad_cases1.drop_duplicates(subset=['traceId'])
        patterns1 = analyze_bad_case_patterns(all_bad_cases1, "Dataset1")
    else:
        patterns1 = {}

    if len(all_bad_cases2) > 0:
        all_bad_cases2 = all_bad_cases2.drop_duplicates(subset=['traceId'])
        patterns2 = analyze_bad_case_patterns(all_bad_cases2, "Dataset2")
    else:
        patterns2 = {}

    # 결과 저장
    results = {
        'summary': {
            'dataset1': {
                'name': 'OpenAI RRF MultiQuery (Top 8)',
                'total_cases': len(pivot_df1),
                'score_distribution': dist1,
                'bad_cases_summary': [
                    {
                        'metric': metric,
                        'count': bc_info['count'],
                        'percentage': bc_info['percentage'],
                        'mean_score': bc_info['mean_score']
                    } for metric, bc_info in bad_cases1.items()
                ],
                'patterns': patterns1
            },
            'dataset2': {
                'name': 'BGE-M3 RRF Reranker (k=6)',
                'total_cases': len(pivot_df2),
                'score_distribution': dist2,
                'bad_cases_summary': [
                    {
                        'metric': metric,
                        'count': bc_info['count'],
                        'percentage': bc_info['percentage'],
                        'mean_score': bc_info['mean_score']
                    } for metric, bc_info in bad_cases2.items()
                ],
                'patterns': patterns2
            },
            'comparison': comparison
        },
        'detailed_bad_cases': {
            'dataset1': detailed1,
            'dataset2': detailed2
        }
    }

    # JSON 파일로 저장
    output_file = OUTPUT_DIR / "bad_case_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 분석 결과 저장: {output_file}")

    # Bad Cases CSV 저장
    if len(all_bad_cases1) > 0:
        bad_csv1 = OUTPUT_DIR / "bad_cases_dataset1.csv"
        all_bad_cases1.to_csv(bad_csv1, index=False, encoding='utf-8')
        print(f"✓ Bad cases CSV 저장: {bad_csv1} ({len(all_bad_cases1)} cases)")

    if len(all_bad_cases2) > 0:
        bad_csv2 = OUTPUT_DIR / "bad_cases_dataset2.csv"
        all_bad_cases2.to_csv(bad_csv2, index=False, encoding='utf-8')
        print(f"✓ Bad cases CSV 저장: {bad_csv2} ({len(all_bad_cases2)} cases)")

    # 요약 마크다운 생성
    create_summary_markdown(results, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("분석 완료!")
    print(f"{'='*80}")

def create_summary_markdown(results, output_dir):
    """분석 결과 마크다운 리포트 생성"""
    md_content = """# Bad Case 심층 분석 리포트

## 개요

이 리포트는 RAG 시스템 평가에서 성능이 낮은 케이스들을 분석하여,
그 원인이 모델의 한계인지, 프롬프트 개선으로 해결 가능한지 파악하기 위해 작성되었습니다.

**분석 대상:**
- Dataset 1: Executive Report V1 (final2)
- Dataset 2: Executive Report V1 (final - reference)

**Bad Case 정의:** Score ≤ 0.5인 케이스

---

"""

    # Dataset 1 분석
    ds1 = results['summary']['dataset1']
    md_content += f"""## 1. {ds1['name']}

### 기본 통계
- **전체 케이스 수**: {ds1['total_cases']:,}

### 점수 분포
"""

    for metric, stats in ds1['score_distribution'].items():
        bad_count = stats['score_0'] + stats['score_0.5']
        md_content += f"""
#### {metric}
- **평균**: {stats['mean']:.3f} (±{stats['std']:.3f})
- **범위**: [{stats['min']:.1f}, {stats['max']:.1f}]
- **중앙값**: {stats['median']:.3f}
- **점수 분포**:
  - Score 0.0: {stats['score_0']} ({stats['score_0']/stats['count']*100:.1f}%)
  - Score ≤0.5 (but >0): {stats['score_0.5']} ({stats['score_0.5']/stats['count']*100:.1f}%)
  - Score 1.0: {stats['score_1.0']} ({stats['score_1.0']/stats['count']*100:.1f}%)
  - **Total Bad Cases**: {bad_count} ({bad_count/stats['count']*100:.1f}%)
"""

    md_content += "\n### Bad Cases 상세 (Score ≤ 0.5)\n"

    for bc in ds1['bad_cases_summary']:
        md_content += f"""
#### {bc['metric']}
- **Bad case 수**: {bc['count']} ({bc['percentage']:.1f}%)
- **평균 점수**: {bc['mean_score']:.3f}
"""

    if ds1['patterns'] and 'query_analysis' in ds1['patterns']:
        md_content += "\n### Bad Case 패턴 분석\n"
        patterns = ds1['patterns']
        qa = patterns['query_analysis']
        md_content += f"""
#### Query 특성
- **평균 길이**: {qa['avg_length']:.1f} 문자
- **길이 범위**: [{qa['min_length']}, {qa['max_length']}]

**샘플 쿼리 (Bad Cases)**:
"""
        for i, q in enumerate(qa['sample_queries'], 1):
            md_content += f"\n{i}. `{q[:200]}{'...' if len(q) > 200 else ''}`\n"

    # Dataset 2 분석
    ds2 = results['summary']['dataset2']
    md_content += f"""

---

## 2. {ds2['name']}

### 기본 통계
- **전체 케이스 수**: {ds2['total_cases']:,}

### 점수 분포
"""

    for metric, stats in ds2['score_distribution'].items():
        bad_count = stats['score_0'] + stats['score_0.5']
        md_content += f"""
#### {metric}
- **평균**: {stats['mean']:.3f} (±{stats['std']:.3f})
- **범위**: [{stats['min']:.1f}, {stats['max']:.1f}]
- **중앙값**: {stats['median']:.3f}
- **점수 분포**:
  - Score 0.0: {stats['score_0']} ({stats['score_0']/stats['count']*100:.1f}%)
  - Score ≤0.5 (but >0): {stats['score_0.5']} ({stats['score_0.5']/stats['count']*100:.1f}%)
  - Score 1.0: {stats['score_1.0']} ({stats['score_1.0']/stats['count']*100:.1f}%)
  - **Total Bad Cases**: {bad_count} ({bad_count/stats['count']*100:.1f}%)
"""

    md_content += "\n### Bad Cases 상세 (Score ≤ 0.5)\n"

    for bc in ds2['bad_cases_summary']:
        md_content += f"""
#### {bc['metric']}
- **Bad case 수**: {bc['count']} ({bc['percentage']:.1f}%)
- **평균 점수**: {bc['mean_score']:.3f}
"""

    if ds2['patterns'] and 'query_analysis' in ds2['patterns']:
        md_content += "\n### Bad Case 패턴 분석\n"
        patterns = ds2['patterns']
        qa = patterns['query_analysis']
        md_content += f"""
#### Query 특성
- **평균 길이**: {qa['avg_length']:.1f} 문자
- **길이 범위**: [{qa['min_length']}, {qa['max_length']}]

**샘플 쿼리 (Bad Cases)**:
"""
        for i, q in enumerate(qa['sample_queries'], 1):
            md_content += f"\n{i}. `{q[:200]}{'...' if len(q) > 200 else ''}`\n"

    # 데이터셋 비교
    comp = results['summary']['comparison']
    md_content += f"""

---

## 3. 데이터셋 비교

### 기본 정보
- **Dataset 1 크기**: {comp['dataset1_size']:,}
- **Dataset 2 크기**: {comp['dataset2_size']:,}
"""

    if 'metric_comparison' in comp:
        md_content += "\n### 메트릭 비교\n\n"
        md_content += "| Metric | Dataset 1 평균 | Dataset 2 평균 | Dataset 1 Bad Cases | Dataset 2 Bad Cases | 개선율 |\n"
        md_content += "|--------|----------------|----------------|---------------------|---------------------|--------|\n"

        for metric, stats in comp['metric_comparison'].items():
            d1_mean = f"{stats['dataset1_mean']:.3f}" if stats['dataset1_mean'] is not None else "N/A"
            d2_mean = f"{stats['dataset2_mean']:.3f}" if stats['dataset2_mean'] is not None else "N/A"

            # 개선율 계산
            if stats['dataset1_mean'] and stats['dataset2_mean']:
                improvement = (stats['dataset2_mean'] - stats['dataset1_mean']) / stats['dataset1_mean'] * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"

            md_content += f"| {metric} | {d1_mean} | {d2_mean} | {stats['dataset1_bad_cases']} | {stats['dataset2_bad_cases']} | {improvement_str} |\n"

    md_content += """

---

## 4. 주요 발견사항

### 4.1 Bad Case 특성 분석

"""

    # Bad cases 상세 분석
    detailed1 = results['detailed_bad_cases']['dataset1']
    detailed2 = results['detailed_bad_cases']['dataset2']

    if detailed1:
        md_content += """
#### Dataset 1 상위 Bad Cases

다음은 점수가 가장 낮은 케이스들입니다:

"""
        for i, case in enumerate(detailed1[:10], 1):
            md_content += f"""
**Case {i}**: {case['metric']} = {case['score']:.2f}
- Query: `{case.get('query', 'N/A')[:200]}{'...' if len(case.get('query', '')) > 200 else ''}`
- Trace: `{case.get('traceName', 'N/A')}`
"""
            # 다른 메트릭 점수도 표시
            other_metrics = [k for k in case.keys() if k.startswith('other_')]
            if other_metrics:
                md_content += "- Other metrics: "
                md_content += ", ".join([f"{k.replace('other_', '')}: {case[k]:.2f}" for k in other_metrics])
                md_content += "\n"

    if detailed2:
        md_content += """

#### Dataset 2 상위 Bad Cases

다음은 점수가 가장 낮은 케이스들입니다:

"""
        for i, case in enumerate(detailed2[:10], 1):
            md_content += f"""
**Case {i}**: {case['metric']} = {case['score']:.2f}
- Query: `{case.get('query', 'N/A')[:200]}{'...' if len(case.get('query', '')) > 200 else ''}`
- Trace: `{case.get('traceName', 'N/A')}`
"""
            # 다른 메트릭 점수도 표시
            other_metrics = [k for k in case.keys() if k.startswith('other_')]
            if other_metrics:
                md_content += "- Other metrics: "
                md_content += ", ".join([f"{k.replace('other_', '')}: {case[k]:.2f}" for k in other_metrics])
                md_content += "\n"

    md_content += """

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
"""

    # 마크다운 파일 저장
    md_file = output_dir / "BAD_CASE_ANALYSIS.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✓ 분석 리포트 저장: {md_file}")

if __name__ == "__main__":
    main()
