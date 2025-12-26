#!/bin/bash
# 4가지 Retrieval 전략 성능 평가 배치 스크립트
#
# 평가 전략:
#   1. RRF
#   2. RRF + MultiQuery
#   3. RRF + MultiQuery + LongContext
#   4. RRF + LongContext + TimeWeighted
#
# 사용법:
#   chmod +x scripts/evaluate_4_strategies.sh
#   ./scripts/evaluate_4_strategies.sh

set -e  # 에러 발생시 중단

echo "======================================================================"
echo "🚀 4가지 Retrieval 전략 성능 평가 시작"
echo "======================================================================"

# 가상환경 활성화
source .venv/bin/activate

# 평가 데이터셋 경로
DATASET="/home/work/rag/Project/rag-report-generator/data/evaluation/merged_qa_dataset.json"

# 공통 파라미터
TOP_K=10
VERSION="v1"

echo ""
echo "📊 평가 설정:"
echo "   - Dataset: $DATASET"
echo "   - Top-K: $TOP_K"
echo "   - Version: $VERSION"
echo ""

# 1. RRF (Baseline)
echo ""
echo "======================================================================"
echo "1️⃣  RRF (Baseline) 평가 중..."
echo "======================================================================"
python scripts/evaluate_retriever.py \
    --retriever ensemble_rrf \
    --dataset "$DATASET" \
    --top-k $TOP_K \
    --version $VERSION

echo "✅ RRF 평가 완료"
sleep 2

# 2. RRF + MultiQuery
echo ""
echo "======================================================================"
echo "2️⃣  RRF + MultiQuery 평가 중..."
echo "======================================================================"
python scripts/evaluate_retriever.py \
    --retriever multiquery \
    --base-retriever ensemble_rrf \
    --num-queries 3 \
    --dataset "$DATASET" \
    --top-k $TOP_K \
    --version $VERSION

echo "✅ RRF + MultiQuery 평가 완료"
sleep 2

# 3. RRF + MultiQuery + LongContext
echo ""
echo "======================================================================"
echo "3️⃣  RRF + MultiQuery + LongContext 평가 중..."
echo "======================================================================"
python scripts/evaluate_retriever.py \
    --retriever multiquery \
    --base-retriever ensemble_rrf_longcontext \
    --num-queries 3 \
    --dataset "$DATASET" \
    --top-k $TOP_K \
    --version $VERSION

echo "✅ RRF + MultiQuery + LongContext 평가 완료"
sleep 2

# 4. RRF + LongContext + TimeWeighted
echo ""
echo "======================================================================"
echo "4️⃣  RRF + LongContext + TimeWeighted 평가 중..."
echo "======================================================================"
python scripts/evaluate_retriever.py \
    --retriever ensemble_rrf_timeweighted_longcontext \
    --decay-rate 0.01 \
    --dataset "$DATASET" \
    --top-k $TOP_K \
    --version $VERSION

echo "✅ RRF + LongContext + TimeWeighted 평가 완료"
sleep 2

echo ""
echo "======================================================================"
echo "✅ 모든 평가 완료!"
echo "======================================================================"
echo ""
echo "📊 다음 단계:"
echo "   1. Langfuse 대시보드에서 결과 확인: https://cloud.langfuse.com"
echo "   2. 각 전략별 stats 파일 확인:"
echo "      - data/evaluation/ensemble_rrf_evaluation_stats.json"
echo "      - data/evaluation/multiquery_ensemble_rrf_evaluation_stats.json"
echo "      - data/evaluation/multiquery_ensemble_rrf_longcontext_evaluation_stats.json"
echo "      - data/evaluation/ensemble_rrf_timeweighted_longcontext_0.01_evaluation_stats.json"
echo ""
echo "   3. 비교 분석을 위해 Langfuse에서 CSV export 후:"
echo "      python scripts/compare_evaluation_results.py"
echo ""
