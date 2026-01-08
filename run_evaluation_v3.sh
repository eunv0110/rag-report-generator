#!/bin/bash

# 메모리 단편화 방지 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 프로젝트 디렉토리로 이동
cd /home/work/rag/Project/rag-report-generator

echo ""
echo "=================================="
echo "최종보고서 평가 시작 (V3 프롬프트)"
echo "=================================="
echo "프롬프트: executive_report_v3 (개선된 균형 버전)"
echo "버전 태그: optimal_v3"
echo ""

python experiments/evaluators/evaluation/evaluate_final.py \
  --report-type executive \
  --version optimal_v4 \
  --max-workers 1

echo ""
echo "=================================="
echo "평가 완료!"
echo "=================================="
