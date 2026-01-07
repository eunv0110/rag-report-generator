#!/bin/bash

# 메모리 단편화 방지 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 프로젝트 디렉토리로 이동
cd /home/work/rag/Project/rag-report-generator

echo ""
echo "=================================="
echo "최종보고서 평가 시작"
echo "=================================="
python experiments/evaluators/evaluation/evaluate_final.py --report-type executive --max-workers 4

echo ""
echo "=================================="
echo "모든 평가 완료!"
echo "=================================="
