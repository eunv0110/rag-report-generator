#!/bin/bash
cd /home/work/rag/Project/rag-report-generator
source .venv/bin/activate

echo "========================================"
echo "주간 보고서 평가 (10개 리트리버)"
echo "========================================"
echo ""

echo "=== 1/10: Upstage + RRF + MultiQuery + LC (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers upstage_rrf_multiquery_lc --version v1

echo ""
echo "=== 2/10: OpenAI + RRF + MultiQuery + LC (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers openai_rrf_multiquery_lc --version v1

echo ""
echo "=== 3/10: Qwen + RRF + MultiQuery + LC (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers qwen_rrf_multiquery_lc --version v1

echo ""
echo "=== 4/10: BGE-M3 + RRF Ensemble (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers bge_m3_rrf_ensemble --version v1

echo ""
echo "=== 5/10: BGE-M3 + RRF + MultiQuery + LC (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers bge_m3_rrf_multiquery_lc --version v1

echo ""
echo "=== 6/10: OpenAI + RRF + LC + Time (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers openai_rrf_lc_time --version v1

echo ""
echo "=== 7/10: Qwen + RRF Ensemble (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers qwen_rrf_ensemble --version v1

echo ""
echo "=== 8/10: Upstage + RRF Ensemble (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers upstage_rrf_ensemble --version v1

echo ""
echo "=== 9/10: OpenAI + RRF + MultiQuery (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers openai_rrf_multiquery --version v1

echo ""
echo "=== 10/10: Gemini + RRF + MultiQuery (k: 6,8,10,12) ==="
python evaluators/evaluate_report_types.py --report-type weekly --retrievers gemini_rrf_multiquery --version v1

echo ""
echo "✅ 주간 보고서 평가 완료!"
echo "총 평가: 10개 리트리버 × 4개 Top-K = 40개 조합"
