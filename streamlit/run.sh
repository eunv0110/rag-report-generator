#!/bin/bash

# RAG 보고서 생성기 Streamlit 앱 실행 스크립트

echo "🚀 RAG 보고서 생성기 Streamlit 앱을 시작합니다..."
echo ""

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.." || exit 1

# 가상환경 활성화
if [ -d ".venv" ]; then
    echo "✅ 가상환경을 활성화합니다..."
    source .venv/bin/activate
else
    echo "⚠️  가상환경을 찾을 수 없습니다. .venv 폴더를 확인하세요."
    exit 1
fi

# Streamlit 앱 실행
echo "📊 Streamlit 앱을 시작합니다..."
echo ""
echo "🌐 브라우저에서 다음 주소로 접속하세요:"
echo "   http://localhost:8501"
echo ""
echo "⚠️  주의: API 서버가 실행 중이어야 합니다 (http://localhost:8000)"
echo "   API 서버 실행: python run_api.py"
echo ""
echo "종료하려면 Ctrl+C를 누르세요."
echo ""

streamlit run streamlit/app.py --server.port=8501 --server.address=0.0.0.0
