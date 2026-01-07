"""API 요청/응답 스키마 정의"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ReportRequest(BaseModel):
    """보고서 생성 요청"""
    report_type: str = Field(..., description="보고서 타입 (weekly 또는 executive)")
    question: Optional[str] = Field(None, description="질문 (미지정 시 기본 질문 사용)")
    date_range: Optional[str] = Field(None, description="날짜 범위 (예: '이번 주', '12월 2주차')")
    start_date: Optional[str] = Field(None, description="시작 날짜 (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="종료 날짜 (YYYY-MM-DD)")
    author: Optional[str] = Field(None, description="보고서 작성자")
    output_format: Optional[str] = Field("json", description="출력 형식 (json, pdf, docx)")

    class Config:
        json_schema_extra = {
            "example": {
                "report_type": "weekly",
                "question": "이번 주 주요 성과와 진행 중인 업무는 무엇인가요?",
                "date_range": "이번 주",
                "author": "김은비",
                "output_format": "pdf"
            }
        }


class ImageInfo(BaseModel):
    """이미지 정보"""
    path: str
    description: str
    source: str


class QuestionResult(BaseModel):
    """질문별 답변 결과"""
    question_id: int
    question: str
    title: Optional[str] = None
    date_filter: Optional[str] = None
    num_docs: int = 0
    doc_titles: List[str] = []
    images: List[ImageInfo] = []
    answer: str
    success: bool
    error: Optional[str] = None


class RetrieverConfig(BaseModel):
    """Retriever 설정 정보"""
    name: str
    display_name: str
    type: str
    embedding: str
    top_k: int
    use_reranker: bool


class LLMConfig(BaseModel):
    """LLM 설정 정보"""
    name: str
    display_name: str
    model_id: str


class ReportMetadata(BaseModel):
    """보고서 메타데이터"""
    report_type: str
    date: str
    author: str
    generation_time: float
    model: str
    cost: float
    timestamp: str


class ReportData(BaseModel):
    """보고서 전체 데이터"""
    report_type: str
    title: str
    generated_at: str
    retriever: Dict[str, Any]
    llm: Dict[str, Any]
    global_date_filter: Optional[str] = None
    num_questions: int
    results: List[QuestionResult]
    author: str
    created_date: str
    created_datetime: str


class ReportResponse(BaseModel):
    """보고서 생성 응답"""
    code: int = Field(1, description="응답 코드 (1: 성공)")
    message: str = Field("성공하였습니다.", description="응답 메시지")
    result: Dict[str, Any] = Field(..., description="결과 데이터")
    success: bool = Field(True, description="성공 여부")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 1,
                "message": "성공하였습니다.",
                "result": {
                    "data": {
                        "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                        "report": "# 주간 업무 보고서\n\n**작성일:** 2025-01-06\n**작성자:** 김은비\n\n## 1. 주요 성과\n\n이번 주 RAG 시스템 성능 최적화를 완료하였습니다.",
                        "metadata": {
                            "report_type": "weekly",
                            "date": "2025-01-06",
                            "author": "김은비",
                            "generation_time": 3.2,
                            "model": "deepseek-v3.1",
                            "cost": 0.05,
                            "timestamp": "2025-01-06T10:30:00"
                        }
                    }
                },
                "success": True
            }
        }


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    version: str
    timestamp: str
