"""Report generation router"""
import time
import uuid
import getpass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.api.schemas import ReportRequest, ReportResponse
from app.scripts.report_generator import ReportGenerator
from app.scripts.document_generator import DocumentGenerator
from app.utils.dates import parse_date_range

router = APIRouter(tags=["Reports"])


def calculate_generation_cost(report_data: Dict[str, Any]) -> float:
    """보고서 생성 비용 계산 (예시)"""
    # 실제로는 LLM API 호출 비용을 계산해야 함
    # 여기서는 간단하게 문서 수 기반으로 예시 비용 계산
    total_docs = sum(result.get('num_docs', 0) for result in report_data.get('results', []))
    cost_per_doc = 0.001  # 문서당 예상 비용
    return round(total_docs * cost_per_doc, 4)


@router.post("/generate-report")
async def generate_report(request: ReportRequest):
    """
    보고서 생성 API

    - **report_type**: 보고서 타입 (weekly 또는 executive)
    - **question**: 질문 (선택사항, 미지정 시 기본 질문 사용)
    - **date_range**: 날짜 범위 (예: '이번 주', '12월 2주차')
    - **start_date**: 시작 날짜 (YYYY-MM-DD)
    - **end_date**: 종료 날짜 (YYYY-MM-DD)
    - **author**: 보고서 작성자 (선택사항)
    - **output_format**: 출력 형식 (json, pdf, docx) - 기본값: json
    """
    try:
        start_time = time.time()
        trace_id = str(uuid.uuid4())

        # 날짜 필터 파싱
        date_filter = parse_date_range(
            date_input=request.date_range,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # 보고서 생성기 초기화
        generator = ReportGenerator(
            config_path=None,  # 기본 설정 파일 사용
            report_type=request.report_type
        )

        # 질문 설정
        if request.question:
            questions = [request.question]
        else:
            # 설정 파일의 기본 질문 사용
            questions = generator.default_questions

        if not questions:
            raise HTTPException(
                status_code=400,
                detail="질문이 제공되지 않았으며, 기본 질문도 없습니다."
            )

        # 작성자 정보 설정
        author = request.author if request.author else getpass.getuser()

        # 보고서 생성
        report_data = generator.generate_report(questions, date_filter)

        # 작성자 및 작성일자 정보 추가
        report_data["author"] = author
        report_data["created_date"] = datetime.now().strftime("%Y-%m-%d")
        report_data["created_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 생성 시간 및 비용 계산
        generation_time = round(time.time() - start_time, 2)
        cost = calculate_generation_cost(report_data)

        # 메타데이터 구성
        metadata = {
            "report_type": request.report_type,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": author,
            "generation_time": generation_time,
            "model": report_data['llm']['model_id'],
            "cost": cost,
            "timestamp": datetime.now().isoformat()
        }

        # 출력 형식에 따라 처리
        output_format = request.output_format.lower() if request.output_format else "json"

        if output_format in ["pdf", "docx"]:
            # 문서 파일 생성
            reports_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)

            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{request.report_type}_report_{timestamp}.{output_format}"
            file_path = reports_dir / filename

            # 문서 생성
            doc_generator = DocumentGenerator()

            if output_format == "pdf":
                doc_generator.generate_pdf_report(report_data, str(file_path))
            else:  # docx
                doc_generator.generate_word_report(report_data, str(file_path))

            # 파일 다운로드 응답
            media_type = "application/pdf" if output_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

            return FileResponse(
                path=str(file_path),
                media_type=media_type,
                filename=filename,
                headers={
                    "X-Trace-ID": trace_id,
                    "X-Generation-Time": str(generation_time),
                    "X-Cost": str(cost)
                }
            )
        else:
            # JSON 응답 (기본)
            # 마크다운 형식으로 변환 (결과 조합)
            report_title = "주간 업무 보고서" if request.report_type == "weekly" else "최종 보고서"
            markdown_parts = [
                f"# {report_title}\n",
                f"**작성일:** {report_data['created_date']}",
                f"**작성자:** {author}\n"
            ]

            for result in report_data.get('results', []):
                if result.get('success', False):
                    section_title = result.get('title') or "항목"
                    answer = result.get('answer', '')
                    markdown_parts.append(f"## {section_title}\n")
                    markdown_parts.append(f"{answer}\n")

            report_markdown = "\n".join(markdown_parts)

            # 응답 구성
            response_data = {
                "code": 1,
                "message": "성공하였습니다.",
                "result": {
                    "data": {
                        "trace_id": trace_id,
                        "report": report_markdown,
                        "metadata": metadata
                    }
                },
                "success": True
            }

            return JSONResponse(content=response_data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)
