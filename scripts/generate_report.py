

#!/usr/bin/env python3
"""통합 보고서 생성 CLI

사용법:
    # 주간 보고서 생성
    python generate_report.py --type weekly --questions "9월 첫째주 업무 요약해줘" --output weekly_report.docx

    # 최종 보고서 생성
    python generate_report.py --type executive --questions "10월 최종 보고서 만들어줘" --output executive_report.pdf

    # 여러 질문
    python generate_report.py --type weekly --questions "9월 업무 요약" "10월 업무 요약" --output report.docx

    # 날짜 범위 지정
    python generate_report.py --type weekly --questions "업무 요약" --date-range "이번 주" --output report.docx
"""

import sys
import argparse
import getpass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.scripts.report_generator import ReportGenerator
from app.scripts.document_generator import DocumentGenerator
from app.utils.dates import parse_date_range


def create_argument_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description="통합 보고서 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 주간 보고서 생성 (Word)
  python generate_report.py --type weekly --questions "9월 첫째주 업무 요약해줘" --output weekly_report.docx

  # 최종 보고서 생성 (PDF)
  python generate_report.py --type executive --questions "10월 최종 보고서 만들어줘" --output executive_report.pdf

  # 여러 질문으로 보고서 생성
  python generate_report.py --type weekly --questions "9월 업무" "10월 업무" --output report.docx

  # 날짜 범위 지정
  python generate_report.py --type weekly --questions "업무 요약" --date-range "이번 주" --output report.docx
        """
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["weekly", "executive"],
        required=True,
        help="보고서 타입: weekly(주간 보고서), executive(최종 보고서)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs='+',
        required=True,
        help="질문 리스트 (공백으로 구분)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 파일 경로 (.docx 또는 .pdf)"
    )
    parser.add_argument(
        "--date-range",
        type=str,
        default=None,
        help="날짜 범위 (예: '이번 주', '지난주', '12월 2주차')"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="시작 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="종료 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="JSON만 생성하고 문서는 생성하지 않음"
    )
    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="보고서 작성자 (미지정 시 시스템 사용자명 사용)"
    )

    return parser


def print_report_header(report_type: str, questions: List[str], date_filter: Optional[Tuple],
                       retriever_name: str, llm_name: str) -> None:
    """보고서 생성 시작 정보 출력"""
    report_title = "주간 보고서" if report_type == "weekly" else "최종 보고서"

    print(f"\n📊 {report_title} 생성 시작...")
    print(f"🔧 설정: {retriever_name} + {llm_name}")

    print(f"\n📝 질문 ({len(questions)}개):")
    for i, question in enumerate(questions, 1):
        print(f"  {i}. {question}")

    if date_filter:
        print(f"\n📅 날짜 필터: {date_filter[0][:10]} ~ {date_filter[1][:10]}")

    print()


def add_metadata_to_report(report_data: Dict, author: Optional[str]) -> Dict:
    """보고서에 메타데이터 추가"""
    report_author = author if author else getpass.getuser()
    current_time = datetime.now()

    report_data["author"] = report_author
    report_data["created_date"] = current_time.strftime("%Y-%m-%d")
    report_data["created_datetime"] = current_time.strftime("%Y-%m-%d %H:%M:%S")

    return report_data


def resolve_output_path(output: str, base_dir: Path = None) -> Path:
    """출력 파일 경로 결정"""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / 'data' / 'reports'

    base_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path.name

    return output_path


def generate_json_path(output_path: Path) -> str:
    """JSON 파일 경로 생성"""
    json_path = str(output_path).replace('.docx', '.json').replace('.pdf', '.json')

    if json_path == str(output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"{output_path}_{timestamp}.json"

    return json_path


def generate_document(report_data: Dict, output_path: Path, json_only: bool) -> None:
    """문서 파일 생성 (.docx 또는 .pdf)"""
    if json_only:
        return

    print(f"\n📄 문서 생성 중...")
    doc_generator = DocumentGenerator()

    output_str = str(output_path)

    if output_str.endswith('.pdf'):
        doc_generator.generate_pdf_report(report_data, output_str)
    elif output_str.endswith('.docx'):
        doc_generator.generate_word_report(report_data, output_str)
    else:
        print("⚠️ 지원되지 않는 파일 형식입니다. .docx 또는 .pdf를 사용하세요.")


def print_completion_message(json_path: str, output_path: Path, json_only: bool) -> None:
    """완료 메시지 출력"""
    print("\n" + "=" * 100)
    print("✅ 보고서 생성 완료!")
    print("=" * 100)
    print(f"📄 JSON: {json_path}")
    if not json_only:
        print(f"📄 문서: {output_path}")
    print()


def main():
    """메인 실행 함수"""
    # CLI 인자 파싱
    parser = create_argument_parser()
    args = parser.parse_args()

    # 날짜 필터 파싱
    date_filter = parse_date_range(
        date_input=args.date_range,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 보고서 생성기 초기화
    generator = ReportGenerator(report_type=args.type)

    # 시작 정보 출력
    print_report_header(
        report_type=args.type,
        questions=args.questions,
        date_filter=date_filter,
        retriever_name=generator.retriever_config['display_name'],
        llm_name=generator.llm_config['display_name']
    )

    # 보고서 데이터 생성
    report_data = generator.generate_report(args.questions, date_filter)

    # 메타데이터 추가
    report_data = add_metadata_to_report(report_data, args.author)

    # 출력 경로 결정
    output_path = resolve_output_path(args.output)

    # JSON 파일 저장
    json_path = generate_json_path(output_path)
    generator.save_json(report_data, json_path)

    # 문서 파일 생성
    generate_document(report_data, output_path, args.json_only)

    # 완료 메시지 출력
    print_completion_message(json_path, output_path, args.json_only)


if __name__ == "__main__":
    main()
