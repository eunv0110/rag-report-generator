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

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from report_generator.report_generator import ReportGenerator
from report_generator.document_generator import DocumentGenerator
from utils.date_utils import parse_date_range


def main():
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

    args = parser.parse_args()

    # 날짜 필터 파싱
    date_filter = parse_date_range(
        date_input=args.date_range,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 통합 보고서 생성기 초기화
    generator = ReportGenerator(report_type=args.type)

    report_title = "주간 보고서" if args.type == "weekly" else "최종 보고서"
    print(f"\n📊 {report_title} 생성 시작...")
    print(f"🔧 설정: {generator.retriever_config['display_name']} + {generator.llm_config['display_name']}")

    # 질문 출력
    print(f"\n📝 질문 ({len(args.questions)}개):")
    for i, q in enumerate(args.questions, 1):
        print(f"  {i}. {q}")

    if date_filter:
        print(f"\n📅 날짜 필터: {date_filter[0][:10]} ~ {date_filter[1][:10]}")

    print()

    # 보고서 생성
    report_data = generator.generate_report(args.questions, date_filter)

    # 작성자 및 작성일자 정보 추가
    author = args.author if args.author else getpass.getuser()
    report_data["author"] = author
    report_data["created_date"] = datetime.now().strftime("%Y-%m-%d")
    report_data["created_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 출력 경로를 data/reports 디렉토리로 설정
    reports_dir = Path(__file__).parent.parent / 'data' / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 출력 파일명이 상대 경로나 파일명만 있는 경우 reports 디렉토리에 저장
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = reports_dir / output_path.name

    # JSON 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = str(output_path).replace('.docx', '.json').replace('.pdf', '.json')
    if json_path == str(output_path):
        json_path = f"{output_path}_{timestamp}.json"

    generator.save_json(report_data, json_path)

    # 문서 생성
    if not args.json_only:
        print(f"\n📄 문서 생성 중...")
        doc_generator = DocumentGenerator()

        if str(output_path).endswith('.pdf'):
            doc_generator.generate_pdf_report(report_data, str(output_path))
        elif str(output_path).endswith('.docx'):
            doc_generator.generate_word_report(report_data, str(output_path))
        else:
            print("⚠️ 지원되지 않는 파일 형식입니다. .docx 또는 .pdf를 사용하세요.")
            print(f"💡 JSON 파일을 사용하세요: {json_path}")

    print("\n" + "=" * 100)
    print("✅ 보고서 생성 완료!")
    print("=" * 100)
    print(f"📄 JSON: {json_path}")
    if not args.json_only:
        print(f"📄 문서: {output_path}")
    print()


if __name__ == "__main__":
    main()
