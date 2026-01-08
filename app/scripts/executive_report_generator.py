#!/usr/bin/env python3
"""최종 보고서 생성기 (Backward Compatibility Wrapper)

이 파일은 하위 호환성을 위한 wrapper입니다.
내부적으로 통합된 ReportGenerator를 사용합니다.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가 (직접 실행 시)
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

# 통합 ReportGenerator 임포트
try:
    from .report_generator import ReportGenerator
except ImportError:
    from app.report_generator.report_generator import ReportGenerator

from app.utils.dates import parse_date_range


class ExecutiveReportGenerator(ReportGenerator):
    """최종 보고서 생성기 (Executive Report)

    하위 호환성을 위한 wrapper 클래스.
    내부적으로 ReportGenerator(report_type='executive')를 사용합니다.
    """

    def __init__(self, config_path=None):
        """
        Args:
            config_path: 설정 파일 경로. 지정하지 않으면 기본 경로 사용.
        """
        if config_path is None:
            # 기본 executive config 사용
            super().__init__(report_type='executive')
        else:
            # 커스텀 config 사용
            super().__init__(config_path=config_path)


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description="최종 보고서 생성기 (OpenAI + DeepSeek-V3.1)")
    parser.add_argument("--questions", type=str, nargs='+', help="질문 리스트")
    parser.add_argument("--date-range", type=str, help="날짜 범위 (예: '이번 주', '12월 2주차')")
    parser.add_argument("--start-date", type=str, help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로")

    args = parser.parse_args()

    # 날짜 필터 파싱
    date_filter = parse_date_range(
        date_input=args.date_range,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 보고서 생성기 초기화
    generator = ExecutiveReportGenerator()

    # 질문 설정
    if args.questions:
        questions = args.questions
    else:
        # 설정 파일의 기본 질문 사용
        questions = generator.default_questions

    # 보고서 생성
    report_data = generator.generate_report(questions, date_filter)

    # JSON 저장
    if args.output:
        generator.save_json(report_data, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"data/reports/executive_report_{timestamp}.json"
        generator.save_json(report_data, default_output)


if __name__ == "__main__":
    main()
