#!/usr/bin/env python3
"""날짜 관련 유틸리티 함수"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
import os


def parse_date_range(
    date_input: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """
    날짜 범위를 파싱합니다.

    두 가지 방식 지원:
    1. 명시적 날짜: start_date, end_date로 직접 지정 (우선순위 높음)
    2. 자연어: date_input으로 "이번 주", "지난주" 등 입력

    Args:
        date_input: 자연어 날짜 표현 ("이번 주", "지난주", "12월 첫째주" 등)
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)

    Returns:
        (start_date, end_date) 튜플 또는 None

    Examples:
        >>> parse_date_range(start_date="2025-12-01", end_date="2025-12-07")
        ('2025-12-01T00:00:00.000Z', '2025-12-07T23:59:59.999Z')

        >>> parse_date_range(date_input="이번 주")
        ('2025-12-22T00:00:00.000Z', '2025-12-28T23:59:59.999Z')
    """

    # 1. 명시적 날짜가 제공된 경우 (우선순위)
    if start_date or end_date:
        return parse_explicit_dates(start_date, end_date)

    # 2. 자연어 입력 파싱
    if date_input:
        return parse_natural_language_date(date_input)

    return None


def parse_explicit_dates(
    start_date: Optional[str],
    end_date: Optional[str]
) -> Optional[Tuple[str, str]]:
    """명시적 날짜를 ISO 형식으로 변환"""

    if not start_date and not end_date:
        return None

    # 기본값 설정
    if not start_date:
        start_date = "1970-01-01"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        # YYYY-MM-DD 형식 검증
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # ISO 8601 형식으로 변환 (Notion 시간 형식)
        start_iso = start_dt.strftime("%Y-%m-%dT00:00:00.000Z")
        end_iso = end_dt.strftime("%Y-%m-%dT23:59:59.999Z")

        return (start_iso, end_iso)

    except ValueError as e:
        print(f"❌ 날짜 형식 오류: {e}")
        print("   올바른 형식: YYYY-MM-DD (예: 2025-12-01)")
        return None


def parse_natural_language_date(date_input: str) -> Optional[Tuple[str, str]]:
    """
    자연어 날짜 표현을 파싱합니다.

    지원하는 표현:
    - "이번 주" / "이번주"
    - "지난주" / "지난 주"
    - "저번주" / "저번 주"
    - "N주 전" (예: "2주 전")
    - "MM월 N주차" (예: "12월 2주차")
    - "MM월 첫째주" / "MM월 둘째주" 등
    """

    date_input = date_input.strip()
    today = datetime.now()

    # 1. "이번 주" / "이번주"
    if re.match(r"이번\s*주", date_input):
        return get_this_week()

    # 2. "지난주" / "저번주"
    if re.match(r"(지난|저번)\s*주", date_input):
        return get_last_week()

    # 3. "N주 전" (예: "2주 전")
    match = re.match(r"(\d+)\s*주\s*전", date_input)
    if match:
        weeks_ago = int(match.group(1))
        return get_n_weeks_ago(weeks_ago)

    # 4. "MM월 N주차" (예: "12월 2주차")
    match = re.match(r"(\d+)\s*월\s*(\d+)\s*주차", date_input)
    if match:
        month = int(match.group(1))
        week_num = int(match.group(2))
        return get_week_of_month(month, week_num)

    # 5. "MM월 첫째주" / "둘째주" 등
    match = re.match(r"(\d+)\s*월\s*(첫째|둘째|셋째|넷째|다섯째)\s*주", date_input)
    if match:
        month = int(match.group(1))
        week_name = match.group(2)
        week_map = {"첫째": 1, "둘째": 2, "셋째": 3, "넷째": 4, "다섯째": 5}
        week_num = week_map.get(week_name)
        if week_num:
            return get_week_of_month(month, week_num)

    # 6. LLM을 사용한 자연어 파싱 (fallback)
    return parse_with_llm(date_input)


def get_this_week() -> Tuple[str, str]:
    """이번 주의 시작일(월요일)과 종료일(일요일) 반환"""
    today = datetime.now()
    # 월요일 = 0, 일요일 = 6
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)

    start_iso = start.strftime("%Y-%m-%dT00:00:00.000Z")
    end_iso = end.strftime("%Y-%m-%dT23:59:59.999Z")

    return (start_iso, end_iso)


def get_last_week() -> Tuple[str, str]:
    """지난 주의 시작일과 종료일 반환"""
    today = datetime.now()
    last_week_start = today - timedelta(days=today.weekday() + 7)
    last_week_end = last_week_start + timedelta(days=6)

    start_iso = last_week_start.strftime("%Y-%m-%dT00:00:00.000Z")
    end_iso = last_week_end.strftime("%Y-%m-%dT23:59:59.999Z")

    return (start_iso, end_iso)


def get_n_weeks_ago(n: int) -> Tuple[str, str]:
    """N주 전의 시작일과 종료일 반환"""
    today = datetime.now()
    target_week_start = today - timedelta(days=today.weekday() + 7 * n)
    target_week_end = target_week_start + timedelta(days=6)

    start_iso = target_week_start.strftime("%Y-%m-%dT00:00:00.000Z")
    end_iso = target_week_end.strftime("%Y-%m-%dT23:59:59.999Z")

    return (start_iso, end_iso)


def get_week_of_month(month: int, week_num: int, year: Optional[int] = None) -> Optional[Tuple[str, str]]:
    """
    특정 월의 N주차 범위 반환

    Args:
        month: 월 (1-12)
        week_num: 주차 (1-5)
        year: 연도 (None이면 현재 연도)
    """
    if year is None:
        year = datetime.now().year

    # 해당 월의 첫째 날
    first_day = datetime(year, month, 1)

    # 첫째 주 월요일 찾기
    days_to_monday = (7 - first_day.weekday()) % 7
    if days_to_monday == 0 and first_day.weekday() != 0:
        days_to_monday = 7
    first_monday = first_day + timedelta(days=days_to_monday)

    # N주차 계산
    target_week_start = first_monday + timedelta(weeks=week_num - 1)
    target_week_end = target_week_start + timedelta(days=6)

    # 해당 월을 벗어나는지 확인
    if target_week_start.month != month:
        print(f"⚠️ {year}년 {month}월에는 {week_num}주차가 없습니다.")
        return None

    start_iso = target_week_start.strftime("%Y-%m-%dT00:00:00.000Z")
    end_iso = target_week_end.strftime("%Y-%m-%dT23:59:59.999Z")

    return (start_iso, end_iso)


def extract_date_filter_from_question(question: str) -> Optional[Tuple[str, str]]:
    """
    질문에서 날짜 관련 표현을 추출하여 날짜 필터 생성

    Args:
        question: 질문 텍스트

    Returns:
        (start_date, end_date) 튜플 또는 None

    Examples:
        >>> extract_date_filter_from_question("9월 첫째주 주요 업무 내용을 요약해줘")
        ('2025-09-01T00:00:00.000Z', '2025-09-07T23:59:59.999Z')

        >>> extract_date_filter_from_question("이번 주 일정 알려줘")
        ('2025-12-23T00:00:00.000Z', '2025-12-29T23:59:59.999Z')
    """
    # 날짜 관련 패턴 정의 (우선순위 순서대로)
    patterns = [
        # "N월 N주차" 패턴
        (r'(\d{1,2})월\s*(\d{1})주차', lambda m: f"{m.group(1)}월 {m.group(2)}주차"),
        # "N월 첫째주/둘째주/셋째주/넷째주" 패턴
        (r'(\d{1,2})월\s*(첫째주|둘째주|셋째주|넷째주|다섯째주)', lambda m: f"{m.group(1)}월 {m.group(2)}"),
        # "N월" 패턴
        (r'(\d{1,2})월', lambda m: f"{m.group(1)}월"),
        # "이번 주/이번주" 패턴
        (r'이번\s*주', lambda m: "이번 주"),
        # "이번 달/이번달" 패턴
        (r'이번\s*달', lambda m: "이번 달"),
        # "지난 주/지난주" 패턴
        (r'지난\s*주', lambda m: "지난 주"),
        # "지난 달/지난달" 패턴
        (r'지난\s*달', lambda m: "지난 달"),
        # "최근 N주/N주간" 패턴
        (r'최근\s*(\d{1,2})\s*주', lambda m: f"최근 {m.group(1)}주"),
        (r'(\d{1,2})\s*주간', lambda m: f"최근 {m.group(1)}주"),
    ]

    # 각 패턴을 순서대로 확인
    for pattern, formatter in patterns:
        match = re.search(pattern, question)
        if match:
            date_str = formatter(match)
            try:
                # parse_date_range를 사용해 날짜 범위로 변환
                date_range = parse_date_range(date_input=date_str)
                if date_range:
                    return date_range
            except Exception as e:
                print(f"⚠️ 날짜 파싱 실패 ('{date_str}'): {e}")
                continue

    return None


def parse_with_llm(date_input: str) -> Optional[Tuple[str, str]]:
    """
    LLM을 사용하여 자연어 날짜를 파싱합니다.

    예: "크리스마스 주", "연말", "여름 방학" 등
    """
    try:
        from langchain.chat_models import init_chat_model
        from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT
        import json

        os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
        os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

        model = init_chat_model(
            "azure_ai:gpt-4.1",
            temperature=0,
            max_completion_tokens=200
        )

        today = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""오늘 날짜는 {today}입니다.

다음 자연어 날짜 표현을 시작일과 종료일로 변환해주세요: "{date_input}"

반드시 다음 JSON 형식으로만 응답하세요:
{{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD"
}}

예시:
- "크리스마스 주" → {{"start_date": "2025-12-22", "end_date": "2025-12-28"}}
- "연말" → {{"start_date": "2025-12-25", "end_date": "2025-12-31"}}
"""

        response = model.invoke([{"role": "user", "content": prompt}])
        result = json.loads(response.content.strip())

        start_date = result.get("start_date")
        end_date = result.get("end_date")

        if start_date and end_date:
            return parse_explicit_dates(start_date, end_date)

    except Exception as e:
        print(f"⚠️ LLM 날짜 파싱 실패: {e}")

    return None


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("날짜 파싱 유틸리티 테스트")
    print("=" * 60)

    test_cases = [
        # 명시적 날짜
        ({"start_date": "2025-12-01", "end_date": "2025-12-07"}, "명시적 날짜"),

        # 자연어
        ({"date_input": "이번 주"}, "이번 주"),
        ({"date_input": "지난주"}, "지난주"),
        ({"date_input": "2주 전"}, "2주 전"),
        ({"date_input": "12월 2주차"}, "12월 2주차"),
        ({"date_input": "12월 첫째주"}, "12월 첫째주"),
    ]

    for kwargs, label in test_cases:
        print(f"\n📅 테스트: {label}")
        result = parse_date_range(**kwargs)
        if result:
            print(f"   시작: {result[0]}")
            print(f"   종료: {result[1]}")
        else:
            print("   ❌ 파싱 실패")
