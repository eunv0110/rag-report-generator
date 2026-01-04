"""파일 유틸리티 함수

JSON 파일 읽기/쓰기 등 파일 관련 유틸리티를 제공합니다.
"""

import json
from typing import Union, List, Dict, Any
from pathlib import Path


def save_json(data: Union[List[Dict[str, Any]], Dict[str, Any]], filepath: str) -> None:
    """JSON 파일로 저장

    Args:
        data: 저장할 데이터 (딕셔너리 또는 리스트)
        filepath: 저장할 파일 경로

    Raises:
        IOError: 파일 저장 실패 시
    """
    try:
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 저장: {filepath}")
    except Exception as e:
        raise IOError(f"JSON 파일 저장 실패 ({filepath}): {e}") from e


def load_json(filepath: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """JSON 파일 로드

    Args:
        filepath: 로드할 파일 경로

    Returns:
        로드된 JSON 데이터

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        json.JSONDecodeError: JSON 파싱 실패 시
    """
    filepath_obj = Path(filepath)
    if not filepath_obj.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)