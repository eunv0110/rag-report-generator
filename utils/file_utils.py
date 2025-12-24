import json
from typing import List, Dict
from pathlib import Path

def save_json(data: List[Dict], filepath: str):
    """JSON 파일로 저장"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 저장: {filepath}")

def load_json(filepath: str) -> List[Dict]:
    """JSON 파일 로드"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)