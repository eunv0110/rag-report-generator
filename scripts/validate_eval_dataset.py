#!/usr/bin/env python3
"""평가 데이터셋 검증 스크립트"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict, Any
from utils.file_utils import load_json


def validate_qa_dataset(file_path: str) -> Dict[str, Any]:
    """
    QA 데이터셋 검증

    Args:
        file_path: 검증할 파일 경로

    Returns:
        검증 결과
    """
    print(f"🔍 검증 중: {file_path}")

    try:
        data = load_json(file_path)
    except Exception as e:
        return {
            "valid": False,
            "error": f"파일 로드 실패: {e}",
            "stats": {}
        }

    if not isinstance(data, list):
        return {
            "valid": False,
            "error": "데이터는 리스트 형식이어야 합니다",
            "stats": {}
        }

    errors = []
    warnings = []
    stats = {
        "total_items": len(data),
        "valid_items": 0,
        "missing_question": 0,
        "missing_ground_truth": 0,
        "short_question": 0,
        "short_answer": 0,
        "has_metadata": 0,
        "categories": set(),
        "difficulties": set()
    }

    for i, item in enumerate(data, 1):
        item_id = item.get("id", f"item_{i}")

        # 필수 필드 체크
        if "question" not in item:
            errors.append(f"{item_id}: 'question' 필드 누락")
            stats["missing_question"] += 1
            continue

        if "ground_truth" not in item:
            errors.append(f"{item_id}: 'ground_truth' 필드 누락")
            stats["missing_ground_truth"] += 1
            continue

        question = item["question"]
        ground_truth = item["ground_truth"]

        # TODO 마커 체크
        if "[TODO" in question or "TODO]" in question:
            warnings.append(f"{item_id}: 질문에 TODO 마커가 남아있음")

        if "[TODO" in ground_truth or "TODO]" in ground_truth:
            warnings.append(f"{item_id}: 답변에 TODO 마커가 남아있음")

        # 길이 체크
        if len(question.strip()) < 10:
            warnings.append(f"{item_id}: 질문이 너무 짧음 ({len(question)}자)")
            stats["short_question"] += 1

        if len(ground_truth.strip()) < 20:
            warnings.append(f"{item_id}: 답변이 너무 짧음 ({len(ground_truth)}자)")
            stats["short_answer"] += 1

        # 메타데이터 체크
        if "metadata" in item:
            stats["has_metadata"] += 1
            metadata = item["metadata"]

            if "category" in metadata:
                stats["categories"].add(metadata["category"])

            if "difficulty" in metadata:
                stats["difficulties"].add(metadata["difficulty"])

        stats["valid_items"] += 1

    # 통계 변환
    stats["categories"] = list(stats["categories"])
    stats["difficulties"] = list(stats["difficulties"])

    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 검증 결과")
    print("=" * 60)

    print(f"\n총 항목 수: {stats['total_items']}")
    print(f"유효한 항목: {stats['valid_items']}")
    print(f"메타데이터 포함: {stats['has_metadata']}")

    if stats["categories"]:
        print(f"\n카테고리: {', '.join(stats['categories'])}")

    if stats["difficulties"]:
        print(f"난이도: {', '.join(stats['difficulties'])}")

    # 에러 출력
    if errors:
        print(f"\n❌ 에러 ({len(errors)}개):")
        for error in errors[:10]:  # 최대 10개만 표시
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 외 {len(errors) - 10}개")

    # 경고 출력
    if warnings:
        print(f"\n⚠️  경고 ({len(warnings)}개):")
        for warning in warnings[:10]:  # 최대 10개만 표시
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... 외 {len(warnings) - 10}개")

    # 최종 판정
    is_valid = len(errors) == 0 and stats["valid_items"] > 0

    if is_valid:
        print("\n✅ 데이터셋이 유효합니다!")
    else:
        print("\n❌ 데이터셋에 문제가 있습니다. 위의 에러를 수정해주세요.")

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats
    }


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="평가 데이터셋 검증")
    parser.add_argument(
        "file_path",
        type=str,
        help="검증할 JSON 파일 경로"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="경고도 에러로 처리"
    )

    args = parser.parse_args()

    result = validate_qa_dataset(args.file_path)

    if args.strict and result["warnings"]:
        print("\n⚠️  --strict 모드: 경고가 있어 실패 처리됩니다")
        sys.exit(1)

    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
