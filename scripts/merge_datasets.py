#!/usr/bin/env python3
"""데이터셋 병합

여러 QA 데이터셋 파일을 하나로 병합하는 스크립트
중복 제거 옵션 제공
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_datasets(
    dataset1_path: str,
    dataset2_path: str,
    output_path: str,
    remove_duplicates: bool = True
):
    """
    두 개의 QA 데이터셋 병합

    Args:
        dataset1_path: 첫 번째 데이터셋 경로
        dataset2_path: 두 번째 데이터셋 경로
        output_path: 출력 파일 경로
        remove_duplicates: 중복 질문 제거 여부
    """
    print("=" * 80)
    print("🔗 QA 데이터셋 병합")
    print("=" * 80)

    # 데이터셋 로드
    print(f"\n📂 데이터셋 1 로드: {dataset1_path}")
    dataset1 = load_dataset(dataset1_path)
    print(f"   ✅ {len(dataset1)}개 항목")

    print(f"\n📂 데이터셋 2 로드: {dataset2_path}")
    dataset2 = load_dataset(dataset2_path)
    print(f"   ✅ {len(dataset2)}개 항목")

    # 병합
    merged = dataset1 + dataset2
    print(f"\n🔗 병합 완료: {len(merged)}개 항목")

    # 중복 제거 (옵션)
    if remove_duplicates:
        print(f"\n🔍 중복 질문 제거 중...")
        seen_questions = set()
        unique_data = []
        duplicates = 0

        for item in merged:
            question = item.get('question', '').strip()
            if question and question not in seen_questions:
                seen_questions.add(question)
                unique_data.append(item)
            else:
                duplicates += 1

        print(f"   ✅ 중복 제거: {duplicates}개 항목 제거됨")
        print(f"   ✅ 최종: {len(unique_data)}개 항목")
        merged = unique_data

    # 통계
    print(f"\n📊 병합된 데이터셋 통계:")

    # 카테고리 분포
    from collections import Counter
    categories = [item.get('metadata', {}).get('category', 'unknown') for item in merged]
    category_counts = Counter(categories)

    print(f"\n   카테고리 분포:")
    for cat, count in category_counts.most_common():
        print(f"      {cat}: {count}개 ({count/len(merged)*100:.1f}%)")

    # 소스 분포
    sources = [item.get('metadata', {}).get('source', 'unknown') for item in merged]
    source_counts = Counter(sources)

    print(f"\n   소스 분포:")
    for src, count in source_counts.most_common():
        print(f"      {src}: {count}개 ({count/len(merged)*100:.1f}%)")

    # 답변 길이 통계
    answer_lengths = [len(item['ground_truth']) for item in merged]
    avg_len = sum(answer_lengths) / len(answer_lengths)
    min_len = min(answer_lengths)
    max_len = max(answer_lengths)

    print(f"\n   답변 길이:")
    print(f"      평균: {avg_len:.0f}자")
    print(f"      최소: {min_len}자")
    print(f"      최대: {max_len}자")

    # 저장
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n💾 저장 완료: {output_path}")
    print(f"   총 {len(merged)}개 항목")

    print("\n" + "=" * 80)
    print("✅ 병합 완료!")
    print("=" * 80)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="QA 데이터셋 병합",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 병합 (중복 제거)
  python merge_qa_datasets.py \\
    --dataset1 data/evaluation/llm_generated_qa_v2.json \\
    --dataset2 data/evaluation/concise_qa_new.json \\
    --output data/evaluation/merged_qa_v3.json

  # 중복 유지
  python merge_qa_datasets.py \\
    --dataset1 data/evaluation/llm_generated_qa_v2.json \\
    --dataset2 data/evaluation/concise_qa_new.json \\
    --output data/evaluation/merged_qa_v3.json \\
    --keep-duplicates
        """
    )
    parser.add_argument(
        "--dataset1",
        type=str,
        required=True,
        help="첫 번째 데이터셋 경로"
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        required=True,
        help="두 번째 데이터셋 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 파일 경로"
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="중복 질문을 유지 (기본값: 제거)"
    )

    args = parser.parse_args()

    merge_datasets(
        dataset1_path=args.dataset1,
        dataset2_path=args.dataset2,
        output_path=args.output,
        remove_duplicates=not args.keep_duplicates
    )


if __name__ == "__main__":
    main()
