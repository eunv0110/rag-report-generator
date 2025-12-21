#!/usr/bin/env python3
"""평가 데이터셋 자동 생성 스크립트"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
from typing import List, Dict, Any, Optional
from utils.file_utils import load_json, save_json
from config.settings import DATA_DIR


def load_notion_data() -> List[Dict[str, Any]]:
    """Notion 데이터 로드"""
    data_file = DATA_DIR / "notion_data.json"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Notion 데이터가 없습니다: {data_file}\n"
            "먼저 'python scripts/build_vectordb.py'를 실행하세요."
        )
    return load_json(str(data_file))


def extract_text_from_page(page: Dict[str, Any], max_length: int = 500) -> str:
    """페이지에서 텍스트 추출"""
    # content 필드가 문자열로 제공되는 경우
    content = page.get("content", "")

    if isinstance(content, str):
        # 이미지 태그 제거
        import re
        content = re.sub(r'\[Image:.*?\]', '', content)

        # 길이 제한
        if len(content) > max_length:
            content = content[:max_length] + "..."

        return content.strip()

    # 기존 blocks 형식 지원 (호환성)
    texts = []
    for block in page.get("blocks", []):
        block_type = block.get("type")

        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3"]:
            rich_texts = block.get(block_type, {}).get("rich_text", [])
            for rt in rich_texts:
                text = rt.get("plain_text", "").strip()
                if text:
                    texts.append(text)

        elif block_type == "bulleted_list_item":
            rich_texts = block.get("bulleted_list_item", {}).get("rich_text", [])
            for rt in rich_texts:
                text = rt.get("plain_text", "").strip()
                if text:
                    texts.append(f"• {text}")

    full_text = " ".join(texts)

    # 길이 제한
    if len(full_text) > max_length:
        full_text = full_text[:max_length] + "..."

    return full_text


def generate_manual_template(
    pages: List[Dict[str, Any]],
    num_samples: int = 10,
    output_file: str = "data/evaluation/manual_qa_template.json"
) -> List[Dict[str, Any]]:
    """
    수동 작성용 QA 템플릿 생성

    Args:
        pages: Notion 페이지 리스트
        num_samples: 생성할 샘플 수
        output_file: 저장 경로

    Returns:
        템플릿 데이터
    """
    print(f"\n📝 수동 작성용 템플릿 생성 중... (샘플 수: {num_samples})")

    # 랜덤으로 페이지 선택
    selected_pages = random.sample(pages, min(num_samples, len(pages)))

    template_data = []

    for i, page in enumerate(selected_pages, 1):
        page_title = page.get("title", "Untitled")
        page_id = page.get("page_id", "")
        content_preview = extract_text_from_page(page, max_length=300)

        template_data.append({
            "id": f"qa_{i}",
            "question": f"[TODO: {page_title}에 대한 질문을 작성하세요]",
            "ground_truth": "[TODO: 정답을 작성하세요]",
            "context_page_id": page_id,
            "context_page_title": page_title,
            "content_preview": content_preview,
            "metadata": {
                "category": "[TODO: 카테고리]",
                "difficulty": "medium",
                "source": "manual"
            }
        })

    # 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(template_data, str(output_path))

    print(f"✅ 템플릿 저장: {output_path}")
    print(f"\n💡 다음 단계:")
    print(f"   1. {output_path} 파일을 열기")
    print(f"   2. [TODO] 부분을 실제 질문/답변으로 수정")
    print(f"   3. content_preview를 참고하여 작성")

    return template_data


def generate_simple_qa_from_headings(
    pages: List[Dict[str, Any]],
    num_samples: int = 20,
    output_file: str = "data/evaluation/auto_qa_from_headings.json"
) -> List[Dict[str, Any]]:
    """
    제목/헤딩 기반 간단한 QA 생성

    Args:
        pages: Notion 페이지 리스트
        num_samples: 생성할 샘플 수
        output_file: 저장 경로

    Returns:
        QA 데이터
    """
    print(f"\n🤖 제목 기반 자동 QA 생성 중... (샘플 수: {num_samples})")

    qa_data = []

    for page in pages:
        page_title = page.get("title", "").strip()
        page_id = page.get("page_id", "")

        if not page_title:
            continue

        # 페이지에서 헤딩과 내용 추출
        headings_with_content = []
        current_heading = None
        current_content = []

        for block in page.get("blocks", []):
            block_type = block.get("type")

            if block_type in ["heading_1", "heading_2", "heading_3"]:
                # 이전 헤딩 저장
                if current_heading and current_content:
                    headings_with_content.append({
                        "heading": current_heading,
                        "content": " ".join(current_content)
                    })

                # 새 헤딩 시작
                rich_texts = block.get(block_type, {}).get("rich_text", [])
                current_heading = " ".join([rt.get("plain_text", "") for rt in rich_texts]).strip()
                current_content = []

            elif block_type == "paragraph":
                rich_texts = block.get("paragraph", {}).get("rich_text", [])
                text = " ".join([rt.get("plain_text", "") for rt in rich_texts]).strip()
                if text:
                    current_content.append(text)

        # 마지막 헤딩 저장
        if current_heading and current_content:
            headings_with_content.append({
                "heading": current_heading,
                "content": " ".join(current_content)
            })

        # QA 생성
        # 1. 페이지 제목 기반 질문
        qa_data.append({
            "question": f"{page_title}에 대해 설명해주세요.",
            "ground_truth": extract_text_from_page(page, max_length=500),
            "context_page_id": page_id,
            "metadata": {
                "category": "page_summary",
                "difficulty": "easy",
                "source": "auto_title"
            }
        })

        # 2. 각 헤딩 기반 질문
        for heading_data in headings_with_content[:3]:  # 최대 3개
            heading = heading_data["heading"]
            content = heading_data["content"]

            if len(content) < 50:  # 너무 짧은 내용은 스킵
                continue

            qa_data.append({
                "question": f"{page_title}의 '{heading}' 섹션에 대해 설명해주세요.",
                "ground_truth": content[:500],
                "context_page_id": page_id,
                "metadata": {
                    "category": "section_explanation",
                    "difficulty": "medium",
                    "source": "auto_heading"
                }
            })

        if len(qa_data) >= num_samples:
            break

    # 샘플 수만큼 자르기
    qa_data = qa_data[:num_samples]

    # 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(qa_data, str(output_path))

    print(f"✅ 자동 QA 생성 완료: {len(qa_data)}개")
    print(f"✅ 저장 위치: {output_path}")

    return qa_data


def generate_qa_with_llm(
    pages: List[Dict[str, Any]],
    num_samples: int = 10,
    llm_provider: str = "openai",
    output_file: str = "data/evaluation/llm_generated_qa.json"
) -> List[Dict[str, Any]]:
    """
    LLM을 사용한 고품질 QA 생성

    Args:
        pages: Notion 페이지 리스트
        num_samples: 생성할 샘플 수
        llm_provider: LLM 제공자 (openai, azure, openrouter)
        output_file: 저장 경로

    Returns:
        QA 데이터
    """
    print(f"\n🤖 LLM 기반 QA 생성 중... (샘플 수: {num_samples})")

    try:
        # LLM 클라이언트 초기화
        if llm_provider == "openai":
            from openai import OpenAI
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")

            client = OpenAI(api_key=api_key)
            model = "gpt-4o-mini"

        elif llm_provider == "azure":
            from openai import OpenAI
            from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT

            if not AZURE_AI_CREDENTIAL:
                raise ValueError("AZURE_AI_CREDENTIAL이 설정되지 않았습니다")

            client = OpenAI(
                api_key=AZURE_AI_CREDENTIAL,
                base_url=AZURE_AI_ENDPOINT
            )
            # Azure AI에서 사용 가능한 모델 사용
            model = "gpt-5.1"

        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {llm_provider}")

        qa_data = []
        selected_pages = random.sample(pages, min(num_samples, len(pages)))

        for page in selected_pages:
            page_title = page.get("title", "Untitled")
            page_id = page.get("page_id", "")
            content = extract_text_from_page(page, max_length=1000)

            if len(content) < 100:
                print(f"  ⏭️  {page_title}: 내용이 너무 짧음 ({len(content)}자), 스킵")
                continue

            # LLM 프롬프트
            prompt = f"""다음 문서 내용을 읽고 평가용 질문-답변 쌍을 생성해주세요.

문서 제목: {page_title}
문서 내용:
{content}

요구사항:
1. 문서 내용을 잘 평가할 수 있는 구체적인 질문 1개 생성
2. 질문에 대한 정확하고 상세한 답변 작성
3. JSON 형식으로 응답

응답 형식:
{{
  "question": "질문 내용",
  "ground_truth": "답변 내용"
}}
"""

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "당신은 RAG 시스템 평가를 위한 고품질 QA 데이터를 생성하는 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)

                qa_data.append({
                    "question": result.get("question", ""),
                    "ground_truth": result.get("ground_truth", ""),
                    "context_page_id": page_id,
                    "metadata": {
                        "category": "llm_generated",
                        "difficulty": "medium",
                        "source": f"llm_{llm_provider}",
                        "page_title": page_title
                    }
                })

                print(f"  ✅ {page_title}: QA 생성 완료")

            except Exception as e:
                print(f"  ❌ {page_title}: QA 생성 실패 - {e}")
                continue

        # 저장
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(qa_data, str(output_path))

        print(f"\n✅ LLM 기반 QA 생성 완료: {len(qa_data)}개")
        print(f"✅ 저장 위치: {output_path}")

        return qa_data

    except Exception as e:
        print(f"\n❌ LLM 기반 QA 생성 실패: {e}")
        print("\n💡 Tip: API 키를 확인하고 다시 시도하거나 --method auto 옵션을 사용하세요")
        return []


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="평가 데이터셋 생성")
    parser.add_argument(
        "--method",
        type=str,
        choices=["manual", "auto", "llm"],
        default="auto",
        help="생성 방법: manual(수동 템플릿), auto(자동), llm(LLM 사용)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="생성할 샘플 수"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "azure"],
        default="azure",
        help="LLM 제공자 (llm 방법 사용 시)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="출력 파일 경로"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("📊 평가 데이터셋 생성 시작")
    print("=" * 60)

    # Notion 데이터 로드
    pages = load_notion_data()
    print(f"\n✅ Notion 데이터 로드: {len(pages)}개 페이지")

    # 생성 방법에 따라 실행
    if args.method == "manual":
        output = args.output or "data/evaluation/manual_qa_template.json"
        generate_manual_template(pages, args.num_samples, output)

    elif args.method == "auto":
        output = args.output or "data/evaluation/auto_qa_from_headings.json"
        generate_simple_qa_from_headings(pages, args.num_samples, output)

    elif args.method == "llm":
        output = args.output or f"data/evaluation/llm_generated_qa_{args.llm_provider}.json"
        generate_qa_with_llm(pages, args.num_samples, args.llm_provider, output)

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
