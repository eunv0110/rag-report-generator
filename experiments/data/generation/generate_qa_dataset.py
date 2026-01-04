#!/usr/bin/env python3
"""통합 QA 데이터셋 생성 스크립트 - 여러 QA 타입 지원"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
from typing import List, Dict, Any, Optional, Literal
from openai import AzureOpenAI, OpenAI
from config.settings import (
    AZURE_AI_CREDENTIAL,
    AZURE_AI_ENDPOINT,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DATA_DIR
)
from utils.files import load_json, save_json

# 프롬프트 템플릿 임포트
sys.path.insert(0, str(Path(__file__).parent.parent / "prompts" / "templates" / "data"))
from qa_prompts import (
    SIMPLE_QUESTION_PROMPT,
    SIMPLE_ANSWER_PROMPT,
    CONCISE_QUESTION_PROMPT,
    CONCISE_ANSWER_PROMPT,
    STANDARD_QA_PROMPT,
    HARD_QA_PROMPT,
    USER_SCENARIO_QUESTION_PROMPT,
    USER_SCENARIO_ANSWER_PROMPT
)


# ============================================================================
# 상수 정의
# ============================================================================
QA_TYPE = Literal["simple", "concise", "standard", "hard", "user_scenario"]

DEFAULT_NUM_SAMPLES = 20
MAX_TEXT_LENGTH = 1000
MIN_CONTENT_LENGTH = 100
DEFAULT_OUTPUT_DIR = "data/evaluation"

# Hard QA 설정 (여러 문서 참조)
MIN_DOCS_PER_QUESTION = 1
MAX_DOCS_PER_QUESTION = 3
MULTI_DOC_PROBABILITY = 0.5


# ============================================================================
# QA 타입별 설정
# ============================================================================
QA_TYPE_CONFIG = {
    "simple": {
        "description": "간단한 질문-답변 (3-5문장)",
        "question_prompt": SIMPLE_QUESTION_PROMPT,
        "answer_prompt": SIMPLE_ANSWER_PROMPT,
        "llm_provider": "azure",
        "model": "gpt-4.1",
        "temperature": 0.5,
        "max_answer_tokens": 400,
        "category": "simple_qa",
        "difficulty": "easy"
    },
    "concise": {
        "description": "간결한 질문-답변 (100-300자)",
        "question_prompt": CONCISE_QUESTION_PROMPT,
        "answer_prompt": CONCISE_ANSWER_PROMPT,
        "llm_provider": "azure",
        "model": "gpt-4.1",
        "temperature": 0.3,
        "max_answer_tokens": 200,
        "category": "concise_qa",
        "difficulty": "medium",
        "max_answer_length": 400
    },
    "standard": {
        "description": "표준 질문-답변 (100-300단어)",
        "question_prompt": STANDARD_QA_PROMPT,
        "answer_prompt": None,  # 단일 프롬프트 사용
        "llm_provider": "azure",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_answer_tokens": 600,
        "category": "llm_generated",
        "difficulty": "medium"
    },
    "hard": {
        "description": "어려운 질문-답변 (여러 문서 참조)",
        "question_prompt": HARD_QA_PROMPT,
        "answer_prompt": None,  # 단일 프롬프트 사용
        "llm_provider": "openrouter",
        "model": "openai/gpt-4o",
        "temperature": 0.8,
        "max_answer_tokens": 800,
        "category": "llm_generated_hard",
        "difficulty": "hard",
        "multi_doc": True
    },
    "user_scenario": {
        "description": "사용자 시나리오 기반 질문-답변",
        "question_prompt": USER_SCENARIO_QUESTION_PROMPT,
        "answer_prompt": USER_SCENARIO_ANSWER_PROMPT,
        "llm_provider": "azure",
        "model": "gpt-4.1",
        "temperature": 0.5,
        "max_answer_tokens": 400,
        "category": "simple_user_scenario",
        "difficulty": "easy"
    }
}


# ============================================================================
# 데이터 로드 함수
# ============================================================================
def load_notion_data() -> List[Dict[str, Any]]:
    """Notion 데이터 로드"""
    data_file = DATA_DIR / "notion_data.json"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Notion 데이터가 없습니다: {data_file}\n"
            "먼저 'python scripts/build_vectordb.py'를 실행하세요."
        )

    data = load_json(str(data_file))

    # 데이터가 리스트인 경우
    if isinstance(data, list):
        pages = data
    else:
        pages = data.get("pages", [])

    # 제목이 있는 페이지만 필터링
    return [page for page in pages if page.get("title")]


# ============================================================================
# 텍스트 처리 함수
# ============================================================================
def extract_text_from_page(page: Dict[str, Any], max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    페이지에서 텍스트 추출 및 정제

    Args:
        page: Notion 페이지 데이터
        max_length: 최대 텍스트 길이

    Returns:
        추출된 텍스트
    """
    content = page.get("content", "")

    # Case 1: 문자열 형식의 content
    if isinstance(content, str):
        return _clean_string_content(content, max_length)

    # Case 2: blocks 형식의 content
    blocks = page.get("blocks", [])
    if blocks:
        content = _extract_from_blocks(blocks)
        if len(content) > max_length:
            content = content[:max_length] + "..."
        return content

    return ""


def _clean_string_content(content: str, max_length: int) -> str:
    """문자열 content 정제"""
    import re

    # 이미지 태그 제거
    content = re.sub(r'\[Image:.*?\]', '', content)

    # 길이 제한
    if len(content) > max_length:
        content = content[:max_length] + "..."

    return content.strip()


def _extract_from_blocks(blocks: List[Dict]) -> str:
    """blocks 형식에서 텍스트 추출"""
    texts = []

    for block in blocks:
        block_type = block.get("type")

        # 일반 텍스트 블록
        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3"]:
            texts.extend(_extract_rich_text(block.get(block_type, {})))

        # 리스트 아이템
        elif block_type == "bulleted_list_item":
            rich_texts = _extract_rich_text(block.get("bulleted_list_item", {}))
            texts.extend([f"• {text}" for text in rich_texts])

    return " ".join(texts)


def _extract_rich_text(block_content: Dict) -> List[str]:
    """rich_text에서 텍스트 추출"""
    texts = []

    for rt in block_content.get("rich_text", []):
        text = rt.get("plain_text", "").strip()
        if text:
            texts.append(text)

    return texts


# ============================================================================
# LLM 클라이언트 초기화
# ============================================================================
def initialize_llm_client(provider: str):
    """
    LLM 클라이언트 초기화

    Args:
        provider: "azure" 또는 "openrouter"

    Returns:
        (client, model) 튜플
    """
    if provider == "azure":
        if not AZURE_AI_CREDENTIAL:
            raise ValueError("AZURE_AI_CREDENTIAL이 설정되지 않았습니다")

        client = AzureOpenAI(
            api_key=AZURE_AI_CREDENTIAL,
            azure_endpoint=AZURE_AI_ENDPOINT,
            api_version="2024-02-01"
        )
        model = "gpt-4.1"

    elif provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY가 설정되지 않았습니다")

        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )
        model = "openai/gpt-4o"

    else:
        raise ValueError(f"지원하지 않는 LLM provider: {provider}")

    return client, model


# ============================================================================
# QA 생성 함수 (2단계: 질문 + 답변)
# ============================================================================
def generate_two_step_qa(
    page: Dict[str, Any],
    client,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    2단계 QA 생성 (질문 생성 -> 답변 생성)

    simple, concise, user_scenario 타입에서 사용
    """
    title = page.get("title", "Untitled")
    page_id = page.get("page_id", "")
    content = extract_text_from_page(page)

    if len(content) < MIN_CONTENT_LENGTH:
        return None

    # 내용이 너무 길면 제한
    max_content = 800 if config["category"] == "concise_qa" else 1000
    if len(content) > max_content:
        content = content[:max_content] + "..."

    try:
        # 1. 질문 생성
        question_prompt = config["question_prompt"].format(
            title=title,
            content=content
        )

        question_response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": question_prompt}],
            temperature=0.7,
            max_tokens=200,
            response_format={"type": "json_object"}
        )

        question_data = json.loads(question_response.choices[0].message.content)
        question = question_data.get("question", "")

        if not question:
            return None

        # 2. 답변 생성
        answer_prompt = config["answer_prompt"].format(
            question=question,
            content=content
        )

        # concise 타입은 시스템 메시지 추가
        messages = []
        if config["category"] == "concise_qa":
            messages.append({
                "role": "system",
                "content": "당신은 간결하고 핵심적인 답변을 제공하는 전문가입니다. 100-300자 이내로만 답변하세요."
            })

        messages.append({"role": "user", "content": answer_prompt})

        answer_response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_answer_tokens"]
        )

        answer = answer_response.choices[0].message.content.strip()

        # 답변 길이 검증 (concise 타입)
        if "max_answer_length" in config and len(answer) > config["max_answer_length"]:
            return None

        return {
            "question": question,
            "ground_truth": answer,
            "context_page_id": page_id,
            "metadata": {
                "category": config["category"],
                "difficulty": config["difficulty"],
                "source": f"llm_{config['llm_provider']}",
                "page_title": title,
                "answer_length": len(answer),
                "key_topics": question_data.get("key_topics", [])
            }
        }

    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return None


# ============================================================================
# QA 생성 함수 (1단계: 통합)
# ============================================================================
def generate_one_step_qa(
    page: Dict[str, Any],
    client,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    1단계 QA 생성 (질문과 답변을 한 번에 생성)

    standard 타입에서 사용
    """
    title = page.get("title", "Untitled")
    page_id = page.get("page_id", "")
    content = extract_text_from_page(page)

    if len(content) < MIN_CONTENT_LENGTH:
        return None

    try:
        prompt = config["question_prompt"].format(
            page_title=title,
            content=content
        )

        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=config["temperature"],
            max_tokens=config["max_answer_tokens"],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return {
            "question": result.get("question", ""),
            "ground_truth": result.get("ground_truth", ""),
            "context_page_id": page_id,
            "metadata": {
                "category": config["category"],
                "difficulty": result.get("difficulty", config["difficulty"]),
                "source": f"llm_{config['llm_provider']}",
                "page_title": title,
                "key_concepts": result.get("key_concepts", [])
            }
        }

    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return None


# ============================================================================
# Hard QA 생성 함수 (여러 문서 참조)
# ============================================================================
def select_documents_for_question(
    pages: List[Dict[str, Any]],
    used_page_ids: set
) -> List[Dict[str, Any]]:
    """질문 생성을 위한 문서 선택 (1-3개)"""
    available_pages = [p for p in pages if p.get("page_id") not in used_page_ids]

    if not available_pages:
        return []

    # 문서 개수 결정
    if random.random() < MULTI_DOC_PROBABILITY and len(available_pages) >= 2:
        num_docs = random.randint(2, min(MAX_DOCS_PER_QUESTION, len(available_pages)))
    else:
        num_docs = 1

    return random.sample(available_pages, num_docs)


def format_documents_for_prompt(pages: List[Dict[str, Any]]) -> str:
    """여러 문서를 프롬프트 형식으로 포맷팅"""
    if len(pages) == 1:
        page = pages[0]
        return f"- 문서 제목: {page.get('title', 'Untitled')}\n- 문서 내용:\n{extract_text_from_page(page)}"

    # 여러 문서인 경우
    formatted_docs = []
    for i, page in enumerate(pages, 1):
        title = page.get('title', 'Untitled')
        content = extract_text_from_page(page, max_length=600)
        formatted_docs.append(f"문서 {i}: \"{title}\"\n내용: {content}")

    return "\n\n".join(formatted_docs)


def generate_hard_qa(
    pages: List[Dict[str, Any]],
    client,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Hard QA 생성 (여러 문서 참조 가능)"""
    total_content_length = sum(len(extract_text_from_page(p)) for p in pages)

    if total_content_length < MIN_CONTENT_LENGTH:
        return None

    try:
        formatted_documents = format_documents_for_prompt(pages)
        prompt = config["question_prompt"].format(documents=formatted_documents)

        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=config["temperature"],
            max_tokens=config["max_answer_tokens"],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        page_titles = [p.get("title", "Untitled") for p in pages]
        page_ids = [p.get("page_id", "") for p in pages]

        return {
            "question": result.get("question", ""),
            "ground_truth": result.get("ground_truth", ""),
            "context_page_id": page_ids,
            "metadata": {
                "category": config["category"],
                "difficulty": config["difficulty"],
                "source": f"llm_{config['llm_provider']}",
                "page_titles": page_titles,
                "num_referenced_docs": len(pages),
                "referenced_docs": result.get("referenced_docs", page_titles),
                "reasoning_type": result.get("reasoning_type", "unknown")
            }
        }

    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return None


# ============================================================================
# 메인 QA 생성 함수
# ============================================================================
def generate_qa_dataset(
    qa_type: QA_TYPE,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    QA 데이터셋 생성

    Args:
        qa_type: QA 타입 (simple, concise, standard, hard, user_scenario)
        num_samples: 생성할 샘플 수
        output_path: 출력 파일 경로

    Returns:
        생성된 QA 데이터 리스트
    """
    if qa_type not in QA_TYPE_CONFIG:
        raise ValueError(f"지원하지 않는 QA 타입: {qa_type}")

    config = QA_TYPE_CONFIG[qa_type]

    print("=" * 60)
    print(f"📝 {config['description']} 데이터셋 생성")
    print("=" * 60)

    # Notion 데이터 로드
    print(f"\n📂 Notion 데이터 로드 중...")
    pages = load_notion_data()
    print(f"✅ {len(pages)}개 페이지 로드")

    # LLM 클라이언트 초기화
    client, model = initialize_llm_client(config["llm_provider"])
    print(f"✅ LLM 클라이언트 초기화: {config['llm_provider']} ({model})")

    # QA 생성
    print(f"\n🤖 QA 생성 중 (목표: {num_samples}개)...\n")

    qa_data = []
    used_page_ids = set()
    attempted = 0
    max_attempts = num_samples * 3

    random.shuffle(pages)

    # Hard QA는 여러 문서 처리
    if config.get("multi_doc", False):
        while len(qa_data) < num_samples and attempted < max_attempts:
            attempted += 1

            selected_pages = select_documents_for_question(pages, used_page_ids)
            if not selected_pages:
                break

            for page in selected_pages:
                used_page_ids.add(page.get("page_id"))

            print(f"[{len(qa_data)+1}/{num_samples}] {len(selected_pages)}개 문서 ", end="", flush=True)

            qa_item = generate_hard_qa(selected_pages, client, config)
            if qa_item:
                qa_data.append(qa_item)
                titles = [p.get("title", "Untitled")[:30] for p in selected_pages]
                print(f"✅ {', '.join(titles)}")
            else:
                print("⏭️ 스킵")

    # 다른 QA 타입은 단일 문서 처리
    else:
        for page in pages:
            if len(qa_data) >= num_samples or attempted >= max_attempts:
                break

            attempted += 1
            print(f"[{len(qa_data)+1}/{num_samples}] ", end="", flush=True)

            # 2단계 QA 생성
            if config["answer_prompt"]:
                qa_item = generate_two_step_qa(page, client, config)
            # 1단계 QA 생성
            else:
                qa_item = generate_one_step_qa(page, client, config)

            if qa_item:
                qa_data.append(qa_item)
                print(f"✅ {page.get('title', 'Untitled')[:40]}")
            else:
                print("⏭️ 스킵")

    # 저장
    if output_path is None:
        output_path = f"{DEFAULT_OUTPUT_DIR}/{qa_type}_qa_dataset.json"

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_json(qa_data, str(output_path_obj))

    # 결과 출력
    print(f"\n{'=' * 60}")
    print(f"✅ QA 생성 완료: {len(qa_data)}개")
    print(f"📁 저장 위치: {output_path}")
    print(f"{'=' * 60}")

    # 통계 출력
    _print_statistics(qa_data, qa_type)

    return qa_data


def _print_statistics(qa_data: List[Dict[str, Any]], qa_type: str):
    """통계 출력"""
    if not qa_data:
        return

    avg_q_len = sum(len(item["question"]) for item in qa_data) / len(qa_data)
    avg_a_len = sum(len(item["ground_truth"]) for item in qa_data) / len(qa_data)

    print(f"\n📊 통계:")
    print(f"   평균 질문 길이: {avg_q_len:.0f}자")
    print(f"   평균 답변 길이: {avg_a_len:.0f}자")

    # Concise 타입 추가 통계
    if qa_type == "concise":
        min_a_len = min(len(item["ground_truth"]) for item in qa_data)
        max_a_len = max(len(item["ground_truth"]) for item in qa_data)
        print(f"   답변 길이 범위: {min_a_len}~{max_a_len}자")

        short = sum(1 for item in qa_data if len(item["ground_truth"]) < 150)
        medium = sum(1 for item in qa_data if 150 <= len(item["ground_truth"]) <= 250)
        long = sum(1 for item in qa_data if len(item["ground_truth"]) > 250)

        print(f"\n   답변 길이 분포:")
        print(f"      짧음 (<150자): {short}개 ({short/len(qa_data)*100:.1f}%)")
        print(f"      적정 (150-250자): {medium}개 ({medium/len(qa_data)*100:.1f}%)")
        print(f"      긺 (>250자): {long}개 ({long/len(qa_data)*100:.1f}%)")

    # Hard 타입 추가 통계
    if qa_type == "hard":
        single_doc = sum(1 for item in qa_data if item["metadata"]["num_referenced_docs"] == 1)
        multi_doc = sum(1 for item in qa_data if item["metadata"]["num_referenced_docs"] > 1)

        print(f"\n   문서 참조 통계:")
        print(f"      단일 문서: {single_doc}개 ({single_doc/len(qa_data)*100:.1f}%)")
        print(f"      여러 문서: {multi_doc}개 ({multi_doc/len(qa_data)*100:.1f}%)")

        if multi_doc > 0:
            avg_docs = sum(
                item["metadata"]["num_referenced_docs"]
                for item in qa_data if item["metadata"]["num_referenced_docs"] > 1
            ) / multi_doc
            print(f"      평균 참조 문서 수 (여러 문서 질문): {avg_docs:.1f}개")

    # 샘플 출력
    print(f"\n📋 샘플 QA:")
    for i, item in enumerate(qa_data[:3], 1):
        print(f"\n   [{i}] Q: {item['question']}")
        answer_preview = item['ground_truth'][:100] + "..." if len(item['ground_truth']) > 100 else item['ground_truth']
        print(f"       A ({len(item['ground_truth'])}자): {answer_preview}")


# ============================================================================
# 메인 함수
# ============================================================================
def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="통합 QA 데이터셋 생성 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QA 타입:
  simple          : 간단한 질문-답변 (3-5문장)
  concise         : 간결한 질문-답변 (100-300자)
  standard        : 표준 질문-답변 (100-300단어)
  hard            : 어려운 질문-답변 (여러 문서 참조)
  user_scenario   : 사용자 시나리오 기반 질문-답변

사용 예시:
  # Simple QA 20개 생성
  python scripts/generate_qa_dataset.py --type simple --num-samples 20

  # Concise QA 50개 생성
  python scripts/generate_qa_dataset.py --type concise --num-samples 50

  # Hard QA 30개 생성 (커스텀 출력 경로)
  python scripts/generate_qa_dataset.py --type hard --num-samples 30 --output data/custom_hard_qa.json

  # 여러 타입 순차 생성
  python scripts/generate_qa_dataset.py --type simple --num-samples 20
  python scripts/generate_qa_dataset.py --type concise --num-samples 30
        """
    )

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=list(QA_TYPE_CONFIG.keys()),
        help="QA 타입 선택"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"생성할 샘플 수 (기본값: {DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="출력 파일 경로 (기본값: data/evaluation/{type}_qa_dataset.json)"
    )

    args = parser.parse_args()

    try:
        generate_qa_dataset(
            qa_type=args.type,
            num_samples=args.num_samples,
            output_path=args.output
        )

        print("\n✨ 모든 작업이 완료되었습니다!")

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
