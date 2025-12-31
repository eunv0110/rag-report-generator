#!/usr/bin/env python3
"""통합 보고서 생성기

설정 파일을 통해 다양한 타입의 보고서를 생성할 수 있습니다:
- Weekly Report: weekly_report_config.yaml
- Executive Report: executive_report_config.yaml
"""

import sys
import os
import json
import yaml
import time
import gc
import re
import argparse
import getpass
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 환경 변수 로드
load_dotenv()

from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT
from utils.langfuse_utils import get_langfuse_client
from utils.date_utils import parse_date_range, extract_date_filter_from_question
from retrievers.ensemble_retriever import get_ensemble_retriever
from retrievers.multiquery_retriever import get_multiquery_retriever


class ReportGenerator:
    """통합 보고서 생성기

    설정 파일을 통해 Retriever, LLM, 프롬프트를 유연하게 설정할 수 있습니다.
    """

    def __init__(self, config_path: Optional[str] = None, report_type: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로. 지정하지 않으면 report_type에 따라 기본 경로 사용.
            report_type: 보고서 타입 ("weekly", "executive"). config_path가 없을 때만 사용.
        """
        # 설정 파일 로드
        if config_path is None:
            if report_type == "weekly":
                config_path = Path(__file__).parent.parent / "config" / "weekly_report_config.yaml"
            elif report_type == "executive":
                config_path = Path(__file__).parent.parent / "config" / "executive_report_config.yaml"
            else:
                raise ValueError("config_path 또는 report_type ('weekly' 또는 'executive') 중 하나를 지정해야 합니다.")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 설정 적용
        self.report_type = config['report_type']
        self.retriever_config = config['retriever']
        self.llm_config = config['llm']
        self.paths = config['paths']
        self.default_questions = config.get('default_questions', [])

        self.langfuse = get_langfuse_client()
        self.qdrant_lock = self.paths['qdrant_lock']

    def load_prompt(self, prompt_file: str) -> str:
        """프롬프트 파일 로드"""
        prompt_path = Path(__file__).parent.parent / self.paths['prompts_base'] / prompt_file
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_title_from_answer(self, answer: str) -> str:
        """답변에서 제목 추출 ([TITLE]...[/TITLE] 태그 사용)"""
        match = re.search(r'\[TITLE\](.*?)\[/TITLE\]', answer, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _remove_title_tag_from_answer(self, answer: str) -> str:
        """답변에서 제목 태그 제거"""
        return re.sub(r'\[TITLE\].*?\[/TITLE\]\s*', '', answer, flags=re.DOTALL).strip()

    def retrieve_documents(self, question: str, date_filter: Optional[tuple] = None) -> List[Any]:
        """문서 검색 - 설정에 따라 RRF Ensemble 또는 RRF MultiQuery 사용"""
        # Qdrant 락 파일 정리
        if os.path.exists(self.qdrant_lock):
            try:
                os.remove(self.qdrant_lock)
            except:
                pass

        # 가비지 컬렉션
        gc.collect()
        time.sleep(0.5)

        # 환경 변수 설정
        os.environ["MODEL_PRESET"] = self.retriever_config['embedding']
        os.environ["USE_EMBEDDING_CACHE"] = "true"

        # 기본 RRF Ensemble 리트리버 생성
        rrf_config = self.retriever_config.get('rrf', {})
        base_retriever = get_ensemble_retriever(
            k=self.retriever_config['top_k'],
            bm25_weight=rrf_config.get('bm25_weight', 0.5),
            dense_weight=rrf_config.get('dense_weight', 0.5),
            date_filter=date_filter,
            use_mmr=rrf_config.get('use_mmr', True),
            lambda_mult=rrf_config.get('lambda_mult', 0.5)
        )

        # MultiQuery 타입이면 래핑
        if self.retriever_config['type'] == 'rrf_multiquery':
            multiquery_config = self.retriever_config.get('multiquery', {})
            retriever = get_multiquery_retriever(
                base_retriever=base_retriever,
                num_queries=multiquery_config.get('num_queries', 3),
                temperature=multiquery_config.get('temperature', 0.7)
            )
        else:
            retriever = base_retriever

        print(f"🔍 검색 중... ({self.retriever_config['display_name']})")

        # 문서 검색
        docs = retriever.invoke(question)

        print(f"📄 검색된 문서 수: {len(docs)}")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc.metadata.get('page_title', 'Unknown')}")

        return docs

    def generate_answer(self, question: str, docs: List[Any]) -> str:
        """LLM으로 답변 생성"""
        # Context 구성
        context_parts = []
        for doc in docs:
            title = doc.metadata.get('page_title', 'Unknown')
            content = doc.page_content
            context_parts.append(f"[{title}]\n{content}\n")

        context_text = "\n".join(context_parts)

        # 프롬프트 로드
        system_prompt = self.load_prompt("system_prompt.txt")
        answer_generation_template = self.load_prompt("answer_generation_prompt.txt")

        # 템플릿에 변수 대입
        user_prompt = answer_generation_template.replace("{context}", context_text).replace("{question}", question)

        # Azure AI 설정
        os.environ['AZURE_AI_CREDENTIAL'] = AZURE_AI_CREDENTIAL
        os.environ['AZURE_AI_ENDPOINT'] = AZURE_AI_ENDPOINT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        print(f"💬 답변 생성 중... ({self.llm_config['display_name']})")

        # LLM 생성
        generation_config = self.llm_config.get('generation', {})
        model = init_chat_model(
            self.llm_config['model_id'],
            temperature=generation_config.get('temperature', 0),
            max_completion_tokens=generation_config.get('max_completion_tokens', 1000)
        )

        # Langfuse로 답변 생성 기록
        with self.langfuse.start_as_current_observation(
            as_type='generation',
            name=f"generation_{self.llm_config['name']}",
            model=self.llm_config['model_id'],
            input={"question": question, "context": context_text[:500] + "..." if len(context_text) > 500 else context_text},
            metadata={"llm": self.llm_config['name'], "num_docs": len(docs)}
        ) as generation:
            response = model.invoke(messages)
            answer = response.content

            # 토큰 사용량 추출
            usage_dict = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_dict = {
                    "input": response.usage_metadata.get('input_tokens', 0),
                    "output": response.usage_metadata.get('output_tokens', 0),
                    "total": response.usage_metadata.get('total_tokens', 0)
                }

            generation.update(
                output={"answer": answer},
                usage=usage_dict if usage_dict else None
            )

            # Trace 업데이트
            self.langfuse.update_current_trace(
                tags=[f"{self.report_type}_report", self.retriever_config['name'], self.llm_config['name']],
                output={"answer": answer}
            )

        print(f"✅ 답변 생성 완료\n")

        return answer

    def generate_report(self, questions: List[str], global_date_filter: Optional[tuple] = None) -> Dict[str, Any]:
        """보고서 생성

        Args:
            questions: 질문 리스트
            global_date_filter: 전역 날짜 필터 (명시적으로 지정된 경우, 질문별 추출보다 우선)

        Returns:
            보고서 데이터
        """
        report_title = "주간 보고서" if self.report_type == "weekly" else "최종 보고서 (Executive Report)"

        print("\n" + "=" * 100)
        print(f"📊 {report_title} 생성 시작")
        print("=" * 100)
        print(f"🔧 설정: {self.retriever_config['display_name']} + {self.llm_config['display_name']}")
        if global_date_filter:
            print(f"📅 전역 날짜 필터: {global_date_filter[0][:10]} ~ {global_date_filter[1][:10]}")
        print(f"📝 질문 수: {len(questions)}")
        print()

        results = []

        for i, question in enumerate(questions, 1):
            print(f"\n{'=' * 100}")
            print(f"질문 {i}/{len(questions)}")
            print(f"{'=' * 100}")
            print(f"❓ {question}\n")

            try:
                # 날짜 필터 결정: 전역 필터가 있으면 그것 사용, 없으면 질문에서 추출
                if global_date_filter:
                    date_filter = global_date_filter
                else:
                    date_filter = extract_date_filter_from_question(question)
                    if date_filter:
                        print(f"📅 감지된 날짜 필터: {date_filter[0][:10]} ~ {date_filter[1][:10]}\n")

                # 문서 검색
                docs = self.retrieve_documents(question, date_filter)

                # 답변 생성 (Langfuse 자동 추적)
                answer = self.generate_answer(question, docs)

                # 제목 추출
                title = self._extract_title_from_answer(answer)
                answer = self._remove_title_tag_from_answer(answer)

                if title:
                    print(f"📋 제목: {title}")
                print(f"📝 답변:\n{answer}\n")

                # 이미지 메타데이터 추출
                images = []
                for doc in docs:
                    if doc.metadata.get('has_image', False):
                        image_paths = doc.metadata.get('image_paths', [])
                        image_descriptions = doc.metadata.get('image_descriptions', [])
                        for img_path, img_desc in zip(image_paths, image_descriptions):
                            images.append({
                                'path': img_path,
                                'description': img_desc,
                                'source': doc.metadata.get('page_title', 'Unknown')
                            })

                if images:
                    print(f"🖼️  첨부 이미지: {len(images)}개")
                    for img in images:
                        print(f"  - {img['path']} (출처: {img['source']})")
                    print()

                results.append({
                    "question_id": i,
                    "question": question,
                    "title": title,  # LLM이 생성한 제목 추가
                    "date_filter": f"{date_filter[0][:10]} ~ {date_filter[1][:10]}" if date_filter else None,
                    "num_docs": len(docs),
                    "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in docs],
                    "images": images,  # 이미지 정보 추가
                    "answer": answer,
                    "success": True
                })

            except Exception as e:
                print(f"❌ 오류 발생: {e}\n")
                traceback.print_exc()

                results.append({
                    "question_id": i,
                    "question": question,
                    "error": str(e),
                    "success": False
                })

        # Langfuse flush
        self.langfuse.flush()

        report_data = {
            "report_type": self.report_type,
            "title": "주간 보고서" if self.report_type == "weekly" else "최종 보고서",
            "generated_at": datetime.now().isoformat(),
            "retriever": self.retriever_config,
            "llm": self.llm_config,
            "global_date_filter": f"{global_date_filter[0][:10]} ~ {global_date_filter[1][:10]}" if global_date_filter else None,
            "num_questions": len(questions),
            "results": results
        }

        print("\n" + "=" * 100)
        print(f"✅ {report_title} 생성 완료!")
        print("=" * 100)

        return report_data

    def save_json(self, report_data: Dict[str, Any], output_path: str):
        """JSON 파일로 저장"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"💾 JSON 저장: {output_file}")


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description="통합 보고서 생성기")
    parser.add_argument("--report-type", type=str, choices=['weekly', 'executive'], required=True,
                       help="보고서 타입 (weekly 또는 executive)")
    parser.add_argument("--config", type=str, help="설정 파일 경로 (지정하지 않으면 report-type에 따라 기본값 사용)")
    parser.add_argument("--questions", type=str, nargs='+', help="질문 리스트")
    parser.add_argument("--date-range", type=str, help="날짜 범위 (예: '이번 주', '12월 2주차')")
    parser.add_argument("--start-date", type=str, help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로")
    parser.add_argument("--author", type=str, default=None, help="보고서 작성자 (미지정 시 시스템 사용자명 사용)")

    args = parser.parse_args()

    # 날짜 필터 파싱
    date_filter = parse_date_range(
        date_input=args.date_range,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 보고서 생성기 초기화
    generator = ReportGenerator(config_path=args.config, report_type=args.report_type)

    # 질문 설정
    if args.questions:
        questions = args.questions
    else:
        # 설정 파일의 기본 질문 사용
        questions = generator.default_questions

    # 작성자 정보 설정
    author = args.author if args.author else getpass.getuser()

    # 보고서 생성
    report_data = generator.generate_report(questions, date_filter)

    # 작성자 및 작성일자 정보 추가
    report_data["author"] = author
    report_data["created_date"] = datetime.now().strftime("%Y-%m-%d")
    report_data["created_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # JSON 저장
    if args.output:
        generator.save_json(report_data, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"data/reports/{args.report_type}_report_{timestamp}.json"
        generator.save_json(report_data, default_output)


if __name__ == "__main__":
    main()
