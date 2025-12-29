#!/usr/bin/env python3
"""최종 보고서 생성기 (Executive Report)

설정: OpenAI + RRF MultiQuery (Top 8) + DeepSeek-V3.1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.chat_models import init_chat_model

from config.settings import AZURE_AI_CREDENTIAL, AZURE_AI_ENDPOINT
from utils.langfuse_utils import get_langfuse_client
from utils.date_utils import parse_date_range, extract_date_filter_from_question
from retrievers.ensemble_retriever import get_ensemble_retriever
from retrievers.multiquery_retriever import get_multiquery_retriever


class ExecutiveReportGenerator:
    """최종 보고서 생성기 (Executive Report)

    설정:
    - Retriever: OpenAI + RRF MultiQuery (Top 8)
    - LLM: DeepSeek-V3.1
    """

    def __init__(self):
        self.retriever_config = {
            'name': 'openai_rrf_multiquery',
            'display_name': 'OpenAI + RRF MultiQuery (Top 8)',
            'embedding': 'openai-large',
            'type': 'rrf_multiquery',
            'top_k': 8,
            'description': 'Faithfulness + MultiQuery'
        }

        self.llm_config = {
            'name': 'deepseek_v31',
            'display_name': 'DeepSeek-V3.1',
            'model_id': 'azure_ai:DeepSeek-V3.1',
            'description': 'DeepSeek 최신 버전'
        }

        self.langfuse = get_langfuse_client()
        self.qdrant_lock = "/home/work/rag/Project/rag-report-generator/data/qdrant_data/openai-large/.lock"

    def load_prompt(self, prompt_file: str) -> str:
        """프롬프트 파일 로드"""
        prompt_path = Path(__file__).parent.parent / "prompts" / "templates" / "service" / "executive_report" / prompt_file
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def retrieve_documents(self, question: str, date_filter: Optional[tuple] = None) -> List[Any]:
        """문서 검색 - RRF MultiQuery"""
        import time
        import gc

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
        base_retriever = get_ensemble_retriever(
            k=self.retriever_config['top_k'],
            bm25_weight=0.5,
            dense_weight=0.5,
            date_filter=date_filter
        )

        # MultiQuery로 래핑
        retriever = get_multiquery_retriever(
            base_retriever=base_retriever,
            num_queries=3,
            temperature=0.7
        )

        print(f"🔍 검색 중... ({self.retriever_config['display_name']})")

        # 문서 검색
        docs = retriever.invoke(question)

        print(f"📄 검색된 문서 수: {len(docs)}")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc.metadata.get('page_title', 'Unknown')}")

        return docs

    def generate_answer(self, question: str, docs: List[Any]) -> str:
        """DeepSeek-V3.1로 답변 생성"""
        # Context 구성
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('page_title', 'Unknown')
            content = doc.page_content
            context_parts.append(f"[문서 {i}] {title}\n{content}\n")

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
        model = init_chat_model(
            self.llm_config['model_id'],
            temperature=0,
            max_completion_tokens=1000
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
                tags=["executive_report", self.retriever_config['name'], self.llm_config['name']],
                output={"answer": answer}
            )

        print(f"✅ 답변 생성 완료\n")

        return answer

    def generate_report(self, questions: List[str], global_date_filter: Optional[tuple] = None) -> Dict[str, Any]:
        """최종 보고서 생성

        Args:
            questions: 질문 리스트
            global_date_filter: 전역 날짜 필터 (명시적으로 지정된 경우, 질문별 추출보다 우선)

        Returns:
            보고서 데이터
        """
        print("\n" + "=" * 100)
        print("📊 최종 보고서 생성 시작 (Executive Report)")
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

                print(f"📝 답변:\n{answer}\n")

                results.append({
                    "question_id": i,
                    "question": question,
                    "date_filter": f"{date_filter[0][:10]} ~ {date_filter[1][:10]}" if date_filter else None,
                    "num_docs": len(docs),
                    "doc_titles": [doc.metadata.get('page_title', 'Unknown') for doc in docs],
                    "answer": answer,
                    "success": True
                })

            except Exception as e:
                print(f"❌ 오류 발생: {e}\n")
                import traceback
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
            "report_type": "executive",
            "generated_at": datetime.now().isoformat(),
            "retriever": self.retriever_config,
            "llm": self.llm_config,
            "global_date_filter": f"{global_date_filter[0][:10]} ~ {global_date_filter[1][:10]}" if global_date_filter else None,
            "num_questions": len(questions),
            "results": results
        }

        print("\n" + "=" * 100)
        print("✅ 최종 보고서 생성 완료!")
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
    import argparse

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

    # 질문 설정
    if args.questions:
        questions = args.questions
    else:
        # 기본 질문
        questions = [
            "10월 최종 보고서 만들어줘",
            "지금까지 한 것 중에 중요 요인들로 최종 보고서 만들어줘",
            "추천시스템 최종 보고서 만들어줘"
        ]

    # 보고서 생성
    generator = ExecutiveReportGenerator()
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
