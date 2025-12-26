#!/usr/bin/env python3
"""S12: Mixed Retrieval Vector DB 구축 스크립트 (요약+원문 혼합)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import *
from models.embeddings.factory import get_embedder
from models.vision.factory import get_vision_model
from core.chunker import process_page_data
from utils.file_utils import load_json
from utils.langfuse_utils import get_langfuse_client, trace_operation
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import uuid
from tqdm import tqdm


def init_mixed_collection(
    client: QdrantClient,
    collection_name: str,
    dimension: int,
    recreate: bool = False
):
    """
    Mixed Vector DB용 Qdrant 컬렉션 초기화

    Args:
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
        dimension: 벡터 차원
        recreate: 기존 컬렉션 삭제 후 재생성 여부
    """
    if recreate:
        try:
            client.delete_collection(collection_name)
            print(f"  ✓ 기존 컬렉션 삭제: {collection_name}")
        except:
            pass

    # 컬렉션 존재 여부 확인
    collections = [col.name for col in client.get_collections().collections]

    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
        print(f"  ✓ 컬렉션 생성: {collection_name}")
    else:
        print(f"  ✓ 기존 컬렉션 사용: {collection_name}")


def summarize_chunk(chunk_text: str, llm, max_length: int = 200) -> str:
    """
    단일 청크를 요약

    Args:
        chunk_text: 요약할 텍스트
        llm: LLM 모델
        max_length: 최대 요약 길이 (자)

    Returns:
        요약된 텍스트
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 텍스트를 간결하게 요약하는 전문가입니다.
주어진 텍스트의 핵심 정보를 보존하면서 {length}자 이내로 요약하세요.

요약 시 주의사항:
- 중요한 키워드와 개념은 반드시 포함
- 구체적인 수치, 날짜, 이름 등은 가능한 보존
- 핵심 내용만 추출
- 불필요한 부연 설명 제거"""),
        ("user", "다음 텍스트를 요약하세요:\n\n{text}")
    ])

    chain = prompt | llm
    result = chain.invoke({"text": chunk_text, "length": max_length})

    return result.content.strip()


def store_mixed_to_qdrant(
    chunks,
    summaries: list,
    summary_embeddings: list,
    original_embeddings: list,
    client: QdrantClient,
    collection_name: str
):
    """
    요약본과 원문을 모두 Qdrant에 저장 (별도 포인트로)

    Args:
        chunks: 원본 청크 리스트
        summaries: 요약본 리스트
        summary_embeddings: 요약본 임베딩 리스트
        original_embeddings: 원본 임베딩 리스트
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
    """
    points = []

    for chunk, summary, summary_emb, original_emb in zip(chunks, summaries, summary_embeddings, original_embeddings):
        # 공통 메타데이터
        base_metadata = {
            "page_id": chunk.page_id,
            "page_title": chunk.page_title,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path,
            "has_image": chunk.has_image,
            "image_descriptions": chunk.image_descriptions,
        }

        # 1. 요약본 포인트
        summary_payload = {
            **base_metadata,
            "chunk_id": str(uuid.uuid4()),
            "text": summary,
            "combined_text": summary,
            "original_text": chunk.combined_text,
            "properties": {
                "content_type": "summary",
                "summary_length": len(summary),
                "original_length": len(chunk.combined_text)
            }
        }

        summary_point = PointStruct(
            id=str(uuid.uuid4()),
            vector=summary_emb.tolist() if hasattr(summary_emb, 'tolist') else summary_emb,
            payload=summary_payload
        )
        points.append(summary_point)

        # 2. 원문 포인트
        original_payload = {
            **base_metadata,
            "chunk_id": str(uuid.uuid4()),
            "text": chunk.combined_text,
            "combined_text": chunk.combined_text,
            "summary_text": summary,
            "properties": {
                "content_type": "original",
                "original_length": len(chunk.combined_text),
                "summary_length": len(summary)
            }
        }

        original_point = PointStruct(
            id=str(uuid.uuid4()),
            vector=original_emb.tolist() if hasattr(original_emb, 'tolist') else original_emb,
            payload=original_payload
        )
        points.append(original_point)

    # 배치로 저장
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )

    print(f"  ✓ {len(chunks)}개 청크 → {len(points)}개 포인트 저장 완료 (요약+원문)")


def main(
    force_recreate: bool = False,
    limit: int = None,
    collection_name: str = "notion_mixed",
    summary_length: int = 200
):
    """
    S12: Mixed Retrieval Vector DB 구축 메인 함수

    Args:
        force_recreate: True면 전체 재생성
        limit: 처리할 페이지 수 제한 (None이면 전체)
        collection_name: 저장할 컬렉션 이름
        summary_length: 요약 최대 길이 (자)
    """
    print("=" * 60)
    print("🔀 S12: Mixed Retrieval Vector DB 구축 시작")
    print("   (요약+원문 혼합 - 균형형)")
    if limit:
        print(f"📊 제한: {limit}개 페이지만 처리")
    print(f"📦 컬렉션: {collection_name}")
    print(f"✂️ 요약 길이: {summary_length}자")
    print("=" * 60)

    # Langfuse 초기화
    get_langfuse_client()

    # 전체 프로세스를 Langfuse로 트레이싱
    with trace_operation(
        name="mixed_vectordb_build",
        metadata={
            "force_recreate": force_recreate,
            "limit": limit,
            "collection_name": collection_name,
            "summary_length": summary_length
        }
    ) as trace:

        data_file = DATA_DIR / "notion_data.json"

        # 1. 모델 초기화
        print("\n📦 모델 로딩...")
        if trace:
            model_span = trace.span(name="model_initialization")

        embedder = get_embedder()
        vision_model = get_vision_model()
        llm = init_chat_model("azure_ai:gpt-5.1", temperature=0.1)
        qdrant_client = QdrantClient(path=QDRANT_PATH)

        if trace:
            model_span.end()

        # 2. 데이터 로드
        print("\n📂 데이터 로딩...")
        if not data_file.exists():
            print("❌ notion_data.json 파일이 없습니다!")
            print("먼저 build_vectordb.py를 실행하여 데이터를 수집하세요.")
            return

        all_data = load_json(str(data_file))
        pages_to_process = all_data[:limit] if limit else all_data
        print(f"  ✓ {len(pages_to_process)}개 페이지 로드")

        # 3. 페이지별로 청크 생성
        print("\n✂️ 청킹...")
        if trace:
            chunking_span = trace.span(name="chunking")

        all_chunks = []
        for page in tqdm(pages_to_process, desc="청킹"):
            chunks = process_page_data(page, embedder, vision_model)
            all_chunks.extend(chunks)

        if trace:
            chunking_span.end(metadata={"total_chunks": len(all_chunks)})

        print(f"  ✓ 총 {len(all_chunks)}개 청크 생성")

        if not all_chunks:
            print("❌ 청크가 생성되지 않았습니다!")
            return

        # 4. 청크 요약 생성
        print(f"\n📝 청크 요약 생성 중 ({summary_length}자 이내)...")
        if trace:
            summarization_span = trace.span(name="summarization")

        summaries = []
        for chunk in tqdm(all_chunks, desc="요약"):
            try:
                summary = summarize_chunk(chunk.combined_text, llm, summary_length)
                summaries.append(summary)
            except Exception as e:
                print(f"  ⚠ 요약 실패, 원본 사용: {str(e)[:50]}...")
                # 요약 실패 시 원본의 처음 부분 사용
                summaries.append(chunk.combined_text[:summary_length])

        if trace:
            summarization_span.end(metadata={
                "total_summaries": len(summaries),
                "avg_summary_length": sum(len(s) for s in summaries) / len(summaries)
            })

        print(f"  ✓ {len(summaries)}개 요약 생성 완료")
        avg_summary_len = sum(len(s) for s in summaries) / len(summaries)
        print(f"  ✓ 평균 요약 길이: {avg_summary_len:.1f}자")

        # 5. 임베딩 생성 (요약본 + 원문)
        print(f"\n🔢 임베딩 생성 중 (요약본 + 원문)...")
        if trace:
            embedding_span = trace.span(name="embedding_generation")

        # 요약본 임베딩
        print("  → 요약본 임베딩...")
        summary_embeddings = embedder.embed_texts(summaries)

        # 원문 임베딩
        print("  → 원문 임베딩...")
        original_texts = [c.combined_text for c in all_chunks]
        original_embeddings = embedder.embed_texts(original_texts)

        if trace:
            embedding_span.end(metadata={
                "num_summary_embeddings": len(summary_embeddings),
                "num_original_embeddings": len(original_embeddings),
                "embedding_dimension": len(summary_embeddings[0])
            })

        print(f"  ✓ 요약 임베딩: {len(summary_embeddings)}개")
        print(f"  ✓ 원문 임베딩: {len(original_embeddings)}개")

        # 6. Qdrant에 저장
        print(f"\n💾 Qdrant 저장 중 (컬렉션: {collection_name})...")
        if trace:
            storage_span = trace.span(name="qdrant_storage")

        # 임베딩 차원 확인
        embedding_dimension = len(summary_embeddings[0])

        # 컬렉션 초기화
        init_mixed_collection(
            qdrant_client,
            collection_name,
            embedding_dimension,
            recreate=force_recreate
        )

        # 요약본 + 원문 저장
        store_mixed_to_qdrant(
            all_chunks,
            summaries,
            summary_embeddings,
            original_embeddings,
            qdrant_client,
            collection_name
        )

        if trace:
            storage_span.end()

        # 7. 검증
        print("\n🔍 저장 검증...")
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"  ✓ 저장된 포인트 수: {collection_info.points_count}")
        print(f"  ✓ 청크당 포인트 수: {collection_info.points_count // len(all_chunks)} (요약+원문)")

        # 통계
        original_total_len = sum(len(c.combined_text) for c in all_chunks)
        summary_total_len = sum(len(s) for s in summaries)
        avg_original_len = original_total_len / len(all_chunks)

        print("\n" + "=" * 60)
        print("🎉 S12: Mixed Retrieval Vector DB 구축 완료!")
        print(f"📦 컬렉션: {collection_name}")
        print(f"📊 총 청크 수: {len(all_chunks)}")
        print(f"📍 총 포인트 수: {collection_info.points_count} (요약 {len(summaries)} + 원문 {len(original_embeddings)})")
        print(f"📝 평균 요약 길이: {avg_summary_len:.1f}자")
        print(f"📄 평균 원문 길이: {avg_original_len:.1f}자")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S12: Mixed Retrieval Vector DB 구축")
    parser.add_argument("--force", action="store_true", help="전체 재생성")
    parser.add_argument("--limit", type=int, default=None, help="처리할 페이지 수 제한")
    parser.add_argument("--collection", type=str, default="notion_mixed", help="컬렉션 이름")
    parser.add_argument("--summary-length", type=int, default=600, help="요약 최대 길이 (자)")

    args = parser.parse_args()

    main(
        force_recreate=args.force,
        limit=args.limit,
        collection_name=args.collection,
        summary_length=args.summary_length
    )
