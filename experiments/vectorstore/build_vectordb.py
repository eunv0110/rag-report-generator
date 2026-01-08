#!/usr/bin/env python3
"""Vector DB 구축 스크립트 (LangChain 호환, 증분 업데이트 지원)"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import *
from models.embeddings.factory import get_embedder
from models.vision.factory import get_vision_model
from core.data_collector import NotionDataSourceCollector
from core.chunker import process_page_data
from services.incremental_sync import (
    check_existing_data,
    collect_missing_pages,
    update_changed_pages
)
from utils.files import save_json, load_json
from utils.langfuse import get_langfuse_client, trace_operation
from utils.embedding_cache import CachedEmbedder

# LangChain imports
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def get_langchain_embeddings(embedder) -> Embeddings:
    """
    기존 embedder를 LangChain Embeddings로 래핑

    Note: OpenRouterEmbedder는 이미 Embeddings를 상속받으므로 그대로 반환
    """
    # embedder가 이미 Embeddings 인터페이스를 구현하고 있으면 그대로 반환
    if isinstance(embedder, Embeddings):
        return embedder

    # 그렇지 않으면 wrapper 생성
    class CustomEmbeddings(Embeddings):
        def __init__(self, embedder):
            self.embedder = embedder

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """문서 임베딩"""
            if hasattr(self.embedder, 'embed_documents'):
                return self.embedder.embed_documents(texts)
            elif hasattr(self.embedder, 'embed_texts'):
                return self.embedder.embed_texts(texts)
            else:
                raise AttributeError("embedder에 embed_documents 또는 embed_texts 메서드가 없습니다")

        def embed_query(self, text: str) -> List[float]:
            """쿼리 임베딩"""
            if hasattr(self.embedder, 'embed_query'):
                return self.embedder.embed_query(text)
            elif hasattr(self.embedder, 'embed_texts'):
                return self.embedder.embed_texts([text])[0]
            else:
                raise AttributeError("embedder에 embed_query 또는 embed_texts 메서드가 없습니다")

    return CustomEmbeddings(embedder)


def chunks_to_documents(chunks) -> List[Document]:
    """청크를 LangChain Document로 변환"""
    documents = []
    for chunk in chunks:
        # 메타데이터 준비
        metadata = {
            "chunk_id": chunk.chunk_id,
            "page_id": chunk.page_id,
            "page_title": chunk.page_title,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path,
            "has_image": chunk.has_image,
            "image_paths": chunk.image_paths,
            "image_descriptions": chunk.image_descriptions,
        }

        # properties가 있으면 추가
        if hasattr(chunk, 'properties') and chunk.properties:
            props = {}
            for k, v in chunk.properties.items():
                # 날짜 필드 처리: dict에서 start, end 추출
                if k == "날짜" and isinstance(v, dict):
                    props["날짜_start"] = v.get('start', '')
                    props["날짜_end"] = v.get('end', '')
                    props["날짜"] = v  # 원본도 유지
                else:
                    props[k] = v if isinstance(v, (str, int, float, bool, list, dict)) else str(v)
            metadata["properties"] = props

        # Document 생성
        doc = Document(
            page_content=chunk.combined_text,
            metadata=metadata
        )
        documents.append(doc)

    return documents


def check_qdrant_collection(client: QdrantClient) -> dict:
    """Qdrant 컬렉션의 데이터 확인"""
    try:
        collections = client.get_collections().collections
        exists = any(c.name == QDRANT_COLLECTION for c in collections)

        if not exists:
            print(f"❌ Qdrant 컬렉션 없음: {QDRANT_COLLECTION}")
            return {"exists": False, "count": 0, "page_ids": set()}

        info = client.get_collection(QDRANT_COLLECTION)
        count = info.points_count

        # 모든 page_id 추출
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=["page_id"]
        )

        page_ids = {point.payload.get("page_id") for point in scroll_result[0] if point.payload.get("page_id")}

        print(f"✅ Qdrant 데이터: {count}개 청크, {len(page_ids)}개 페이지")

        return {
            "exists": True,
            "count": count,
            "page_ids": page_ids
        }
    except Exception as e:
        print(f"⚠️ Qdrant 확인 실패: {e}")
        return {"exists": False, "count": 0, "page_ids": set()}


def delete_page_from_vectorstore(vectorstore: Qdrant, page_id: str):
    """LangChain Qdrant에서 특정 페이지의 모든 청크 삭제"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    vectorstore.client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="page_id",
                    match=MatchValue(value=page_id)
                )
            ]
        )
    )
    print(f"  🗑️ 페이지 삭제: {page_id}")


def main(force_recreate: bool = False, check_updates: bool = True, limit: int = None, input_file: str = None):
    """
    Vector DB 구축 메인 함수

    Args:
        force_recreate: True면 전체 재생성
        check_updates: True면 수정된 페이지도 확인
        limit: 처리할 페이지 수 제한 (None이면 전체)
        input_file: 사용할 입력 JSON 파일 경로 (None이면 기본 파일 사용)
    """
    print("=" * 60)
    print("🚀 Vector DB 구축 시작")
    if limit:
        print(f"📊 제한: {limit}개 페이지만 처리")
    if input_file:
        print(f"📁 입력 파일: {input_file}")
    print("=" * 60)

    # Langfuse 초기화
    get_langfuse_client()

    # 전체 프로세스를 Langfuse로 트레이싱
    with trace_operation(
        name="vectordb_build",
        metadata={
            "force_recreate": force_recreate,
            "check_updates": check_updates,
            "limit": limit,
            "db_name": DB_NAME
        }
    ) as trace:

        # 입력 파일 결정
        if input_file:
            data_file = Path(input_file)
            if not data_file.exists():
                raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")
            print(f"✅ 기존 JSON 파일 사용: {data_file}")
        else:
            data_file = DATA_DIR / "notion_data.json"

        # 1. 모델 초기화
        print("\n📦 모델 로딩...")
        if trace:
            model_span = trace.span(name="model_initialization")

        base_embedder = get_embedder()
        # 캐시된 임베더로 래핑
        cached_embedder = CachedEmbedder(
            embedder=base_embedder,
            model_name=DB_NAME
        )
        embedder = cached_embedder.cached_embedder  # LangChain Embeddings 인터페이스
        langchain_embeddings = get_langchain_embeddings(embedder)
        vision_model = get_vision_model()

        # Qdrant client는 필요할 때만 생성 (중복 생성 방지)
        qdrant_client = None

        if trace:
            model_span.end()

        # 2. 데이터 수집 (증분)
        if trace:
            collection_span = trace.span(
                name="data_collection",
                metadata={"mode": "force_recreate" if force_recreate else "incremental"}
            )

        # input_file이 지정된 경우 해당 파일에서 직접 로드
        if input_file:
            print(f"\n📂 기존 JSON 파일 로딩: {data_file}")
            all_data = load_json(str(data_file))
            pages_to_index = all_data
            print(f"✅ {len(all_data)}개 페이지 로드 완료")

        elif force_recreate:
            print("\n♻️ 전체 재수집 모드")
            if data_file.exists():
                from datetime import datetime
                backup = data_file.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                data_file.rename(backup)
                print(f"   백업: {backup}")

            collector = NotionDataSourceCollector(NOTION_TOKEN, DATA_SOURCE_ID)
            all_data = collector.collect_all(limit=limit)
            save_json(all_data, str(data_file))
            pages_to_index = all_data

        else:
            # 기존 데이터 확인
            existing_info = check_existing_data(str(data_file))

            if not existing_info["exists"]:
                print("\n📥 초기 수집 시작...")
                collector = NotionDataSourceCollector(NOTION_TOKEN, DATA_SOURCE_ID)
                all_data = collector.collect_all(limit=limit)
                save_json(all_data, str(data_file))
                pages_to_index = all_data

            else:
                # 새 페이지 수집
                collector = NotionDataSourceCollector(NOTION_TOKEN, DATA_SOURCE_ID)
                new_data = collect_missing_pages(
                    collector,
                    existing_info["page_ids"],
                    str(data_file),
                    limit=limit
                )

                # 수정된 페이지 업데이트
                updated_page_ids = set()
                if check_updates:
                    all_data = load_json(str(data_file))
                    old_data = all_data.copy()
                    all_data = update_changed_pages(collector, all_data, str(data_file))

                    # 어떤 페이지가 업데이트됐는지 추적
                    for old, new in zip(old_data, all_data):
                        if old.get("last_edited_time") != new.get("last_edited_time"):
                            updated_page_ids.add(new["page_id"])
                else:
                    all_data = load_json(str(data_file))

                # 인덱싱할 페이지 결정
                pages_to_index = [
                    p for p in all_data
                    if p["page_id"] in updated_page_ids or
                       p in new_data
                ]

        if trace:
            collection_span.end(metadata={
                "total_pages_to_index": len(pages_to_index)
            })

        if not pages_to_index:
            print("\n✅ 인덱싱할 페이지 없음 (모두 최신 상태)")
            return

        # limit 적용
        if limit and len(pages_to_index) > limit:
            print(f"\n⚠️  {len(pages_to_index)}개 페이지 중 {limit}개만 처리")
            pages_to_index = pages_to_index[:limit]

        print(f"\n📝 {len(pages_to_index)}개 페이지 인덱싱...")

        # 3. Qdrant 초기화 (체크용 client 생성)
        if qdrant_client is None:
            qdrant_client = QdrantClient(path=QDRANT_PATH)
        qdrant_info = check_qdrant_collection(qdrant_client)

        if force_recreate or not qdrant_info["exists"]:
            # 전체 재생성
            if trace:
                chunking_span = trace.span(name="chunking")

            # 청크 캐시 파일 경로
            chunks_cache_file = DATA_DIR / f"chunks_cache_{DB_NAME}.pkl"

            # 캐시된 청크가 있는지 확인
            if chunks_cache_file.exists() and not force_recreate:
                print(f"\n💾 캐시된 청크 로딩: {chunks_cache_file}")
                import pickle
                with open(chunks_cache_file, 'rb') as f:
                    all_chunks = pickle.load(f)
                print(f"✅ {len(all_chunks)}개 청크 로드 완료")
            else:
                # 청킹 수행
                all_chunks = []

                # 병렬 처리로 청킹 속도 향상
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import os

                max_workers = min(8, os.cpu_count() or 4)  # CPU 코어 수에 맞춰 조정
                print(f"  병렬 처리 (workers: {max_workers})")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 모든 페이지를 병렬로 처리
                    future_to_page = {
                        executor.submit(process_page_data, page, embedder, vision_model): page
                        for page in pages_to_index
                    }

                    for future in as_completed(future_to_page):
                        page = future_to_page[future]
                        try:
                            chunks = future.result()
                            all_chunks.extend(chunks)
                            print(f"  ✓ {page.get('title', 'Untitled')}: {len(chunks)}개 청크")
                        except Exception as e:
                            print(f"  ✗ {page.get('title', 'Untitled')}: 에러 - {e}")

                # 청크를 캐시에 저장
                print(f"\n💾 청크 캐싱: {chunks_cache_file}")
                import pickle
                with open(chunks_cache_file, 'wb') as f:
                    pickle.dump(all_chunks, f)
                print(f"✅ {len(all_chunks)}개 청크 저장 완료")

            if trace:
                chunking_span.end(metadata={"total_chunks": len(all_chunks)})

            if all_chunks:
                # LangChain Documents로 변환
                if trace:
                    conversion_span = trace.span(name="document_conversion")

                print(f"\n📄 LangChain Documents로 변환 중...")
                documents = chunks_to_documents(all_chunks)

                if trace:
                    conversion_span.end(metadata={"num_documents": len(documents)})

                # LangChain Qdrant vectorstore 생성 및 저장
                if trace:
                    storage_span = trace.span(name="qdrant_storage")

                print(f"\n💾 LangChain Qdrant 벡터스토어 생성 중...")

                # 기존 client 닫기 (중복 접근 방지)
                if qdrant_client:
                    del qdrant_client
                    qdrant_client = None

                # 새로운 client 생성 및 컬렉션 설정
                new_client = QdrantClient(path=QDRANT_PATH)

                # 컬렉션이 있으면 삭제
                try:
                    new_client.delete_collection(QDRANT_COLLECTION)
                    print(f"  🗑️ 기존 컬렉션 삭제: {QDRANT_COLLECTION}")
                except Exception:
                    pass

                # 벡터 차원 확인 (첫 번째 문서로)
                sample_embedding = langchain_embeddings.embed_query(documents[0].page_content)
                vector_dim = len(sample_embedding)
                print(f"  📏 벡터 차원: {vector_dim}")

                # 컬렉션 생성
                from qdrant_client.models import Distance, VectorParams
                new_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
                )
                print(f"  ✅ 컬렉션 생성: {QDRANT_COLLECTION}")

                # vectorstore 생성 및 문서 추가
                vectorstore = Qdrant(
                    client=new_client,
                    collection_name=QDRANT_COLLECTION,
                    embeddings=langchain_embeddings,
                )

                # 문서 추가
                vectorstore.add_documents(documents)
                print(f"✅ {len(documents)}개 문서 저장 완료")

                if trace:
                    storage_span.end()
        else:
            # 증분 업데이트: 변경된 페이지만 재인덱싱
            print("\n🔄 변경된 페이지 재인덱싱...")

            if trace:
                incremental_span = trace.span(name="incremental_update")

            # 기존 vectorstore 로드
            vectorstore = Qdrant(
                client=qdrant_client,
                collection_name=QDRANT_COLLECTION,
                embeddings=langchain_embeddings,
            )

            total_chunks = 0
            for page in pages_to_index:
                page_id = page["page_id"]

                # 기존 청크 삭제
                delete_page_from_vectorstore(vectorstore, page_id)

                # 새 청크 생성
                chunks = process_page_data(page, embedder, vision_model)

                if chunks:
                    # LangChain Documents로 변환 및 추가
                    documents = chunks_to_documents(chunks)
                    vectorstore.add_documents(documents)
                    total_chunks += len(chunks)

                    print(f"  ✅ {page.get('title', 'Untitled')}: {len(chunks)}개 청크 업데이트")

            if trace:
                incremental_span.end(metadata={
                    "pages_updated": len(pages_to_index),
                    "total_chunks": total_chunks
                })

        # 캐시 통계 출력
        print("\n📊 임베딩 캐시 통계:")
        cached_embedder.print_stats()

        print("\n" + "=" * 60)
        print("🎉 Vector DB 구축 완료!")
        print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="전체 재생성")
    parser.add_argument("--no-updates", action="store_true", help="수정 체크 안 함")
    parser.add_argument("--limit", type=int, default=None, help="처리할 페이지 수 제한")
    parser.add_argument("--input-file", type=str, default=None, help="사용할 JSON 파일 경로")
    args = parser.parse_args()

    main(force_recreate=args.force, check_updates=not args.no_updates, limit=args.limit, input_file=args.input_file)
