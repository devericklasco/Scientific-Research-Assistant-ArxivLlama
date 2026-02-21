from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from tqdm import tqdm

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def normalize_authors(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def compute_paper_hash(paper_data: Dict) -> str:
    payload = {
        "arxiv_id": paper_data.get("arxiv_id", ""),
        "title": paper_data.get("title", ""),
        "published": paper_data.get("published", ""),
        "authors": normalize_authors(paper_data.get("authors", [])),
        "chunks": paper_data.get("chunks", []),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def iter_chunk_records(paper_data: Dict) -> Iterable[Dict]:
    arxiv_id = paper_data.get("arxiv_id", "")
    title = paper_data.get("title", "Untitled Paper")
    file_path = paper_data.get("file_path", "")
    published = paper_data.get("published", "")
    authors = normalize_authors(paper_data.get("authors", []))
    chunks = paper_data.get("chunks", [])

    if chunks and isinstance(chunks[0], str):
        for i, text in enumerate(chunks):
            yield {
                "id_seed": f"{arxiv_id}_{i}",
                "text": text,
                "metadata": {
                    "paper_id": arxiv_id,
                    "title": title,
                    "chunk_id": i,
                    "file_path": file_path,
                    "arxiv_id": arxiv_id,
                    "authors": ", ".join(authors),
                    "published": published,
                    "page_start": -1,
                    "page_end": -1,
                    "section": "content",
                },
            }
        return

    for i, chunk in enumerate(chunks):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        yield {
            "id_seed": f"{arxiv_id}_{chunk.get('chunk_id', i)}",
            "text": text,
            "metadata": {
                "paper_id": arxiv_id,
                "title": title,
                "chunk_id": int(chunk.get("chunk_id", i)),
                "file_path": file_path,
                "arxiv_id": arxiv_id,
                "authors": ", ".join(authors),
                "published": published,
                "page_start": int(chunk.get("page_start", -1)),
                "page_end": int(chunk.get("page_end", -1)),
                "section": str(chunk.get("section", "content")),
            },
        }


def _index_path() -> Path:
    path = Path(os.getenv("INDEX_PATH", "./data/indices"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_vector_index(force_rebuild: bool = False) -> Tuple[VectorStoreIndex | None, int]:
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        return None, 0

    print("🚀 Starting index creation...")
    start_time = time.time()

    index_path = _index_path()
    faiss_store_path = index_path / "faiss_vector_store"
    faiss_index_file = index_path / "faiss_index.bin"
    if force_rebuild:
        if faiss_store_path.exists():
            shutil.rmtree(faiss_store_path, ignore_errors=True)
        if faiss_index_file.exists():
            faiss_index_file.unlink(missing_ok=True)

    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    chunk_files = list(chunk_path.glob("*.json"))
    if not chunk_files:
        print(f"❌ No JSON files found in {chunk_path}")
        return None, 0

    print(f"📚 Processing {len(chunk_files)} papers...")
    nodes: List[TextNode] = []
    for chunk_file in tqdm(chunk_files, desc="Processing papers"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            try:
                paper_data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON: {chunk_file}")
                continue

        for record in iter_chunk_records(paper_data):
            doc_id = hashlib.md5(record["id_seed"].encode()).hexdigest()
            nodes.append(
                TextNode(
                    id_=doc_id,
                    text=record["text"],
                    metadata=record["metadata"],
                )
            )

    if not nodes:
        print("❌ No valid chunks available for indexing")
        return None, 0

    print(f"🧩 Total chunks to index: {len(nodes)}")

    faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    vector_store.persist(persist_path=str(faiss_store_path))
    faiss.write_index(faiss_index, str(faiss_index_file))

    elapsed = time.time() - start_time
    vector_count = len(nodes)
    print(f"\n✅ Index created successfully!")
    print(f"   Vectors: {vector_count} | Time: {elapsed:.1f}s")
    return index, vector_count


if __name__ == "__main__":
    index, vector_count = create_vector_index(force_rebuild=True)
    if index and vector_count > 0:
        print(f"\n🎉 Index ready with {vector_count} vectors")
