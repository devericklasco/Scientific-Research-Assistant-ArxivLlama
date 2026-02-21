from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "arxiv_papers"
MANIFEST_FILENAME = "index_manifest.json"

def get_collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION_NAME", COLLECTION_NAME)

def get_embedding_backend() -> str:
    backend = os.getenv("EMBEDDING_BACKEND", "openai").strip().lower()
    return backend if backend in {"openai", "local"} else "openai"

def ensure_writable_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        probe.unlink(missing_ok=True)
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        return path
    except OSError:
        fallback = Path("/tmp") / f"chroma_db_fallback_{int(time.time())}"
        fallback.mkdir(parents=True, exist_ok=True)
        os.chmod(fallback, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        return fallback

def normalize_authors(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []

def compute_paper_hash(paper_data: Dict) -> str:
    hash_payload = {
        "arxiv_id": paper_data.get("arxiv_id", ""),
        "title": paper_data.get("title", ""),
        "published": paper_data.get("published", ""),
        "authors": normalize_authors(paper_data.get("authors", [])),
        "chunks": paper_data.get("chunks", []),
    }
    return hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
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

def load_manifest(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
    except (OSError, json.JSONDecodeError):
        pass
    return {}

def save_manifest(manifest_path: Path, manifest: Dict[str, str]) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

def _delete_paper_chunks(chroma_collection, arxiv_id: str) -> None:
    try:
        chroma_collection.delete(where={"arxiv_id": arxiv_id})
    except Exception:
        # Safe no-op when ids do not exist.
        pass

def create_vector_index(force_rebuild: bool = False) -> Tuple[VectorStoreIndex | None, int]:
    embedding_backend = get_embedding_backend()
    if embedding_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        return None, 0

    print("🚀 Starting index creation...")
    start_time = time.time()
    
    # Determine storage location
    persist_dir = "/tmp/chroma_db" if "STREAMLIT_SERVER" in os.environ else os.getenv("INDEX_PATH", "./data/indices/chroma_db")
    
    # Ensure directory exists with proper permissions
    persist_path = ensure_writable_directory(Path(persist_dir))
    persist_dir = str(persist_path)
    manifest_path = persist_path / MANIFEST_FILENAME
    collection_name = get_collection_name()
    embedding_model_name = EMBED_MODEL if embedding_backend == "openai" else LOCAL_EMBED_MODEL
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(
            is_persistent=True,
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    if force_rebuild:
        try:
            chroma_client.delete_collection(collection_name)
            print("♻️ Deleted existing collection")
        except Exception:
            pass
        if manifest_path.exists():
            manifest_path.unlink()

    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model_name,
        },
    )
    
    # Get chunk path
    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    chunk_files = list(chunk_path.glob("*.json"))
    
    if not chunk_files:
        print(f"❌ No JSON files found in {chunk_path}")
        return None, 0
    
    print(f"📚 Processing {len(chunk_files)} papers...")
    previous_manifest = load_manifest(manifest_path)
    next_manifest: Dict[str, str] = {}
    changed_papers: List[Tuple[str, Dict, str]] = []
    current_papers = set()

    for chunk_file in tqdm(chunk_files, desc="Collecting chunks"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            try:
                paper_data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON: {chunk_file}")
                continue

        arxiv_id = paper_data.get("arxiv_id", chunk_file.stem)
        current_papers.add(arxiv_id)
        paper_hash = compute_paper_hash(paper_data)
        next_manifest[arxiv_id] = paper_hash
        if force_rebuild or previous_manifest.get(arxiv_id) != paper_hash:
            changed_papers.append((arxiv_id, paper_data, paper_hash))

    removed_papers = set(previous_manifest.keys()) - current_papers
    for arxiv_id in removed_papers:
        _delete_paper_chunks(chroma_collection, arxiv_id)

    if changed_papers:
        print(f"♻️ Updating {len(changed_papers)} changed papers (incremental indexing)...")
    else:
        print("✅ No content changes detected; reusing existing vectors.")

    all_ids: List[str] = []
    all_documents: List[str] = []
    all_metadatas: List[Dict] = []
    for arxiv_id, paper_data, _ in tqdm(changed_papers, desc="Preparing changed papers"):
        _delete_paper_chunks(chroma_collection, arxiv_id)
        for record in iter_chunk_records(paper_data):
            doc_id = hashlib.md5(record["id_seed"].encode()).hexdigest()
            all_ids.append(doc_id)
            all_documents.append(record["text"])
            all_metadatas.append(record["metadata"])

    if all_documents:
        if embedding_backend == "openai":
            embed_func = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=EMBED_MODEL
            )
        else:
            embed_func = embedding_functions.DefaultEmbeddingFunction()
        print("🧬 Generating embeddings in batches...")
        all_embeddings = []
        embed_batch_size = 100
        for i in tqdm(range(0, len(all_documents), embed_batch_size), desc="Embedding"):
            batch_docs = all_documents[i : i + embed_batch_size]
            all_embeddings.extend(embed_func(batch_docs))

        print("📥 Inserting vectors into database...")
        insert_batch_size = 200
        for i in tqdm(range(0, len(all_ids), insert_batch_size), desc="Indexing"):
            try:
                chroma_collection.add(
                    ids=all_ids[i : i + insert_batch_size],
                    documents=all_documents[i : i + insert_batch_size],
                    embeddings=all_embeddings[i : i + insert_batch_size],
                    metadatas=all_metadatas[i : i + insert_batch_size],
                )
            except Exception as exc:
                if "readonly database" in str(exc).lower():
                    raise RuntimeError(
                        f"ChromaDB path is read-only at '{persist_dir}'. "
                        "Use a writable per-session INDEX_PATH (for example under /tmp or data/sessions)."
                    ) from exc
                raise

    save_manifest(manifest_path, next_manifest)
    
    # Create LlamaIndex wrapper
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL) if os.getenv("OPENAI_API_KEY") else None
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # Calculate metrics
    vector_count = chroma_collection.count()
    elapsed = time.time() - start_time
    
    print(f"\n✅ Index created successfully!")
    print(f"   Vectors: {vector_count} | Time: {elapsed:.1f}s")
    
    return index, vector_count

if __name__ == "__main__":
    # Test run
    index, vector_count = create_vector_index()
    if index and vector_count > 0:
        print("\n🔍 Testing query...")
        query_engine = index.as_query_engine(similarity_top_k=1)
        test_response = query_engine.query("What is the main topic?")
        print(f"   Test response: {test_response.response[:150]}...")
        print("\n🎉 Index ready for use!")
        print(f"💡 Estimated embedding cost: ${(vector_count * 300 / 1_000_000) * 0.02:.6f}")
