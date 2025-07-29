
import os
import shutil
import stat
import time
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# ── Load .env so OPENAI_API_KEY and CHUNK_PATH, INDEX_PATH work ─────────
load_dotenv()

# Which embedding model to use
EMBED_MODEL = "text-embedding-3-small"

def create_vector_index():
    # ── 1) Bail if no API key ────────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        return None, 0

    print("🚀 Starting index creation...")
    start_time = time.time()

    # ── 2) Decide where to store ChromaDB files ─────────────────────────
    #    On Streamlit Cloud we put under /tmp (writable), otherwise use INDEX_PATH
    if "STREAMLIT_SERVER" in os.environ:
        persist_dir = "/tmp/chroma_db"
    else:
        persist_dir = os.getenv("INDEX_PATH", "./data/indices/chroma_db")

    # ── 3) Wipe out any old index folder + ensure full permissions ───────
    persist_path = Path(persist_dir)
    if persist_path.exists():
        shutil.rmtree(persist_path, ignore_errors=True)
    persist_path.mkdir(parents=True, exist_ok=True)
    os.chmod(persist_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # ── 4) Initialize persistent Chroma client ──────────────────────────
    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(
            is_persistent=True,
            anonymized_telemetry=False,
            allow_reset=True,
        )
    )
    chroma_collection = chroma_client.get_or_create_collection(
        name="arxiv_papers",
        metadata={"hnsw:space": "cosine"}
    )

    # ── 5) Load all your JSON chunk files into memory (no embeddings yet) ─
    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    chunk_files = list(chunk_path.glob("*.json"))
    if not chunk_files:
        print(f"❌ No JSON files found in {chunk_path}. Run PDF processor first.")
        return None, 0

    print(f"📚 Loading {len(chunk_files)} papers…")
    ids, documents, metadatas = [], [], []
    for chunk_file in tqdm(chunk_files, desc="Reading JSONs"):
        with open(chunk_file, 'r') as f:
            paper_data = json.load(f)
        arxiv_id    = paper_data.get("arxiv_id", chunk_file.stem)
        authors_str = ", ".join(paper_data.get("authors", []))
        for i, chunk_text in enumerate(paper_data["chunks"]):
            # unique doc_id per chunk
            chunk_id = f"{arxiv_id}_{i}"
            doc_id   = hashlib.md5(chunk_id.encode()).hexdigest()
            ids.append(doc_id)
            documents.append(chunk_text)
            metadatas.append({
                "paper_id": arxiv_id,
                "title":     paper_data.get("title", "Untitled Paper"),
                "chunk_id":  i,
                "file_path": paper_data.get("file_path", ""),
                "arxiv_id":  arxiv_id,
                "authors":   authors_str,
                "published": paper_data.get("published", "")
            })
    print(f"🧩 Total chunks loaded: {len(ids)}")

    # ── 6) Batch‑embed & insert to Chroma ───────────────────────────────
    embed_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBED_MODEL
    )
    batch_size = 500
    for i in tqdm(range(0, len(ids), batch_size), desc="Embedding & Indexing"):
        batch_ids    = ids[i : i+batch_size]
        batch_docs   = documents[i : i+batch_size]
        batch_metas  = metadatas[i : i+batch_size]
        # single API call per batch
        batch_embeds = embed_func(batch_docs)
        chroma_collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_metas
        )

    # ── 7) Wrap Chroma collection in a LlamaIndex VectorStoreIndex ──────
    vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model     = OpenAIEmbedding(model=EMBED_MODEL)
    index           = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # ── 8) Report metrics ────────────────────────────────────────────────
    vector_count = chroma_collection.count()
    approx_tokens = 300
    total_tokens  = vector_count * approx_tokens
    cost_estimate = (total_tokens / 1000) * 0.00002
    elapsed       = time.time() - start_time

    print(f"\n✅ Index creation complete!")
    print(f"   Vectors: {vector_count}")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Tokens: {total_tokens}")
    print(f"   Est. cost: ${cost_estimate:.6f}")

    return index, vector_count


if __name__ == "__main__":
    # standalone test run from CLI
    idx_path = Path("./data/indices/chroma_db")
    if idx_path.exists():
        print("🧹 Cleaning old index…")
        shutil.rmtree(idx_path)
    index, count = create_vector_index()
    if index:
        print(f"🎉 Created {count} vectors at {idx_path}")
