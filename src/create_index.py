import os
import json
import hashlib
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import faiss
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode

# ─── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()
CHUNK_PATH  = Path(os.getenv("CHUNK_PATH",  "./data/chunks"))
INDEX_ROOT  = Path(os.getenv("INDEX_PATH", "./data/indices"))
STORE_DIR   = INDEX_ROOT / "faiss_store"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536

# ─── Index builder ─────────────────────────────────────────────────────────────

def create_vector_index():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("❌ OPENAI_API_KEY not set in environment")

    print("🚀 Starting index creation…")
    t0 = time.time()

    # 1) Clear any old store
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    # 2) Create an empty FAISS index and wrap in LlamaIndex
    faiss_idx    = faiss.IndexFlatIP(EMBED_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_idx)

    # 3) Build a StorageContext _in memory_ using that vector_store
    storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)
    embed_model  = OpenAIEmbedding(model=EMBED_MODEL)

    # 4) Load your chunk files
    chunk_files = list(CHUNK_PATH.glob("*.json"))
    if not chunk_files:
        raise RuntimeError(f"❌ No JSON chunks found in {CHUNK_PATH}")

    print(f"📚 Processing {len(chunk_files)} files…")
    nodes = []
    for cf in tqdm(chunk_files, desc="Reading chunks"):
        data = json.loads(cf.read_text())
        arxiv_id = data.get("arxiv_id", cf.stem)
        authors  = ", ".join(data.get("authors", []))

        for i, txt in enumerate(data.get("chunks", [])):
            uid = hashlib.md5(f"{arxiv_id}_{i}".encode()).hexdigest()
            nodes.append(TextNode(
                id_=uid,
                text=txt,
                metadata={
                    "paper_id": arxiv_id,
                    "title":     data.get("title", "Untitled"),
                    "chunk_id":  i,
                    "file_path": data.get("file_path", ""),
                    "arxiv_id":  arxiv_id,
                    "authors":   authors,
                    "published": data.get("published", ""),
                }
            ))

    # 5) Build the VectorStoreIndex _in memory_
    print(f"🧩 Indexing {len(nodes)} chunks…")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_ctx,
        embed_model=embed_model,
        show_progress=True
    )

    # 6) Persist _all_ storage artifacts to disk:
    #    - faiss_index.bin
    #    - default__vector_store.json
    #    - docstore.json
    index.storage_context.persist(persist_dir=str(STORE_DIR))

    # 7) Report metrics
    n = len(nodes)
    tokens = n * 300
    cost   = tokens / 1e6 * 0.02
    elapsed = time.time() - t0

    print(f"\n✅ Done: {n} vectors in {elapsed:.1f}s")
    print(f"   Estimated tokens: {tokens}")
    print(f"   Embedding cost: ${cost:.6f}")
    print(f"   Store directory: {STORE_DIR}")

    return index, n

# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    idx, count = create_vector_index()
    if count > 0:
        print("\n🔍 Quick test query…")
        qe   = idx.as_query_engine(similarity_top_k=1)
        resp = qe.query("What is the main topic?")
        print("→", resp.response[:150], "…")
