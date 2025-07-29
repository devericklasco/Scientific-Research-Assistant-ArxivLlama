import os
import time
import json
import hashlib
import shutil
import stat
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()
EMBED_MODEL = "text-embedding-3-small"

def debug_perms(path: Path):
    mode = path.stat().st_mode
    print(f"🔍 perms for {path}: {oct(mode)}")

def create_vector_index():
    # 1) API key check
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        return None, 0

    print("🚀 Starting index creation…")
    start_time = time.time()

    # 2) Choose storage
    if "STREAMLIT_SERVER" in os.environ:
        persist_dir = Path("/tmp/chroma_db")
    else:
        persist_dir = Path(os.getenv("INDEX_PATH", "./data/indices/chroma_db"))
    print(f"📁 Using persist_dir = {persist_dir.resolve()}")

    # 3) Before wipe: does it exist?
    if persist_dir.exists():
        print(f"⚠️ {persist_dir} already exists!")
        print("Contents before delete:", persist_dir.iterdir())
        debug_perms(persist_dir)
    else:
        print(f"ℹ️ {persist_dir} does not exist yet.")

    # 4) Wipe it out
    try:
        shutil.rmtree(persist_dir, ignore_errors=False)
        print(f"✅ Removed {persist_dir}")
    except Exception as e:
        print(f"❌ Error removing {persist_dir}: {e}")

    # 5) Make directory
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {persist_dir}")
    except Exception as e:
        print(f"❌ Error creating {persist_dir}: {e}")

    # 6) Check perms & set them
    try:
        debug_perms(persist_dir)
        os.chmod(persist_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        print(f"🔧 Applied chmod 0o777 to {persist_dir}")
        debug_perms(persist_dir)
    except Exception as e:
        print(f"❌ chmod error: {e}")

    # 7) Initialize Chroma client
    try:
        chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chromadb.Settings(
                is_persistent=True,
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print("✅ Initialized PersistentClient")
    except Exception as e:
        print(f"❌ Error initializing Chroma client: {e}")
        return None, 0

    try:
        chroma_collection = chroma_client.get_or_create_collection(
            name="arxiv_papers",
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ Got/created collection 'arxiv_papers'")
    except Exception as e:
        print(f"❌ get_or_create_collection error: {e}")
        return None, 0

    # 8) Load JSON chunks
    chunk_path  = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    chunk_files = list(chunk_path.glob("*.json"))
    print(f"📂 Looking for JSONs in {chunk_path.resolve()}")
    print(f"📑 Found {len(chunk_files)} files")
    if not chunk_files:
        return None, 0

    ids, documents, metadatas = [], [], []
    for cf in tqdm(chunk_files, desc="Reading JSONs"):
        try:
            paper = json.loads(cf.read_text())
        except Exception as e:
            print(f"⚠️ Skipping {cf}: {e}")
            continue
        aid = paper.get("arxiv_id", cf.stem)
        auth = ", ".join(paper.get("authors", []))
        for i, txt in enumerate(paper.get("chunks", [])):
            digest = hashlib.md5(f"{aid}_{i}".encode()).hexdigest()
            ids.append(digest)
            documents.append(txt)
            metadatas.append({
                "paper_id": aid,
                "title": paper.get("title", "Untitled"),
                "chunk_id": i,
                "file_path": paper.get("file_path", ""),
                "arxiv_id": aid,
                "authors": auth,
                "published": paper.get("published", "")
            })
    print(f"🧩 Loaded total chunks: {len(ids)}")

    # 9) Embed + index
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBED_MODEL
    )
    batch_sz = 500
    for i in tqdm(range(0, len(ids), batch_sz), desc="Embedding & Indexing"):
        batch_ids   = ids[i:i+batch_sz]
        batch_docs  = documents[i:i+batch_sz]
        batch_mets  = metadatas[i:i+batch_sz]
        try:
            batch_embs = embed_fn(batch_docs)
            chroma_collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_mets
            )
        except Exception as e:
            print(f"❌ Error in batch {i//batch_sz}: {e}")
            # inspect persist_dir right after failure
            print("Contents now:", list(persist_dir.iterdir()))
            debug_perms(persist_dir)
            return None, 0

    # 10) Wrap in LlamaIndex
    try:
        vs = ChromaVectorStore(chroma_collection=chroma_collection)
        sc = StorageContext.from_defaults(vector_store=vs)
        em = OpenAIEmbedding(model=EMBED_MODEL)
        index = VectorStoreIndex.from_vector_store(vs, storage_context=sc, embed_model=em)
    except Exception as e:
        print(f"❌ LlamaIndex wrapper error: {e}")
        return None, 0

    vec_count = chroma_collection.count()
    elapsed   = time.time() - start_time
    print(f"\n✅ Complete: {vec_count} vectors in {elapsed:.1f}s")

    return index, vec_count


if __name__ == "__main__":
    idx, cnt = create_vector_index()
    if idx:
        print(f"🎉 CLI run created {cnt} vectors.")
