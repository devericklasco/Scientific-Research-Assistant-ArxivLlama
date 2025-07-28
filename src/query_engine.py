import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import json
import time
from dotenv import load_dotenv

import tiktoken
import faiss

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from src.citation_generator import generate_apa_citation

# ─── Load env & config ─────────────────────────────────────────────────────────

load_dotenv()
EMBED_MODEL = "text-embedding-3-small"
STORE_DIR   = Path(os.getenv("INDEX_PATH", "./data/indices")) / "faiss_store"

# ─── Utils ───────────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

# ─── Core initialization ────────────────────────────────────────────────────────

def initialize_engine():
    """Initialize the FAISS-based query engine"""
    if not STORE_DIR.exists():
        raise RuntimeError(f"No persisted index found at {STORE_DIR}")

    # ⚙️ Clean up any corrupted JSON that isn’t valid UTF‑8
    bad = STORE_DIR / "image__vector_store.json"
    if bad.exists():
        bad.unlink()  # remove the file so LlamaIndex won’t try to JSON‑decode it

    # 1) Load **all** storage: FAISS index, vector‑store metadata, docstore, index store, etc.
    storage_context = StorageContext.from_defaults(persist_dir=str(STORE_DIR))

    # 2) Wire up your embedding model for any downstream uses
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)

    # 3) Reconstruct the **full** VectorStoreIndex (embeddings + text) from that context
    index = VectorStoreIndex.from_storage_context(storage_context)

    # 4) Spin up the query engine
    llm = OpenAI(model="gpt-4o", temperature=0.1)
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode="compact",
    )
    return query_engine

# ─── Helpers ─────────────────────────────────────────────────────────────────────

def get_paper_recommendations(query_engine, topic: str, num_papers: int = 3) -> str:
    prompt = (
        f"Based on the research topic: '{topic}', recommend {num_papers} papers from the collection. "
        "For each recommendation, include:\n"
        "1. Paper title\n"
        "2. Brief justification (1 sentence)\n"
        "3. Key contribution\n"
        "Format as markdown bullet points."
    )
    response = query_engine.query(prompt)
    return response.response

def load_paper_metadata():
    metadata = {}
    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    for json_file in chunk_path.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            metadata[data["arxiv_id"]] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return metadata

# ─── CLI Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing research assistant…")
    engine = initialize_engine()
    paper_metadata = load_paper_metadata()
    print("✅ System ready. Type your questions below.")
    print("   Type 'exit' to quit or '!recommend' for paper recommendations\n")

    total_cost = 0.0
    while True:
        query = input("📝 Your research question: ")
        if query.lower() in ["exit", "quit"]:
            break

        if query.startswith("!recommend"):
            topic = query.replace("!recommend", "").strip() or "machine learning"
            print(f"\n🔍 Recommendations for '{topic}':")
            print(get_paper_recommendations(engine, topic))
            continue

        # Run the query and track cost/time
        start = time.time()
        response = engine.query(query)
        elapsed = time.time() - start

        ctx_text = " ".join(n.text for n in response.source_nodes) if response.source_nodes else ""
        in_toks = count_tokens(query + ctx_text)
        out_toks = count_tokens(response.response)
        # GPT-4o pricing: $5/1M in, $15/1M out
        cost = (in_toks * 5 + out_toks * 15) / 1_000_000
        total_cost += cost

        # Print answer
        print(f"\n💡 Answer ({elapsed:.2f}s, ${cost:.6f}):\n{response.response}\n")

        # Print sources if available
        if response.source_nodes:
            print("🔍 Sources:")
            for i, node in enumerate(response.source_nodes, 1):
                md = node.metadata or {}
                pid = md.get("arxiv_id", "unknown")
                pm = paper_metadata.get(pid, {})
                title   = pm.get("title", md.get("title", "Untitled"))
                authors = pm.get("authors", md.get("authors", "Unknown"))
                print(f"{i}. {title} [{pid}]")
                print(f"   Authors: {authors}")
                print(f"   Score: {node.score:.3f}")
                print(f"   Excerpt: {node.text[:120]}…")
                print(f"   Citation: {generate_apa_citation(pm or md)}\n")
        else:
            print("🔍 No sources found.\n")

    print(f"Session ended. Total cost: ${total_cost:.6f}")
