from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode  # Add this import
import faiss
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # Dimension for text-embedding-3-small

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
    
    # Initialize FAISS index
    faiss_index = faiss.IndexFlatIP(EMBED_DIM)  # Inner product for cosine similarity
    # Save FAISS index to disk - UPDATED PERSISTENCE
    faiss.write_index(faiss_index, str(INDEX_PATH / "faiss_index.bin"))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # Also save the vector store configuration
    vector_store.persist(persist_path=str(INDEX_PATH / "faiss_vector_store"))
    
    # Get chunk path
    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    chunk_files = list(chunk_path.glob("*.json"))
    
    if not chunk_files:
        print(f"❌ No JSON files found in {chunk_path}")
        return None, 0
    
    # Prepare nodes for insertion
    print(f"📚 Processing {len(chunk_files)} papers...")
    nodes = []
    
    for chunk_file in tqdm(chunk_files, desc="Processing papers"):
        with open(chunk_file, 'r') as f:
            try:
                paper_data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON: {chunk_file}")
                continue

        arxiv_id = paper_data.get("arxiv_id", chunk_file.stem)
        
        # Create nodes for each chunk
        for i, chunk_text in enumerate(paper_data["chunks"]):
            # Create unique ID for each chunk
            chunk_id = f"{arxiv_id}_{i}"
            doc_id = hashlib.md5(chunk_id.encode()).hexdigest()

            authors_list = paper_data.get("authors", [])
            authors_str = ", ".join(authors_list)
            
            # Create TextNode with metadata
            node = TextNode(
                id_=doc_id,
                text=chunk_text,
                metadata={
                    "paper_id": arxiv_id,
                    "title": paper_data.get("title", "Untitled Paper"),
                    "chunk_id": i,
                    "file_path": paper_data.get("file_path", ""),
                    "arxiv_id": arxiv_id,
                    "authors": authors_str,
                    "published": paper_data.get("published", "")
                }
            )
            nodes.append(node)
    
    print(f"🧩 Total chunks to index: {len(nodes)}")
    
    # Create storage context and embedding model
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    # Create index - use nodes directly
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True  # Show progress bar during indexing
    )
    
    # Save FAISS index to disk
    vector_store.persist(persist_path=str(INDEX_PATH / "faiss_index"))
    
    # Calculate cost metrics
    vector_count = len(nodes)
    approx_tokens_per_chunk = 300
    total_tokens = vector_count * approx_tokens_per_chunk
    
    # Updated pricing for text-embedding-3-small ($0.02 per 1M tokens)
    embedding_cost = (total_tokens / 1_000_000) * 0.02
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Index created successfully!")
    print(f"   Vectors: {vector_count} | Time: {elapsed:.1f}s")
    
    return index, vector_count

if __name__ == "__main__":
    # Clear previous index data
    faiss_index_path = INDEX_PATH / "faiss_index"
    if faiss_index_path.exists():
        print("🧹 Cleaning up previous index data...")
        shutil.rmtree(faiss_index_path)
    
    index, vector_count = create_vector_index()
    if index and vector_count > 0:
        print("\n🔍 Testing query...")
        query_engine = index.as_query_engine(similarity_top_k=1)
        test_response = query_engine.query("What is the main topic?")
        print(f"   Test response: {test_response.response[:150]}...")
        print("\n🎉 Index is ready for querying!")
        
        print(f"\n✅ Successfully created index with {vector_count} vectors")
        print(f"🔑 Vector store persisted at: {INDEX_PATH / 'faiss_index'}")
        print(f"💡 Estimated embedding cost: ${(vector_count * 300 / 1_000_000) * 0.02:.6f}")
