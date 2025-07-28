import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode  # Add this import
import faiss
from tqdm import tqdm
import json
import hashlib
import time
import shutil

# Load environment variables
load_dotenv()

# Configuration
CHUNK_PATH = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./data/indices"))
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # Dimension for text-embedding-3-small

def create_vector_index():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        return None, 0

    print("🚀 Starting index creation...")
    start_time = time.time()
   
    
    # Create storage directory
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize FAISS index
    faiss_index = faiss.IndexFlatIP(EMBED_DIM)  # Inner product for cosine similarity
    # Save FAISS index to disk - UPDATED PERSISTENCE
    faiss.write_index(faiss_index, str(INDEX_PATH / "faiss_index.bin"))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # Also save the vector store configuration
    vector_store.persist(persist_path=str(INDEX_PATH / "faiss_vector_store"))
    
    # Load and process chunk files
    chunk_files = list(CHUNK_PATH.glob("*.json"))
    
    if not chunk_files:
        print(f"❌ No JSON files found in {CHUNK_PATH}. Run PDF processor first.")
        return None, 0
    
    # Prepare nodes for insertion
    print(f"📚 Processing {len(chunk_files)} papers...")
    nodes = []
    
    for chunk_file in tqdm(chunk_files, desc="Processing papers"):
        with open(chunk_file, 'r') as f:
            try:
                paper_data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON file: {chunk_file}")
                continue
                
        # Safely get arxiv_id with fallback
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
    
    print(f"\n✅ Index creation complete!")
    print(f"   Total vectors: {vector_count}")
    print(f"   Time taken: {elapsed:.1f} seconds")
    print(f"   Estimated tokens processed: {total_tokens}")
    print(f"   Estimated embedding cost: ${embedding_cost:.6f}")
    
    return index, vector_count

if __name__ == "__main__":
    # Clear previous index data
    faiss_index_path = INDEX_PATH / "faiss_index"
    if faiss_index_path.exists():
        print("🧹 Cleaning up previous index data...")
        shutil.rmtree(faiss_index_path)
    
    index, vector_count = create_vector_index()
    
    if index and vector_count > 0:
        print("\n🔍 Testing query functionality...")
        query_engine = index.as_query_engine(similarity_top_k=1)
        test_response = query_engine.query("What is the main topic?")
        print(f"   Test response: {test_response.response[:150]}...")
        print("\n🎉 Index is ready for querying!")
        
        print(f"\n✅ Successfully created index with {vector_count} vectors")
        print(f"🔑 Vector store persisted at: {INDEX_PATH / 'faiss_index'}")
        print(f"💡 Estimated embedding cost: ${(vector_count * 300 / 1_000_000) * 0.02:.6f}")