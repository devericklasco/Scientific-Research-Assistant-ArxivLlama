import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from chromadb.utils import embedding_functions
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

def create_vector_index():
    print("🚀 Starting index creation...")
    start_time = time.time()
    
    # Create storage directory
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(INDEX_PATH / "chroma_db"))
    
    # Clean up existing collection if it exists
    if "arxiv_papers" in [c.name for c in chroma_client.list_collections()]:
        print("♻️ Removing existing collection...")
        chroma_client.delete_collection("arxiv_papers")
    
    # Create new collection with embedding function
    embed_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBED_MODEL
    )
    
    chroma_collection = chroma_client.get_or_create_collection(
        name="arxiv_papers",
        embedding_function=embed_func,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Load and process chunk files
    chunk_files = list(CHUNK_PATH.glob("*.json"))
    
    if not chunk_files:
        print(f"❌ No JSON files found in {CHUNK_PATH}. Run PDF processor first.")
        return None, 0
    
    # Prepare data for insertion
    print(f"📚 Processing {len(chunk_files)} papers...")
    ids = []
    documents = []
    metadatas = []
    
    for chunk_file in tqdm(chunk_files, desc="Processing papers"):
        with open(chunk_file, 'r') as f:
            paper_data = json.load(f)
            
        # Create entries for each chunk
        for i, chunk_text in enumerate(paper_data["chunks"]):
            # Create unique ID for each chunk
            chunk_id = f"{paper_data['paper_id']}_{i}"
            doc_id = hashlib.md5(chunk_id.encode()).hexdigest()
            
            ids.append(doc_id)
            documents.append(chunk_text)
            metadatas.append({
                "paper_id": paper_data["paper_id"],
                "title": paper_data["title"],
                "chunk_id": i,
                "file_path": paper_data["file_path"]
            })
    
    print(f"🧩 Total chunks to index: {len(ids)}")
    
    # Batch insertion
    batch_size = 100
    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing chunks"):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        chroma_collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas
        )
    
    # Create LlamaIndex wrapper
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Configure embedding model for LlamaIndex
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    
    # Calculate cost metrics
    vector_count = chroma_collection.count()
    approx_tokens_per_chunk = 300  # Average tokens per 512-character chunk
    total_tokens = vector_count * approx_tokens_per_chunk
    embedding_cost = (total_tokens / 1000) * 0.00002  # text-embedding-3-small pricing
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Index creation complete!")
    print(f"   Total vectors: {vector_count}")
    print(f"   Time taken: {elapsed:.1f} seconds")
    print(f"   Estimated tokens processed: {total_tokens}")
    print(f"   Estimated embedding cost: ${embedding_cost:.6f}")
    
    return index, vector_count

if __name__ == "__main__":
    # Clear previous index data
    chroma_db_path = INDEX_PATH / "chroma_db"
    if chroma_db_path.exists():
        print("🧹 Cleaning up previous index data...")
        shutil.rmtree(chroma_db_path)
    
    index, vector_count = create_vector_index()
    
    if index and vector_count > 0:
        print("\n🔍 Testing query functionality...")
        query_engine = index.as_query_engine(similarity_top_k=1)
        test_response = query_engine.query("What is the main topic?")
        print(f"   Test response: {test_response.response[:150]}...")
        print("\n🎉 Index is ready for querying!")
        
        # Final summary metrics
        print(f"\n✅ Successfully created index with {vector_count} vectors")
        print(f"🔑 Vector store persisted at: {INDEX_PATH / 'chroma_db'}")
        print(f"💡 Estimated embedding cost: ${vector_count * 0.0000001:.6f} (simplified)")