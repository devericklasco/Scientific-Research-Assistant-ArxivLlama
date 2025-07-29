import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import json
import hashlib
import time
import shutil
import stat

# Load environment variables
load_dotenv()

# Configuration
EMBED_MODEL = "text-embedding-3-small"

def create_vector_index():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        return None, 0

    print("🚀 Starting index creation...")
    start_time = time.time()
    
    # Determine storage location
    persist_dir = "/tmp/chroma_db" if "STREAMLIT_SERVER" in os.environ else os.getenv("INDEX_PATH", "./data/indices/chroma_db")
    
    # Ensure clean directory with proper permissions
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    # Set permissions (read/write/execute for all)
    os.chmod(persist_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(
            is_persistent=True,
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Clean up existing collection
    try:
        chroma_client.delete_collection("arxiv_papers")
        print("♻️ Deleted existing collection")
    except Exception as e:
        print(f"ℹ️ No existing collection to delete: {str(e)}")
    
    # Create new collection
    chroma_collection = chroma_client.create_collection(
        name="arxiv_papers",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Get chunk path
    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    chunk_files = list(chunk_path.glob("*.json"))
    
    if not chunk_files:
        print(f"❌ No JSON files found in {chunk_path}")
        return None, 0
    
    print(f"📚 Processing {len(chunk_files)} papers...")
    
    # Initialize data structures
    all_ids = []
    all_documents = []
    all_metadatas = []
    
    # Create embedding function
    embed_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBED_MODEL
    )
    
    # First pass: Collect all chunks
    for chunk_file in tqdm(chunk_files, desc="Collecting chunks"):
        with open(chunk_file, 'r') as f:
            try:
                paper_data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON: {chunk_file}")
                continue
                
        arxiv_id = paper_data.get("arxiv_id", chunk_file.stem)
        authors_str = ", ".join(paper_data.get("authors", []))
        
        for i, chunk_text in enumerate(paper_data["chunks"]):
            chunk_id = f"{arxiv_id}_{i}"
            doc_id = hashlib.md5(chunk_id.encode()).hexdigest()
            
            all_ids.append(doc_id)
            all_documents.append(chunk_text)
            all_metadatas.append({
                "paper_id": arxiv_id,
                "title": paper_data.get("title", "Untitled"),
                "chunk_id": i,
                "file_path": paper_data.get("file_path", ""),
                "arxiv_id": arxiv_id,
                "authors": authors_str,
                "published": paper_data.get("published", "")
            })
    
    # Batch embedding generation
    print("🧬 Generating embeddings in batches...")
    all_embeddings = []
    embed_batch_size = 100  # Optimal for OpenAI API
    
    for i in tqdm(range(0, len(all_documents), embed_batch_size), desc="Embedding"):
        batch_docs = all_documents[i:i+embed_batch_size]
        batch_embeddings = embed_func(batch_docs)
        all_embeddings.extend(batch_embeddings)
    
    # Batch insertion to ChromaDB
    print("📥 Inserting vectors into database...")
    insert_batch_size = 200
    
    for i in tqdm(range(0, len(all_ids), insert_batch_size), desc="Indexing"):
        batch_ids = all_ids[i:i+insert_batch_size]
        batch_docs = all_documents[i:i+insert_batch_size]
        batch_embeddings = all_embeddings[i:i+insert_batch_size]
        batch_metas = all_metadatas[i:i+insert_batch_size]
        
        chroma_collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas
        )
    
    # Create LlamaIndex wrapper
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
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