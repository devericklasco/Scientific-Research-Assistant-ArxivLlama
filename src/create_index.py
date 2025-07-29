import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Document
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
    
    # Determine storage location based on environment
    if "STREAMLIT_SERVER" in os.environ:
        persist_dir = "/tmp/chroma_db"
    else:
        persist_dir = os.getenv("INDEX_PATH", "./data/indices/chroma_db")
    
    # Ensure clean directory
    persist_path = Path(persist_dir)
    if persist_path.exists():
        shutil.rmtree(persist_path, ignore_errors=True)
    persist_path.mkdir(parents=True, exist_ok=True)
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
    
    # Create or get collection WITHOUT embedding function
    chroma_collection = chroma_client.get_or_create_collection(
        name="arxiv_papers",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Get chunk path from environment
    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    
    # Load and process chunk files
    chunk_files = list(chunk_path.glob("*.json"))
    
    if not chunk_files:
        print(f"❌ No JSON files found in {chunk_path}. Run PDF processor first.")
        return None, 0
    
    # Prepare data for insertion
    print(f"📚 Processing {len(chunk_files)} papers...")
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    # Create embedding function
    embed_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBED_MODEL
    )
    
    for chunk_file in tqdm(chunk_files, desc="Processing papers"):
        with open(chunk_file, 'r') as f:
            try:
                paper_data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON file: {chunk_file}")
                continue
                
        # Safely get arxiv_id with fallback
        arxiv_id = paper_data.get("arxiv_id", chunk_file.stem)
        
        # Create entries for each chunk
        for i, chunk_text in enumerate(paper_data["chunks"]):
            # Create unique ID for each chunk
            chunk_id = f"{arxiv_id}_{i}"
            doc_id = hashlib.md5(chunk_id.encode()).hexdigest()

            authors_list = paper_data.get("authors", [])
            authors_str = ", ".join(authors_list)
            
            # Generate embedding
            embedding = embed_func([chunk_text])[0]
            
            ids.append(doc_id)
            documents.append(chunk_text)
            embeddings.append(embedding)
            metadatas.append({
                "paper_id": arxiv_id,
                "title": paper_data.get("title", "Untitled Paper"),
                "chunk_id": i,
                "file_path": paper_data.get("file_path", ""),
                "arxiv_id": arxiv_id,
                "authors": authors_str,
                "published": paper_data.get("published", "")
            })
    
    print(f"🧩 Total chunks to index: {len(ids)}")
    
    # Batch insertion
    batch_size = 100
    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing chunks"):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        chroma_collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas
        )
    
    # Create LlamaIndex wrapper
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Explicitly create embedding model
    embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    # Create index with explicit embed_model
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # Calculate cost metrics
    vector_count = chroma_collection.count()
    approx_tokens_per_chunk = 300
    total_tokens = vector_count * approx_tokens_per_chunk
    embedding_cost = (total_tokens / 1000) * 0.00002
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Index creation complete!")
    print(f"   Total vectors: {vector_count}")
    print(f"   Time taken: {elapsed:.1f} seconds")
    print(f"   Estimated tokens processed: {total_tokens}")
    print(f"   Estimated embedding cost: ${embedding_cost:.6f}")
    
    return index, vector_count

if __name__ == "__main__":
    # Clear previous index data
    chroma_db_path = Path("./data/indices/chroma_db")
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
        
        print(f"\n✅ Successfully created index with {vector_count} vectors")
        print(f"🔑 Vector store persisted at: ./data/indices/chroma_db")
        print(f"💡 Estimated embedding cost: ${vector_count * 0.0000001:.6f}")