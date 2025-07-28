import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import tiktoken
import time
# import faiss
import json
from src.citation_generator import generate_apa_citation
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
EMBED_MODEL = "text-embedding-3-small"
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./data/indices"))

def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def initialize_engine():
    """Initialize the FAISS-based query engine"""
    # Load FAISS index from disk
    # Load FAISS index directly
    faiss_index = faiss.read_index(str(INDEX_PATH / "faiss_index.bin"))
    # vector_store = FaissVectorStore.from_persist_dir(str(INDEX_PATH / "faiss_index"))
    # Create vector store
    vector_store = FaissVectorStore.from_persist_dir(
        persist_dir=str(INDEX_PATH / "faiss_vector_store"))
    vector_store.faiss_index = faiss_index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    
    # Configure embedding model
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    # Create index
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context
    )
    
    # Create query engine with GPT-4o
    llm = OpenAI(model="gpt-4o", temperature=0.1)
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode="compact"
    )
    return query_engine

def get_paper_recommendations(query_engine, topic: str, num_papers: int = 3) -> str:
    """Get paper recommendations based on research topic"""
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
    """Load paper metadata from chunk files"""
    metadata = {}
    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    
    for json_file in chunk_path.glob("*.json"):
        with open(json_file, 'r') as f:
            try:
                paper_data = json.load(f)
                metadata[paper_data["arxiv_id"]] = paper_data
            except (json.JSONDecodeError, KeyError):
                continue
    return metadata

# CLI functionality removed since app.py is the main interface
# FAISS doesn't support direct metadata retrieval by ID like Chroma did
# Metadata is now managed through load_paper_metadata() and session state in app.py

if __name__ == "__main__":
    print("Initializing research assistant...")
    engine = initialize_engine()
    paper_metadata = load_paper_metadata()
    print("✅ System ready. Type your questions about the research papers.")
    print("   Type 'exit' to quit or '!recommend' for paper recommendations\n")
    
    total_cost = 0.0
    
    while True:
        query = input("\n📝 Your research question: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        if query.startswith("!recommend"):
            topic = query.replace("!recommend", "").strip() or "machine learning"
            print(f"\n🔍 Getting recommendations for: {topic}")
            recommendations = get_paper_recommendations(engine, topic)
            print(f"\n📚 Recommended Papers:\n{recommendations}")
            continue
     
        # Track query cost
        start_time = time.time()
        response = engine.query(query)
        elapsed = time.time() - start_time
        
        context_text = " ".join([n.text for n in response.source_nodes]) if response.source_nodes else ""
        input_tokens = count_tokens(query + context_text)
        output_tokens = count_tokens(response.response)
        
        # Updated pricing for GPT-4o (May 2024 pricing)
        input_cost_per_token = 5 / 1_000_000  # $5 per 1M tokens
        output_cost_per_token = 15 / 1_000_000  # $15 per 1M tokens
        cost = (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)
        total_cost += cost
        
        print(f"\n💡 Answer ({elapsed:.1f}s, ${cost:.6f}):")
        print(response.response)
        
        if response.source_nodes:
            print("\n🔍 Sources:")
            for i, source in enumerate(response.source_nodes, 1):
                metadata = source.metadata or {}
                source_id = metadata.get("arxiv_id", "unknown")
                
                # Get full metadata from loaded paper data
                paper_meta = paper_metadata.get(source_id, {})
                title = paper_meta.get("title", metadata.get("title", "Untitled Paper"))
                authors = paper_meta.get("authors", metadata.get("authors", "Unknown authors"))
                
                print(f"{i}. [{source_id}] {title}")
                print(f"   Authors: {authors}")
                print(f"   Relevance: {source.score or 0.0:.3f}")
                print(f"   Excerpt: {source.text[:150]}...")
                
                # Generate citation
                citation = generate_apa_citation(paper_meta or metadata)
                print(f"   Citation: {citation[:60]}...")
        else:
            print("\n🔍 No sources found for this response")
    
    print(f"\nℹ️ Total session cost: ${total_cost:.6f}")