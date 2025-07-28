import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import chromadb
from dotenv import load_dotenv
import os
import tiktoken
import time
import json
from src.citation_generator import generate_apa_citation
from typing import Tuple, Optional, Union
from llama_index.core.query_engine import BaseQueryEngine

# Load environment variables
load_dotenv()

# Configuration
EMBED_MODEL = "text-embedding-3-small"

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def create_query_engine_from_index(index: VectorStoreIndex) -> BaseQueryEngine:
    """Create query engine directly from a VectorStoreIndex"""
    # Create query engine with GPT-4o
    llm = OpenAI(model="gpt-4o", temperature=0.1)
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode="compact"
    )

def initialize_engine() -> Tuple[Optional[BaseQueryEngine], Optional[VectorStoreIndex], Optional[chromadb.Collection]]:
    """Initialize query engine from persisted ChromaDB collection"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        return None, None, None

    # Determine storage location based on environment
    if "STREAMLIT_SERVER" in os.environ:
        persist_dir = "/tmp/chroma_db"
    else:
        persist_dir = os.getenv("INDEX_PATH", "./data/indices/chroma_db")
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    try:
        # Try to get existing collection
        chroma_collection = chroma_client.get_collection("arxiv_papers")
    except chromadb.errors.NotFoundError:
        print(f"❌ Collection 'arxiv_papers' not found at {persist_dir}")
        return None, None, None
    
    # Create LlamaIndex components
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    # Create index
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # Create query engine
    query_engine = create_query_engine_from_index(index)
    
    return query_engine, index, chroma_collection

def get_paper_recommendations(query_engine: BaseQueryEngine, topic: str, num_papers: int = 3) -> str:
    """Get paper recommendations based on a topic"""
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

def get_paper_metadata(chroma_collection: chromadb.Collection, paper_id: str) -> dict:
    """Get metadata for a specific paper"""
    result = chroma_collection.get(ids=[paper_id], include=["metadatas"])
    if result and result["metadatas"]:
        return result["metadatas"][0]
    return {}

if __name__ == "__main__":
    print("Initializing research assistant...")
    engine, index, chroma_collection = initialize_engine()
    
    if not engine:
        print("❌ Failed to initialize query engine. Please create an index first.")
        exit(1)
    
    print("✅ System ready. Type your questions about the research papers.")
    print("   Type 'exit' to quit, '!recommend' for paper recommendations,")
    print("   or '!cite <paper_id>' to generate a citation.\n")
    
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
            
        if query.startswith("!cite"):
            try:
                paper_id = query.split()[1]
                metadata = get_paper_metadata(chroma_collection, paper_id)
                if metadata:
                    citation = generate_apa_citation(metadata)
                    print(f"\n📝 APA Citation for {metadata.get('title', 'paper')}:")
                    print(citation)
                else:
                    print(f"❌ Paper ID {paper_id} not found")
            except IndexError:
                print("❌ Please specify a paper ID: !cite <paper_id>")
            continue
            
        # Track query cost
        start_time = time.time()
        response = engine.query(query)
        elapsed = time.time() - start_time
        
        context_text = " ".join([n.text for n in response.source_nodes]) if response.source_nodes else ""
        input_tokens = count_tokens(query + context_text)
        output_tokens = count_tokens(response.response)
        
        cost = (input_tokens/1000000*5) + (output_tokens/1000000*15)
        total_cost += cost
        
        print(f"\n💡 Answer ({elapsed:.1f}s, ${cost:.6f}):")
        print(response.response)
        
        if response.source_nodes:
            print("\n🔍 Sources:")
            for i, source in enumerate(response.source_nodes, 1):
                metadata = source.metadata or {}
                source_id = metadata.get("paper_id", "unknown")
                title = metadata.get("title", "Untitled Paper")
                score = source.score or 0.0
                
                print(f"{i}. [{source_id}] {title}")
                print(f"   Relevance: {score:.3f}")
                print(f"   Excerpt: {source.text[:150]}...")
        else:
            print("\n🔍 No sources found for this response")
    
    print(f"\nℹ️ Total session cost: ${total_cost:.6f}")