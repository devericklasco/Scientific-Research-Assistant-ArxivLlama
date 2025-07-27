from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import chromadb
from dotenv import load_dotenv
from pathlib import Path
import os
import tiktoken
import time  # Added missing import

# Load environment variables
load_dotenv()

def count_tokens(text: str) -> int:
    """Count tokens using OpenAI's tokenizer"""
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def initialize_engine():
    # Configuration
    INDEX_PATH = Path(os.getenv("INDEX_PATH", "./data/indices"))
    EMBED_MODEL = "text-embedding-3-small"
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(INDEX_PATH / "chroma_db"))
    chroma_collection = chroma_client.get_collection("arxiv_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Configure embedding model
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    # Load index
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
    return query_engine, llm

if __name__ == "__main__":
    print("Initializing research assistant...")
    engine, llm = initialize_engine()
    print("✅ System ready. Type your questions about the research papers.")
    print("   Type 'exit' to quit.\n")
    
    total_cost = 0.0
    
    while True:
        query = input("\n📝 Your research question: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        # Track query cost
        start_time = time.time()
        response = engine.query(query)
        elapsed = time.time() - start_time
        
        # Calculate token usage and cost
        input_tokens = count_tokens(query + " ".join([n.text for n in response.source_nodes]))
        output_tokens = count_tokens(response.response)
        
        # GPT-4o pricing: $5/1M input tokens, $15/1M output tokens
        cost = (input_tokens/1000000*5) + (output_tokens/1000000*15)
        total_cost += cost
        
        # Show response
        print(f"\n💡 Answer ({elapsed:.1f}s, ${cost:.6f}):")
        print(response.response)
        
        # Show sources
        print("\n🔍 Sources:")
        for i, source in enumerate(response.source_nodes, 1):
            source_id = source.metadata.get("paper_id", "unknown")
            title = source.metadata.get("title", "Untitled Paper")
            print(f"{i}. [{source_id}] {title}")
            print(f"   Relevance: {source.score:.3f}")
            print(f"   Excerpt: {source.text[:150]}...")
    
    print(f"\nℹ️ Total session cost: ${total_cost:.6f}")