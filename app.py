import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import streamlit as st
from src.query_engine import initialize_engine, get_paper_recommendations
from src.create_index import create_vector_index
from src.pdf_processor import process_papers
from src.arxiv_downloader import search_and_download_papers
from src.citation_generator import generate_apa_citation
from pathlib import Path
import os
import time
import json
import shutil

# Configuration
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./data/indices"))
CHUNK_PATH = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
PAPERS_PATH = Path(os.getenv("DATA_PATH", "./data/papers"))

# Configure Streamlit
st.set_page_config(page_title="ArxivLlama", page_icon="🦙", layout="wide")
st.title("🦙 ArxivLlama - RAG Powered Scientific Research Assistant")

def cleanup_faiss_index():
    """Clean up previous index files"""
    files_to_remove = [
        INDEX_PATH / "faiss_index.bin",
        INDEX_PATH / "docstore.json",
        INDEX_PATH / "graph_store.json",
        INDEX_PATH / "index_store.json",
        INDEX_PATH / "vector_store.json"
    ]
    for file_path in files_to_remove:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            st.error(f"Couldn't remove {file_path}: {str(e)}")
            return False
    return True


# Initialize session state
if "engine" not in st.session_state:
    st.session_state.engine = None
    st.session_state.index_ready = False
    st.session_state.paper_metadata = {}

def load_paper_metadata():
    """Load paper metadata from chunk files"""
    metadata = {}
    for json_file in CHUNK_PATH.glob("*.json"):
        with open(json_file, 'r') as f:
            try:
                meta = json.load(f)
                metadata[meta["arxiv_id"]] = meta
            except (json.JSONDecodeError, KeyError):
                continue
    return metadata

# Sidebar for setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    st.header("📥 Paper Management")
    
    # Paper download section
    with st.expander("Download Papers from ArXiv"):
        topic = st.text_input("Research Topic", placeholder="e.g., large language models")
        max_results = st.slider("Max Papers", 1, 50, 10)
        
        if st.button("Download Papers", key="download_btn"):
            if not topic:
                st.warning("Please enter a research topic")
            elif not api_key:
                st.warning("Please enter your OpenAI API key")
            else:
                with st.spinner(f"Downloading {max_results} papers..."):
                    papers = search_and_download_papers(topic, max_results)
                    if papers:
                        st.success(f"Downloaded {len(papers)} papers!")
                        st.session_state.paper_metadata.update(
                            {meta['arxiv_id']: meta for _, meta in papers.items()}
                        )
                    else:
                        st.error("Failed to download papers")
    
    # Processing section
    if st.button("Process PDFs", key="process_btn"):
        if not api_key:
            st.warning("Please enter your OpenAI API key")
        else:
            with st.spinner("Extracting text and creating chunks..."):
                result = process_papers()
                if result:
                    st.success(f"Processed {len(result)} papers into chunks!")
                    st.session_state.paper_metadata = load_paper_metadata()
                else:
                    st.error("No papers processed - check PDF directory")

    # Index creation section
    if st.button("Create Vector Index", key="index_btn"):
        if not api_key:
            st.warning("Please enter your OpenAI API key")
        else:
            with st.spinner("Creating semantic index (this may take a few minutes)..."):
                # Clear previous index
                faiss_index_path = INDEX_PATH / "faiss_index"
                try:
                    # Handle both directory and file cases
                    if faiss_index_path.exists():
                        if faiss_index_path.is_dir():
                            shutil.rmtree(faiss_index_path)
                        else:
                            faiss_index_path.unlink()
                    
                    # Create parent directory if it doesn't exist
                    INDEX_PATH.mkdir(parents=True, exist_ok=True)
                    
                    index, vector_count = create_vector_index()
                    st.session_state.engine = initialize_engine()
                    st.session_state.index_ready = True
                    st.session_state.paper_metadata = load_paper_metadata()
                    st.success(f"Index created with {vector_count} vectors!")
                except Exception as e:
                    st.error(f"Index creation failed: {str(e)}")
                    st.error("Please check the index directory permissions")

# Main chat interface
if st.session_state.index_ready:
    st.subheader("💬 Research Assistant")
    st.caption("Ask questions about your research papers")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources & Citations"):
                    for i, source in enumerate(message["sources"], 1):
                        metadata = source["metadata"]
                        st.markdown(f"**Source {i}:**")
                        st.caption(f"**Title:** {metadata.get('title', 'Untitled')}")
                        st.caption(f"**Authors:** {metadata.get('authors', 'Unknown')}")
                        st.caption(f"**Published:** {metadata.get('published', 'Unknown')}")
                        
                        # Generate and display citation
                        citation = generate_apa_citation(metadata)
                        st.code(citation, language="text")
                        
                        st.caption(f"**Excerpt:** {source['text'][:200]}...")
                        st.divider()
    
    # Accept user input
    if prompt := st.chat_input("Your research question"):
        if not prompt.strip():
            st.warning("Please enter a question")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    start_time = time.time()
                    response = st.session_state.engine.query(prompt)
                    elapsed = time.time() - start_time
                    
                    # Display response
                    full_response += f"{response.response}\n\n"
                    message_placeholder.markdown(full_response)
                    
                    # Prepare sources for display
                    sources = []
                    if response.source_nodes:
                        for source in response.source_nodes:
                            paper_id = source.metadata.get("arxiv_id", "")
                            metadata = st.session_state.paper_metadata.get(
                                paper_id, 
                                source.metadata
                            )
                            sources.append({
                                "text": source.text,
                                "metadata": metadata,
                                "score": source.score or 0.0
                            })
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
else:
    st.info("Please configure and create an index using the sidebar controls")

# Paper recommendation section
st.divider()
st.subheader("📚 Paper Recommendations")

if st.session_state.index_ready:
    rec_topic = st.text_input("Enter research topic for recommendations", 
                            placeholder="e.g., transformer architectures")
    num_papers = st.slider("Number of recommendations", 1, 10, 3)
    
    if st.button("Get Recommendations"):
        if not rec_topic.strip():
            st.warning("Please enter a research topic")
        else:
            with st.spinner("Finding relevant papers..."):
                try:
                    recommendations = get_paper_recommendations(
                        st.session_state.engine, 
                        rec_topic,
                        num_papers
                    )
                    st.markdown(recommendations)
                except Exception as e:
                    st.error(f"Failed to get recommendations: {str(e)}")

# Citation generator section
st.divider()
st.subheader("📝 Citation Generator")

if st.session_state.paper_metadata:
    paper_options = {meta['arxiv_id']: meta['title'] 
                    for meta in st.session_state.paper_metadata.values()}
    selected_id = st.selectbox("Select paper to cite", options=list(paper_options.keys()),
                            format_func=lambda id: f"{paper_options[id]} ({id})")
    
    if selected_id:
        metadata = st.session_state.paper_metadata.get(selected_id, {})
        citation = generate_apa_citation(metadata)
        
        st.code(citation, language="text")
        
        if st.button("Copy to Clipboard"):
            st.session_state.copied = True
            st.code(citation, language="text")
            st.success("Citation copied to clipboard!")
else:
    st.info("Download and process papers to generate citations")