__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Disable ChromaDB telemetry
import chromadb
# chromadb.utils.embedding_functions.DEFAULT_EMBEDDING_FUNC = None
# chromadb.utils.embedding_functions.DefaultEmbeddingFunction = None
# chromadb.Client = None
import streamlit as st
from src.query_engine import initialize_engine, get_paper_recommendations
from src.create_index import create_vector_index
from src.pdf_processor import process_papers
from src.arxiv_downloader import search_and_download_papers
from src.citation_generator import generate_apa_citation
from src.query_engine import create_query_engine_from_index
from pathlib import Path
import os
import time
import json
import shutil
import atexit
import stat

# Configure Streamlit
st.set_page_config(page_title="ArxivLlama", page_icon="🦙", layout="wide")
st.title("🦙 ArxivLlama - RAG Powered Scientific Research Assistant")

# Initialize session state
if "engine" not in st.session_state:
    st.session_state.engine = None
    st.session_state.chroma_collection = None
    st.session_state.index_ready = False
    st.session_state.paper_metadata = {}
    st.session_state.chroma_path = ""

# Cleanup function for Streamlit Cloud
@atexit.register
def cleanup():
    if "STREAMLIT_SERVER" in os.environ:
        # Clean temporary ChromaDB directory
        chroma_path = Path("/tmp/chroma_db")
        if chroma_path.exists():
            shutil.rmtree(chroma_path, ignore_errors=True)
        # Clean temporary papers
        papers_path = Path("/tmp/papers")
        if papers_path.exists():
            shutil.rmtree(papers_path, ignore_errors=True)
        # Clean temporary chunks
        chunks_path = Path("/tmp/chunks")
        if chunks_path.exists():
            shutil.rmtree(chunks_path, ignore_errors=True)

# Helper function to clear previous downloads
def clear_previous_downloads():
    # Determine storage location based on environment
    if "STREAMLIT_SERVER" in os.environ:
        papers_path = Path("/tmp/papers")
        chunks_path = Path("/tmp/chunks")
        chroma_path = Path("/tmp/chroma_db")
    else:
        papers_path = Path(os.getenv("DATA_PATH", "./data/papers"))
        chunks_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
        chroma_path = Path(os.getenv("INDEX_PATH", "./data/indices/chroma_db"))
    
    # Clear directories with error handling
    for path in [papers_path, chunks_path, chroma_path]:
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except Exception as e:
            st.error(f"Error clearing {path}: {str(e)}")
    
    # Clear session metadata
    st.session_state.paper_metadata = {}
    st.session_state.index_ready = False
    st.session_state.engine = None
    
    # Set environment variables for other modules
    os.environ["DATA_PATH"] = str(papers_path)
    os.environ["CHUNK_PATH"] = str(chunks_path)
    os.environ["INDEX_PATH"] = str(chroma_path)
    
    return papers_path, chunks_path, chroma_path

# Sidebar for setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value="", help="Enter your OpenAI API key")
    
    # Only set environment variable when needed
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
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
                with st.spinner("Preparing directories..."):
                    # Clear previous downloads
                    papers_path, chunks_path, chroma_path = clear_previous_downloads()
                    
                with st.spinner(f"Downloading {max_results} papers..."):
                    papers = search_and_download_papers(topic, max_results)
                    st.success(f"Downloaded {len(papers)} papers!")
                    st.session_state.paper_metadata.update(
                        {meta['arxiv_id']: meta for _, meta in papers.items()}
                    )
    
    # Processing section
    if st.button("Process PDFs", key="process_btn"):
        if not api_key:
            st.warning("Please enter your OpenAI API key")
        else:
            with st.spinner("Extracting text and creating chunks..."):
                result = process_papers()
                st.success(f"Processed {len(result)} papers into chunks!")
                
                # Load metadata into session state
                papers_path = Path(os.getenv("DATA_PATH", "./data/papers"))
                for json_file in papers_path.glob("*.json"):
                    with open(json_file, 'r') as f:
                        meta = json.load(f)
                        st.session_state.paper_metadata[meta["arxiv_id"]] = meta

    if st.button("Create Vector Index", key="index_btn"):
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Please enter your OpenAI API key")
        else:
            with st.spinner("Creating semantic index…"):
                index, vector_count = create_vector_index()

            if index:
                st.success(f"Indexed {vector_count} vectors!")
                # build & stash your query engine
                qe = create_query_engine_from_index(index)
                st.session_state.engine      = qe
                st.session_state.index_ready = True
            else:
                st.error("Failed to create index!")
    # if st.button("Create Vector Index", key="index_btn"):
    #     if not api_key:
    #         st.warning("Please enter your OpenAI API key")
    #     else:
    #         with st.spinner("Creating semantic index (this may take a few minutes)…"):
    #             # THIS single call both wipes+prepares the folder AND builds the index
    #             index, vector_count = create_vector_index()

    #         if index:
    #             st.success(f"Indexed {vector_count} vectors!")
    #             # build & store your query engine
    #             query_engine = create_query_engine_from_index(index)
    #             st.session_state.engine = query_engine
    #             st.session_state.index_ready = True

    #             # reload your chunk metadata
    #             chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    #             for json_file in chunk_path.glob("*.json"):
    #                 with open(json_file, 'r') as f:
    #                     meta = json.load(f)
    #                     st.session_state.paper_metadata[meta["arxiv_id"]] = meta
    #         else:
    #             st.error("Failed to create index!")
    # Index creation section
    # if st.button("Create Vector Index", key="index_btn"):
    #     index, count = create_vector_index()
    #     st.success(f"Indexed {count} vectors.")
    #     if not api_key:
    #         st.warning("Please enter your OpenAI API key")
    #     else:
    #         with st.spinner("Creating semantic index (this may take a few minutes)..."):
    #             # Get the chroma path from environment
    #             chroma_path = Path(os.getenv("INDEX_PATH", "./data/indices/chroma_db"))
                
    #             # Clear any existing index
    #             if chroma_path.exists():
    #                 shutil.rmtree(chroma_path, ignore_errors=True)
                
    #             # Create directory with full permissions
    #             chroma_path.mkdir(parents=True, exist_ok=True)
    #             os.chmod(chroma_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                
    #             # Create index
    #             index, vector_count = create_vector_index()
                
    #             # Create query engine directly from the new index
    #             if index:
    #                 from src.query_engine import create_query_engine_from_index
    #                 query_engine = create_query_engine_from_index(index)
                    
    #                 # Store in session state
    #                 st.session_state.engine = query_engine
    #                 st.session_state.index_ready = True
    #                 st.session_state.chroma_path = str(chroma_path)
    #                 st.success(f"Index created with {vector_count} vectors!")
                    
    #                 # Load chunk metadata
    #                 chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    #                 for json_file in chunk_path.glob("*.json"):
    #                     with open(json_file, 'r') as f:
    #                         meta = json.load(f)
    #                         st.session_state.paper_metadata[meta["arxiv_id"]] = meta
    #             else:
    #                 st.error("Failed to create index!")
    

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
                        st.caption(f"**Authors:** {', '.join(metadata.get('authors', []))}")
                        st.caption(f"**Published:** {metadata.get('published', 'Unknown')}")
                        
                        # Generate and display citation
                        citation = generate_apa_citation(metadata)
                        st.code(citation, language="text")
                        
                        st.caption(f"**Excerpt:** {source['text'][:200]}...")
                        st.divider()
    
    # Accept user input
    if prompt := st.chat_input("Your research question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
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
        if not rec_topic:
            st.warning("Please enter a research topic")
        else:
            with st.spinner("Finding relevant papers..."):
                recommendations = get_paper_recommendations(
                    st.session_state.engine, 
                    rec_topic,
                    num_papers
                )
                st.markdown(recommendations)

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
        
        # Use Streamlit's clipboard component
        st.caption("Click the button below to copy the citation to clipboard")
        if st.button("Copy to Clipboard", key="copy_citation"):
            st.session_state.copied = True
            st.code(citation, language="text")
            
            # Add JavaScript clipboard functionality
            st.markdown(
                f"""
                <script>
                navigator.clipboard.writeText(`{citation}`);
                </script>
                """,
                unsafe_allow_html=True
            )
            st.success("Citation copied to clipboard!")
else:
    st.info("Download and process papers to generate citations")