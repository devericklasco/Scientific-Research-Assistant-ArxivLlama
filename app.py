import warnings
from pathlib import Path
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from src.arxiv_downloader import search_and_download_papers
from src.citation_generator import (
    build_citation_bundle,
    generate_apa_citation,
    generate_bibtex_citation,
)
from src.create_index import create_vector_index
from src.pdf_processor import process_papers
from src.query_engine import create_query_engine_from_index, get_paper_recommendations

warnings.filterwarnings("ignore", category=UserWarning)

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
if "paper_metadata" not in st.session_state:
    st.session_state.paper_metadata = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "claims_records" not in st.session_state:
    st.session_state.claims_records = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4().hex[:12]
if "workspace_id" not in st.session_state:
    st.session_state.workspace_id = f"ws_boot_{uuid4().hex[:8]}"
if "embedding_backend" not in st.session_state:
    st.session_state.embedding_backend = "openai"
if "last_cleanup_check_ts" not in st.session_state:
    st.session_state.last_cleanup_check_ts = 0.0


def _safe_chmod(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except OSError:
        pass


def _sessions_base() -> Path:
    base = Path("/tmp/arxivllama_sessions") if "STREAMLIT_SERVER" in os.environ else Path("./data/sessions")
    base.mkdir(parents=True, exist_ok=True)
    _safe_chmod(base)
    return base


def _session_root() -> Path:
    base = _sessions_base()
    root = base / st.session_state.session_id
    root.mkdir(parents=True, exist_ok=True)
    _safe_chmod(root)
    return root


def _workspace_root(workspace_id: str) -> Path:
    return _session_root() / "workspaces" / workspace_id


def _workspace_state_path(workspace_root: Path) -> Path:
    return workspace_root / WORKSPACE_STATE_FILENAME


def _touch_workspace_activity(workspace_root: Path, backend: str) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    state = {
        "last_active_epoch": now,
        "last_active_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now)),
        "backend": backend,
    }
    try:
        with open(_workspace_state_path(workspace_root), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


def _read_workspace_last_active(workspace_root: Path) -> float:
    state_path = _workspace_state_path(workspace_root)
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            ts = float(state.get("last_active_epoch", 0))
            if ts > 0:
                return ts
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    try:
        return workspace_root.stat().st_mtime
    except OSError:
        return 0.0


def cleanup_stale_workspaces(active_workspace: Path | None = None) -> int:
    now = time.time()
    base = _sessions_base()
    removed = 0
    active_workspace_str = str(active_workspace.resolve()) if active_workspace is not None else ""
    for workspaces_dir in base.glob("*/workspaces"):
        if not workspaces_dir.is_dir():
            continue
        for workspace_root in workspaces_dir.iterdir():
            if not workspace_root.is_dir():
                continue
            if active_workspace_str and str(workspace_root.resolve()) == active_workspace_str:
                continue
            last_active = _read_workspace_last_active(workspace_root)
            if last_active <= 0:
                continue
            if (now - last_active) > WORKSPACE_TTL_SECONDS:
                shutil.rmtree(workspace_root, ignore_errors=True)
                removed += 1
        # remove empty workspaces dir
        try:
            if workspaces_dir.exists() and not any(workspaces_dir.iterdir()):
                workspaces_dir.rmdir()
        except OSError:
            pass
        # remove empty session dir
        session_root = workspaces_dir.parent
        try:
            if session_root.exists() and not any(session_root.iterdir()):
                session_root.rmdir()
        except OSError:
            pass
    return removed


def activate_workspace(workspace_id: str, embedding_backend: str | None = None, reset_state: bool = False):
    backend = (embedding_backend or st.session_state.embedding_backend or "openai").strip().lower()
    if backend not in {"openai", "local"}:
        backend = "openai"

    workspace_root = _workspace_root(workspace_id)
    papers_path = workspace_root / "papers"
    chunks_path = workspace_root / "chunks"
    chroma_path = workspace_root / "indices" / "chroma_db"

    for path in [papers_path, chunks_path, chroma_path]:
        path.mkdir(parents=True, exist_ok=True)
        _safe_chmod(path)

    collection_name = f"arxiv_papers_{st.session_state.session_id}_{workspace_id}".replace("-", "_")
    os.environ["DATA_PATH"] = str(papers_path)
    os.environ["CHUNK_PATH"] = str(chunks_path)
    os.environ["INDEX_PATH"] = str(chroma_path)
    os.environ["CHROMA_COLLECTION_NAME"] = collection_name
    os.environ["EMBEDDING_BACKEND"] = backend

    st.session_state.workspace_id = workspace_id
    st.session_state.workspace_root = str(workspace_root)
    st.session_state.embedding_backend = backend
    _touch_workspace_activity(workspace_root, backend)

    if reset_state:
        st.session_state.paper_metadata = {}
        st.session_state.index_ready = False
        st.session_state.engine = None
        st.session_state.messages = []
        st.session_state.claims_records = []

    return papers_path, chunks_path, chroma_path


def start_new_workspace(embedding_backend: str | None = None):
    workspace_id = f"ws_{int(time.time())}_{uuid4().hex[:6]}"
    return activate_workspace(
        workspace_id=workspace_id,
        embedding_backend=embedding_backend,
        reset_state=True,
    )


def render_copy_button(text: str, copy_id: str, label: str = "Copy") -> None:
    element_id = re.sub(r"[^a-zA-Z0-9_-]", "_", copy_id)
    label_safe = html.escape(label)
    text_json = json.dumps(text)
    component_html = f"""
    <div style="margin: 0.1rem 0 0.5rem 0;">
      <button id="{element_id}" style="border: 1px solid #ccc; border-radius: 8px; padding: 0.35rem 0.7rem; background: white; cursor: pointer;">
        {label_safe}
      </button>
    </div>
    <script>
      const button = document.getElementById("{element_id}");
      const textToCopy = {text_json};
      async function copyText() {{
        try {{
          if (navigator.clipboard && window.isSecureContext) {{
            await navigator.clipboard.writeText(textToCopy);
          }} else {{
            const textarea = document.createElement("textarea");
            textarea.value = textToCopy;
            textarea.style.position = "fixed";
            textarea.style.left = "-9999px";
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();
            document.execCommand("copy");
            document.body.removeChild(textarea);
          }}
          button.textContent = "Copied";
        }} catch (err) {{
          button.textContent = "Copy failed";
        }}
      }}
      button.addEventListener("click", copyText);
    </script>
    """
    components.html(component_html, height=45)


def load_metadata_from_disk(reset: bool = True) -> None:
    if reset:
        st.session_state.paper_metadata = {}
    papers_path = Path(os.getenv("DATA_PATH", "./data/papers"))
    for json_file in papers_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.session_state.paper_metadata[meta["arxiv_id"]] = meta
        except (OSError, json.JSONDecodeError, KeyError):
            continue


activate_workspace(
    st.session_state.workspace_id,
    embedding_backend=st.session_state.embedding_backend,
    reset_state=False,
)

if (time.time() - st.session_state.last_cleanup_check_ts) >= WORKSPACE_CLEANUP_COOLDOWN_SECONDS:
    cleanup_stale_workspaces(active_workspace=Path(st.session_state.workspace_root))
    st.session_state.last_cleanup_check_ts = time.time()


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
    embedding_backend = st.radio(
        "Embedding Backend",
        options=["openai", "local"],
        index=0 if st.session_state.embedding_backend == "openai" else 1,
        help="Use OpenAI for production-quality retrieval, or local ONNX embeddings for local/dev usage.",
    )
    if embedding_backend != st.session_state.embedding_backend:
        st.session_state.embedding_backend = embedding_backend
        os.environ["EMBEDDING_BACKEND"] = embedding_backend
        st.info("Embedding backend changed. Start a new workspace and re-index to avoid dimension mismatch.")

    api_key = st.text_input("OpenAI API Key", type="password", value="", help="Enter your OpenAI API key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    st.divider()
    st.header("📥 Paper Management")
    with st.expander("Download Papers from ArXiv", expanded=True):
        topic = st.text_input("Research Topic", placeholder="e.g., large language models")
        max_results = st.slider("Max Papers", 1, 50, 12)
        max_age_years = st.slider("Only include papers from last N years", 1, 15, 3)
        sort_by_recent = st.checkbox("Prioritize newest papers", value=True)

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
    st.caption("Answers are grounded with source metadata and page-level evidence when available.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View Sources & Citations"):
                    for i, source in enumerate(message["sources"], start=1):
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
    st.info("Please configure and create an index using the sidebar controls.")


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

    if st.session_state.claims_records:
        st.dataframe(st.session_state.claims_records, use_container_width=True)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(st.session_state.claims_records[0].keys()))
        writer.writeheader()
        writer.writerows(st.session_state.claims_records)
        st.download_button(
            label="Download Evidence Table (CSV)",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="claims_evidence_table.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download Evidence Table (JSON)",
            data=json.dumps(st.session_state.claims_records, indent=2).encode("utf-8"),
            file_name="claims_evidence_table.json",
            mime="application/json",
        )


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
    st.info("Download and process papers to generate citations.")
