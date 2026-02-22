from __future__ import annotations

import sys

# Prefer pysqlite3 when available (helps older environments), but fall back to
# stdlib sqlite3 for Python/platforms where pysqlite3-binary wheels are absent.
try:
    import pysqlite3  # type: ignore

    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

import csv
import html
import io
import json
import os
import re
import shutil
import stat
import time
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

st.set_page_config(page_title="ArxivLlama", page_icon="🦙", layout="wide")
st.title("🦙 ArxivLlama - RAG Powered Scientific Research Assistant")

WORKSPACE_TTL_SECONDS = int(os.getenv("WORKSPACE_TTL_SECONDS", 2 * 60 * 60))
WORKSPACE_CLEANUP_COOLDOWN_SECONDS = int(os.getenv("WORKSPACE_CLEANUP_COOLDOWN_SECONDS", 60))
WORKSPACE_STATE_FILENAME = ".workspace_state.json"

if "engine" not in st.session_state:
    st.session_state.engine = None
if "index_ready" not in st.session_state:
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
            else:
                with st.spinner("Preparing a new isolated workspace..."):
                    start_new_workspace(embedding_backend=embedding_backend)
                with st.spinner(f"Downloading up to {max_results} papers..."):
                    papers = search_and_download_papers(
                        query=topic,
                        max_results=max_results,
                        max_age_days=max_age_years * 365,
                        sort_by_recent=sort_by_recent,
                        dedupe=True,
                    )
                st.success(f"Downloaded {len(papers)} papers.")
                st.session_state.paper_metadata.update({m["arxiv_id"]: m for m in papers.values()})

    if st.button("Process PDFs", key="process_btn"):
        with st.spinner("Extracting text and creating token-aware chunks..."):
            result = process_papers()
        st.success(f"Processed {len(result)} papers into chunk records.")
        load_metadata_from_disk(reset=True)

    force_rebuild = st.checkbox("Force full re-index (skip incremental mode)", value=False)
    if st.button("Create Vector Index", key="index_btn"):
        if embedding_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
            st.warning("OpenAI backend selected: please enter your OpenAI API key.")
        else:
            with st.spinner("Creating semantic index..."):
                index, vector_count = create_vector_index(force_rebuild=force_rebuild)
            if index:
                st.success(f"Index ready with {vector_count} vectors.")
                st.session_state.engine = create_query_engine_from_index(index)
                st.session_state.index_ready = True
            else:
                st.error("Failed to create index.")


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
                        st.markdown(f"**Source {i}:** {metadata.get('title', 'Untitled')}")
                        st.caption(
                            f"Authors: {', '.join(metadata.get('authors', [])) if isinstance(metadata.get('authors'), list) else metadata.get('authors', 'Unknown')}"
                        )
                        st.caption(
                            f"Published: {metadata.get('published', 'Unknown')} | "
                            f"Page: {metadata.get('page_start', 'N/A')} | Section: {metadata.get('section', 'content')} | "
                            f"Score: {source.get('score', 0.0):.3f}"
                        )
                        st.code(generate_apa_citation(metadata), language="text")
                        st.caption(f"Excerpt: {source['text'][:260]}...")
                        st.divider()

    if prompt := st.chat_input("Ask a question about your indexed papers"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            start_time = time.time()
            response = st.session_state.engine.query(prompt)
            elapsed = time.time() - start_time
            placeholder.markdown(f"{response.response}\n\n_Time: {elapsed:.2f}s_")
            if hasattr(response, "grounded") and not response.grounded:
                st.warning("Low-confidence answer: insufficient evidence in retrieved context.")

            sources = []
            if getattr(response, "source_nodes", None):
                for source in response.source_nodes:
                    paper_id = source.metadata.get("arxiv_id", "")
                    metadata = st.session_state.paper_metadata.get(paper_id, source.metadata)
                    sources.append(
                        {
                            "text": source.text,
                            "metadata": metadata,
                            "score": source.score or 0.0,
                        }
                    )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.response,
                    "sources": sources,
                }
            )
else:
    st.info("Please configure and create an index using the sidebar controls.")


st.divider()
st.subheader("📚 Paper Recommendations")
if st.session_state.index_ready:
    rec_topic = st.text_input("Recommendation topic", placeholder="e.g., retrieval-augmented generation")
    num_papers = st.slider("Number of recommendations", 1, 10, 4)
    if st.button("Get Recommendations"):
        if not rec_topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Finding relevant papers..."):
                st.markdown(get_paper_recommendations(st.session_state.engine, rec_topic, num_papers))


st.divider()
st.subheader("🧪 Multi-Paper Comparison")
if st.session_state.index_ready:
    compare_topic = st.text_input("Comparison topic", placeholder="e.g., compare retrieval strategies for hallucination reduction")
    compare_count = st.slider("Papers to compare", 2, 6, 3)
    if st.button("Compare Papers"):
        if not compare_topic:
            st.warning("Please enter a comparison topic.")
        else:
            with st.spinner("Building comparison table..."):
                comparison = st.session_state.engine.compare_papers(compare_topic, num_papers=compare_count)
            st.markdown(comparison)


st.divider()
st.subheader("🧾 Claims & Evidence Export")
if st.session_state.index_ready:
    claim_topic = st.text_input("Claim extraction topic", placeholder="e.g., key claims on efficient LLM inference")
    max_claims = st.slider("Max claims", 1, 12, 6)
    if st.button("Extract Claims"):
        if not claim_topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Extracting claims with evidence..."):
                st.session_state.claims_records = st.session_state.engine.extract_claims_with_evidence(
                    claim_topic, max_claims=max_claims
                )

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
    paper_ids = sorted(st.session_state.paper_metadata.keys())
    selected_id = st.selectbox(
        "Select paper to cite",
        options=paper_ids,
        format_func=lambda pid: f"{st.session_state.paper_metadata[pid].get('title', 'Untitled')} ({pid})",
    )
    metadata = st.session_state.paper_metadata[selected_id]
    apa_citation = generate_apa_citation(metadata)
    bibtex_citation = generate_bibtex_citation(metadata)
    st.caption("APA")
    st.code(apa_citation, language="text")
    render_copy_button(apa_citation, copy_id=f"copy_apa_{selected_id}", label="Copy APA")
    st.caption("BibTeX")
    st.code(bibtex_citation, language="bibtex")
    render_copy_button(bibtex_citation, copy_id=f"copy_bib_{selected_id}", label="Copy BibTeX")

    bundle_bytes = build_citation_bundle(list(st.session_state.paper_metadata.values()))
    st.download_button(
        label="Download Citation Bundle (APA + BibTeX + Metadata)",
        data=bundle_bytes,
        file_name="citation_bundle.zip",
        mime="application/zip",
    )
else:
    st.info("Download and process papers to generate citations.")
