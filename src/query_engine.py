from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import tiktoken
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))
MIN_GROUNDING_SCORE = float(os.getenv("MIN_GROUNDING_SCORE", 0.2))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", 0.62))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", 0.23))
RECENCY_WEIGHT = float(os.getenv("RECENCY_WEIGHT", 0.15))

def get_collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION_NAME", "arxiv_papers")

def get_embedding_backend() -> str:
    backend = os.getenv("EMBEDDING_BACKEND", "openai").strip().lower()
    return backend if backend in {"openai", "local"} else "openai"


@dataclass
class ScoredSourceNode:
    text: str
    metadata: Dict
    score: float
    semantic_score: float
    lexical_score: float
    recency_score: float


@dataclass
class GroundedResponse:
    response: str
    source_nodes: List[ScoredSourceNode]
    grounded: bool


def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def tokenize_for_search(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(t) >= 2]


def lexical_overlap_score(query: str, document: str) -> float:
    q_tokens = tokenize_for_search(query)
    d_tokens = tokenize_for_search(document)
    if not q_tokens or not d_tokens:
        return 0.0
    q_set = set(q_tokens)
    d_set = set(d_tokens)
    overlap = len(q_set & d_set)
    return overlap / max(1, len(q_set))


def compute_recency_score(published: str) -> float:
    if not published:
        return 0.0
    try:
        published_dt = datetime.strptime(published, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    age_days = (datetime.now(timezone.utc) - published_dt).days
    if age_days <= 0:
        return 1.0
    if age_days >= 3650:
        return 0.0
    return 1.0 - (age_days / 3650.0)


def combine_scores(semantic_score: float, lexical_score: float, recency_score: float) -> float:
    return (
        SEMANTIC_WEIGHT * semantic_score
        + LEXICAL_WEIGHT * lexical_score
        + RECENCY_WEIGHT * recency_score
    )


def should_return_insufficient_evidence(source_nodes: List[ScoredSourceNode]) -> bool:
    if not source_nodes:
        return True
    strongest = max(node.score for node in source_nodes)
    return strongest < MIN_GROUNDING_SCORE


class HybridResearchEngine:
    def __init__(
        self,
        chroma_collection: chromadb.Collection,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    ) -> None:
        self.collection = chroma_collection
        self.similarity_top_k = similarity_top_k
        self.embedding_backend = get_embedding_backend()
        self._llm = None
        self._query_embedder = None
        if os.getenv("OPENAI_API_KEY"):
            self._llm = OpenAI(model="gpt-4o", temperature=0.1)
        if self.embedding_backend == "openai" and os.getenv("OPENAI_API_KEY"):
            self._query_embedder = OpenAIEmbedding(model=EMBED_MODEL)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[ScoredSourceNode]:
        k = top_k or self.similarity_top_k
        try:
            if self.embedding_backend == "openai" and self._query_embedder is not None:
                query_embedding = self._query_embedder.get_text_embedding(query)
                raw = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(k * 4, 20),
                    include=["documents", "metadatas", "distances"],
                )
            else:
                raw = self.collection.query(
                    query_texts=[query],
                    n_results=max(k * 4, 20),
                    include=["documents", "metadatas", "distances"],
                )
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise RuntimeError(
                    "Embedding dimension mismatch between your index and query backend. "
                    "Set EMBEDDING_BACKEND consistently and recreate the vector index."
                ) from exc
            raise
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        candidates: List[ScoredSourceNode] = []
        for i, text in enumerate(documents):
            metadata = metadatas[i] or {}
            distance = float(distances[i]) if i < len(distances) else 1.0
            semantic_score = max(0.0, 1.0 - distance)
            lex_score = lexical_overlap_score(query, text)
            recency = compute_recency_score(str(metadata.get("published", "")))
            final_score = combine_scores(semantic_score, lex_score, recency)
            candidates.append(
                ScoredSourceNode(
                    text=text,
                    metadata=metadata,
                    score=final_score,
                    semantic_score=semantic_score,
                    lexical_score=lex_score,
                    recency_score=recency,
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:k]

    def _fallback_answer(self, question: str, sources: List[ScoredSourceNode]) -> str:
        snippets = []
        for idx, source in enumerate(sources[:3], start=1):
            title = source.metadata.get("title", "Untitled")
            snippets.append(f"[S{idx}] {title}: {source.text[:240]}...")
        return (
            "OpenAI API key is not configured, so this is an extractive summary only.\n\n"
            f"Question: {question}\n\n"
            + "\n".join(snippets)
        )

    def _build_grounded_prompt(self, question: str, sources: List[ScoredSourceNode]) -> str:
        context_lines = []
        for idx, source in enumerate(sources, start=1):
            md = source.metadata
            title = md.get("title", "Untitled")
            published = md.get("published", "Unknown")
            page_start = md.get("page_start", -1)
            section = md.get("section", "content")
            context_lines.append(
                f"[S{idx}] title={title} | published={published} | page={page_start} | section={section}\n{source.text}"
            )
        context = "\n\n".join(context_lines)
        return (
            "You are a careful research assistant. Answer ONLY from the provided sources.\n"
            "Rules:\n"
            "1) If evidence is insufficient, say exactly: 'Insufficient evidence in indexed papers.'\n"
            "2) Cite every key claim with source tags like [S1], [S2].\n"
            "3) Keep answer concise and technical.\n\n"
            f"Question:\n{question}\n\n"
            f"Sources:\n{context}\n"
        )

    def query(self, question: str) -> GroundedResponse:
        sources = self.retrieve(question, top_k=self.similarity_top_k)
        if should_return_insufficient_evidence(sources):
            return GroundedResponse(
                response=(
                    "Insufficient evidence in indexed papers. "
                    "Try rephrasing the question, expanding the paper set, or lowering recency filters."
                ),
                source_nodes=sources,
                grounded=False,
            )

        if self._llm is None:
            return GroundedResponse(
                response=self._fallback_answer(question, sources),
                source_nodes=sources,
                grounded=True,
            )

        prompt = self._build_grounded_prompt(question, sources)
        response = self._llm.complete(prompt).text
        return GroundedResponse(response=response.strip(), source_nodes=sources, grounded=True)

    def recommend_papers(self, topic: str, num_papers: int = 3) -> str:
        sources = self.retrieve(topic, top_k=max(num_papers * 4, 12))
        by_paper: Dict[str, ScoredSourceNode] = {}
        for source in sources:
            paper_id = source.metadata.get("arxiv_id", "unknown")
            existing = by_paper.get(paper_id)
            if existing is None or source.score > existing.score:
                by_paper[paper_id] = source

        ranked = sorted(by_paper.values(), key=lambda s: s.score, reverse=True)[:num_papers]
        bullets = []
        for source in ranked:
            md = source.metadata
            bullets.append(
                f"- **{md.get('title', 'Untitled')}** ({md.get('published', 'Unknown')}) "
                f"[{md.get('arxiv_id', 'N/A')}]\n"
                f"  - Why: score={source.score:.3f}, section={md.get('section', 'content')}\n"
                f"  - Evidence: {source.text[:220]}..."
            )
        return "\n".join(bullets) if bullets else "No recommendations available."

    def compare_papers(self, comparison_topic: str, num_papers: int = 3) -> str:
        sources = self.retrieve(comparison_topic, top_k=max(12, num_papers * 4))
        by_paper: Dict[str, ScoredSourceNode] = {}
        for source in sources:
            paper_id = source.metadata.get("arxiv_id", "unknown")
            current = by_paper.get(paper_id)
            if current is None or source.score > current.score:
                by_paper[paper_id] = source
        ranked = sorted(by_paper.values(), key=lambda s: s.score, reverse=True)[:num_papers]
        if not ranked:
            return "No papers found for comparison."

        rows = ["| Paper | Published | Signal | Evidence |", "|---|---|---|---|"]
        for node in ranked:
            md = node.metadata
            rows.append(
                f"| {md.get('title', 'Untitled')} | {md.get('published', 'Unknown')} | "
                f"{md.get('section', 'content')} (score {node.score:.2f}) | {node.text[:140].replace('|', ' ')}... |"
            )
        return "\n".join(rows)

    def extract_claims_with_evidence(self, topic: str, max_claims: int = 5) -> List[Dict]:
        sources = self.retrieve(topic, top_k=max(max_claims * 3, 12))
        claims: List[Dict] = []
        for source in sources[:max_claims]:
            md = source.metadata
            sentence = source.text.split(".")[0].strip()
            if not sentence:
                sentence = source.text[:120]
            claims.append(
                {
                    "claim": sentence,
                    "evidence": source.text[:260],
                    "arxiv_id": md.get("arxiv_id", ""),
                    "title": md.get("title", ""),
                    "published": md.get("published", ""),
                    "page_start": md.get("page_start", -1),
                    "page_end": md.get("page_end", -1),
                    "score": round(source.score, 4),
                }
            )
        return claims


def _extract_collection_from_index(index: VectorStoreIndex) -> Optional[chromadb.Collection]:
    vector_store = getattr(index, "_vector_store", None) or getattr(index, "vector_store", None)
    if vector_store is None:
        return None
    for attr in ("_collection", "collection", "chroma_collection"):
        collection = getattr(vector_store, attr, None)
        if collection is not None:
            return collection
    return None


def create_query_engine_from_index(index: VectorStoreIndex):
    chroma_collection = _extract_collection_from_index(index)
    if chroma_collection is None:
        if os.getenv("OPENAI_API_KEY"):
            llm = OpenAI(model="gpt-4o", temperature=0.1)
            return index.as_query_engine(llm=llm, similarity_top_k=3, response_mode="compact")
        return index.as_query_engine(similarity_top_k=3, response_mode="compact")
    return HybridResearchEngine(chroma_collection=chroma_collection)


def initialize_engine() -> Tuple[Optional[HybridResearchEngine], Optional[VectorStoreIndex], Optional[chromadb.Collection]]:
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        return None, None, None

    persist_dir = "/tmp/chroma_db" if "STREAMLIT_SERVER" in os.environ else os.getenv("INDEX_PATH", "./data/indices/chroma_db")
    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(is_persistent=True, anonymized_telemetry=False),
    )

    try:
        chroma_collection = chroma_client.get_collection(get_collection_name())
    except chromadb.errors.NotFoundError:
        print(f"❌ Collection '{get_collection_name()}' not found at {persist_dir}")
        return None, None, None

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    query_engine = HybridResearchEngine(chroma_collection=chroma_collection)
    return query_engine, index, chroma_collection


def get_paper_recommendations(query_engine, topic: str, num_papers: int = 3) -> str:
    if hasattr(query_engine, "recommend_papers"):
        return query_engine.recommend_papers(topic, num_papers=num_papers)

    prompt = (
        f"Based on the research topic: '{topic}', recommend {num_papers} papers from the collection. "
        "For each recommendation, include:\n"
        "1. Paper title\n"
        "2. Brief justification (1 sentence)\n"
        "3. Key contribution\n"
        "Format as markdown bullet points."
    )
    response = query_engine.query(prompt)
    return response.response if hasattr(response, "response") else str(response)


def get_paper_metadata(chroma_collection: chromadb.Collection, paper_id: str) -> dict:
    result = chroma_collection.get(ids=[paper_id], include=["metadatas"])
    if result and result["metadatas"]:
        return result["metadatas"][0]
    return {}


if __name__ == "__main__":
    print("Initializing research assistant...")
    engine, index, chroma_collection = initialize_engine()
    if not engine:
        print("❌ Failed to initialize query engine. Please create an index first.")
        raise SystemExit(1)

    print("✅ System ready. Type your questions about the research papers.")
    print("Type 'exit' to quit, '!recommend' for paper recommendations, '!compare' for comparisons.")

    while True:
        query = input("\n📝 Your research question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if query.startswith("!recommend"):
            topic = query.replace("!recommend", "").strip() or "machine learning"
            print(engine.recommend_papers(topic))
            continue
        if query.startswith("!compare"):
            topic = query.replace("!compare", "").strip() or "method comparisons"
            print(engine.compare_papers(topic))
            continue

        answer = engine.query(query)
        print(f"\n💡 Answer:\n{answer.response}")
        if answer.source_nodes:
            print("\n🔍 Sources:")
            for i, source in enumerate(answer.source_nodes, start=1):
                md = source.metadata or {}
                print(
                    f"{i}. [{md.get('arxiv_id', 'unknown')}] {md.get('title', 'Untitled')} "
                    f"(score={source.score:.3f}, page={md.get('page_start', -1)})"
                )
