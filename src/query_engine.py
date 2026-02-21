from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import tiktoken
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI

load_dotenv()

DEFAULT_SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))
MIN_GROUNDING_SCORE = float(os.getenv("MIN_GROUNDING_SCORE", 0.2))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", 0.62))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", 0.23))
RECENCY_WEIGHT = float(os.getenv("RECENCY_WEIGHT", 0.15))


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
    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K) -> None:
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.retriever = index.as_retriever(similarity_top_k=max(similarity_top_k * 4, 20))
        self._llm = OpenAI(model="gpt-4o", temperature=0.1) if os.getenv("OPENAI_API_KEY") else None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[ScoredSourceNode]:
        k = top_k or self.similarity_top_k
        candidates: List[ScoredSourceNode] = []
        for node_with_score in self.retriever.retrieve(query):
            text = node_with_score.text
            metadata = node_with_score.metadata or {}
            raw_score = float(node_with_score.score or 0.0)
            semantic_score = min(max(raw_score, 0.0), 1.0)
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
            snippets.append(f"[S{idx}] {title}: {source.text[:220]}...")
        return (
            "OpenAI API key is not configured, so this is an extractive summary only.\n\n"
            f"Question: {question}\n\n"
            + "\n".join(snippets)
        )

    def _build_grounded_prompt(self, question: str, sources: List[ScoredSourceNode]) -> str:
        source_blocks = []
        for idx, s in enumerate(sources, start=1):
            md = s.metadata
            source_blocks.append(
                f"[S{idx}] title={md.get('title','Untitled')} | published={md.get('published','Unknown')} | "
                f"section={md.get('section','content')} | page={md.get('page_start', -1)}\n{s.text}"
            )
        context = "\n\n".join(source_blocks)
        return (
            "Answer only from sources. Cite key statements with [S1], [S2], etc. "
            "If insufficient evidence, say exactly: 'Insufficient evidence in indexed papers.'\n\n"
            f"Question:\n{question}\n\nSources:\n{context}"
        )

    def query(self, question: str) -> GroundedResponse:
        sources = self.retrieve(question, top_k=self.similarity_top_k)
        if should_return_insufficient_evidence(sources):
            return GroundedResponse(
                response=(
                    "Insufficient evidence in indexed papers. "
                    "Try rephrasing the question, expanding papers, or adjusting filters."
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
        answer = self._llm.complete(prompt).text.strip()
        return GroundedResponse(response=answer, source_nodes=sources, grounded=True)

    def recommend_papers(self, topic: str, num_papers: int = 3) -> str:
        sources = self.retrieve(topic, top_k=max(num_papers * 4, 12))
        by_paper: Dict[str, ScoredSourceNode] = {}
        for s in sources:
            pid = s.metadata.get("arxiv_id", "unknown")
            if pid not in by_paper or s.score > by_paper[pid].score:
                by_paper[pid] = s
        ranked = sorted(by_paper.values(), key=lambda x: x.score, reverse=True)[:num_papers]
        if not ranked:
            return "No recommendations available."
        lines = []
        for s in ranked:
            md = s.metadata
            lines.append(
                f"- **{md.get('title','Untitled')}** ({md.get('published','Unknown')}) "
                f"[{md.get('arxiv_id','N/A')}]\n"
                f"  - Why: score={s.score:.3f}, section={md.get('section','content')}\n"
                f"  - Evidence: {s.text[:220]}..."
            )
        return "\n".join(lines)

    def compare_papers(self, comparison_topic: str, num_papers: int = 3) -> str:
        sources = self.retrieve(comparison_topic, top_k=max(12, num_papers * 4))
        by_paper: Dict[str, ScoredSourceNode] = {}
        for s in sources:
            pid = s.metadata.get("arxiv_id", "unknown")
            if pid not in by_paper or s.score > by_paper[pid].score:
                by_paper[pid] = s
        ranked = sorted(by_paper.values(), key=lambda x: x.score, reverse=True)[:num_papers]
        if not ranked:
            return "No papers found for comparison."
        rows = ["| Paper | Published | Signal | Evidence |", "|---|---|---|---|"]
        for s in ranked:
            md = s.metadata
            rows.append(
                f"| {md.get('title','Untitled')} | {md.get('published','Unknown')} | "
                f"{md.get('section','content')} (score {s.score:.2f}) | {s.text[:140].replace('|', ' ')}... |"
            )
        return "\n".join(rows)

    def extract_claims_with_evidence(self, topic: str, max_claims: int = 5) -> List[Dict]:
        sources = self.retrieve(topic, top_k=max(max_claims * 3, 12))
        claims: List[Dict] = []
        for s in sources[:max_claims]:
            md = s.metadata
            claim = s.text.split(".")[0].strip() or s.text[:120]
            claims.append(
                {
                    "claim": claim,
                    "evidence": s.text[:260],
                    "arxiv_id": md.get("arxiv_id", ""),
                    "title": md.get("title", ""),
                    "published": md.get("published", ""),
                    "page_start": md.get("page_start", -1),
                    "page_end": md.get("page_end", -1),
                    "score": round(s.score, 4),
                }
            )
        return claims


def create_query_engine_from_index(index: VectorStoreIndex) -> HybridResearchEngine:
    return HybridResearchEngine(index=index)


def initialize_engine():
    return None, None, None


def get_paper_recommendations(query_engine, topic: str, num_papers: int = 3) -> str:
    if hasattr(query_engine, "recommend_papers"):
        return query_engine.recommend_papers(topic, num_papers=num_papers)
    response = query_engine.query(
        f"Recommend {num_papers} papers about {topic} from the indexed collection with short reasons."
    )
    return response.response if hasattr(response, "response") else str(response)
