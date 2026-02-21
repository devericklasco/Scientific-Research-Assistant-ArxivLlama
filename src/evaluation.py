from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_benchmark(path: str | Path) -> List[Dict]:
    benchmark_path = Path(path)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Benchmark file must contain a list of evaluation rows.")
    return data


def precision_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    expected_set = set(expected_ids)
    hits = sum(1 for rid in top if rid in expected_set)
    return hits / len(top)


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    expected_set = set(expected_ids)
    if not expected_set:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = len(top & expected_set)
    return hits / len(expected_set)


def evaluate_retrieval(engine, benchmark_rows: List[Dict], k: int = 5) -> Dict:
    rows = []
    p_scores = []
    r_scores = []
    exact_hits = 0

    for row in benchmark_rows:
        query = row["query"]
        expected = row.get("expected_arxiv_ids", [])
        sources = engine.retrieve(query, top_k=k)
        retrieved = [s.metadata.get("arxiv_id", "") for s in sources]
        p_k = precision_at_k(retrieved, expected, k=k)
        r_k = recall_at_k(retrieved, expected, k=k)
        if any(rid in set(expected) for rid in retrieved[:k]):
            exact_hits += 1
        p_scores.append(p_k)
        r_scores.append(r_k)
        rows.append(
            {
                "query": query,
                "expected_arxiv_ids": expected,
                "retrieved_arxiv_ids": retrieved[:k],
                "precision_at_k": round(p_k, 4),
                "recall_at_k": round(r_k, 4),
            }
        )

    total = max(1, len(benchmark_rows))
    return {
        "k": k,
        "num_queries": len(benchmark_rows),
        "mean_precision_at_k": round(sum(p_scores) / total, 4),
        "mean_recall_at_k": round(sum(r_scores) / total, 4),
        "hit_rate_at_k": round(exact_hits / total, 4),
        "rows": rows,
    }
