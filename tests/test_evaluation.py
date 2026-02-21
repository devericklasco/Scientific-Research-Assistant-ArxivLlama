from dataclasses import dataclass

from src.evaluation import evaluate_retrieval, precision_at_k, recall_at_k


@dataclass
class DummySource:
    metadata: dict


class DummyEngine:
    def __init__(self, outputs):
        self.outputs = outputs

    def retrieve(self, query, top_k=5):
        ids = self.outputs.get(query, [])[:top_k]
        return [DummySource(metadata={"arxiv_id": rid}) for rid in ids]


def test_precision_and_recall_at_k():
    retrieved = ["a", "b", "c"]
    expected = ["b", "x"]
    assert precision_at_k(retrieved, expected, k=2) == 0.5
    assert recall_at_k(retrieved, expected, k=2) == 0.5


def test_evaluate_retrieval():
    benchmark = [
        {"query": "q1", "expected_arxiv_ids": ["a"]},
        {"query": "q2", "expected_arxiv_ids": ["x"]},
    ]
    engine = DummyEngine(outputs={"q1": ["a", "b"], "q2": ["c", "d"]})
    report = evaluate_retrieval(engine, benchmark_rows=benchmark, k=2)
    assert report["num_queries"] == 2
    assert 0.0 <= report["mean_precision_at_k"] <= 1.0
    assert 0.0 <= report["mean_recall_at_k"] <= 1.0
