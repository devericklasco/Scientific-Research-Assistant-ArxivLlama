from src.query_engine import (
    ScoredSourceNode,
    combine_scores,
    compute_recency_score,
    lexical_overlap_score,
    should_return_insufficient_evidence,
)


def test_lexical_overlap_score_behaves_reasonably():
    high = lexical_overlap_score("graph neural network", "A graph neural network model")
    low = lexical_overlap_score("graph neural network", "quantum chemistry experiment")
    assert high > low
    assert 0.0 <= low <= 1.0


def test_compute_recency_score_recent_higher_than_old():
    recent = compute_recency_score("2025-12-01")
    old = compute_recency_score("2010-01-01")
    assert recent > old


def test_combine_scores_weighted_sum():
    score = combine_scores(semantic_score=1.0, lexical_score=0.5, recency_score=0.0)
    assert 0.0 < score < 1.0


def test_should_return_insufficient_evidence():
    weak = [
        ScoredSourceNode(
            text="x",
            metadata={},
            score=0.01,
            semantic_score=0.01,
            lexical_score=0.01,
            recency_score=0.01,
        )
    ]
    strong = [
        ScoredSourceNode(
            text="x",
            metadata={},
            score=0.9,
            semantic_score=0.9,
            lexical_score=0.9,
            recency_score=0.9,
        )
    ]
    assert should_return_insufficient_evidence(weak) is True
    assert should_return_insufficient_evidence(strong) is False
