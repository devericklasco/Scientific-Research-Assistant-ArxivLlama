from src.create_index import compute_paper_hash, iter_chunk_records, normalize_authors


def test_normalize_authors_handles_string_and_list():
    assert normalize_authors("Alice, Bob") == ["Alice", "Bob"]
    assert normalize_authors(["Alice", " Bob "]) == ["Alice", "Bob"]
    assert normalize_authors(None) == []


def test_iter_chunk_records_legacy_shape():
    paper = {
        "arxiv_id": "1111.1111v1",
        "title": "Legacy",
        "authors": "Alice, Bob",
        "published": "2024-01-01",
        "chunks": ["chunk one", "chunk two"],
    }
    records = list(iter_chunk_records(paper))
    assert len(records) == 2
    assert records[0]["metadata"]["authors"] == "Alice, Bob"
    assert records[0]["metadata"]["page_start"] == -1


def test_iter_chunk_records_v2_shape():
    paper = {
        "arxiv_id": "2222.2222v1",
        "title": "Modern",
        "authors": ["Alice", "Bob"],
        "published": "2025-01-01",
        "chunks": [
            {"chunk_id": 7, "text": "content", "page_start": 2, "page_end": 3, "section": "method"}
        ],
    }
    records = list(iter_chunk_records(paper))
    assert len(records) == 1
    assert records[0]["metadata"]["chunk_id"] == 7
    assert records[0]["metadata"]["page_start"] == 2
    assert records[0]["metadata"]["section"] == "method"


def test_compute_paper_hash_changes_when_content_changes():
    base = {
        "arxiv_id": "3333.3333v1",
        "title": "A",
        "authors": ["A"],
        "published": "2024-01-01",
        "chunks": ["one"],
    }
    changed = dict(base)
    changed["chunks"] = ["two"]
    assert compute_paper_hash(base) != compute_paper_hash(changed)
