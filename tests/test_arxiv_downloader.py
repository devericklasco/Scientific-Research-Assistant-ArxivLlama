import tempfile
from pathlib import Path

from src.arxiv_downloader import build_arxiv_query, load_existing_metadata, save_metadata


def test_build_arxiv_query_with_explicit_dates():
    query = build_arxiv_query(
        base_query="graph neural networks",
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_age_days=None,
    )
    assert "graph neural networks" in query
    assert "submittedDate:[202401010000 TO 202412312359]" in query


def test_build_arxiv_query_without_dates():
    query = build_arxiv_query(base_query="llm serving", max_age_days=None)
    assert query == "llm serving"


def test_load_existing_metadata_reads_valid_json():
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        payload = {"arxiv_id": "1234.5678v1", "title": "Test Paper"}
        save_metadata(payload, temp_path)
        bad_file = temp_path / "bad.json"
        bad_file.write_text("{not-json}", encoding="utf-8")

        existing = load_existing_metadata(temp_path)
        assert "1234.5678v1" in existing
        assert existing["1234.5678v1"]["title"] == "Test Paper"
