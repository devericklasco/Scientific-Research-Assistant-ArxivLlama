import io
import zipfile

from src.citation_generator import (
    build_citation_bundle,
    generate_apa_citation,
    generate_bibtex_citation,
)


def test_generate_apa_citation_from_string_authors():
    metadata = {
        "authors": "Alice Smith, Bob Jones",
        "published": "2024-05-01",
        "title": "A Study",
        "arxiv_id": "2405.12345v1",
    }
    citation = generate_apa_citation(metadata)
    assert "Alice Smith, & Bob Jones" in citation
    assert "(2024)" in citation
    assert "A Study." in citation


def test_generate_bibtex_citation_contains_arxiv_fields():
    metadata = {
        "authors": ["Alice Smith"],
        "published": "2023-01-01",
        "title": "Efficient Retrieval",
        "arxiv_id": "2301.00001v2",
    }
    bib = generate_bibtex_citation(metadata)
    assert "@article{" in bib
    assert "archivePrefix" in bib
    assert "2301.00001v2" in bib


def test_build_citation_bundle_creates_expected_files():
    papers = [
        {
            "authors": ["Alice Smith"],
            "published": "2023-01-01",
            "title": "Efficient Retrieval",
            "arxiv_id": "2301.00001v2",
        }
    ]
    bundle = build_citation_bundle(papers)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zip_file:
        names = set(zip_file.namelist())
    assert {"citations_apa.txt", "citations.bib", "papers_metadata.json"} <= names
