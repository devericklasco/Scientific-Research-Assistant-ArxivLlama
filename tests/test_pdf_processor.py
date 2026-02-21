from src.pdf_processor import (
    build_chunk_records,
    detect_section_label,
    split_into_token_windows,
    tokenize_text,
)


def test_detect_section_label():
    assert detect_section_label("Abstract\nThis paper introduces...") == "abstract"
    assert detect_section_label("1 Introduction\nWe study...") == "introduction"
    assert detect_section_label("5 Conclusion\nIn summary...") == "conclusion"


def test_split_into_token_windows_with_overlap():
    text = " ".join(["token"] * 120)
    windows = split_into_token_windows(text=text, chunk_tokens=40, overlap_tokens=10)
    assert len(windows) >= 3

    first_tail = windows[0].split()[-8:]
    second_head = windows[1].split()[:8]
    assert first_tail == second_head


def test_build_chunk_records_supports_legacy_and_v2_shapes():
    legacy_sections = [("content", "A short legacy section.")]
    legacy_records = build_chunk_records(legacy_sections, chunk_tokens=50, overlap_tokens=5)
    assert legacy_records
    assert legacy_records[0]["page_start"] == 1

    v2_sections = [{"page_number": 4, "section": "method", "text": "Method section text."}]
    v2_records = build_chunk_records(v2_sections, chunk_tokens=50, overlap_tokens=5)
    assert v2_records
    assert v2_records[0]["page_start"] == 4
    assert v2_records[0]["section"] == "method"
