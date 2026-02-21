from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_TOKENS", 350))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", 60))
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "cl100k_base")
_TOKENIZER = None

def get_data_path() -> Path:
    data_path = Path(os.getenv("DATA_PATH", "./data/papers"))
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path

def get_chunk_path() -> Path:
    chunk_path = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
    chunk_path.mkdir(parents=True, exist_ok=True)
    return chunk_path

def load_metadata(pdf_path: Path) -> dict:
    """Load paper metadata from JSON file"""
    metadata_path = pdf_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {
        "title": pdf_path.stem,
        "authors": [],
        "published": "",
        "arxiv_id": pdf_path.stem
    }

def clean_text(text: str) -> str:
    """Normalize spacing and strip low-value control characters."""
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    return text.strip()

def detect_section_label(text: str) -> str:
    """Lightweight section detector from leading lines."""
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    header_region = " ".join(lines[:4])
    if header_region.startswith("abstract"):
        return "abstract"
    section_patterns = [
        ("introduction", r"\b(?:\d+\.?\s*)?introduction\b"),
        ("related_work", r"\b(?:\d+\.?\s*)?(related work|background)\b"),
        ("method", r"\b(?:\d+\.?\s*)?(method|approach|model)\b"),
        ("experiments", r"\b(?:\d+\.?\s*)?(experiment|evaluation|results)\b"),
        ("discussion", r"\b(?:\d+\.?\s*)?discussion\b"),
        ("conclusion", r"\b(?:\d+\.?\s*)?(conclusion|limitations|future work)\b"),
        ("references", r"\b(?:references|bibliography)\b"),
    ]
    for label, pattern in section_patterns:
        if re.search(pattern, header_region):
            return label
    return "content"

def tokenize_text(text: str) -> List:
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = tiktoken.get_encoding(TOKENIZER_NAME)
        except Exception:
            _TOKENIZER = False
    if _TOKENIZER is False:
        return text.split()
    return _TOKENIZER.encode(text)

def decode_tokens(tokens: List) -> str:
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = tiktoken.get_encoding(TOKENIZER_NAME)
        except Exception:
            _TOKENIZER = False
    if _TOKENIZER is False:
        return " ".join(tokens)
    return _TOKENIZER.decode(tokens)

def split_into_token_windows(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    token_ids = tokenize_text(text)
    if not token_ids:
        return []
    windows = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start : start + chunk_tokens]
        if not chunk:
            continue
        windows.append(decode_tokens(chunk).strip())
        if start + chunk_tokens >= len(token_ids):
            break
    return [w for w in windows if w]

def extract_sections(pdf_path: Path) -> list:
    """Extract page-level cleaned text and section labels."""
    doc = fitz.open(pdf_path)
    sections = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        cleaned = clean_text(text)
        if not cleaned:
            continue
        sections.append(
            {
                "page_number": page_num,
                "section": detect_section_label(cleaned),
                "text": cleaned,
            }
        )

    return sections

def chunk_text(sections: list, chunk_size: int = CHUNK_SIZE) -> list:
    """Backwards-compatible helper returning plain chunk strings."""
    chunk_records = build_chunk_records(
        sections=sections,
        chunk_tokens=chunk_size,
        overlap_tokens=CHUNK_OVERLAP,
    )
    return [record["text"] for record in chunk_records]

def build_chunk_records(
    sections: List[Dict],
    chunk_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """Create token-aware chunk records with page and section metadata."""
    chunk_records: List[Dict] = []
    chunk_id = 0

    for idx, item in enumerate(sections, start=1):
        if isinstance(item, dict):
            page = int(item.get("page_number", idx))
            section = item.get("section", "content")
            text = item.get("text", "")
        else:
            section, text = item
            page = idx
        for window in split_into_token_windows(text, chunk_tokens, overlap_tokens):
            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "text": window,
                    "page_start": page,
                    "page_end": page,
                    "section": section,
                }
            )
            chunk_id += 1
    return chunk_records

def process_papers():
    data_path = get_data_path()
    chunk_path = get_chunk_path()
    processed = {}
    for pdf_file in tqdm(list(data_path.glob("*.pdf")), desc="Processing PDFs"):
        try:
            # Load metadata
            metadata = load_metadata(pdf_file)
            
            # Process content
            sections = extract_sections(pdf_file)
            chunks = build_chunk_records(
                sections=sections,
                chunk_tokens=CHUNK_SIZE,
                overlap_tokens=CHUNK_OVERLAP,
            )
            
            # Save chunks to JSON
            chunk_data = {
                "paper_id": metadata["arxiv_id"],
                "title": metadata["title"],
                "file_path": str(pdf_file),
                "authors": metadata.get("authors", []),
                "published": metadata["published"],
                "arxiv_id": metadata["arxiv_id"],
                "chunk_version": "v2_token_overlap",
                "chunk_size_tokens": CHUNK_SIZE,
                "chunk_overlap_tokens": CHUNK_OVERLAP,
                "chunks": chunks
            }
            
            output_file = chunk_path / f"{metadata['arxiv_id']}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2)
            
            processed[metadata["arxiv_id"]] = {
                "path": str(pdf_file),
                "num_chunks": len(chunks),
                "title": metadata["title"]
            }
        except Exception as e:
            print(f"Failed to process {pdf_file.name}: {str(e)}")
    
    return processed

if __name__ == "__main__":
    print("Starting PDF processing...")
    processed_papers = process_papers()
    total_chunks = sum(meta["num_chunks"] for meta in processed_papers.values())
    print(f"✅ Processed {len(processed_papers)} papers into {total_chunks} text chunks")
    print(f"📁 Chunk data saved to: {get_chunk_path()}")
