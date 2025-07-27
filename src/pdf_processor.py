import fitz  # PyMuPDF
from pathlib import Path
import os
from dotenv import load_dotenv
from tqdm import tqdm
import re
import json

load_dotenv()

# Configuration
DATA_PATH = Path(os.getenv("DATA_PATH", "./data/papers"))
CHUNK_PATH = Path(os.getenv("CHUNK_PATH", "./data/chunks"))
CHUNK_SIZE = 512  # Optimal for GPT models

# Create directories if they don't exist
DATA_PATH.mkdir(parents=True, exist_ok=True)
CHUNK_PATH.mkdir(parents=True, exist_ok=True)

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
    """Remove unwanted characters and formatting"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    return text.strip()

def extract_sections(pdf_path: Path) -> list:
    """Extract structured sections from PDF"""
    doc = fitz.open(pdf_path)
    sections = []
    
    for page in doc:
        text = page.get_text()
        cleaned = clean_text(text)
        
        # Simple section detection (improve with NLP later)
        if cleaned.startswith("Abstract"):
            sections.append(("abstract", cleaned))
        elif "Introduction" in cleaned[:100]:
            sections.append(("introduction", cleaned))
        elif "Conclusion" in cleaned[:100]:
            sections.append(("conclusion", cleaned))
        else:
            sections.append(("content", cleaned))
    
    return sections

def chunk_text(sections: list, chunk_size: int = CHUNK_SIZE) -> list:
    """Split text into manageable chunks"""
    chunks = []
    current_chunk = ""
    
    for _, text in sections:
        words = text.split()
        for word in words:
            if len(current_chunk) + len(word) < chunk_size:
                current_chunk += word + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = word + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def process_papers():
    processed = {}
    for pdf_file in tqdm(list(DATA_PATH.glob("*.pdf")), desc="Processing PDFs"):
        try:
            # Load metadata
            metadata = load_metadata(pdf_file)
            
            # Process content
            sections = extract_sections(pdf_file)
            chunks = chunk_text(sections)
            
            # Save chunks to JSON
            chunk_data = {
                "paper_id": metadata["arxiv_id"],
                "title": metadata["title"],
                "file_path": str(pdf_file),
                # "authors": metadata["authors"],
                "authors": ", ".join(metadata["authors"]),  # Store as string early
                "published": metadata["published"],
                "arxiv_id": metadata["arxiv_id"],
                "chunks": chunks
            }
            
            output_file = CHUNK_PATH / f"{metadata['arxiv_id']}.json"
            with open(output_file, 'w') as f:
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
    print(f"📁 Chunk data saved to: {CHUNK_PATH}")