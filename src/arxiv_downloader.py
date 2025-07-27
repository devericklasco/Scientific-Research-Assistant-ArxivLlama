import os
import arxiv
from dotenv import load_dotenv
from pathlib import Path
import requests
import time
from tqdm import tqdm
import json  # Added for metadata saving

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = Path(os.getenv("DATA_PATH", "./data/papers"))
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", 10))
DATA_PATH.mkdir(parents=True, exist_ok=True)

def download_paper(paper: arxiv.Result, output_dir: Path) -> Path:
    """Download a paper PDF and return its path"""
    paper_url = paper.pdf_url
    filename = f"{paper.get_short_id()}.pdf"
    filepath = output_dir / filename
    
    # Download the paper
    response = requests.get(paper_url)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    return filepath

def save_metadata(paper_meta: dict, output_dir: Path):
    """Save paper metadata to JSON file"""
    filename = f"{paper_meta['arxiv_id']}.json"
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(paper_meta, f, indent=2)

def search_and_download_papers(query: str, max_results: int = MAX_RESULTS) -> dict:
    """Search ArXiv and download papers"""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = client.results(search)
    
    downloaded_papers = {}
    
    for paper in tqdm(results, total=max_results, desc="Downloading papers"):
        try:
            # Download paper
            filepath = download_paper(paper, DATA_PATH)
            
            # Store metadata
            paper_meta = {
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "file_path": str(filepath),
                "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "arxiv_id": paper.get_short_id()
            }
            
            # Save metadata to JSON
            save_metadata(paper_meta, DATA_PATH)
            
            downloaded_papers[paper.entry_id] = paper_meta
        except Exception as e:
            print(f"Failed to download {paper.title}: {str(e)}")
    
    return downloaded_papers

if __name__ == "__main__":
    query = input("Enter research topic to search on ArXiv: ")
    results = search_and_download_papers(query)
    
    print(f"\nDownloaded {len(results)} papers:")
    for i, (paper_id, meta) in enumerate(results.items(), 1):
        print(f"{i}. {meta['title']}")
        print(f"   Authors: {', '.join(meta['authors'][:3])}{' et al.' if len(meta['authors']) > 3 else ''}")
        print(f"   arXiv ID: {meta['arxiv_id']}")
        print(f"   Saved at: {meta['file_path']}\n")