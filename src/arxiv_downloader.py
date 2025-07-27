import os
import arxiv
from dotenv import load_dotenv
from pathlib import Path
import requests
import time
from tqdm import tqdm

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

def search_and_download_papers(query: str, max_results: int = MAX_RESULTS) -> dict:
    """Search ArXiv and download papers"""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    downloaded_papers = {}
    
    for paper in tqdm(search.results(), total=max_results, desc="Downloading papers"):
        try:
            # Download paper
            filepath = download_paper(paper, DATA_PATH)
            
            # Store metadata
            downloaded_papers[paper.entry_id] = {
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "filepath": str(filepath),
                "download_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
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
        print(f"   Saved at: {meta['filepath']}\n")