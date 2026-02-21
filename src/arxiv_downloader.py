from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import arxiv
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Configuration
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", 10))
DEFAULT_MAX_AGE_DAYS = int(os.getenv("ARXIV_MAX_AGE_DAYS", 365 * 3))

def get_data_path() -> Path:
    data_path = Path(os.getenv("DATA_PATH", "./data/papers"))
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path

def _safe_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

def _build_submitted_date_clause(start_date: Optional[str], end_date: Optional[str]) -> Optional[str]:
    start = _safe_datetime(start_date)
    end = _safe_datetime(end_date)
    if not start and not end:
        return None
    if not start:
        start = datetime(1990, 1, 1, tzinfo=timezone.utc)
    if not end:
        end = datetime.now(timezone.utc)
    return f"submittedDate:[{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359]"

def build_arxiv_query(
    base_query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_age_days: Optional[int] = DEFAULT_MAX_AGE_DAYS,
) -> str:
    """Create an arXiv query string with optional date constraints."""
    if max_age_days and not start_date:
        start = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        start_date = start.strftime("%Y-%m-%d")
    date_clause = _build_submitted_date_clause(start_date=start_date, end_date=end_date)
    if not date_clause:
        return base_query
    return f"({base_query}) AND {date_clause}"

def _build_requests_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session

def download_paper(
    paper: arxiv.Result,
    output_dir: Path,
    session: requests.Session,
    request_timeout: float = 30.0,
) -> Path:
    """Download a paper PDF and return its path."""
    paper_url = paper.pdf_url
    filename = f"{paper.get_short_id()}.pdf"
    filepath = output_dir / filename

    response = session.get(paper_url, timeout=request_timeout, stream=True)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return filepath

def save_metadata(paper_meta: dict, output_dir: Path):
    """Save paper metadata to JSON file."""
    filename = f"{paper_meta['arxiv_id']}.json"
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(paper_meta, f, indent=2)

def load_existing_metadata(output_dir: Path) -> Dict[str, dict]:
    existing: Dict[str, dict] = {}
    for meta_file in output_dir.glob("*.json"):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            arxiv_id = data.get("arxiv_id")
            if arxiv_id:
                existing[arxiv_id] = data
        except (OSError, json.JSONDecodeError):
            continue
    return existing

def search_and_download_papers(
    query: str,
    max_results: int = MAX_RESULTS,
    max_age_days: Optional[int] = DEFAULT_MAX_AGE_DAYS,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by_recent: bool = True,
    dedupe: bool = True,
    request_timeout: float = 30.0,
) -> dict:
    """Search ArXiv and download papers with recency-aware filtering."""
    data_path = get_data_path()
    client = arxiv.Client()
    arxiv_query = build_arxiv_query(
        base_query=query,
        start_date=start_date,
        end_date=end_date,
        max_age_days=max_age_days,
    )
    search = arxiv.Search(
        query=arxiv_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate if sort_by_recent else arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = client.results(search)

    downloaded_papers = {}
    existing = load_existing_metadata(data_path) if dedupe else {}
    seen_ids = set()
    session = _build_requests_session()

    for paper in tqdm(results, total=max_results, desc="Downloading papers"):
        try:
            arxiv_id = paper.get_short_id()
            if dedupe and (arxiv_id in existing or arxiv_id in seen_ids):
                continue

            if max_age_days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
                if paper.published.replace(tzinfo=timezone.utc) < cutoff:
                    continue

            filepath = download_paper(paper, data_path, session=session, request_timeout=request_timeout)
            paper_meta = {
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "file_path": str(filepath),
                "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "arxiv_id": arxiv_id,
                "primary_category": paper.primary_category,
                "categories": list(paper.categories),
            }

            save_metadata(paper_meta, data_path)
            downloaded_papers[paper.entry_id] = paper_meta
            seen_ids.add(arxiv_id)
        except Exception as e:
            print(f"Failed to download {paper.title}: {str(e)}")

    return downloaded_papers

if __name__ == "__main__":
    query = input("Enter research topic to search on ArXiv: ")
    results = search_and_download_papers(query, max_age_days=DEFAULT_MAX_AGE_DAYS)

    print(f"\nDownloaded {len(results)} papers:")
    for i, (paper_id, meta) in enumerate(results.items(), 1):
        print(f"{i}. {meta['title']}")
        print(f"   Authors: {', '.join(meta['authors'][:3])}{' et al.' if len(meta['authors']) > 3 else ''}")
        print(f"   arXiv ID: {meta['arxiv_id']}")
        print(f"   Saved at: {meta['file_path']}\n")
