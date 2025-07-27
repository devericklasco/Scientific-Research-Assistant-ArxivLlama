import re
from datetime import datetime
from typing import Dict

def generate_apa_citation(metadata: Dict) -> str:
    """
    Generate APA citation from paper metadata
    Format: Author(s). (Year). Title. arXiv preprint arXiv:ID
    """
    # Extract authors (if available)
    authors = metadata.get("authors", [])
    if not authors:
        authors = ["Anonymous"]
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",")]  # Convert string to list 
    # Format author list
    if len(authors) > 1:
        author_str = ", ".join(authors[:-1]) + ", & " + authors[-1]
    else:
        author_str = authors[0]
    
    # Extract year from publication date
    published = metadata.get("published", "")
    year = ""
    if published:
        try:
            # Try to parse various date formats
            for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
                try:
                    dt = datetime.strptime(published, fmt)
                    year = str(dt.year)
                    break
                except ValueError:
                    continue
        except Exception:
            pass
    
    if not year:
        year = "n.d."  # No date available
    
    # Extract title
    title = metadata.get("title", "Untitled Paper").strip()
    if not title.endswith("."):
        title += "."
    
    # Extract arXiv ID
    arxiv_id = metadata.get("paper_id", "")
    if not arxiv_id.startswith("arXiv:"):
        # Try to extract from file path
        file_path = metadata.get("file_path", "")
        if file_path:
            # Look for pattern like 1234.56789v1.pdf
            match = re.search(r"/(\d{4}\.\d{4,5}(v\d+)?)\.pdf$", file_path)
            if match:
                arxiv_id = f"arXiv:{match.group(1)}"
    
    # Construct citation
    citation = f"{author_str} ({year}). {title}"
    if arxiv_id:
        citation += f" {arxiv_id}."
    
    return citation