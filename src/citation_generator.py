import io
import json
import re
import zipfile
from datetime import datetime
from typing import Dict, List

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

def _normalize_authors(metadata: Dict) -> List[str]:
    authors = metadata.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    if not authors:
        return ["Anonymous"]
    return authors

def _extract_year(metadata: Dict) -> str:
    published = metadata.get("published", "")
    if not published:
        return "n.d."
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return str(datetime.strptime(published, fmt).year)
        except ValueError:
            continue
    return "n.d."

def _extract_arxiv_id(metadata: Dict) -> str:
    arxiv_id = metadata.get("arxiv_id") or metadata.get("paper_id", "")
    if arxiv_id:
        return str(arxiv_id).replace("arXiv:", "")
    file_path = metadata.get("file_path", "")
    if file_path:
        match = re.search(r"/(\d{4}\.\d{4,5}(v\d+)?)\.pdf$", file_path)
        if match:
            return match.group(1)
    return ""

def generate_bibtex_citation(metadata: Dict) -> str:
    authors = " and ".join(_normalize_authors(metadata))
    title = metadata.get("title", "Untitled Paper").replace("{", "").replace("}", "")
    year = _extract_year(metadata)
    arxiv_id = _extract_arxiv_id(metadata)
    citation_key = f"{_normalize_authors(metadata)[0].split()[-1].lower()}_{year}_{arxiv_id or 'paper'}"
    lines = [
        f"@article{{{citation_key},",
        f"  title={{ {title} }},",
        f"  author={{ {authors} }},",
        f"  year={{ {year} }},",
    ]
    if arxiv_id:
        lines.append(f"  eprint={{ {arxiv_id} }},")
        lines.append("  archivePrefix={ arXiv },")
    lines.append("}")
    return "\n".join(lines)

def build_citation_bundle(papers: List[Dict]) -> bytes:
    apa_lines = []
    bibtex_lines = []
    for paper in papers:
        apa_lines.append(generate_apa_citation(paper))
        bibtex_lines.append(generate_bibtex_citation(paper))

    metadata_json = json.dumps(papers, indent=2).encode("utf-8")
    bundle_stream = io.BytesIO()
    with zipfile.ZipFile(bundle_stream, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("citations_apa.txt", "\n".join(apa_lines))
        zip_file.writestr("citations.bib", "\n\n".join(bibtex_lines))
        zip_file.writestr("papers_metadata.json", metadata_json)
    bundle_stream.seek(0)
    return bundle_stream.read()
