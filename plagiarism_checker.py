"""
plagiarism_checker.py — Plagiarism & Reference Checker.

Features:
  • Plagiarism score via CrossRef + PubMed text similarity
  • Reference extraction from PDF text
  • DOI validation
  • Citation style generator (APA/MLA/IEEE)
"""

from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher
from typing import Any

import requests

logger = logging.getLogger(__name__)

CROSSREF_API   = "https://api.crossref.org/works"
PUBMED_SEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
DOI_LOOKUP     = "https://doi.org/"


# ---------------------------------------------------------------------------
# Reference Extractor
# ---------------------------------------------------------------------------

# Common reference patterns in academic papers
_REF_PATTERNS = [
    # [1] Author, Title, Journal, Year
    r'\[\d+\]\s+([A-Z][^[]+?)(?=\[\d+\]|$)',
    # numbered list: 1. Author...
    r'^\s*\d+\.\s+([A-Z][^\n]{20,})',
    # DOI pattern
    r'(?:doi:|DOI:|https?://doi\.org/)([^\s,\]]+)',
]

_DOI_RE      = re.compile(r'(?:doi:|DOI:|https?://doi\.org/)(10\.\d{4,}/[^\s,\]>"]+)', re.IGNORECASE)
_AUTHOR_YEAR = re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+et\s+al\.?,?\s+\(?(20\d{2}|19\d{2})\)?')


def extract_references(text: str) -> list[dict[str, Any]]:
    """
    Extract references/citations from paper text.
    Returns list of {raw, doi, authors, year, type}
    """
    refs = []
    seen = set()

    # 1. Extract DOIs
    for m in _DOI_RE.finditer(text):
        doi = m.group(1).rstrip('.')
        if doi not in seen:
            seen.add(doi)
            refs.append({"raw": m.group(0), "doi": doi, "type": "doi", "authors": "", "year": ""})

    # 2. Extract author+year citations
    for m in _AUTHOR_YEAR.finditer(text):
        raw = m.group(0)
        if raw not in seen:
            seen.add(raw)
            refs.append({"raw": raw, "doi": "", "type": "citation",
                         "authors": m.group(1), "year": m.group(2)})

    # 3. Try to find References section and parse it
    ref_section_match = re.search(
        r'(?:References|Bibliography|REFERENCES)\s*\n(.*?)(?:\n\n\n|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if ref_section_match:
        ref_text = ref_section_match.group(1)
        # Parse numbered references
        for m in re.finditer(r'(\d+)\.\s+(.{30,200})', ref_text):
            raw = m.group(2).strip()
            if raw not in seen:
                seen.add(raw)
                # Try to extract year
                yr = re.search(r'(20\d{2}|19\d{2})', raw)
                refs.append({"raw": raw[:200], "doi": "", "type": "numbered",
                             "authors": "", "year": yr.group(1) if yr else ""})

    return refs[:30]  # cap at 30


def validate_doi(doi: str) -> dict[str, Any]:
    """Check if a DOI actually resolves and get metadata."""
    if not doi:
        return {"valid": False, "error": "No DOI provided"}
    try:
        url = f"https://api.crossref.org/works/{doi.strip()}"
        resp = requests.get(url, timeout=8,
                            headers={"User-Agent": "PharmacyRAG/1.0 (research@example.com)"})
        if resp.status_code == 200:
            data = resp.json().get("message", {})
            title = data.get("title", [""])[0] if data.get("title") else "Unknown"
            authors = data.get("author", [])
            author_str = ", ".join(
                f"{a.get('family','')} {a.get('given','')[:1]}".strip()
                for a in authors[:3]
            )
            year = ""
            pub = data.get("published", {})
            if pub and pub.get("date-parts"):
                year = str(pub["date-parts"][0][0])
            journal = ""
            ct = data.get("container-title", [])
            if ct:
                journal = ct[0]
            return {
                "valid": True, "doi": doi, "title": title,
                "authors": author_str, "year": year, "journal": journal,
                "url": f"https://doi.org/{doi}",
            }
        else:
            return {"valid": False, "doi": doi, "error": f"DOI not found (HTTP {resp.status_code})"}
    except Exception as e:
        return {"valid": False, "doi": doi, "error": str(e)}


# ---------------------------------------------------------------------------
# Plagiarism Checker
# ---------------------------------------------------------------------------

def _text_similarity(a: str, b: str) -> float:
    """Simple sequence similarity between two text snippets."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower()[:500], b.lower()[:500]).ratio()


def _get_crossref_abstract(query: str) -> list[str]:
    """Fetch abstracts/titles from CrossRef for comparison."""
    try:
        params = {"query": query[:200], "rows": 5,
                  "select": "title,abstract",
                  "mailto": "research@example.com"}
        resp = requests.get(CROSSREF_API, params=params, timeout=8)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        texts = []
        for item in items:
            title = item.get("title", [""])[0] if item.get("title") else ""
            abstract = item.get("abstract", "")
            # Strip HTML tags from abstract
            abstract = re.sub(r'<[^>]+>', '', abstract)
            texts.append(f"{title} {abstract}")
        return texts
    except Exception as e:
        logger.warning("CrossRef fetch error: %s", e)
        return []


def _get_pubmed_abstracts(query: str) -> list[str]:
    """Fetch abstracts from PubMed for comparison."""
    try:
        search_resp = requests.get(PUBMED_SEARCH, params={
            "db": "pubmed", "term": query[:150],
            "retmax": 5, "retmode": "json"
        }, timeout=8)
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch_resp = requests.get(PUBMED_FETCH, params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "json"
        }, timeout=8)
        result_data = fetch_resp.json().get("result", {})
        texts = []
        for uid in ids:
            item = result_data.get(uid, {})
            title = item.get("title", "")
            source = item.get("source", "")
            texts.append(f"{title} {source}")
        return texts
    except Exception as e:
        logger.warning("PubMed fetch error: %s", e)
        return []


def compute_plagiarism_score(chunks: list[dict], filename: str) -> dict[str, Any]:
    """
    Compute plagiarism score for uploaded document.

    Method:
    1. Take key sentences from document
    2. Search CrossRef + PubMed for similar content
    3. Compute text similarity scores
    4. Return overall plagiarism percentage + matched sources
    """
    if not chunks:
        return {"score": 0, "status": "No content", "matches": []}

    # Take sample sentences from document
    sample_texts = []
    for chunk in chunks[:10]:
        text = chunk.get("text", "")
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 60]
        sample_texts.extend(sentences[:2])

    if not sample_texts:
        return {"score": 0, "status": "No extractable text", "matches": []}

    # Use first few sentences as search query
    search_query = " ".join(sample_texts[:3])[:200]

    # Fetch comparison texts
    crossref_texts = _get_crossref_abstract(search_query)
    pubmed_texts   = _get_pubmed_abstracts(search_query)
    all_sources    = crossref_texts + pubmed_texts

    if not all_sources:
        return {
            "score": 0,
            "status": "✅ No matches found in CrossRef/PubMed",
            "matches": [],
            "sentences_checked": len(sample_texts),
        }

    # Compare each sample sentence against fetched sources
    matches = []
    total_sim = 0.0
    checked = 0

    for sent in sample_texts[:8]:
        best_sim = 0.0
        best_src = ""
        for src_text in all_sources:
            sim = _text_similarity(sent, src_text)
            if sim > best_sim:
                best_sim = sim
                best_src = src_text[:100]

        if best_sim > 0.3:  # 30% similarity threshold
            matches.append({
                "sentence": sent[:150],
                "similarity": round(best_sim * 100, 1),
                "matched_source": best_src,
            })
        total_sim += best_sim
        checked += 1
        time.sleep(0.05)

    overall_score = round((total_sim / max(checked, 1)) * 100, 1)

    if overall_score < 15:
        status = "✅ Original — Low similarity"
        color  = "green"
    elif overall_score < 30:
        status = "⚠️ Moderate similarity — Review recommended"
        color  = "orange"
    else:
        status = "🔴 High similarity — Possible plagiarism"
        color  = "red"

    return {
        "score":             overall_score,
        "status":            status,
        "color":             color,
        "matches":           matches,
        "sentences_checked": checked,
        "sources_compared":  len(all_sources),
    }


# ---------------------------------------------------------------------------
# Citation Generator
# ---------------------------------------------------------------------------

def generate_citation(doi_data: dict, style: str = "APA") -> str:
    """Generate citation string in APA, MLA, or IEEE format."""
    if not doi_data.get("valid"):
        return "Citation unavailable — invalid DOI"

    authors = doi_data.get("authors", "Unknown Author")
    year    = doi_data.get("year", "n.d.")
    title   = doi_data.get("title", "Untitled")
    journal = doi_data.get("journal", "")
    doi     = doi_data.get("doi", "")
    url     = doi_data.get("url", "")

    if style == "APA":
        return f"{authors} ({year}). {title}. *{journal}*. https://doi.org/{doi}"
    elif style == "MLA":
        return f'{authors}. "{title}." *{journal}*, {year}, https://doi.org/{doi}'
    elif style == "IEEE":
        return f'{authors}, "{title}," *{journal}*, {year}. [Online]. Available: https://doi.org/{doi}'
    return f"{authors} ({year}). {title}. {journal}. DOI: {doi}"
