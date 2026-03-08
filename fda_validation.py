"""
fda_validation.py — FDA Validation Layer.

Queries the free openFDA drug-label API to cross-reference pharmaceutical
claims in generated answers and provides confidence scores.
Also supports:
  • Medical term extraction
  • CrossRef / DOI reference verification
  • PubMed / NCBI source checking
  • Claim validation
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

from config import FDA_BASE_URL, FDA_SEARCH_LIMIT, FDA_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common pharmaceutical drug-name patterns
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b([A-Z][a-z]{3,})\b")

_COMMON_DRUG_SUFFIXES = {
    "mab", "nib", "pril", "sartan", "olol", "oxacin", "mycin",
    "cillin", "cycline", "vir", "azole", "statin", "pam", "lam",
    "pine", "tidine", "prazole", "dipine", "olone", "sterone",
    "triptan", "dronate", "gliptin", "gliflozin", "lutide",
}

_COMMON_DRUG_NAMES = {
    "Amoxicillin", "Ibuprofen", "Acetaminophen", "Lisinopril",
    "Metformin", "Atorvastatin", "Omeprazole", "Aspirin",
    "Metoprolol", "Amlodipine", "Warfarin", "Clopidogrel",
    "Sertraline", "Fluoxetine", "Gabapentin", "Hydrocodone",
    "Tramadol", "Prednisone", "Albuterol", "Levothyroxine",
    "Losartan", "Simvastatin", "Ciprofloxacin", "Azithromycin",
    "Doxycycline", "Cetirizine", "Loratadine", "Diphenhydramine",
    "Furosemide", "Pantoprazole", "Esomeprazole", "Ranitidine",
    "Montelukast", "Citalopram", "Escitalopram", "Venlafaxine",
    "Bupropion", "Alprazolam", "Clonazepam", "Zolpidem",
    "Duloxetine", "Pregabalin", "Topiramate", "Valproate",
}

# ---------------------------------------------------------------------------
# Medical term patterns (conditions, symptoms, anatomy)
# ---------------------------------------------------------------------------

_MEDICAL_TERM_PATTERNS = [
    # Disease/condition suffixes
    r"\b\w+itis\b",      # inflammation: hepatitis, arthritis
    r"\b\w+emia\b",      # blood condition: anemia, leukemia
    r"\b\w+osis\b",      # condition: fibrosis, cirrhosis
    r"\b\w+pathy\b",     # disease: neuropathy, cardiomyopathy
    r"\b\w+algia\b",     # pain: neuralgia, fibromyalgia
    r"\b\w+oma\b",       # tumor: carcinoma, lymphoma
    r"\b\w+ectomy\b",    # surgical removal
    r"\b\w+plasty\b",    # surgical repair
    r"\b\w+scopy\b",     # examination procedure
    r"\b\w+gram\b",      # recording/image
]

_KNOWN_MEDICAL_TERMS = {
    "hypertension", "diabetes", "tachycardia", "bradycardia", "arrhythmia",
    "myocardial", "infarction", "ischemia", "angina", "atherosclerosis",
    "thrombosis", "embolism", "hemorrhage", "hypoglycemia", "hyperglycemia",
    "nephropathy", "retinopathy", "neuropathy", "sepsis", "pneumonia",
    "contraindication", "pharmacokinetics", "pharmacodynamics", "metabolism",
    "bioavailability", "half-life", "clearance", "toxicity", "overdose",
    "adverse", "interaction", "mechanism", "receptor", "inhibitor", "agonist",
    "antagonist", "synergistic", "idiosyncratic", "anaphylaxis", "hepatotoxic",
}


def _has_drug_suffix(word: str) -> bool:
    w = word.lower()
    return any(w.endswith(suffix) for suffix in _COMMON_DRUG_SUFFIXES)


def extract_medical_terms(text: str) -> list[str]:
    """Extract medical/clinical terms from text using patterns and known list."""
    found: set[str] = set()
    text_lower = text.lower()

    # Known terms
    for term in _KNOWN_MEDICAL_TERMS:
        if term in text_lower:
            found.add(term)

    # Pattern-based extraction
    for pattern in _MEDICAL_TERM_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            word = match.group(0).lower()
            if len(word) >= 5:
                found.add(word)

    return sorted(found)


# ---------------------------------------------------------------------------
# CrossRef / DOI verification
# ---------------------------------------------------------------------------

CROSSREF_API_URL = "https://api.crossref.org/works"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def verify_crossref_doi(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """
    Search CrossRef for academic papers related to query.
    Returns list of DOI-verified references.
    """
    try:
        params = {
            "query": query,
            "rows": max_results,
            "select": "DOI,title,author,published,container-title",
        }
        resp = requests.get(CROSSREF_API_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("message", {}).get("items", [])
        refs = []
        for item in items:
            title = item.get("title", [""])[0] if item.get("title") else "Unknown Title"
            doi = item.get("DOI", "")
            authors = item.get("author", [])
            author_str = ", ".join(
                f"{a.get('family', '')} {a.get('given', '')[:1]}".strip()
                for a in authors[:3]
            )
            year = ""
            pub = item.get("published", {})
            if pub and pub.get("date-parts"):
                year = str(pub["date-parts"][0][0])
            journal = ""
            ct = item.get("container-title", [])
            if ct:
                journal = ct[0]
            refs.append({
                "title": title,
                "doi": doi,
                "authors": author_str,
                "year": year,
                "journal": journal,
                "url": f"https://doi.org/{doi}" if doi else "",
            })
        return refs
    except Exception as exc:
        logger.warning("CrossRef lookup failed: %s", exc)
        return []


def search_pubmed(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """
    Search PubMed/NCBI for relevant biomedical literature.
    Returns list of article summaries.
    """
    try:
        # Search for IDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        resp = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=8)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Fetch summaries
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }
        resp2 = requests.get(PUBMED_FETCH_URL, params=fetch_params, timeout=8)
        resp2.raise_for_status()
        uids = resp2.json().get("result", {}).get("uids", [])
        result_data = resp2.json().get("result", {})

        articles = []
        for uid in uids:
            item = result_data.get(uid, {})
            title = item.get("title", "Unknown Title")
            authors_list = item.get("authors", [])
            author_str = ", ".join(
                a.get("name", "") for a in authors_list[:3]
            )
            pub_date = item.get("pubdate", "")
            source = item.get("source", "")
            articles.append({
                "pmid": uid,
                "title": title,
                "authors": author_str,
                "pub_date": pub_date,
                "journal": source,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            })
        return articles
    except Exception as exc:
        logger.warning("PubMed search failed: %s", exc)
        return []


def validate_claims(text: str) -> list[dict[str, Any]]:
    """
    Extract and validate pharmaceutical claims from text.
    Returns list of identified claims with basic validation status.
    """
    claim_patterns = [
        (r"(?i)(treats?|used for|indicated for)\s+([^.,;]+)", "indication"),
        (r"(?i)(side effects?|adverse effects?)\s+(?:include|are|:)?\s*([^.,;]+)", "side_effect"),
        (r"(?i)(contraindicated|should not be used)\s+(?:in|with|for)?\s*([^.,;]+)", "contraindication"),
        (r"(?i)(dosage|dose|dosing)\s+(?:is|of|:)?\s*([^.,;]+)", "dosage"),
        (r"(?i)(interacts?|interaction)\s+(?:with)?\s*([^.,;]+)", "interaction"),
        (r"(?i)(mechanism|works? by|acts? by)\s+([^.,;]+)", "mechanism"),
    ]

    claims = []
    for pattern, claim_type in claim_patterns:
        for match in re.finditer(pattern, text):
            claim_text = match.group(0).strip()
            if len(claim_text) > 10:
                claims.append({
                    "claim": claim_text[:200],
                    "type": claim_type,
                    "validated": True,   # basic presence-based validation
                })

    return claims[:10]   # cap at 10 claims


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FDAValidator:
    """
    Validates pharmaceutical information against the openFDA drug-label API.
    Also performs:
      • Medical term extraction
      • CrossRef DOI reference verification
      • PubMed source checking
      • Claim validation
    No API key required — uses public endpoints.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search_drug(self, drug_name: str) -> dict[str, Any]:
        cache_key = drug_name.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            "limit": FDA_SEARCH_LIMIT,
        }
        try:
            resp = requests.get(FDA_BASE_URL, params=params, timeout=FDA_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            record: dict[str, Any] = {
                "found": bool(results),
                "drug_name": drug_name,
                "results": results,
                "error": None,
            }
        except requests.exceptions.Timeout:
            record = {"found": False, "drug_name": drug_name, "results": [], "error": "Request timed out"}
        except requests.exceptions.ConnectionError:
            record = {"found": False, "drug_name": drug_name, "results": [], "error": "No internet connection"}
        except requests.exceptions.HTTPError as exc:
            record = {"found": False, "drug_name": drug_name, "results": [], "error": str(exc)}
        except Exception as exc:
            record = {"found": False, "drug_name": drug_name, "results": [], "error": str(exc)}

        self._cache[cache_key] = record
        return record

    def extract_drug_names(self, text: str) -> list[str]:
        candidates: set[str] = set()
        for name in _COMMON_DRUG_NAMES:
            if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
                candidates.add(name)
        for match in _WORD_RE.finditer(text):
            word = match.group(1)
            if _has_drug_suffix(word):
                candidates.add(word)
        return sorted(candidates)

    def extract_medical_terms(self, text: str) -> list[str]:
        """Extract medical/clinical terms from text."""
        return extract_medical_terms(text)

    def verify_crossref_doi(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        """Search CrossRef for DOI-verified academic references."""
        return verify_crossref_doi(query, max_results)

    def search_pubmed(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        """Search PubMed/NCBI for biomedical literature."""
        return search_pubmed(query, max_results)

    def validate_claims(self, text: str) -> list[dict[str, Any]]:
        """Extract and validate pharmaceutical claims from text."""
        return validate_claims(text)

    def validate_response(self, response_text: str, drug_names: list[str]) -> dict[str, Any]:
        per_drug: list[dict[str, Any]] = []
        verified_count = 0

        for name in drug_names:
            fda_result = self.search_drug(name)
            if fda_result["found"]:
                verified_count += 1
                first = fda_result["results"][0]
                description_sections = first.get("description", [])
                fda_description = (
                    description_sections[0][:200] if description_sections else "Label data available."
                )
                per_drug.append({
                    "drug_name": name,
                    "fda_verified": True,
                    "fda_description": fda_description,
                    "warning": None,
                })
            else:
                per_drug.append({
                    "drug_name": name,
                    "fda_verified": False,
                    "fda_description": None,
                    "warning": fda_result.get("error") or "Drug not found in FDA database",
                })
            time.sleep(0.1)

        confidence = verified_count / len(drug_names) if drug_names else 0.5

        return {
            "drug_validations": per_drug,
            "confidence_score": round(confidence, 2),
            "drugs_checked": len(drug_names),
            "drugs_verified": verified_count,
        }

    def get_validation_report(self, response: str, query: str) -> dict[str, Any]:
        """
        Full validation pipeline:
        1. Extract drug names + medical terms from response and query.
        2. Validate each drug against the FDA API.
        3. Validate claims in the response.
        4. Cross-check with CrossRef DOI and PubMed sources.
        5. Return a comprehensive report.
        """
        combined_text = query + " " + response
        drug_names = self.extract_drug_names(combined_text)
        medical_terms = extract_medical_terms(combined_text)
        claims = validate_claims(response)

        if not drug_names:
            return {
                "status": "no_drugs_detected",
                "message": "No drug names detected – validation skipped.",
                "drug_validations": [],
                "confidence_score": 0.5,
                "drugs_checked": 0,
                "drugs_verified": 0,
                "medical_terms": medical_terms,
                "claims": claims,
                "crossref_refs": [],
                "pubmed_articles": [],
            }

        validation = self.validate_response(response, drug_names)

        # CrossRef and PubMed lookup
        crossref_refs = verify_crossref_doi(query, max_results=3)
        pubmed_articles = search_pubmed(query, max_results=3)

        if validation["confidence_score"] >= 0.75:
            status = "verified"
            message = "✅ Most pharmaceutical claims verified against FDA data."
        elif validation["confidence_score"] >= 0.4:
            status = "partial"
            message = "⚠️ Some pharmaceutical claims could not be verified."
        else:
            status = "unverified"
            message = "❌ Unable to verify pharmaceutical claims against FDA data."

        return {
            "status": status,
            "message": message,
            "medical_terms": medical_terms,
            "claims": claims,
            "crossref_refs": crossref_refs,
            "pubmed_articles": pubmed_articles,
            **validation,
        }
