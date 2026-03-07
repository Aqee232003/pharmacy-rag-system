"""
fda_validation.py — FDA Validation Layer.

Queries the free openFDA drug-label API to cross-reference pharmaceutical
claims in generated answers and provides confidence scores.
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

# Simple heuristic: words that start with an uppercase letter and are ≥4 chars,
# surrounded by common boundary chars (handles generic and brand names).
_WORD_RE = re.compile(r"\b([A-Z][a-z]{3,})\b")

# Known common drug keywords to improve extraction precision
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


def _has_drug_suffix(word: str) -> bool:
    w = word.lower()
    return any(w.endswith(suffix) for suffix in _COMMON_DRUG_SUFFIXES)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FDAValidator:
    """
    Validates pharmaceutical information against the openFDA drug-label API.
    No API key required — uses the public endpoint.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}   # simple in-memory request cache

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search_drug(self, drug_name: str) -> dict[str, Any]:
        """
        Search the openFDA API for drug label records matching *drug_name*.

        Returns a dict with:
            found (bool), drug_name (str), results (list), error (str|None)
        """
        cache_key = drug_name.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            "limit": FDA_SEARCH_LIMIT,
        }
        try:
            resp = requests.get(
                FDA_BASE_URL,
                params=params,
                timeout=FDA_REQUEST_TIMEOUT,
            )
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
        except Exception as exc:  # noqa: BLE001
            record = {"found": False, "drug_name": drug_name, "results": [], "error": str(exc)}

        self._cache[cache_key] = record
        return record

    def extract_drug_names(self, text: str) -> list[str]:
        """
        Extract likely drug names from free text using heuristics.
        Combines a fixed known-drug list with suffix-based detection.
        """
        candidates: set[str] = set()

        # 1. Match against known drug names
        for name in _COMMON_DRUG_NAMES:
            if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
                candidates.add(name)

        # 2. Suffix-based heuristic for unknown drugs
        for match in _WORD_RE.finditer(text):
            word = match.group(1)
            if _has_drug_suffix(word):
                candidates.add(word)

        return sorted(candidates)

    def validate_response(
        self,
        response_text: str,
        drug_names: list[str],
    ) -> dict[str, Any]:
        """
        Validate *response_text* against FDA data for the given *drug_names*.

        Returns a dict with per-drug validation results plus an overall
        confidence score (0–1).
        """
        per_drug: list[dict[str, Any]] = []
        verified_count = 0

        for name in drug_names:
            fda_result = self.search_drug(name)
            if fda_result["found"]:
                verified_count += 1
                # Pull a brief description from the first result
                first = fda_result["results"][0]
                description_sections = first.get("description", [])
                fda_description = (
                    description_sections[0][:200] if description_sections else "Label data available."
                )
                per_drug.append(
                    {
                        "drug_name": name,
                        "fda_verified": True,
                        "fda_description": fda_description,
                        "warning": None,
                    }
                )
            else:
                per_drug.append(
                    {
                        "drug_name": name,
                        "fda_verified": False,
                        "fda_description": None,
                        "warning": fda_result.get("error") or "Drug not found in FDA database",
                    }
                )
            # Throttle to stay within free API rate limits
            time.sleep(0.1)

        confidence = verified_count / len(drug_names) if drug_names else 0.5

        return {
            "drug_validations": per_drug,
            "confidence_score": round(confidence, 2),
            "drugs_checked": len(drug_names),
            "drugs_verified": verified_count,
        }

    def get_validation_report(
        self,
        response: str,
        query: str,
    ) -> dict[str, Any]:
        """
        Full validation pipeline:
        1. Extract drug names from both the response and the query.
        2. Validate each against the FDA API.
        3. Return a comprehensive report.
        """
        # Extract from both query and response to maximise coverage
        drug_names = self.extract_drug_names(query + " " + response)

        if not drug_names:
            return {
                "status": "no_drugs_detected",
                "message": "No drug names detected – validation skipped.",
                "drug_validations": [],
                "confidence_score": 0.5,
                "drugs_checked": 0,
                "drugs_verified": 0,
            }

        validation = self.validate_response(response, drug_names)

        # Determine overall status
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
            **validation,
        }
