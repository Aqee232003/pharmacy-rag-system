"""
document_processor.py — Pharmaceutical document loading and chunking.

Loads PDFs, splits them into overlapping chunks, cleans pharmaceutical text,
and extracts metadata (page number, source file, section headers).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports – keep the module importable even when dependencies
# are only partially installed (e.g. during unit-testing without langchain).
# ---------------------------------------------------------------------------
try:
    from langchain_community.document_loaders import PyPDFLoader
    _PYPDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYPDF_AVAILABLE = False
    logger.warning("langchain_community not installed; PDF loading unavailable.")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _SPLITTER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPLITTER_AVAILABLE = False
    logger.warning("langchain_text_splitters not installed; using simple splitter fallback.")

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Section-header patterns common in pharmaceutical documents
_SECTION_PATTERNS = [
    re.compile(r"^\d+\.\s+[A-Z][A-Z\s]+$", re.MULTILINE),   # "1. INDICATIONS"
    re.compile(r"^[A-Z][A-Z\s]{4,}$", re.MULTILINE),         # "DOSAGE AND ADMINISTRATION"
]

_CHEMICAL_FORMULA = re.compile(r"\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+\b")
_EXTRA_WHITESPACE = re.compile(r"[ \t]{2,}")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


def _detect_section(text: str) -> str:
    """Return the first detected section header, or 'unknown'."""
    for pattern in _SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return "unknown"


# ---------------------------------------------------------------------------
# Simple fallback splitter (used when langchain_text_splitters is absent)
# ---------------------------------------------------------------------------

class _SimpleSplitter:
    """Naive fixed-size splitter – only used as a last resort."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Any]) -> list[Any]:
        from dataclasses import dataclass, field

        @dataclass
        class _FakeDoc:
            page_content: str
            metadata: dict = field(default_factory=dict)

        results: list[_FakeDoc] = []
        for doc in documents:
            text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                results.append(_FakeDoc(page_content=chunk, metadata=dict(getattr(doc, "metadata", {}))))
                start += self.chunk_size - self.chunk_overlap
        return results


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PharmacyDocumentProcessor:
    """
    Loads, cleans, and splits pharmaceutical PDF documents into chunks
    ready for embedding and storage.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if _SPLITTER_AVAILABLE:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            self._splitter = _SimpleSplitter(chunk_size, chunk_overlap)  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_pdf(self, file_path: str | Path) -> list[Any]:
        """
        Load a PDF and return a list of LangChain Document objects
        (one per page).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if not _PYPDF_AVAILABLE:
            logger.warning("PyPDFLoader unavailable – returning empty document list.")
            return []

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        logger.info("Loaded %d page(s) from %s", len(documents), file_path.name)
        return documents

    def clean_text(self, text: str) -> str:
        """
        Normalise and clean pharmaceutical text:
        - Unicode normalisation (NFKC)
        - Collapse excessive whitespace
        - Preserve chemical formulae and drug names
        """
        # Normalise unicode (handles ligatures, special dashes, etc.)
        text = unicodedata.normalize("NFKC", text)
        # Collapse multiple spaces/tabs
        text = _EXTRA_WHITESPACE.sub(" ", text)
        # Collapse excessive blank lines
        text = _MULTIPLE_NEWLINES.sub("\n\n", text)
        # Remove non-printable characters (but keep newlines / tabs)
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
        return text.strip()

    def split_documents(self, documents: list[Any]) -> list[dict[str, Any]]:
        """
        Split LangChain Document objects into chunks.

        Returns a list of dicts with keys:
            text, source, page, chunk_index, section
        """
        chunks: list[dict[str, Any]] = []

        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)

        raw_chunks = self._splitter.split_documents(documents)

        for idx, chunk in enumerate(raw_chunks):
            text = chunk.page_content.strip()
            if len(text) < MIN_CHUNK_LENGTH:
                continue

            metadata = chunk.metadata or {}
            chunks.append(
                {
                    "text": text,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", 0),
                    "chunk_index": idx,
                    "section": _detect_section(text),
                }
            )

        logger.info("Produced %d valid chunk(s)", len(chunks))
        return chunks

    def process_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        """
        Full pipeline: load PDF → clean → split → return chunks with metadata.
        """
        documents = self.load_pdf(file_path)
        if not documents:
            return []
        return self.split_documents(documents)

    def process_text(self, text: str, source: str = "inline") -> list[dict[str, Any]]:
        """
        Split a raw text string (no PDF file) into chunks.
        Useful for processing sample data.
        """
        from dataclasses import dataclass, field

        @dataclass
        class _FakeDoc:
            page_content: str
            metadata: dict = field(default_factory=dict)

        fake_doc = _FakeDoc(page_content=text, metadata={"source": source, "page": 0})
        return self.split_documents([fake_doc])
