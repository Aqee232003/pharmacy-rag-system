"""
knowledge_base.py — Pharmacy Knowledge Base Manager.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import UPLOAD_DIR

logger = logging.getLogger(__name__)

# Sample data kept for reference but NOT indexed / counted
SAMPLE_PHARMACY_DATA: list[dict[str, Any]] = []


class PharmacyKnowledgeBase:
    def __init__(self) -> None:
        self._documents: list[dict[str, Any]] = []
        self._indexed_files: list[str] = []
        self._created_at = datetime.now(timezone.utc)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def get_sample_data(self) -> list[dict[str, Any]]:
        """Returns empty — no dummy data."""
        return []

    def add_document(self, doc_path: str | Path) -> dict[str, Any]:
        doc_path = Path(doc_path)
        if not doc_path.exists():
            return {"success": False, "error": f"File not found: {doc_path}"}
        self._indexed_files.append(doc_path.name)
        logger.info("Registered document: %s", doc_path.name)
        return {"success": True, "name": doc_path.name}

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self._documents.extend(chunks)

    def get_documents(self) -> list[str]:
        return list(self._indexed_files)

    def get_stats(self) -> dict[str, Any]:
        topics: dict[str, int] = {}
        for chunk in self._documents:
            topic = chunk.get("topic", "uploaded")
            topics[topic] = topics.get(topic, 0) + 1

        return {
            "total_chunks":    len(self._documents),
            "sample_chunks":   0,
            "uploaded_chunks": len(self._documents),
            "indexed_files":   len(self._indexed_files),
            "topics":          topics,
        }

    def get_all_chunks(self) -> list[dict[str, Any]]:
        return list(self._documents)
