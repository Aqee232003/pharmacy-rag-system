"""
rag_pipeline.py — Core RAG Pipeline.

Components:
  • BioBERT embeddings (dmis-lab/biobert-v1.1) with mean-pooling
  • Pinecone vector database (optional — falls back to in-memory search)
  • Dense Passage Retrieval (DPR) — cosine-similarity ranking
  • Answer generation via HuggingFace BART summariser (with extractive fallback)

The class ``PharmacyRAGPipeline`` orchestrates the full
  Embed → Index → Retrieve → Generate
pipeline.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import numpy as np

from config import (
    BIOBERT_MODEL_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_SEQUENCE_LENGTH,
    MIN_OUTPUT_TOKENS,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
    SIMILARITY_THRESHOLD,
    SUMMARIZATION_MODEL,
    get_pinecone_api_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — handled gracefully so the module can be imported even
# in environments where heavy ML packages are not installed.
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("torch / transformers not installed — BioBERT unavailable.")

try:
    from transformers import pipeline as hf_pipeline

    _HF_PIPELINE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HF_PIPELINE_AVAILABLE = False

try:
    from pinecone import Pinecone, ServerlessSpec

    _PINECONE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PINECONE_AVAILABLE = False
    logger.warning("pinecone-client not installed — vector DB unavailable.")


# ---------------------------------------------------------------------------
# Fallback: simple in-memory vector store
# ---------------------------------------------------------------------------

class _InMemoryVectorStore:
    """Cosine-similarity vector store used when Pinecone is unavailable."""

    def __init__(self) -> None:
        self._vectors: list[dict[str, Any]] = []   # {id, values, metadata}

    def upsert(self, vectors: list[dict[str, Any]]) -> None:
        for v in vectors:
            # Replace existing id if present
            existing = next((i for i, x in enumerate(self._vectors) if x["id"] == v["id"]), None)
            if existing is not None:
                self._vectors[existing] = v
            else:
                self._vectors.append(v)

    def query(self, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if not self._vectors:
            return []
        q = np.array(vector, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._vectors:
            v = np.array(item["values"], dtype=np.float32)
            v_norm = v / (np.linalg.norm(v) + 1e-10)
            score = float(np.dot(q_norm, v_norm))
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": item["id"], "score": score, "metadata": item.get("metadata", {})}
            for score, item in scored[:top_k]
        ]

    def describe_index_stats(self) -> dict[str, Any]:
        return {"total_vector_count": len(self._vectors)}


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class PharmacyRAGPipeline:
    """
    End-to-end RAG pipeline for pharmaceutical query answering.

    Usage::

        pipeline = PharmacyRAGPipeline()
        pipeline.index_documents(chunks, metadatas)
        result = pipeline.query("What are the side effects of metformin?")
    """

    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._biobert: Any = None
        self._summarizer: Any = None
        self._vector_store: Any = None   # Pinecone index or _InMemoryVectorStore
        self._use_pinecone = False
        self._device = "cpu"

        # Cached chunk text for retrieval (keyed by vector id)
        self._id_to_chunk: dict[str, dict[str, Any]] = {}

        self._init_device()
        self._init_biobert()
        self._init_vector_store()
        self._init_summarizer()

    # ------------------------------------------------------------------ #
    # Initialisation helpers                                               #
    # ------------------------------------------------------------------ #

    def _init_device(self) -> None:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            self._device = "cuda"
            logger.info("GPU detected — using CUDA for BioBERT.")
        else:
            self._device = "cpu"
            logger.info("No GPU detected — using CPU for BioBERT.")

    def _init_biobert(self) -> None:
        if not _TORCH_AVAILABLE:
            logger.warning("BioBERT unavailable — embeddings will be random vectors.")
            return
        try:
            logger.info("Loading BioBERT tokenizer and model (%s)…", BIOBERT_MODEL_NAME)
            self._tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
            self._biobert = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)
            self._biobert.to(self._device)
            self._biobert.eval()
            logger.info("BioBERT loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load BioBERT: %s — falling back to random embeddings.", exc)

    def _init_vector_store(self) -> None:
        api_key = get_pinecone_api_key()
        if _PINECONE_AVAILABLE and api_key:
            try:
                pc = Pinecone(api_key=api_key)
                existing = [idx.name for idx in pc.list_indexes()]
                if PINECONE_INDEX_NAME not in existing:
                    logger.info("Creating Pinecone index '%s'…", PINECONE_INDEX_NAME)
                    pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=EMBEDDING_DIMENSION,
                        metric=PINECONE_METRIC,
                        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                    )
                self._vector_store = pc.Index(PINECONE_INDEX_NAME)
                self._use_pinecone = True
                logger.info("Connected to Pinecone index '%s'.", PINECONE_INDEX_NAME)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pinecone init failed: %s — using in-memory store.", exc)
                self._vector_store = _InMemoryVectorStore()
        else:
            if not api_key:
                logger.info("PINECONE_API_KEY not set — using in-memory vector store.")
            self._vector_store = _InMemoryVectorStore()

    def _init_summarizer(self) -> None:
        if not _HF_PIPELINE_AVAILABLE:
            logger.warning("HuggingFace pipeline unavailable — using extractive summarisation.")
            return
        try:
            logger.info("Loading summarisation model (%s)…", SUMMARIZATION_MODEL)
            self._summarizer = hf_pipeline(
                "summarization",
                model=SUMMARIZATION_MODEL,
                device=0 if self._device == "cuda" else -1,
            )
            logger.info("Summarisation model loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load summariser: %s — using extractive fallback.", exc)

    # ------------------------------------------------------------------ #
    # Embedding                                                            #
    # ------------------------------------------------------------------ #

    def create_embedding(self, text: str) -> list[float]:
        """
        Compute a BioBERT embedding for *text* using mean-pooling.
        Falls back to a deterministic pseudo-random vector if BioBERT is
        unavailable (useful for testing without GPU/internet).
        """
        if self._biobert is None or self._tokenizer is None:
            return self._fallback_embedding(text)

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._biobert(**inputs)

            # Mean-pool over the token dimension (ignore padding via attention mask)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()

            return embedding.tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding error: %s — using fallback.", exc)
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> list[float]:
        """
        Deterministic pseudo-random embedding (for demo / offline use).
        Uses a hash of the text so identical queries return the same vector.
        """
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        vec = rng.standard_normal(EMBEDDING_DIMENSION).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.tolist()

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def index_documents(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> int:
        """
        Embed and store document chunks in the vector store.

        *chunks* should be a list of dicts with at least a ``text`` key.
        Returns the number of vectors successfully upserted.
        """
        if not chunks:
            return 0

        upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors: list[dict[str, Any]] = []

            for chunk in batch:
                text = chunk.get("text", "")
                if not text:
                    continue
                chunk_id = chunk.get("id") or str(uuid.uuid4())
                embedding = self.create_embedding(text)

                # Store chunk text for later retrieval
                self._id_to_chunk[chunk_id] = chunk

                meta: dict[str, Any] = {
                    k: v
                    for k, v in chunk.items()
                    if k not in ("text",) and isinstance(v, (str, int, float, bool))
                }
                meta["text"] = text[:1000]   # Pinecone metadata value limit

                vectors.append({"id": chunk_id, "values": embedding, "metadata": meta})

            if vectors:
                if self._use_pinecone:
                    self._vector_store.upsert(
                        vectors=vectors,
                        namespace=PINECONE_NAMESPACE,
                    )
                else:
                    self._vector_store.upsert(vectors)
                upserted += len(vectors)

        logger.info("Indexed %d vector(s).", upserted)
        return upserted

    # ------------------------------------------------------------------ #
    # Retrieval (DPR)                                                      #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Dense Passage Retrieval: embed the query and find the most similar
        document chunks.

        Returns a list of dicts with keys:
            text, score, source, page, chunk_index, section
        """
        query_embedding = self.create_embedding(query)

        if self._use_pinecone:
            raw = self._vector_store.query(
                namespace=PINECONE_NAMESPACE,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )
            matches = raw.get("matches", [])
        else:
            matches = self._vector_store.query(query_embedding, top_k=top_k)

        results: list[dict[str, Any]] = []
        for match in matches:
            score = match.get("score", 0.0)
            if score < SIMILARITY_THRESHOLD:
                continue

            chunk_id = match.get("id", "")
            # Prefer the full text from our local cache
            if chunk_id in self._id_to_chunk:
                chunk = self._id_to_chunk[chunk_id]
                text = chunk.get("text", "")
                meta = {k: v for k, v in chunk.items() if k != "text"}
            else:
                meta = match.get("metadata", {})
                text = meta.pop("text", "")

            results.append(
                {
                    "text": text,
                    "score": round(score, 4),
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "section": meta.get("section", "unknown"),
                    "metadata": meta,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Answer generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate a natural-language answer from the retrieved *context*.

        Uses BART summarisation if available; otherwise falls back to
        an extractive approach (returns the most relevant context sentences).
        """
        if not context.strip():
            return (
                "I could not find relevant information in the knowledge base to "
                "answer your question. Please try rephrasing or upload additional "
                "pharmaceutical documents."
            )

        prompt = (
            f"Question: {query}\n\n"
            f"Relevant pharmaceutical information:\n{context}\n\n"
            "Answer:"
        )

        if self._summarizer is not None:
            try:
                # Truncate to model max tokens
                truncated = prompt[:MAX_INPUT_TOKENS * 4]   # rough char estimate
                result = self._summarizer(
                    truncated,
                    max_length=MAX_OUTPUT_TOKENS,
                    min_length=MIN_OUTPUT_TOKENS,
                    do_sample=False,
                )
                return result[0]["summary_text"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Summariser error: %s — using extractive fallback.", exc)

        # Extractive fallback: pick top sentences from context
        return self._extractive_answer(query, context)

    def _extractive_answer(self, query: str, context: str) -> str:
        """Return the most relevant sentences from context (extractive)."""
        sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 30]
        if not sentences:
            return context[:800]

        q_words = set(query.lower().split())
        scored = []
        for sent in sentences:
            s_words = set(sent.lower().split())
            overlap = len(q_words & s_words) / (len(q_words) + 1)
            scored.append((overlap, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:5]]
        return ". ".join(top) + "."

    # ------------------------------------------------------------------ #
    # Full pipeline                                                        #
    # ------------------------------------------------------------------ #

    def query(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """
        Full RAG pipeline: retrieve → generate.

        Returns a dict with:
            answer (str), sources (list), num_results (int), query (str)
        """
        retrieved = self.retrieve(question, top_k=top_k)

        if not retrieved:
            context = ""
        else:
            context = "\n\n".join(
                f"[Source: {r['source']}, Score: {r['score']:.3f}]\n{r['text']}"
                for r in retrieved
            )

        answer = self.generate_answer(question, context)

        return {
            "query": question,
            "answer": answer,
            "sources": retrieved,
            "num_results": len(retrieved),
        }

    # ------------------------------------------------------------------ #
    # Status                                                               #
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict[str, Any]:
        """Return a health-check dict for the system status dashboard."""
        vector_count = 0
        try:
            stats = self._vector_store.describe_index_stats()
            if self._use_pinecone:
                namespaces = stats.get("namespaces", {})
                ns = namespaces.get(PINECONE_NAMESPACE, {})
                vector_count = ns.get("vector_count", stats.get("total_vector_count", 0))
            else:
                vector_count = stats.get("total_vector_count", 0)
        except Exception:  # noqa: BLE001
            pass

        return {
            "biobert_loaded": self._biobert is not None,
            "summarizer_loaded": self._summarizer is not None,
            "pinecone_connected": self._use_pinecone,
            "vector_store_type": "pinecone" if self._use_pinecone else "in-memory",
            "indexed_vectors": vector_count,
            "device": self._device,
        }
