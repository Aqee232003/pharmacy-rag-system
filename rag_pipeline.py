"""
rag_pipeline.py — Smart RAG Pipeline with PubMed Fallback & Auto KB Update.

Architecture:
  1. User Query → BioBERT Embedding (DPR)
  2. Semantic Search in Knowledge Base (Pinecone / In-Memory)
  3. Similarity Evaluation (threshold = 0.75)
     ├── HIGH SCORE → Use local KB chunks directly
     └── LOW SCORE  → Fallback to PubMed/NCBI
                         → Fetch + Summarize
                         → Store in KB (auto-learning)
                         → Return Answer
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

import numpy as np
import requests

from config import (
    BIOBERT_MODEL_NAME,
    CHARS_PER_TOKEN,
    DEFAULT_TOP_K,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_SEQUENCE_LENGTH,
    MIN_OUTPUT_TOKENS,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_METADATA_TEXT_LIMIT,
    PINECONE_METRIC,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
    SIMILARITY_THRESHOLD,
    SUMMARIZATION_MODEL,
    get_pinecone_api_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Similarity threshold for KB vs Fallback decision
# ---------------------------------------------------------------------------
KB_HIT_THRESHOLD    = 0.75   # score >= this → use KB directly
FALLBACK_THRESHOLD  = 0.50   # score < this  → trigger PubMed fallback

# ---------------------------------------------------------------------------
# PubMed / NCBI API endpoints
# ---------------------------------------------------------------------------
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMM_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# ---------------------------------------------------------------------------
# Optional ML imports
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch/transformers not installed — BioBERT unavailable.")

try:
    from transformers import pipeline as hf_pipeline
    _HF_PIPELINE_AVAILABLE = True
except ImportError:
    _HF_PIPELINE_AVAILABLE = False

try:
    from pinecone import Pinecone, ServerlessSpec
    _PINECONE_AVAILABLE = True
except ImportError:
    _PINECONE_AVAILABLE = False
    logger.warning("pinecone not installed — using in-memory store.")


# ---------------------------------------------------------------------------
# In-Memory Vector Store (fallback when Pinecone not available)
# ---------------------------------------------------------------------------

class _InMemoryVectorStore:
    def __init__(self) -> None:
        self._vectors: list[dict[str, Any]] = []

    def upsert(self, vectors: list[dict[str, Any]]) -> None:
        for v in vectors:
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
        scored = []
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
# PubMed / NCBI Fetcher
# ---------------------------------------------------------------------------

def _search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    try:
        resp = requests.get(PUBMED_SEARCH_URL, params={
            "db": "pubmed", "term": query,
            "retmax": max_results, "retmode": "json",
            "sort": "relevance",
        }, timeout=10)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning("PubMed search failed: %s", e)
        return []


def _fetch_pubmed_abstracts(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch full abstracts from PubMed for given PMIDs."""
    if not pmids:
        return []
    try:
        resp = requests.get(PUBMED_FETCH_URL, params={
            "db": "pubmed", "id": ",".join(pmids),
            "rettype": "abstract", "retmode": "text",
        }, timeout=10)
        resp.raise_for_status()
        raw_text = resp.text

        # Also get metadata (title, authors, year)
        summ_resp = requests.get(PUBMED_SUMM_URL, params={
            "db": "pubmed", "id": ",".join(pmids), "retmode": "json",
        }, timeout=10)
        summ_data = summ_resp.json().get("result", {})

        articles = []
        for pmid in pmids:
            item = summ_data.get(pmid, {})
            title   = item.get("title", "Unknown Title")
            authors = ", ".join(
                a.get("name", "") for a in item.get("authors", [])[:3]
            )
            year    = item.get("pubdate", "")[:4]
            journal = item.get("source", "PubMed")

            # Extract abstract for this PMID from raw text
            abstract = ""
            pattern = rf"PMID-\s*{pmid}.*?AB\s+-\s+(.*?)(?=PMID-|\Z)"
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                abstract = re.sub(r'\s+', ' ', match.group(1)).strip()[:1000]

            if not abstract:
                abstract = f"{title}. Published in {journal}, {year}."

            articles.append({
                "pmid":     pmid,
                "title":    title,
                "authors":  authors,
                "year":     year,
                "journal":  journal,
                "abstract": abstract,
                "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return articles
    except Exception as e:
        logger.warning("PubMed fetch failed: %s", e)
        return []


def _articles_to_chunks(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert PubMed articles to KB-compatible chunks."""
    chunks = []
    for article in articles:
        text = (
            f"Title: {article['title']}\n"
            f"Authors: {article['authors']} ({article['year']})\n"
            f"Journal: {article['journal']}\n"
            f"Abstract: {article['abstract']}\n"
            f"Source URL: {article['url']}"
        )
        chunks.append({
            "id":      f"pubmed_{article['pmid']}",
            "text":    text,
            "source":  f"PubMed:{article['pmid']}",
            "page":    0,
            "topic":   "pubmed_fetched",
            "pmid":    article["pmid"],
            "title":   article["title"],
            "year":    article["year"],
            "journal": article["journal"],
        })
    return chunks


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------

class PharmacyRAGPipeline:
    """
    Smart RAG Pipeline:
      Query → BioBERT Embed → KB Search → Similarity Check
        ├── Score >= 0.75 → Use KB directly
        └── Score < 0.75  → PubMed Fallback → Store in KB → Return
    """

    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._biobert:   Any = None
        self._summarizer: Any = None
        self._vector_store: Any = None
        self._use_pinecone = False
        self._device = "cpu"
        self._id_to_chunk: dict[str, dict[str, Any]] = {}

        # Track which queries already triggered PubMed (avoid duplicates)
        self._pubmed_cache: set[str] = set()

        self._init_device()
        self._init_biobert()
        self._init_vector_store()
        self._init_summarizer()

    # ── Init helpers ──────────────────────────────────────────────────────

    def _init_device(self) -> None:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            self._device = "cuda"
            logger.info("GPU detected — using CUDA.")
        else:
            self._device = "cpu"
            logger.info("No GPU — using CPU.")

    def _init_biobert(self) -> None:
        if not _TORCH_AVAILABLE:
            logger.warning("BioBERT unavailable — using fallback embeddings.")
            return
        try:
            logger.info("Loading BioBERT (%s)…", BIOBERT_MODEL_NAME)
            self._tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
            self._biobert   = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)
            self._biobert.to(self._device)
            self._biobert.eval()
            logger.info("BioBERT loaded.")
        except Exception as e:
            logger.warning("BioBERT load failed: %s", e)

    def _init_vector_store(self) -> None:
        api_key = get_pinecone_api_key()
        if _PINECONE_AVAILABLE and api_key:
            try:
                pc = Pinecone(api_key=api_key)
                existing = [idx.name for idx in pc.list_indexes()]
                if PINECONE_INDEX_NAME not in existing:
                    pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=EMBEDDING_DIMENSION,
                        metric=PINECONE_METRIC,
                        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                    )
                self._vector_store = pc.Index(PINECONE_INDEX_NAME)
                self._use_pinecone = True
                logger.info("Connected to Pinecone '%s'.", PINECONE_INDEX_NAME)
            except Exception as e:
                logger.warning("Pinecone init failed: %s — using in-memory.", e)
                self._vector_store = _InMemoryVectorStore()
        else:
            logger.info("No Pinecone API key — using in-memory store.")
            self._vector_store = _InMemoryVectorStore()

    def _init_summarizer(self) -> None:
        if not _HF_PIPELINE_AVAILABLE:
            return
        try:
            logger.info("Loading summarizer (%s)…", SUMMARIZATION_MODEL)
            self._summarizer = hf_pipeline(
                "summarization",
                model=SUMMARIZATION_MODEL,
                device=0 if self._device == "cuda" else -1,
            )
            logger.info("Summarizer loaded.")
        except Exception as e:
            logger.warning("Summarizer load failed: %s", e)

    # ── Embedding ─────────────────────────────────────────────────────────

    def create_embedding(self, text: str) -> list[float]:
        if self._biobert is None or self._tokenizer is None:
            return self._fallback_embedding(text)
        try:
            inputs = self._tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQUENCE_LENGTH, padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._biobert(**inputs)
            mask = inputs["attention_mask"]
            tok_emb = outputs.last_hidden_state
            mask_exp = mask.unsqueeze(-1).expand(tok_emb.size()).float()
            emb = (torch.sum(tok_emb * mask_exp, 1) /
                   torch.clamp(mask_exp.sum(1), min=1e-9)).squeeze().cpu().numpy()
            return emb.tolist()
        except Exception as e:
            logger.warning("Embedding error: %s", e)
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> list[float]:
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        vec = rng.standard_normal(EMBEDDING_DIMENSION).astype(np.float32)
        return (vec / (np.linalg.norm(vec) + 1e-10)).tolist()

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_documents(self, chunks: list[dict[str, Any]],
                        batch_size: int = EMBEDDING_BATCH_SIZE) -> int:
        if not chunks:
            return 0
        upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            vectors = []
            for chunk in batch:
                text = chunk.get("text", "")
                if not text:
                    continue
                chunk_id  = chunk.get("id") or str(uuid.uuid4())
                embedding = self.create_embedding(text)
                self._id_to_chunk[chunk_id] = chunk
                meta = {
                    k: v for k, v in chunk.items()
                    if k != "text" and isinstance(v, (str, int, float, bool))
                }
                meta["text"] = text[:PINECONE_METADATA_TEXT_LIMIT]
                vectors.append({"id": chunk_id, "values": embedding, "metadata": meta})

            if vectors:
                if self._use_pinecone:
                    self._vector_store.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
                else:
                    self._vector_store.upsert(vectors)
                upserted += len(vectors)

        logger.info("Indexed %d vectors.", upserted)
        return upserted

    # ── KB Retrieval (DPR) ────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Dense Passage Retrieval from local Knowledge Base."""
        query_emb = self.create_embedding(query)

        if self._use_pinecone:
            raw     = self._vector_store.query(
                namespace=PINECONE_NAMESPACE, vector=query_emb,
                top_k=top_k, include_metadata=True,
            )
            matches = raw.get("matches", [])
        else:
            matches = self._vector_store.query(query_emb, top_k=top_k)

        results = []
        for match in matches:
            score    = match.get("score", 0.0)
            chunk_id = match.get("id", "")

            if chunk_id in self._id_to_chunk:
                chunk = self._id_to_chunk[chunk_id]
                text  = chunk.get("text", "")
                meta  = {k: v for k, v in chunk.items() if k != "text"}
            else:
                meta = match.get("metadata", {})
                text = meta.pop("text", "")

            results.append({
                "text":        text,
                "score":       round(score, 4),
                "source":      meta.get("source", "unknown"),
                "page":        meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "section":     meta.get("section", "unknown"),
                "metadata":    meta,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ── PubMed Fallback ───────────────────────────────────────────────────

    def _pubmed_fallback(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Fallback: Search PubMed/NCBI → Fetch abstracts → Store in KB → Return chunks.
        This makes the system auto-learn new pharmaceutical knowledge.
        """
        cache_key = query.lower().strip()[:100]
        if cache_key in self._pubmed_cache:
            logger.info("PubMed cache hit — skipping re-fetch.")
            # Try KB again after previous fetch
            return self.retrieve(query, top_k=top_k)

        logger.info("🔍 KB miss — triggering PubMed fallback for: %s", query[:60])

        # Step 1: Search PubMed
        pmids = _search_pubmed(query, max_results=top_k)
        if not pmids:
            logger.warning("PubMed returned no results for query.")
            return []

        # Step 2: Fetch abstracts
        articles = _fetch_pubmed_abstracts(pmids)
        if not articles:
            return []

        # Step 3: Convert to chunks
        chunks = _articles_to_chunks(articles)

        # Step 4: Store in KB (auto-learning!)
        stored = self.index_documents(chunks)
        logger.info("✅ Stored %d PubMed chunks in KB.", stored)

        # Mark as cached
        self._pubmed_cache.add(cache_key)

        # Step 5: Now retrieve from KB (will find what we just stored)
        results = self.retrieve(query, top_k=top_k)

        # Tag results as PubMed-sourced
        for r in results:
            if r["source"].startswith("PubMed:"):
                r["from_pubmed"] = True
                r["pubmed_url"]  = f"https://pubmed.ncbi.nlm.nih.gov/{r['source'].split(':')[1]}/"

        return results

    # ── Answer Generation ─────────────────────────────────────────────────

    def generate_answer(self, query: str, context: str) -> str:
        if not context.strip():
            return (
                "No relevant information found in the knowledge base or PubMed. "
                "Please try rephrasing your question or upload pharmaceutical documents."
            )

        prompt = (
            f"Question: {query}\n\n"
            f"Relevant pharmaceutical information:\n{context}\n\n"
            "Answer:"
        )

        if self._summarizer is not None:
            try:
                truncated = prompt[:MAX_INPUT_TOKENS * CHARS_PER_TOKEN]
                result    = self._summarizer(
                    truncated,
                    max_length=MAX_OUTPUT_TOKENS,
                    min_length=MIN_OUTPUT_TOKENS,
                    do_sample=False,
                )
                return result[0]["summary_text"]
            except Exception as e:
                logger.warning("Summarizer error: %s — using extractive.", e)

        return self._extractive_answer(query, context)

    def _extractive_answer(self, query: str, context: str) -> str:
        sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 30]
        if not sentences:
            return context[:800]
        q_words = set(query.lower().split())
        scored  = sorted(
            [(len(q_words & set(s.lower().split())) / (len(q_words) + 1), s)
             for s in sentences],
            reverse=True
        )
        return ". ".join(s for _, s in scored[:5]) + "."

    # ── Main Query Entry Point ────────────────────────────────────────────

    def query(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """
        Full Smart RAG Pipeline:

        1. Embed query with BioBERT
        2. Search local Knowledge Base (DPR)
        3. Check similarity score:
           - Score >= 0.75 → Use KB directly ✅
           - Score < 0.75  → PubMed Fallback 🌐
               → Fetch + Store in KB
               → Retrieve again
        4. Generate structured answer
        5. Return answer + sources + metadata
        """
        # Step 1 & 2: KB search
        retrieved = self.retrieve(question, top_k=top_k)

        # Step 3: Similarity evaluation
        top_score     = retrieved[0]["score"] if retrieved else 0.0
        used_fallback = False
        fallback_count = 0

        if top_score < FALLBACK_THRESHOLD:
            # ── LOW SCORE → PubMed Fallback ──────────────────────────────
            logger.info("⚠️ Low KB score (%.3f) — triggering PubMed fallback.", top_score)
            pubmed_results = self._pubmed_fallback(question, top_k=top_k)
            if pubmed_results:
                retrieved     = pubmed_results
                used_fallback = True
                fallback_count = len(pubmed_results)
                logger.info("✅ PubMed fallback returned %d results.", fallback_count)
        elif top_score < KB_HIT_THRESHOLD:
            # ── MEDIUM SCORE → Merge KB + PubMed ─────────────────────────
            logger.info("⚡ Medium KB score (%.3f) — merging KB + PubMed.", top_score)
            pubmed_results = self._pubmed_fallback(question, top_k=3)
            if pubmed_results:
                retrieved     = retrieved + pubmed_results
                used_fallback = True
                fallback_count = len(pubmed_results)
                # Re-sort by score
                retrieved.sort(key=lambda x: x["score"], reverse=True)
                retrieved = retrieved[:top_k]
        else:
            logger.info("✅ Good KB hit (score=%.3f) — using local KB.", top_score)

        # Step 4: Build context
        if retrieved:
            context = "\n\n".join(
                f"[Source: {r['source']}, Score: {r['score']:.3f}]\n{r['text']}"
                for r in retrieved
            )
        else:
            context = ""

        # Step 5: Generate answer
        answer = self.generate_answer(question, context)

        return {
            "query":          question,
            "answer":         answer,
            "sources":        retrieved,
            "num_results":    len(retrieved),
            "top_score":      top_score,
            "used_fallback":  used_fallback,
            "fallback_count": fallback_count,
            "kb_hit":         top_score >= KB_HIT_THRESHOLD,
        }

    # ── Status ────────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        vector_count = 0
        try:
            stats = self._vector_store.describe_index_stats()
            if self._use_pinecone:
                ns = stats.get("namespaces", {}).get(PINECONE_NAMESPACE, {})
                vector_count = ns.get("vector_count", stats.get("total_vector_count", 0))
            else:
                vector_count = stats.get("total_vector_count", 0)
        except Exception:
            pass

        return {
            "biobert_loaded":    self._biobert is not None,
            "summarizer_loaded": self._summarizer is not None,
            "pinecone_connected": self._use_pinecone,
            "vector_store_type": "pinecone" if self._use_pinecone else "in-memory",
            "indexed_vectors":   vector_count,
            "device":            self._device,
            "pubmed_queries_cached": len(self._pubmed_cache),
        }
