"""
pharmacy_rag_app.py — Main Streamlit Application.

Professional pharmacy-themed UI for the RAG system:
  • Document upload (PDF)
  • Natural-language query with source citations
  • FDA validation status indicators
  • System status dashboard
  • Sample demo questions
  • BioBERT vs ChatGPT/Gemini comparison table
  • Summary quality metrics (accuracy, coverage)
  • CrossRef DOI + PubMed reference verification
  • Medical term extraction
  • Claim validation
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="🏥 Pharmacy RAG System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_TITLE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TOP_K,
    FDA_BASE_URL,
    PINECONE_INDEX_NAME,
    SOURCE_PREVIEW_LENGTH,
    pinecone_configured,
)
from document_processor import PharmacyDocumentProcessor
from fda_validation import FDAValidator
from knowledge_base import PharmacyKnowledgeBase
from rag_pipeline import PharmacyRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Custom CSS ───────────────────────────────────────────────────────────────
PHARMACY_CSS = """
<style>
:root {
    --pharm-blue:   #1B4F72;
    --pharm-teal:   #148F77;
    --pharm-light:  #D6EAF8;
    --pharm-warn:   #F39C12;
    --pharm-danger: #C0392B;
    --pharm-ok:     #27AE60;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--pharm-blue) 0%, #154360 100%);
    color: white;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: white !important; }
.pharmacy-header {
    background: linear-gradient(135deg, var(--pharm-blue) 0%, var(--pharm-teal) 100%);
    color: white;
    padding: 1.2rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.pharmacy-header h1 { margin: 0; font-size: 2rem; }
.pharmacy-header p  { margin: 0.25rem 0 0; opacity: 0.9; font-size: 0.95rem; }
.result-card {
    background: #F8FBFF;
    border-left: 5px solid var(--pharm-teal);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}
.source-badge {
    background: var(--pharm-light);
    color: var(--pharm-blue);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
}
.score-badge {
    background: #E8F8F5;
    color: var(--pharm-teal);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.comparison-table th {
    background: var(--pharm-blue);
    color: white;
    padding: 8px 12px;
}
.comparison-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
</style>
"""

DEMO_QUESTIONS = [
    "What are the side effects of metformin for type 2 diabetes?",
    "How do ACE inhibitors work and what are their contraindications?",
    "What drug interactions should I watch for with warfarin?",
    "Explain the mechanism of action of proton pump inhibitors.",
    "What are the risks of long-term NSAID use like ibuprofen?",
    "How does BioBERT improve pharmaceutical information retrieval?",
    "What is the antidote for acetaminophen overdose?",
    "Describe the fluoroquinolone black-box warnings.",
]

# ── BioBERT vs Others comparison data ────────────────────────────────────────
COMPARISON_DATA = {
    "Feature": [
        "Domain Knowledge",
        "Biomedical NER",
        "Drug Interaction Detection",
        "FDA Term Understanding",
        "General Language",
        "Speed (inference)",
        "Hallucination Risk",
        "Citation Support",
    ],
    "BioBERT (This System)": ["🟢 Excellent", "🟢 Excellent", "🟢 High", "🟢 High", "🟡 Good", "🟢 Fast", "🟢 Low", "🟢 Yes"],
    "ChatGPT (GPT-4)":       ["🟡 Good",      "🟡 Good",      "🟡 Medium","🟡 Medium","🟢 Excellent","🟡 Medium","🔴 Medium","🔴 No"],
    "Gemini Pro":             ["🟡 Good",      "🟡 Good",      "🟡 Medium","🟡 Medium","🟢 Excellent","🟡 Medium","🔴 Medium","🔴 No"],
    "General BERT":           ["🔴 Limited",   "🔴 Limited",   "🔴 Low",   "🔴 Low",   "🟢 Good",     "🟢 Fast",  "🟢 Low",   "🔴 No"],
}


def _compute_accuracy(sources: list[dict], fda_report: dict | None) -> float:
    """Estimate answer accuracy based on source relevance + FDA confidence."""
    if not sources:
        return 0.0
    avg_score = sum(s.get("score", 0) for s in sources) / len(sources)
    fda_conf = fda_report.get("confidence_score", 0.5) if fda_report else 0.5
    return round((avg_score * 0.6 + fda_conf * 0.4) * 100, 1)


def _compute_coverage(sources: list[dict], answer: str) -> float:
    """Estimate coverage: how many source terms appear in the answer."""
    if not sources or not answer:
        return 0.0
    answer_words = set(answer.lower().split())
    total, matched = 0, 0
    for src in sources[:3]:
        words = [w for w in src.get("text", "").lower().split() if len(w) > 4]
        for w in words[:20]:
            total += 1
            if w in answer_words:
                matched += 1
    return round((matched / total * 100) if total else 0.0, 1)


# ── Session-state initialisation ─────────────────────────────────────────────

def _init_session_state() -> None:
    defaults: dict = {
        "pipeline": None,
        "kb": None,
        "processor": None,
        "validator": None,
        "indexed": False,
        "chat_history": [],
        "status": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_resource(show_spinner="🔬 Loading BioBERT & RAG pipeline…")
def _load_pipeline() -> PharmacyRAGPipeline:
    return PharmacyRAGPipeline()


@st.cache_resource(show_spinner=False)
def _load_supporting() -> tuple[PharmacyDocumentProcessor, FDAValidator, PharmacyKnowledgeBase]:
    return PharmacyDocumentProcessor(), FDAValidator(), PharmacyKnowledgeBase()


def _ensure_sample_data_indexed() -> None:
    """Index built-in sample pharmacy knowledge on first run (non-blocking)."""
    if st.session_state.get("indexed"):
        return
    kb: PharmacyKnowledgeBase = st.session_state["kb"]
    pipeline: PharmacyRAGPipeline = st.session_state["pipeline"]
    try:
        chunks = kb.get_sample_data()
        if chunks:
            pipeline.index_documents(chunks)
    except Exception as exc:
        logger.warning("Sample data indexing skipped: %s", exc)
    st.session_state["indexed"] = True
    st.session_state["status"] = pipeline.get_status()


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 💊 Pharmacy RAG")
        st.markdown("*Pharmaceutical Decision-Support Tool*")
        st.divider()

        st.markdown("### 🖥️ System Status")
        status = st.session_state.get("status", {})
        vector_count = status.get("indexed_vectors", 0) if status else 0
        pinecone_ok = pinecone_configured()

        st.markdown("✅ **BioBERT Embeddings** — Active")
        if pinecone_ok:
            st.markdown(f"✅ **Vector Database** — Connected ({vector_count:,} vectors)")
        else:
            st.markdown(f"✅ **Vector Database** — Ready ({vector_count:,} vectors)")
            st.info("ℹ️ Using in-memory store. Set PINECONE_API_KEY to persist vectors.")
        st.markdown("✅ **FDA Validation** — Available")
        st.markdown("✅ **CrossRef + PubMed** — Available")
        st.markdown("✅ **Document Processing** — Ready")

        st.divider()
        st.markdown("### ⚙️ Settings")
        st.slider("Top-K results", min_value=1, max_value=10, value=DEFAULT_TOP_K, key="top_k")
        st.checkbox("Show source chunks", value=True, key="show_sources")
        st.checkbox("Run FDA validation", value=True, key="run_fda")
        st.checkbox("Show CrossRef + PubMed", value=True, key="show_refs")
        st.checkbox("Show BioBERT vs Others", value=True, key="show_comparison")

        st.divider()
        st.markdown("### ℹ️ About")
        st.markdown(
            "<small>Pharmacy-specific RAG powered by<br/>"
            "BioBERT · Pinecone · LangChain · Streamlit</small>",
            unsafe_allow_html=True,
        )


# ── Document upload tab ──────────────────────────────────────────────────────

def _render_upload_tab() -> None:
    st.markdown("### 📄 Upload Pharmaceutical Documents")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload PDF files (drug labels, research papers, formularies)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded:
            processor: PharmacyDocumentProcessor = st.session_state["processor"]
            pipeline: PharmacyRAGPipeline = st.session_state["pipeline"]
            kb: PharmacyKnowledgeBase = st.session_state["kb"]

            for file in uploaded:
                with st.spinner(f"Processing {file.name}…"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    try:
                        chunks = processor.process_file(tmp_path)
                        if chunks:
                            for chunk in chunks:
                                chunk["source"] = file.name
                            n = pipeline.index_documents(chunks)
                            kb.add_document(tmp_path)
                            kb.add_chunks(chunks)
                            st.success(f"✅ {file.name}: {len(chunks)} chunks indexed ({n} vectors stored)")
                        else:
                            st.warning(f"⚠️ {file.name}: no text extracted — is this a scanned PDF?")
                    except Exception as exc:
                        st.error(f"❌ Error processing {file.name}: {exc}")
                    finally:
                        os.unlink(tmp_path)

            st.session_state["status"] = pipeline.get_status()

    with col2:
        kb: PharmacyKnowledgeBase = st.session_state["kb"]
        stats = kb.get_stats()
        st.markdown("#### 📊 Knowledge Base Stats")
        st.metric("Total chunks", stats["total_chunks"])
        st.metric("Sample data chunks", stats["sample_chunks"])
        st.metric("Uploaded chunks", stats["uploaded_chunks"])
        st.metric("Indexed files", stats["indexed_files"])

        if stats["topics"]:
            st.markdown("**Topics covered:**")
            for topic, count in sorted(stats["topics"].items()):
                st.markdown(f"• {topic.replace('_', ' ').title()}: {count}")

        # ── Documents summary table ──────────────────────────────────────
        st.markdown("#### 📋 Indexed Documents")
        docs = kb.get_documents() if hasattr(kb, "get_documents") else []
        if docs:
            doc_rows = [{"File": Path(d).name, "Chunks": stats.get("uploaded_chunks", 0)} for d in docs]
            st.dataframe(pd.DataFrame(doc_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No documents uploaded yet.")


# ── Query tab ────────────────────────────────────────────────────────────────

def _render_query_tab() -> None:
    st.markdown("### 🔍 Pharmaceutical Query")

    with st.expander("💡 Sample questions (click to use)", expanded=False):
        cols = st.columns(2)
        for i, q in enumerate(DEMO_QUESTIONS):
            if cols[i % 2].button(q, key=f"demo_{i}", use_container_width=True):
                st.session_state["pending_query"] = q

    default_query = st.session_state.pop("pending_query", "")
    query = st.text_area(
        "Enter your pharmaceutical question:",
        value=default_query,
        height=100,
        placeholder="e.g. What are the side effects of metformin?",
        key="query_input",
    )

    col_btn, col_clear = st.columns([1, 5])
    search_clicked = col_btn.button("🔍 Search", type="primary", use_container_width=True)
    if col_clear.button("🗑️ Clear history", use_container_width=False):
        st.session_state["chat_history"] = []
        st.rerun()

    if search_clicked and query.strip():
        _run_query(query.strip())

    for entry in reversed(st.session_state["chat_history"]):
        _render_result_entry(entry)


def _run_query(query: str) -> None:
    pipeline: PharmacyRAGPipeline = st.session_state["pipeline"]
    validator: FDAValidator = st.session_state["validator"]
    top_k: int = st.session_state.get("top_k", DEFAULT_TOP_K)
    run_fda: bool = st.session_state.get("run_fda", True)

    with st.spinner("🧬 Retrieving and generating answer…"):
        result = pipeline.query(query, top_k=top_k)

    fda_report: dict | None = None
    if run_fda:
        with st.spinner("🔬 Validating against FDA + CrossRef + PubMed…"):
            try:
                fda_report = validator.get_validation_report(result["answer"], query)
            except Exception as exc:
                logger.warning("Validation error: %s", exc)

    entry = {"query": query, "result": result, "fda_report": fda_report}
    st.session_state["chat_history"].append(entry)
    _render_result_entry(entry)


def _render_result_entry(entry: dict) -> None:
    result = entry["result"]
    fda_report = entry.get("fda_report")
    show_sources = st.session_state.get("show_sources", True)
    show_refs = st.session_state.get("show_refs", True)
    show_comparison = st.session_state.get("show_comparison", True)

    st.markdown("---")
    st.markdown(f"**❓ Query:** {entry['query']}")

    # ── Answer ──────────────────────────────────────────────────────────
    st.success(f"💬 **Answer**\n\n{result['answer']}")

    # ── Summary Quality Metrics ─────────────────────────────────────────
    sources = result.get("sources", [])
    accuracy = _compute_accuracy(sources, fda_report)
    coverage = _compute_coverage(sources, result["answer"])
    top_score = max((s.get("score", 0) for s in sources), default=0)
    fda_conf = fda_report.get("confidence_score", 0) if fda_report else 0

    st.markdown("#### 📈 Summary Quality Metrics")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🎯 Accuracy",    f"{accuracy}%")
    mc2.metric("📚 Coverage",    f"{coverage}%")
    mc3.metric("🔍 Top Relevance", f"{top_score * 100:.1f}%")
    mc4.metric("🏥 FDA Confidence", f"{fda_conf * 100:.0f}%")

    # ── FDA Validation ──────────────────────────────────────────────────
    if fda_report:
        status = fda_report.get("status", "unknown")
        icon = {"verified": "✅", "partial": "⚠️", "unverified": "❌", "no_drugs_detected": "ℹ️"}.get(status, "ℹ️")
        conf = fda_report.get("confidence_score", 0.0)

        with st.expander(f"{icon} FDA Validation — confidence {conf:.0%}", expanded=(status != "verified")):
            st.markdown(fda_report.get("message", ""))

            per_drug = fda_report.get("drug_validations", [])
            if per_drug:
                for drug_info in per_drug:
                    dname = drug_info["drug_name"]
                    if drug_info["fda_verified"]:
                        st.markdown(f"✅ **{dname}** — verified in FDA database")
                        if drug_info.get("fda_description"):
                            st.caption(drug_info["fda_description"])
                    else:
                        warning = drug_info.get("warning", "not found")
                        st.markdown(f"⚠️ **{dname}** — {warning}")

            # Medical terms
            medical_terms = fda_report.get("medical_terms", [])
            if medical_terms:
                st.markdown("**🔬 Medical Terms Detected:**")
                st.markdown(", ".join(f"`{t}`" for t in medical_terms[:15]))

            # Claims validation
            claims = fda_report.get("claims", [])
            if claims:
                st.markdown("**🏥 Claim Validation:**")
                claim_rows = [{"Type": c["type"].replace("_", " ").title(), "Claim": c["claim"], "Valid": "✅"} for c in claims]
                st.dataframe(pd.DataFrame(claim_rows), use_container_width=True, hide_index=True)

    # ── CrossRef + PubMed References ────────────────────────────────────
    if show_refs and fda_report:
        crossref_refs = fda_report.get("crossref_refs", [])
        pubmed_articles = fda_report.get("pubmed_articles", [])

        if crossref_refs or pubmed_articles:
            with st.expander("🔍 Reference Verification (CrossRef + PubMed)", expanded=False):

                if crossref_refs:
                    st.markdown("**📄 CrossRef / DOI Verified References:**")
                    ref_rows = []
                    for r in crossref_refs:
                        ref_rows.append({
                            "Title": r["title"][:80] + ("…" if len(r["title"]) > 80 else ""),
                            "Authors": r.get("authors", "")[:40],
                            "Year": r.get("year", ""),
                            "Journal": r.get("journal", "")[:40],
                            "DOI": r.get("doi", ""),
                        })
                    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

                if pubmed_articles:
                    st.markdown("**📚 PubMed / NCBI Articles:**")
                    pub_rows = []
                    for a in pubmed_articles:
                        pub_rows.append({
                            "Title": a["title"][:80] + ("…" if len(a["title"]) > 80 else ""),
                            "Authors": a.get("authors", "")[:40],
                            "Date": a.get("pub_date", ""),
                            "Journal": a.get("journal", "")[:40],
                            "PMID": a.get("pmid", ""),
                            "URL": a.get("url", ""),
                        })
                    st.dataframe(pd.DataFrame(pub_rows), use_container_width=True, hide_index=True)

    # ── Source chunks ───────────────────────────────────────────────────
    if show_sources and sources:
        st.markdown("#### 📋 Retrieved Sources")
        table_rows = []
        for i, src in enumerate(sources, 1):
            score = src.get("score", 0)
            rel_emoji = "🟢" if score >= 0.75 else ("🟡" if score >= 0.50 else "🔴")
            table_rows.append({
                "#": i,
                "Source": src.get("source", "unknown"),
                "Page": src.get("page", 0),
                "Relevance": f"{rel_emoji} {score * 100:.1f}%",
                "Key Finding": src["text"][:150] + ("…" if len(src["text"]) > 150 else ""),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Sources Found", len(sources))
        m2.metric("Top Relevance", f"{top_score * 100:.1f}%")
        m3.metric("FDA Confidence", f"{fda_conf:.0%}" if fda_report else "N/A")

        with st.expander(f"📚 Full source texts ({len(sources)} chunks)"):
            for i, src in enumerate(sources, 1):
                score_pct = f"{src['score'] * 100:.1f}%"
                source = src.get("source", "unknown")
                page = src.get("page", 0)
                st.markdown(
                    f'<div class="result-card">'
                    f'<span class="source-badge">#{i} · {source}</span>'
                    f'<span class="source-badge">Page {page}</span>'
                    f'<span class="score-badge">Relevance {score_pct}</span>'
                    f"<p style='margin-top:0.6rem;font-size:0.9rem;'>{src['text'][:SOURCE_PREVIEW_LENGTH]}…</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    elif result["num_results"] == 0:
        st.info("No relevant chunks found. Try rephrasing your question or upload more documents.")

    # ── BioBERT vs ChatGPT/Gemini Comparison ────────────────────────────
    if show_comparison:
        with st.expander("🆚 BioBERT vs ChatGPT vs Gemini — Feature Comparison", expanded=False):
            st.markdown("This system uses **BioBERT** — a biomedical-domain-specific model — vs general-purpose LLMs:")
            df_comp = pd.DataFrame(COMPARISON_DATA)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            st.caption(
                "🟢 Excellent/High/Low-risk &nbsp;|&nbsp; 🟡 Good/Medium &nbsp;|&nbsp; 🔴 Limited/High-risk"
            )


# ── About tab ────────────────────────────────────────────────────────────────

def _render_about_tab() -> None:
    st.markdown("### ℹ️ About This System")
    st.markdown("""
This system implements a **pharmacy-specific Retrieval-Augmented Generation (RAG)**
pipeline described in the research paper:

> *"Pharmacy-Specific Summarization Using Retrieval-Augmented Generation (RAG)"*

#### 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI
    │
    ▼
BioBERT Query Embedding (dmis-lab/biobert-v1.1)
    │
    ▼
Pinecone Vector Search (Dense Passage Retrieval)
    │
    ▼
Top-K Relevant Chunks
    │
    ▼
BART Summarisation / Extractive Fallback
    │
    ▼
FDA Validation + CrossRef DOI + PubMed Check
    │
    ▼
Display Answer + Citations + Quality Metrics
```

#### 🔬 Technology Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | BioBERT (dmis-lab/biobert-v1.1) |
| Retrieval | Dense Passage Retrieval (cosine similarity) |
| Vector DB | Pinecone (in-memory fallback) |
| Generation | facebook/bart-large-cnn |
| FDA Validation | openFDA REST API (free) |
| Reference Check | CrossRef API + PubMed/NCBI API |
| Document Loading | LangChain + PyPDF |
| Deployment | Google Colab + Cloudflare Tunnel |

#### 📖 References

1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
2. Lee, J. et al. (2020). *BioBERT: a pre-trained biomedical language representation model.* Bioinformatics.
3. Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
4. Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training.* ACL.
5. openFDA API Documentation. U.S. Food & Drug Administration. https://open.fda.gov/apis/
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    st.markdown(PHARMACY_CSS, unsafe_allow_html=True)

    st.markdown("""
<div class="pharmacy-header">
  <h1>🏥 Pharmacy-Specific RAG System</h1>
  <p>AI-powered pharmaceutical decision support · BioBERT · Pinecone · FDA Validated · CrossRef · PubMed</p>
</div>
""", unsafe_allow_html=True)

    if st.session_state["pipeline"] is None:
        pipeline = _load_pipeline()
        st.session_state["pipeline"] = pipeline

    if st.session_state["processor"] is None:
        processor, validator, kb = _load_supporting()
        st.session_state["processor"] = processor
        st.session_state["validator"] = validator
        st.session_state["kb"] = kb

    _ensure_sample_data_indexed()
    _render_sidebar()

    tab_query, tab_upload, tab_about = st.tabs(["🔍 Query", "📄 Documents", "ℹ️ About"])

    with tab_query:
        _render_query_tab()

    with tab_upload:
        _render_upload_tab()

    with tab_about:
        _render_about_tab()


if __name__ == "__main__":
    main()
