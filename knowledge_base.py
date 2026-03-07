"""
knowledge_base.py — Pharmacy Knowledge Base Manager.

Manages the document collection, tracks indexed content, and supplies
sample pharmaceutical data so the demo works immediately — even without
external PDFs or Pinecone.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import UPLOAD_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample pharmaceutical knowledge (covers the most-requested demo topics)
# ---------------------------------------------------------------------------

SAMPLE_PHARMACY_DATA: list[dict[str, Any]] = [
    # ── Drug Interactions ──────────────────────────────────────────────────
    {
        "id": "pharm_001",
        "text": (
            "Drug interactions occur when one drug affects the activity of another. "
            "Warfarin (an anticoagulant) has many clinically significant interactions. "
            "Aspirin and NSAIDs increase bleeding risk when combined with warfarin by "
            "inhibiting platelet function and displacing warfarin from plasma proteins. "
            "Antibiotics such as metronidazole inhibit the CYP2C9 enzyme responsible for "
            "warfarin metabolism, causing elevated INR and bleeding risk. "
            "Patients on warfarin should be regularly monitored with INR blood tests."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "drug_interactions",
            "drug_names": ["Warfarin", "Aspirin", "Metronidazole"],
        },
    },
    {
        "id": "pharm_002",
        "text": (
            "Metformin is the first-line oral antidiabetic agent for type 2 diabetes mellitus. "
            "It works by reducing hepatic glucose production (gluconeogenesis), improving "
            "insulin sensitivity in peripheral tissues, and delaying intestinal glucose absorption. "
            "Common side effects include gastrointestinal upset — nausea, diarrhoea, and abdominal "
            "pain — particularly at initiation. Lactic acidosis is a rare but serious adverse event. "
            "Metformin is contraindicated in significant renal impairment (eGFR < 30 mL/min). "
            "It does not cause hypoglycaemia when used as monotherapy."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "antidiabetics",
            "drug_names": ["Metformin"],
        },
    },
    # ── Antibiotics ────────────────────────────────────────────────────────
    {
        "id": "pharm_003",
        "text": (
            "Amoxicillin is a broad-spectrum aminopenicillin antibiotic effective against many "
            "Gram-positive and selected Gram-negative bacteria. It inhibits bacterial cell-wall "
            "synthesis by binding penicillin-binding proteins (PBPs). Common uses include "
            "community-acquired pneumonia, urinary tract infections, and ear infections (otitis media). "
            "Beta-lactamase-producing organisms are resistant; amoxicillin is often combined with "
            "clavulanate (a beta-lactamase inhibitor) to overcome this. "
            "Hypersensitivity reactions range from mild rash to severe anaphylaxis."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "antibiotics",
            "drug_names": ["Amoxicillin"],
        },
    },
    {
        "id": "pharm_004",
        "text": (
            "Ciprofloxacin is a fluoroquinolone antibiotic that inhibits bacterial DNA gyrase and "
            "topoisomerase IV, enzymes required for DNA replication. It has broad-spectrum activity "
            "against Gram-negative organisms including Pseudomonas aeruginosa, and is used for "
            "urinary tract infections, respiratory infections, and travellers' diarrhoea. "
            "Fluoroquinolones carry an FDA black-box warning for tendon rupture and peripheral "
            "neuropathy. They should be avoided in children and pregnant women due to effects on "
            "developing cartilage. Concurrent antacid use reduces absorption via chelation."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "antibiotics",
            "drug_names": ["Ciprofloxacin"],
        },
    },
    # ── Cardiovascular ────────────────────────────────────────────────────
    {
        "id": "pharm_005",
        "text": (
            "Statins (HMG-CoA reductase inhibitors) are the cornerstone of LDL-cholesterol lowering "
            "therapy. Atorvastatin and rosuvastatin are high-intensity statins recommended in patients "
            "with cardiovascular disease or high 10-year cardiovascular risk. They reduce LDL-C by "
            "30–50% and also possess pleiotropic anti-inflammatory effects. Myopathy and, rarely, "
            "rhabdomyolysis are the most serious adverse effects, with risk increased by CYP3A4 "
            "inhibitors (e.g., clarithromycin, grapefruit juice). "
            "Liver enzymes should be checked before initiating therapy."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "cardiovascular",
            "drug_names": ["Atorvastatin"],
        },
    },
    {
        "id": "pharm_006",
        "text": (
            "ACE inhibitors (e.g., lisinopril, enalapril) block the conversion of angiotensin I to "
            "angiotensin II, reducing vasoconstriction and aldosterone secretion. They are first-line "
            "for hypertension, heart failure with reduced ejection fraction, and diabetic nephropathy. "
            "The characteristic side effect is a dry, persistent cough caused by bradykinin "
            "accumulation. Angioedema is a rare but potentially life-threatening reaction. "
            "ACE inhibitors are contraindicated in pregnancy (teratogenic) and bilateral renal artery "
            "stenosis. Potassium levels must be monitored — hyperkalaemia can occur."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "cardiovascular",
            "drug_names": ["Lisinopril"],
        },
    },
    # ── Pain Management ───────────────────────────────────────────────────
    {
        "id": "pharm_007",
        "text": (
            "Ibuprofen is a non-selective COX-1/COX-2 inhibitor (NSAID) used for pain, fever, and "
            "inflammation. It blocks prostaglandin synthesis, reducing sensitisation of pain receptors. "
            "GI side effects — dyspepsia, gastric ulceration — are common and can be mitigated by "
            "co-prescribing a proton pump inhibitor. Renal impairment can occur, particularly in "
            "volume-depleted patients or with concurrent ACE-inhibitor use (triple whammy). "
            "Cardiovascular risk (MI, stroke) is elevated with long-term use, particularly with "
            "higher doses. Avoid in patients with peptic ulcer disease, severe renal failure, or "
            "the third trimester of pregnancy."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "analgesics",
            "drug_names": ["Ibuprofen"],
        },
    },
    {
        "id": "pharm_008",
        "text": (
            "Acetaminophen (paracetamol) is one of the most widely used analgesics and antipyretics "
            "worldwide. Its exact mechanism is not fully elucidated but involves inhibition of central "
            "COX enzymes and modulation of serotonergic pathways. It lacks clinically significant "
            "anti-inflammatory activity and does not irritate the gastric mucosa, making it safer "
            "for patients with GI issues. The maximum recommended adult dose is 4 g/day; hepatotoxicity "
            "is the major risk with overdose, mediated by the reactive metabolite NAPQI. "
            "N-acetylcysteine (NAC) is the antidote for acetaminophen overdose."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "analgesics",
            "drug_names": ["Acetaminophen"],
        },
    },
    # ── RAG / BioBERT Background ───────────────────────────────────────────
    {
        "id": "pharm_009",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines information retrieval with neural text "
            "generation to produce factually grounded answers. A dense retriever (e.g., DPR) encodes "
            "the query and document chunks into vector space; semantically similar chunks are fetched "
            "and concatenated with the query before being passed to a generative model (e.g., BART). "
            "BioBERT (dmis-lab/biobert-v1.1) is pre-trained on PubMed abstracts and PMC full-text "
            "articles, giving it superior representation of biomedical terminology compared with "
            "general-purpose BERT. Using BioBERT embeddings improves retrieval accuracy for "
            "pharmaceutical and clinical queries."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "rag_background",
            "drug_names": [],
        },
    },
    {
        "id": "pharm_010",
        "text": (
            "Proton pump inhibitors (PPIs) such as omeprazole, pantoprazole, and esomeprazole "
            "irreversibly inhibit the H+/K+-ATPase pump in gastric parietal cells, reducing acid "
            "secretion by up to 95%. They are indicated for GERD, peptic ulcer disease, Helicobacter "
            "pylori eradication (in combination therapy), and NSAID-induced ulcer prophylaxis. "
            "Long-term use is associated with hypomagnesaemia, vitamin B12 deficiency, increased "
            "risk of Clostridioides difficile infection, and possible bone fracture risk. "
            "PPIs are prodrugs activated by the acid environment of the parietal cell canaliculus."
        ),
        "metadata": {
            "source": "sample_knowledge_base",
            "topic": "gastrointestinal",
            "drug_names": ["Omeprazole", "Pantoprazole", "Esomeprazole"],
        },
    },
]

# ---------------------------------------------------------------------------
# Knowledge Base class
# ---------------------------------------------------------------------------

class PharmacyKnowledgeBase:
    """
    Manages the pharmacy document collection.

    Tracks indexed documents, exposes sample data for demos, and provides
    statistics about the current state of the knowledge base.
    """

    def __init__(self) -> None:
        self._documents: list[dict[str, Any]] = []
        self._indexed_files: list[dict[str, Any]] = []
        self._created_at = datetime.now(timezone.utc)

        # Create upload directory if it doesn't exist
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_sample_data(self) -> list[dict[str, Any]]:
        """Return the built-in pharmacy knowledge chunks ready for indexing."""
        return list(SAMPLE_PHARMACY_DATA)

    def add_document(self, doc_path: str | Path) -> dict[str, Any]:
        """
        Register a document path as part of the knowledge base.
        (The caller is responsible for processing and embedding the file.)

        Returns a status dict.
        """
        doc_path = Path(doc_path)
        if not doc_path.exists():
            return {"success": False, "error": f"File not found: {doc_path}"}

        entry = {
            "path": str(doc_path),
            "name": doc_path.name,
            "size_kb": round(doc_path.stat().st_size / 1024, 1),
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        self._indexed_files.append(entry)
        logger.info("Registered document: %s", doc_path.name)
        return {"success": True, "document": entry}

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Append processed chunks to the in-memory document store."""
        self._documents.extend(chunks)

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the current knowledge base."""
        topics: dict[str, int] = {}
        for item in SAMPLE_PHARMACY_DATA:
            topic = item["metadata"].get("topic", "general")
            topics[topic] = topics.get(topic, 0) + 1

        for chunk in self._documents:
            topic = chunk.get("topic", "uploaded")
            topics[topic] = topics.get(topic, 0) + 1

        return {
            "total_chunks": len(SAMPLE_PHARMACY_DATA) + len(self._documents),
            "sample_chunks": len(SAMPLE_PHARMACY_DATA),
            "uploaded_chunks": len(self._documents),
            "indexed_files": len(self._indexed_files),
            "topics": topics,
            "created_at": self._created_at.isoformat(),
        }

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """Return sample data merged with any uploaded document chunks."""
        return self.get_sample_data() + list(self._documents)
