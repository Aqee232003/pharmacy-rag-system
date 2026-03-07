# 🏥 Pharmacy-Specific RAG System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![BioBERT](https://img.shields.io/badge/BioBERT-dmis--lab%2Fbiobert--v1.1-green?logo=huggingface)](https://huggingface.co/dmis-lab/biobert-v1.1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aqee232003/pharmacy-rag-system/blob/main/Pharmacy_RAG_Colab.ipynb)

> AI-powered pharmaceutical decision-support tool that retrieves relevant scientific information and generates accurate, FDA-validated summaries.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Google Colab](#google-colab)
- [API Keys Setup](#api-keys-setup)
- [Usage](#usage)
- [Sample Queries](#sample-queries)
- [How It Works](#how-it-works)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## Overview

This project implements *Pharmacy-Specific Summarization Using Retrieval-Augmented Generation (RAG)* — a system designed to help pharmaceutical researchers quickly retrieve relevant scientific information and generate accurate, well-sourced summaries.

Key differentiators:
- **BioBERT embeddings** (pre-trained on PubMed + PMC) for superior biomedical semantic understanding
- **Dense Passage Retrieval (DPR)** for high-precision document retrieval
- **FDA validation layer** that cross-references answers against the openFDA API
- **Works without API keys** — built-in sample pharmacy knowledge for instant demos

---

## Architecture

```
                     ┌─────────────────────────┐
                     │       Streamlit UI       │
                     │  (pharmacy_rag_app.py)   │
                     └────────────┬────────────┘
                                  │ User Query
                                  ▼
                     ┌─────────────────────────┐
                     │   BioBERT Query Embed   │
                     │  dmis-lab/biobert-v1.1  │
                     └────────────┬────────────┘
                                  │ 768-dim vector
                                  ▼
                     ┌─────────────────────────┐
                     │   Pinecone / In-Memory  │
                     │    Vector DB Search     │
                     └────────────┬────────────┘
                                  │ Top-K chunks
                                  ▼
                     ┌─────────────────────────┐
                     │   BART Summarisation /  │
                     │   Extractive Fallback   │
                     └────────────┬────────────┘
                                  │ Generated answer
                                  ▼
                     ┌─────────────────────────┐
                     │   FDA Validation Layer  │
                     │    (openFDA REST API)   │
                     └────────────┬────────────┘
                                  │ Validated answer + confidence
                                  ▼
                     ┌─────────────────────────┐
                     │  Display + Citations    │
                     └─────────────────────────┘
```

---

## Features

- 🔬 **BioBERT Embeddings** — domain-specific biomedical language model for accurate pharmaceutical retrieval
- 🗄️ **Pinecone Integration** — scalable vector database with graceful in-memory fallback
- 📄 **PDF Upload** — drag-and-drop pharmaceutical documents (drug labels, research papers)
- ✅ **FDA Validation** — automatic cross-reference against the free openFDA drug-label API
- 💬 **Natural Language Answers** — BART-based abstractive summarisation with extractive fallback
- 📚 **Source Citations** — every answer linked to source documents with relevance scores
- 🖥️ **System Status Dashboard** — real-time component health monitoring
- 💡 **Sample Demo Questions** — one-click pharmacy-domain test queries
- 🌐 **Colab + ngrok** — zero-infrastructure deployment for demos and research

---

## Project Structure

```
pharmacy-rag-system/
├── pharmacy_rag_app.py          # 🌐 Main Streamlit application
├── rag_pipeline.py              # 🧬 Core RAG pipeline (BioBERT + Pinecone + BART)
├── document_processor.py        # 📄 PDF loading and chunking
├── fda_validation.py            # ✅ FDA drug-label validation
├── knowledge_base.py            # 📚 Knowledge base manager + sample data
├── config.py                    # ⚙️  All configuration constants
├── requirements.txt             # 📦 Python dependencies
├── .env.example                 # 🔑 Environment variable template
├── Pharmacy_RAG_Colab.ipynb     # 🧪 Complete Google Colab notebook
└── README.md                    # 📖 This file
```

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| UI Framework | Streamlit | ≥ 1.28 |
| Language Model | BioBERT | dmis-lab/biobert-v1.1 |
| Deep Learning | PyTorch + Transformers | ≥ 2.0 / ≥ 4.35 |
| Summarisation | facebook/bart-large-cnn | via HuggingFace |
| Vector Database | Pinecone | ≥ 3.0 |
| Document Loading | LangChain Community | ≥ 0.0.10 |
| Text Splitting | LangChain Text Splitters | ≥ 0.0.1 |
| FDA Validation | openFDA REST API | free, no key |
| PDF Parsing | pypdf / PyPDF2 | ≥ 3.17 |
| Tunnelling | pyngrok | ≥ 7.0 (Colab only) |

---

## Installation

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Aqee232003/pharmacy-rag-system.git
cd pharmacy-rag-system

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys (optional)
cp .env.example .env
# Edit .env and add your keys

# 5. Run the application
streamlit run pharmacy_rag_app.py
```

The app will open at **http://localhost:8501**.

### Google Colab

1. Open [Pharmacy_RAG_Colab.ipynb](Pharmacy_RAG_Colab.ipynb) in Google Colab
2. Enable GPU: *Runtime → Change runtime type → T4 GPU*
3. Run all cells in order
4. The ngrok tunnel URL will appear in Cell 6

---

## API Keys Setup

| Variable | Description | Required |
|---|---|---|
| `PINECONE_API_KEY` | Pinecone vector database | ❌ Optional (falls back to in-memory) |
| `PINECONE_ENVIRONMENT` | Pinecone region (e.g. `us-east-1`) | ❌ Optional |
| `HUGGINGFACE_TOKEN` | HuggingFace Hub access token | ❌ Optional (public models only) |
| `OPENAI_API_KEY` | OpenAI API (alternative generation) | ❌ Optional |
| `NGROK_AUTH_TOKEN` | ngrok tunnel (Colab only) | Required for Colab public URL |

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

> **The app works fully without any API keys** using an in-memory vector store and the built-in sample pharmacy knowledge base.

---

## Usage

### Running a Query

1. Launch the app (`streamlit run pharmacy_rag_app.py`)
2. Navigate to the **🔍 Query** tab
3. Type a pharmaceutical question or click a sample question
4. Click **🔍 Search**
5. View the generated answer, FDA validation status, and source citations

### Uploading Documents

1. Navigate to the **📄 Documents** tab
2. Drag and drop one or more PDF files (drug labels, research papers, clinical guidelines)
3. Wait for processing — chunk count and indexing status are shown
4. Return to the **Query** tab to search across uploaded content

### Adjusting Settings (Sidebar)

- **Top-K results** — number of document chunks retrieved per query
- **Show source chunks** — toggle visibility of retrieved passages
- **Run FDA validation** — enable/disable the openFDA cross-reference step

---

## Sample Queries

| Query | Topic |
|---|---|
| What are the side effects of metformin for type 2 diabetes? | Antidiabetics |
| How do ACE inhibitors work and what are their contraindications? | Cardiovascular |
| What drug interactions should I watch for with warfarin? | Drug Interactions |
| Explain the mechanism of action of proton pump inhibitors. | GI Drugs |
| What are the risks of long-term NSAID use like ibuprofen? | Analgesics |
| What is the antidote for acetaminophen overdose? | Toxicology |
| Describe the fluoroquinolone black-box warnings. | Antibiotics |
| How does BioBERT improve pharmaceutical information retrieval? | RAG Background |

---

## How It Works

1. **Ingest** — PDF documents are loaded via LangChain's `PyPDFLoader`, then split into 1 000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`.

2. **Embed** — Each chunk is encoded into a 768-dimensional vector using **BioBERT** (`dmis-lab/biobert-v1.1`) with mean-pooling over the token dimension (handles variable-length text, respects pharmaceutical terminology).

3. **Store** — Vectors + metadata are upserted into **Pinecone** (or an in-memory cosine-similarity store when Pinecone is unavailable).

4. **Retrieve** — The user's query is embedded with the same BioBERT model. **Dense Passage Retrieval** finds the top-K most similar chunks via cosine similarity.

5. **Generate** — Retrieved chunks are concatenated into a context window and passed to **BART** (`facebook/bart-large-cnn`) for abstractive summarisation. An extractive fallback is used if the model is unavailable.

6. **Validate** — Drug names are extracted from the answer heuristically and queried against the **openFDA** drug-label API. A confidence score (0–1) is computed based on how many names are verified.

7. **Display** — The answer, validation status, and source citations (with relevance scores) are rendered in the Streamlit UI.

---

## References

1. Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
2. Lee, J., Yoon, W., Kim, S., et al. (2020). *BioBERT: a pre-trained biomedical language representation model for biomedical text mining.* Bioinformatics, 36(4), 1234–1240.
3. Karpukhin, V., Oğuz, B., Min, S., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.
4. Lewis, M., Liu, Y., Goyal, N., et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL 2020.
5. U.S. Food & Drug Administration. *openFDA API Documentation.* https://open.fda.gov/apis/

---

## Author

**Aqee232003** — pharmaceutical AI researcher and developer.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.