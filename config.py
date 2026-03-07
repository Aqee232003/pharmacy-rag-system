"""
Configuration constants for the Pharmacy RAG System.
All model names, dimensions, API endpoints, and app settings in one place.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"

# ─── BioBERT / Embedding ──────────────────────────────────────────────────────
BIOBERT_MODEL_NAME = "dmis-lab/biobert-v1.1"
EMBEDDING_DIMENSION = 768        # BioBERT hidden size
MAX_SEQUENCE_LENGTH = 512        # BioBERT max tokens
EMBEDDING_BATCH_SIZE = 8

# ─── Document Processing ──────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 50            # Discard chunks shorter than this

# ─── Pinecone ─────────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "pharmacy-index"
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_NAMESPACE = "pharmacy-docs"

# ─── Metadata / display limits ────────────────────────────────────────────────
PINECONE_METADATA_TEXT_LIMIT = 1000  # Max characters stored in Pinecone metadata field
CHARS_PER_TOKEN = 4                  # Approximate character-to-token ratio for BART truncation
SOURCE_PREVIEW_LENGTH = 400          # Characters shown per source chunk in the UI

# ─── Retrieval ────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3       # Minimum cosine similarity to include a result

# ─── Summarisation / Generation ───────────────────────────────────────────────
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024          # Token budget fed to summariser
MAX_OUTPUT_TOKENS = 512
MIN_OUTPUT_TOKENS = 50

# ─── FDA / openFDA API ────────────────────────────────────────────────────────
FDA_BASE_URL = "https://api.fda.gov/drug/label.json"
FDA_SEARCH_LIMIT = 3
FDA_REQUEST_TIMEOUT = 10         # Seconds

# ─── Streamlit App ────────────────────────────────────────────────────────────
APP_TITLE = "🏥 Pharmacy RAG System"
APP_ICON = "💊"
APP_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# ─── Environment Variable Names ───────────────────────────────────────────────
ENV_PINECONE_API_KEY = "PINECONE_API_KEY"
ENV_PINECONE_ENVIRONMENT = "PINECONE_ENVIRONMENT"
ENV_HUGGINGFACE_TOKEN = "HUGGINGFACE_TOKEN"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_NGROK_AUTH_TOKEN = "NGROK_AUTH_TOKEN"

# ─── Helper: read keys from environment ───────────────────────────────────────

def get_pinecone_api_key() -> str:
    return os.environ.get(ENV_PINECONE_API_KEY, "")


def get_huggingface_token() -> str:
    return os.environ.get(ENV_HUGGINGFACE_TOKEN, "")


def get_ngrok_auth_token() -> str:
    return os.environ.get(ENV_NGROK_AUTH_TOKEN, "")


def pinecone_configured() -> bool:
    return bool(get_pinecone_api_key())
