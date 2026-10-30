import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Ollama (runs locally, no API key)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.45"))

# Local storage
UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_FILE = VECTOR_STORE_DIR / "knowledge_base.npz"

# Question -> response cache
CACHE_ENABLED = _bool(os.getenv("CACHE_ENABLED"), default=True)
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "168"))  # 0 = never expire
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "responses.json"

MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "50")) * 1024 * 1024

FLASK_DEBUG = _bool(os.getenv("FLASK_DEBUG"), default=False)