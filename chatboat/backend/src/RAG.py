"""RAG retrieval step.

This module handles only the "R" in RAG:

    question -> Ollama embedding -> local vector search -> relevant chunks

It does not call the LLM. chatbot.py uses it to get context, and you can
also run it on its own to see which chunks a question retrieves:

    python src/RAG.py "What are the symptoms of anemia?"
"""
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# Allow running this file directly as well as importing it.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from langchain_ollama import OllamaEmbeddings

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    TOP_K,
    RELEVANCE_THRESHOLD,
)
from src import vector_store

NOT_TRAINED_MESSAGE = (
    "The knowledge base has not been trained yet. "
    "Upload documents and click Train Knowledge Base first."
)


class KnowledgeBaseNotReady(RuntimeError):
    """Raised when there is no usable knowledge base to search."""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: object
    score: float

    def to_source(self):
        """Citation info sent to the UI (without the full text)."""
        return {"source": self.source, "page": self.page, "score": round(self.score, 4)}


class Retriever:
    def __init__(
        self,
        top_k=TOP_K,
        threshold=RELEVANCE_THRESHOLD,
        embedding_model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    ):
        self.top_k = top_k
        self.threshold = threshold
        self.embedding_model = embedding_model
        self.base_url = base_url
        self._embeddings = None

    # ---- setup -----------------------------------------------------------

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url,
            )
        return self._embeddings

    def check_ready(self):
        """Make sure a knowledge base exists and matches the embedding model."""
        loaded = vector_store.load_store()
        if loaded is None:
            raise KnowledgeBaseNotReady(NOT_TRAINED_MESSAGE)

        _, _, info = loaded
        trained_model = info.get("embedding_model")
        if trained_model and trained_model != self.embedding_model:
            raise KnowledgeBaseNotReady(
                f"The knowledge base was trained with '{trained_model}', but "
                f"OLLAMA_EMBEDDING_MODEL is now '{self.embedding_model}'. "
                f"Click Train Knowledge Base to rebuild it."
            )
        return info

    # ---- retrieval -------------------------------------------------------

    def embed_query(self, query):
        try:
            return self._get_embeddings().embed_query(query)
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Make sure Ollama is running "
                f"and '{self.embedding_model}' is installed. Details: {exc}"
            ) from exc

    def retrieve(self, query, top_k=None, threshold=None):
        """Return the most relevant chunks for query, best first.

        Chunks scoring below the threshold are dropped, so an empty list
        means the documents don't cover the question.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("Please enter a question.")

        self.check_ready()
        vector = self.embed_query(query)

        results = vector_store.search(
            vector,
            top_k=top_k if top_k is not None else self.top_k,
            threshold=threshold if threshold is not None else self.threshold,
        )

        chunks = []
        seen = set()
        for score, metadata in results:
            text = str(metadata.get("text", "")).strip()
            if not text:
                continue

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "?")

            # Skip exact duplicates (same text on the same page).
            key = (source, page, text)
            if key in seen:
                continue
            seen.add(key)

            chunks.append(RetrievedChunk(text=text, source=source, page=page, score=score))

        return chunks

    # ---- helpers for the generation step --------------------------------

    @staticmethod
    def format_context(chunks):
        """Join chunks into the context block that goes into the LLM prompt."""
        return "\n\n---\n\n".join(
            f"Source: {c.source}, Page: {c.page}\n{c.text}" for c in chunks
        )

    @staticmethod
    def to_sources(chunks):
        return [c.to_source() for c in chunks]


# Shared instance used by chatbot.py.
_default_retriever = None


def get_retriever():
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever


def retrieve(query, top_k=None, threshold=None):
    """Shortcut: retrieve chunks with the default settings from .env."""
    return get_retriever().retrieve(query, top_k=top_k, threshold=threshold)


if __name__ == "__main__":
    # Test retrieval on its own, without the LLM. Useful for tuning
    # RELEVANCE_THRESHOLD: threshold 0 here shows every top match's score.
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    retriever = Retriever()

    try:
        chunks = retriever.retrieve(question, threshold=0.0)
    except Exception as exc:
        print("Error:", exc)
        sys.exit(1)

    if not chunks:
        print("No chunks found.")
        sys.exit(0)

    print(f"\nTop {len(chunks)} chunks (threshold in .env: {retriever.threshold}):\n")
    for i, c in enumerate(chunks, start=1):
        mark = "USED   " if c.score >= retriever.threshold else "IGNORED"
        preview = c.text.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"{i}. [{mark}] score={c.score:.4f}  {c.source} (page {c.page})")
        print(f"   {preview}\n")