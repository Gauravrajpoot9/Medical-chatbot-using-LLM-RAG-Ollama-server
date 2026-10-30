"""RAG generation step.

Retrieval (finding relevant chunks) lives in src/RAG.py.
This module takes those chunks, builds the prompt and asks the Ollama LLM
for an answer.
"""
import sys
from pathlib import Path

# Allow running this file directly (python src/chatbot.py)
# as well as importing it from app.py.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from langchain_ollama import ChatOllama

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    TOP_K,
    RELEVANCE_THRESHOLD,
)
from src.cache import response_cache
from src.RAG import get_retriever, Retriever, KnowledgeBaseNotReady  # noqa: F401 (re-exported for app.py)

NO_MATCH_MESSAGE = (
    "I could not find relevant information about this in the uploaded documents. "
    "Try rephrasing the question, or upload and train documents that cover this topic. "
    "For medical concerns, please consult a qualified healthcare professional."
)

# Change this whenever PROMPT_TEMPLATE changes, so cached answers made with
# the old prompt are no longer used.
PROMPT_VERSION = "3"

PROMPT_TEMPLATE = """
You are a medical information assistant for a document-based RAG chatbot.

Use ONLY the retrieved context below as your source of information.
Do not invent facts that are not supported by the context.

User question:
{query}

Retrieved context:
{context}

Instructions:
1. Answer clearly and simply.
2. Do not present a definitive diagnosis.
3. If the context supports possible causes or conditions, describe them as possibilities.
4. Give precautions only when supported by the context.
5. Do not prescribe medicines or dosages.
6. If symptoms could indicate an emergency, advise the user to seek urgent professional medical care.
7. If the context does not answer the question, say that the uploaded documents do not contain enough information instead of guessing.
8. Encourage consultation with a qualified healthcare professional for diagnosis or treatment.
9. Give only the answer. Do not list sources, file names, page numbers or references.

Answer:
"""

_llm = None


def reset():
    """Kept for app.py compatibility. The local store reloads itself
    automatically when the knowledge base file changes."""
    return None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )
    return _llm


def build_prompt(query, context):
    return PROMPT_TEMPLATE.format(query=query, context=context)


def generate_response(query):
    if not query or not query.strip():
        raise ValueError("Please enter a question.")

    query = query.strip()

    # 1. Retrieval
    chunks = get_retriever().retrieve(query)
    if not chunks:
        return NO_MATCH_MESSAGE, []

    # 2. Augmentation
    prompt = build_prompt(query, Retriever.format_context(chunks))

    # 3. Generation
    try:
        response = get_llm().invoke(prompt)
    except Exception as exc:
        raise RuntimeError(
            f"Ollama could not generate an answer with '{OLLAMA_LLM_MODEL}'. "
            f"Make sure the model is installed. Details: {exc}"
        ) from exc

    text = getattr(response, "content", str(response))
    return text, Retriever.to_sources(chunks)


def _cache_fingerprint(kb_info):
    """Everything that can change an answer. If any of it changes,
    previously cached answers stop matching."""
    return {
        "kb_created_at": kb_info.get("created_at"),
        "kb_count": kb_info.get("count"),
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "llm_model": OLLAMA_LLM_MODEL,
        "top_k": TOP_K,
        "threshold": RELEVANCE_THRESHOLD,
        "prompt_version": PROMPT_VERSION,
    }


def ask(query):
    """Answer a question, using the cache when possible.

    Returns {"response": str, "sources": list, "cached": bool}.
    """
    if not query or not query.strip():
        raise ValueError("Please enter a question.")
    query = query.strip()

    kb_info = get_retriever().check_ready()  # raises KnowledgeBaseNotReady
    fingerprint = _cache_fingerprint(kb_info)

    cached = response_cache.get(query, fingerprint)
    if cached is not None:
        return {"response": cached["response"], "sources": cached["sources"], "cached": True}

    answer, sources = generate_response(query)
    response_cache.set(query, fingerprint, answer, sources)
    return {"response": answer, "sources": sources, "cached": False}


if __name__ == "__main__":
    print("Medical chatbot started. Type 'exit' to stop.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        try:
            result = ask(query)
            print("\nAI:", result["response"])
            if result["cached"]:
                print("(answer from cache)")
        except Exception as exc:
            print("\nError:", exc)