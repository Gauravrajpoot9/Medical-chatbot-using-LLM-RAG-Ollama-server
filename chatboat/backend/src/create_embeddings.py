import logging
import sys
import threading
from pathlib import Path

# Allow running this file directly (python src/create_embeddings.py)
# as well as importing it from app.py.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    UPLOAD_FOLDER,
    VECTOR_STORE_FILE,
)
from src.vector_store import save_store

# pypdf prints long font warnings (e.g. "fontTools is required to fully
# parse the encoding of a CFF Type1 font") for some PDFs. They are not
# errors, so only show real errors in the terminal.
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPORTED = {".pdf", ".txt"}
MIN_PAGE_CHARS = 20
BATCH_SIZE = 50

_training_lock = threading.Lock()


def get_embeddings():
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    try:
        test = embeddings.embed_query("medical chatbot test")
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL} with model "
            f"'{OLLAMA_EMBEDDING_MODEL}'. Make sure Ollama is running and run "
            f"'ollama pull {OLLAMA_EMBEDDING_MODEL}'. Details: {exc}"
        ) from exc

    if len(test) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Embedding dimension mismatch. Ollama returned {len(test)}, "
            f"but EMBEDDING_DIMENSION={EMBEDDING_DIMENSION}. "
            f"Update EMBEDDING_DIMENSION in .env to {len(test)}."
        )
    return embeddings


def load_uploaded_documents():
    """Load text from uploads.

    Returns (documents, files, skipped_pages, unreadable_files) where
    skipped_pages lists PDF pages with no extractable text (usually scanned
    images) and unreadable_files lists files that produced no text at all.
    """
    documents = []
    skipped_pages = []
    unreadable_files = []

    files = sorted(
        p for p in UPLOAD_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    if not files:
        raise RuntimeError("No PDF or TXT files found. Upload documents first.")

    for path in files:
        file_docs = []

        if path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(path))
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        pass
                pages = list(reader.pages)
            except Exception as exc:
                unreadable_files.append({
                    "source": path.name,
                    "reason": f"Could not open PDF ({exc}).",
                })
                continue

            for page_no, page in enumerate(pages, start=1):
                try:
                    text = (page.extract_text() or "").strip()
                except Exception:
                    text = ""
                if len(text) < MIN_PAGE_CHARS:
                    skipped_pages.append({"source": path.name, "page": page_no})
                    continue
                file_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_no},
                    )
                )

            if not file_docs:
                unreadable_files.append({
                    "source": path.name,
                    "reason": "No text layer found. The PDF is probably scanned; "
                              "run OCR on it before uploading.",
                })
        else:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                file_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": 1},
                    )
                )
            else:
                unreadable_files.append({
                    "source": path.name,
                    "reason": "The file is empty.",
                })

        documents.extend(file_docs)

    if not documents:
        details = "; ".join(f"{u['source']}: {u['reason']}" for u in unreadable_files)
        raise RuntimeError(
            "No readable text was found in the uploaded documents. " + details
        )

    return documents, files, skipped_pages, unreadable_files


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _noop_progress(stage, current, total, message):
    pass


def train(progress=None):
    """Build the local knowledge base.

    progress(stage, current, total, message) is called as training moves
    through the stages: reading, chunking, embedding, saving.
    """
    progress = progress or _noop_progress

    if not _training_lock.acquire(blocking=False):
        raise RuntimeError("Training is already running. Please wait for it to finish.")

    try:
        progress("reading", 0, 1, "Connecting to Ollama")
        embeddings = get_embeddings()

        progress("reading", 0, 1, "Reading uploaded documents")
        documents, source_files, skipped_pages, unreadable_files = load_uploaded_documents()
        progress("reading", 1, 1, f"Read {len(source_files)} document(s)")

        progress("chunking", 0, 1, "Splitting text into chunks")
        chunks = create_chunks(documents)
        total = len(chunks)
        progress("chunking", 1, 1, f"Created {total} chunks")
        progress("embedding", 0, total, f"Creating embeddings: 0 of {total} chunks")

        all_vectors = []
        all_metadata = []

        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            texts = [d.page_content for d in batch]
            try:
                vectors = embeddings.embed_documents(texts)
            except Exception as exc:
                raise RuntimeError(
                    f"Ollama failed while embedding chunks {start + 1}-{start + len(batch)}. "
                    f"Details: {exc}"
                ) from exc

            for doc, vector in zip(batch, vectors):
                all_vectors.append(vector)
                all_metadata.append({
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": int(doc.metadata.get("page", 1)),
                })

            done = min(start + BATCH_SIZE, total)
            print(f"Embedded {done}/{total} chunks")
            progress("embedding", done, total, f"Creating embeddings: {done} of {total} chunks")

        # The new knowledge base replaces the old one only after every chunk
        # was embedded successfully, so a failed run keeps the previous one.
        progress("saving", 0, 1, "Saving to the embedding database")
        save_store(
            all_vectors,
            all_metadata,
            {"embedding_model": OLLAMA_EMBEDDING_MODEL},
        )

        return {
            "documents": len(source_files) - len(unreadable_files),
            "chunks": total,
            "vectors": total,
            "store": str(VECTOR_STORE_FILE),
            "skipped_pages": len(skipped_pages),
            "skipped_page_details": skipped_pages[:50],
            "unreadable_files": unreadable_files,
        }
    finally:
        _training_lock.release()


if __name__ == "__main__":
    print(train())