"""Local, offline vector store.

All vectors and chunk texts are saved in one file on disk:
    backend/data/vector_store/knowledge_base.npz

Search uses cosine similarity with NumPy. No server, account or API key
is needed, and the knowledge base survives restarts.
"""
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np

from config import VECTOR_STORE_FILE

_lock = threading.Lock()
_cache = {"mtime": None, "vectors": None, "metadata": None, "info": None}


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def store_exists():
    return VECTOR_STORE_FILE.is_file()


def save_store(vectors, metadata, info):
    """Save vectors + metadata atomically.

    The file is written to a temporary name first and then swapped in,
    so a failed or interrupted training run never corrupts the existing
    knowledge base.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(metadata):
        raise RuntimeError("Vector count does not match chunk count.")

    matrix = _normalize(matrix)
    info = dict(info)
    info.update({
        "count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })

    VECTOR_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = VECTOR_STORE_FILE.with_name(VECTOR_STORE_FILE.stem + ".tmp.npz")

    with open(tmp_path, "wb") as f:
        np.savez(
            f,
            vectors=matrix,
            metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
            info=np.array(json.dumps(info, ensure_ascii=False)),
        )

    with _lock:
        os.replace(tmp_path, VECTOR_STORE_FILE)
        _cache["mtime"] = None  # force reload on next search

    return info


def load_store():
    """Load the store, reusing the in-memory copy while the file is unchanged."""
    if not store_exists():
        return None

    mtime = VECTOR_STORE_FILE.stat().st_mtime
    with _lock:
        if _cache["mtime"] == mtime and _cache["vectors"] is not None:
            return _cache["vectors"], _cache["metadata"], _cache["info"]

        try:
            with np.load(VECTOR_STORE_FILE, allow_pickle=False) as data:
                vectors = np.array(data["vectors"], dtype=np.float32)
                metadata = json.loads(str(data["metadata"]))
                info = json.loads(str(data["info"]))
        except Exception as exc:
            raise RuntimeError(
                f"The local knowledge base file is damaged ({exc}). "
                f"Click Train Knowledge Base to rebuild it."
            ) from exc

        if vectors.shape[0] != len(metadata):
            raise RuntimeError(
                "The local knowledge base is inconsistent. "
                "Click Train Knowledge Base to rebuild it."
            )

        _cache.update({"mtime": mtime, "vectors": vectors,
                       "metadata": metadata, "info": info})
        return vectors, metadata, info


def search(query_vector, top_k=5, threshold=0.0):
    """Return [(score, metadata), ...] sorted by score, best first."""
    loaded = load_store()
    if loaded is None:
        return []

    vectors, metadata, _ = loaded
    if vectors.shape[0] == 0:
        return []

    query = np.asarray(query_vector, dtype=np.float32)
    if query.shape[0] != vectors.shape[1]:
        raise RuntimeError(
            f"Question embedding has {query.shape[0]} dimensions but the knowledge "
            f"base has {vectors.shape[1]}. Click Train Knowledge Base to rebuild it."
        )

    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    query = query / norm

    scores = vectors @ query
    k = min(top_k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]

    return [
        (float(scores[i]), metadata[i])
        for i in top
        if float(scores[i]) >= threshold
    ]