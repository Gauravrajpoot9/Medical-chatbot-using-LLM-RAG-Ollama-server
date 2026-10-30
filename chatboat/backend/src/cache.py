"""Question -> response cache.

Repeated questions are answered instantly from the cache instead of running
retrieval and the LLM again.

- Questions are normalised first, so "What is anemia?" and "what is anemia"
  share one cache entry.
- Each entry is tied to a "fingerprint" of the knowledge base and settings
  (training time, chunk count, models, TOP_K, threshold, prompt version).
  After re-training or changing a setting, old entries simply stop matching.
- The cache is saved to backend/data/cache/responses.json, so it survives
  restarts. It is limited to CACHE_MAX_ENTRIES (oldest used entries are
  dropped first) and entries expire after CACHE_TTL_HOURS (0 = never).
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import CACHE_ENABLED, CACHE_FILE, CACHE_MAX_ENTRIES, CACHE_TTL_HOURS


def normalize_question(question):
    text = unicodedata.normalize("NFKC", question or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" ?.!")


class ResponseCache:
    def __init__(self, path=CACHE_FILE, max_entries=CACHE_MAX_ENTRIES,
                 ttl_hours=CACHE_TTL_HOURS, enabled=CACHE_ENABLED):
        self.path = Path(path)
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = float(ttl_hours) * 3600 if ttl_hours else 0
        self.enabled = enabled
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._loaded = False
        self.hits = 0
        self.misses = 0

    # ---- persistence ------------------------------------------------------

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for key, entry in data.get("entries", []):
                self._entries[key] = entry
        except Exception:
            # A damaged cache file is not important; start with an empty cache.
            self._entries.clear()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"entries": list(self._entries.items())}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    # ---- keys ---------------------------------------------------------------

    @staticmethod
    def make_key(question, fingerprint):
        raw = json.dumps([normalize_question(question), fingerprint],
                         sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _expired(self, entry):
        return bool(self.ttl_seconds) and time.time() - entry["created"] > self.ttl_seconds

    # ---- public API -----------------------------------------------------------

    def get(self, question, fingerprint):
        if not self.enabled:
            return None
        key = self.make_key(question, fingerprint)
        with self._lock:
            self._load()
            entry = self._entries.get(key)
            if entry is None or self._expired(entry):
                if entry is not None:
                    del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)  # mark as recently used
            entry["hits"] = entry.get("hits", 0) + 1
            self.hits += 1
            return entry

    def set(self, question, fingerprint, response, sources):
        if not self.enabled:
            return
        key = self.make_key(question, fingerprint)
        with self._lock:
            self._load()
            self._entries[key] = {
                "question": question,
                "response": response,
                "sources": sources,
                "created": time.time(),
                "hits": 0,
            }
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)  # drop least recently used
            try:
                self._save()
            except OSError:
                pass  # caching is optional; never fail a request because of it

    def clear(self):
        with self._lock:
            self._loaded = True
            count = len(self._entries)
            self._entries.clear()
            self.hits = self.misses = 0
            try:
                if self.path.exists():
                    self.path.unlink()
            except OSError:
                pass
            return count

    def stats(self):
        with self._lock:
            self._load()
            return {
                "enabled": self.enabled,
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_hours": self.ttl_seconds / 3600 if self.ttl_seconds else 0,
                "hits": self.hits,
                "misses": self.misses,
            }


response_cache = ResponseCache()