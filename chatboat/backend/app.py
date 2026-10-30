import re
import threading
import time
import unicodedata
from pathlib import Path

# pyright: reportMissingImports=false
from flask import Flask, jsonify, render_template, request

from config import UPLOAD_FOLDER, MAX_CONTENT_LENGTH, FLASK_DEBUG
from src.create_embeddings import train
from src.chatbot import ask, reset as reset_chatbot, KnowledgeBaseNotReady
from src.cache import response_cache
from src import vector_store

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
# Always serve the newest HTML/CSS/JS so the browser never mixes old and new files.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Change this when the frontend files change; it is added to CSS/JS links
# (style.css?v=...) so browsers download the new versions instead of cached ones.
ASSET_VERSION = "2.4"


@app.context_processor
def inject_asset_version():
    return {"asset_version": ASSET_VERSION}

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILENAME_LENGTH = 150

# Characters not allowed in Windows/Linux file names, plus control characters.
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def clean_filename(raw_name):
    """Make a safe file name while keeping non-English characters.

    Unlike werkzeug's secure_filename, this keeps names such as
    '報告.pdf' or 'تقرير.pdf' instead of stripping them to nothing.
    """
    name = unicodedata.normalize("NFC", raw_name or "")
    name = name.replace("\\", "/").split("/")[-1]          # drop any path
    name = _INVALID_CHARS.sub("_", name)
    name = re.sub(r"\s+", " ", name).strip().strip(".")

    suffix = Path(name).suffix.lower()
    stem = name[: -len(suffix)] if suffix else name
    stem = stem.strip().strip(".") or "document"

    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"

    stem = stem[: MAX_FILENAME_LENGTH - len(suffix)]
    return f"{stem}{suffix}"


def unique_destination(filename):
    """Return a path that does not overwrite an existing upload."""
    destination = UPLOAD_FOLDER / filename
    if not destination.exists():
        return destination

    stem, suffix = destination.stem, destination.suffix
    counter = 1
    while True:
        candidate = UPLOAD_FOLDER / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def list_documents():
    return sorted(
        p.name for p in UPLOAD_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )


def resolve_upload(name):
    """Return the upload path for name, or None if it is not a valid upload."""
    if not name or name != Path(name).name:
        return None
    path = (UPLOAD_FOLDER / name).resolve()
    if path.parent != UPLOAD_FOLDER.resolve():
        return None
    if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None
    return path


@app.route("/")
def home():
    return render_template("index.html")


@app.post("/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "message": "No files selected."}), 400

    uploaded = []
    renamed = []
    rejected = []

    for file in files:
        if not file or not file.filename:
            continue

        filename = clean_filename(file.filename)
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            rejected.append(file.filename)
            continue

        destination = unique_destination(filename)
        file.save(destination)
        uploaded.append(destination.name)
        if destination.name != filename:
            renamed.append({"original": file.filename, "saved_as": destination.name})

    if not uploaded:
        return jsonify({
            "success": False,
            "message": "Only PDF and TXT files are supported.",
            "rejected": rejected,
        }), 400

    message = f"{len(uploaded)} document(s) uploaded."
    if renamed:
        message += f" {len(renamed)} renamed to avoid overwriting existing files."
    if rejected:
        message += f" {len(rejected)} rejected (only PDF and TXT are supported)."
    message += " Click Train Knowledge Base to include them."

    return jsonify({
        "success": True,
        "message": message,
        "uploaded": uploaded,
        "renamed": renamed,
        "rejected": rejected,
    })


@app.get("/documents")
def documents():
    return jsonify({"success": True, "documents": list_documents()})


@app.delete("/documents/<path:name>")
def delete_document(name):
    path = resolve_upload(name)
    if path is None:
        return jsonify({"success": False, "message": "Document not found."}), 404

    try:
        path.unlink()
    except OSError as exc:
        return jsonify({"success": False, "message": f"Could not delete file: {exc}"}), 500

    return jsonify({
        "success": True,
        "message": f"Deleted {name}. Click Train Knowledge Base to remove it from the AI's answers.",
        "documents": list_documents(),
    })


# ---------------------------------------------------------------------------
# Training runs in a background thread so the UI can show live progress.
# ---------------------------------------------------------------------------

# Share of the progress bar given to each stage (embedding is the slow one).
_STAGE_RANGES = {
    "reading": (0, 8),
    "chunking": (8, 12),
    "embedding": (12, 96),
    "saving": (96, 100),
}

_job_lock = threading.Lock()
_job = {
    "state": "idle",        # idle | running | done | error
    "stage": "",
    "current": 0,
    "total": 0,
    "percent": 0,
    "message": "",
    "result": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _update_job(**changes):
    with _job_lock:
        _job.update(changes)


def _job_snapshot():
    with _job_lock:
        snap = dict(_job)
    if snap["started_at"]:
        end = snap["finished_at"] or time.time()
        snap["elapsed_seconds"] = round(end - snap["started_at"], 1)
    return snap


def _on_progress(stage, current, total, message):
    start, end = _STAGE_RANGES.get(stage, (0, 100))
    fraction = (current / total) if total else 0
    _update_job(
        stage=stage,
        current=current,
        total=total,
        percent=round(start + (end - start) * fraction, 1),
        message=message,
    )


def _train_summary(result):
    message = (
        f"Training complete: {result['documents']} documents, "
        f"{result['chunks']} chunks."
    )
    if result["skipped_pages"]:
        message += f" {result['skipped_pages']} page(s) had no readable text and were skipped."
    if result["unreadable_files"]:
        names = ", ".join(u["source"] for u in result["unreadable_files"])
        message += f" Unreadable file(s): {names}."
    return message


def _run_training():
    try:
        result = train(progress=_on_progress)
        reset_chatbot()
        response_cache.clear()  # answers from the old knowledge base are no longer valid
        _update_job(
            state="done",
            stage="done",
            percent=100,
            message=_train_summary(result),
            result=result,
            finished_at=time.time(),
        )
    except Exception as exc:
        _update_job(state="error", error=str(exc), message=str(exc), finished_at=time.time())


@app.post("/train")
def train_documents():
    with _job_lock:
        if _job["state"] == "running":
            return jsonify({"success": False, "message": "Training is already running."}), 409
        _job.update({
            "state": "running",
            "stage": "reading",
            "current": 0,
            "total": 0,
            "percent": 0,
            "message": "Starting training",
            "result": None,
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        })

    threading.Thread(target=_run_training, daemon=True).start()
    return jsonify({"success": True, "message": "Training started."}), 202


@app.get("/train/status")
def train_status():
    return jsonify({"success": True, **_job_snapshot()})


@app.get("/status")
def knowledge_base_status():
    """Summary of the saved knowledge base, shown in both pages."""
    try:
        loaded = vector_store.load_store()
    except Exception as exc:
        return jsonify({"success": True, "trained": False, "message": str(exc)})

    if loaded is None:
        return jsonify({"success": True, "trained": False})

    _, metadata, info = loaded
    return jsonify({
        "success": True,
        "trained": True,
        "chunks": info.get("count", len(metadata)),
        "documents": len({m.get("source") for m in metadata}),
        "created_at": info.get("created_at"),
        "embedding_model": info.get("embedding_model"),
    })


@app.get("/cache")
def cache_stats():
    return jsonify({"success": True, **response_cache.stats()})


@app.post("/cache/clear")
def cache_clear():
    removed = response_cache.clear()
    return jsonify({"success": True, "message": f"Cleared {removed} cached answer(s)."})


@app.route("/qa")
def qa_page():
    return render_template("qa.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    query = str(data.get("message", "")).strip()

    if not query:
        return jsonify({"success": False, "message": "Please enter a question."}), 400

    try:
        result = ask(query)
        return jsonify({"success": True, **result})
    except KnowledgeBaseNotReady as exc:
        return jsonify({"success": False, "message": str(exc)}), 409
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "success": False,
        "message": "File is too large. Check MAX_CONTENT_LENGTH_MB in .env.",
    }), 413


if __name__ == "__main__":
    print(f"Templates folder: {app.template_folder}")
    print(f"Static folder:    {app.static_folder}")
    print("Medical Chatbot UI: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=FLASK_DEBUG)