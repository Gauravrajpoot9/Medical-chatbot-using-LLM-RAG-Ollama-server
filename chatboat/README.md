# Medical AI Chatbot — Flask + Ollama (Fully Offline)

A document-based medical Q&A chatbot that runs entirely on your computer.
No API keys, no accounts, no cloud services.

## Architecture

Browser → Flask → Ollama Embeddings → Local vector store (NumPy file) → Ollama LLM → Browser

Everything runs locally:

- **Ollama** creates embeddings and generates answers on your machine.
- **The knowledge base** is saved as a single file: `backend/data/vector_store/knowledge_base.npz`.
- **Search** uses cosine similarity with NumPy.

You only need the internet **once**, to install Python packages and pull the Ollama models. After that the chatbot works with no internet connection.

## Features

1. Upload one or multiple PDF/TXT documents (non-English file names supported; duplicates are renamed, not overwritten).
2. Delete uploaded documents from the UI.
3. Train the knowledge base using all uploaded documents.
4. Create embeddings locally with Ollama `nomic-embed-text`.
5. Store 768-dimensional vectors in a local file (kept between restarts).
6. Ask questions through the Flask UI (Enter to send, Shift+Enter for a new line).
7. Retrieve relevant document chunks locally, ignoring chunks below a relevance threshold.
8. Generate the answer with Ollama `llama3.2:3b`.

## Requirements

- Windows 10 22H2 or newer
- Python 3.12 recommended
- Ollama
- Internet only for the one-time setup (steps 1–3)

## 1. Install Ollama (one time, needs internet)

Install Ollama for Windows, then open PowerShell:

```powershell
ollama --version
```

Pull the models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Check:

```powershell
ollama list
```

Ollama must be running while using the chatbot.

## 2. Create Python environment

From the project root:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Install packages (one time, needs internet)

```powershell
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure .env (optional)

The defaults work without a `.env` file. To change settings, copy:

```text
backend/.env.example
```

to:

```text
backend/.env
```

Available settings:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

CHUNK_SIZE=800
CHUNK_OVERLAP=100

TOP_K=5
RELEVANCE_THRESHOLD=0.45

FLASK_DEBUG=true
MAX_CONTENT_LENGTH_MB=50
```

No API key is needed. Set `FLASK_DEBUG=false` for anything other than local development.

## 5. Run the UI

From:

```text
Medical_Chatbot_Ollama_Flask/backend
```

run:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 6. How to use

### Upload

Click **Upload Documents**.

Select one or multiple:

```text
.pdf
.txt
```

### Train

Click **Train Knowledge Base**.

A progress bar appears in the **Knowledge Base** panel, directly under the buttons. It shows the percentage, how many chunks have been embedded, the time elapsed, roughly how long is left, and the four stages (read documents, split into chunks, create embeddings, save to database). The colour band at the top of the page shimmers while training runs. If you refresh the page during training, the progress bar picks up where it was.

The application will:

```text
PDF/TXT
↓
Text extraction
↓
Chunking
↓
Ollama nomic-embed-text
↓
768D embeddings
↓
Local vector store (knowledge_base.npz)
```

Every time Train is clicked, the knowledge base is rebuilt from the current uploaded documents. The old knowledge base is replaced only after training finishes successfully, so a failed run keeps the previous one.

PDF pages with no text layer (scanned images) are skipped, and the training message tells you how many pages and which files could not be read. Run OCR on scanned PDFs before uploading.

### Delete

Click ✕ next to a document to delete it, then click **Train Knowledge Base** so it is removed from the AI's answers.

### Ask Medical AI

Click **Ask Medical AI**. A separate Q&A window opens (if your browser blocks pop-ups, the Q&A page opens in the same tab; you can also go to `http://127.0.0.1:5000/qa` directly).

Type a question and press Enter (Shift+Enter adds a new line), or click one of the example questions. Each answer lists the documents and pages it was based on.

Example:

```text
What are the common symptoms of anemia?
```

The system:

```text
Question
↓
Ollama embedding
↓
Local similarity search
↓
Relevant document chunks
↓
Ollama llama3.2:3b
↓
Answer
```

## Important

This is a document-based medical information assistant, not a doctor. The chatbot prompt avoids definitive diagnosis and medication/dosage prescribing. Users should consult qualified healthcare professionals.

## Troubleshooting

### Page looks wrong or mixes old and new parts

The browser is using an old `index.html`, `style.css` or `script.js`. Replace the whole `frontend` folder with the one from the latest ZIP, restart `python app.py`, and press **Ctrl+F5** in the browser. When the app starts, the terminal prints the templates and static folders it is using; check that they point to this project.

### Ollama connection error

Run:

```powershell
ollama list
```

and confirm the models are installed and Ollama is running.

### Embedding dimension mismatch

`nomic-embed-text` returns 768 dimensions. If you switch to another embedding model, set `EMBEDDING_DIMENSION` to its size and click **Train Knowledge Base** again.

### "The knowledge base has not been trained yet"

No knowledge base file exists yet. Upload documents and click **Train Knowledge Base**.

### Changed the embedding model

The chatbot will ask you to re-train, because vectors from different models can't be compared. Click **Train Knowledge Base**.

### Answers say "could not find relevant information" too often (or answer off-topic questions)

Tune `RELEVANCE_THRESHOLD` in `.env`. Lower it (for example `0.35`) if relevant questions are not answered; raise it (for example `0.55`) if unrelated questions still get answers. Restart `python app.py` after changing it.

### Start fresh

Stop the app and delete:

```text
backend/data/vector_store/knowledge_base.npz
```

### No documents found

Upload documents first. They are stored in:

```text
backend/data/uploads/
```

Only PDF and TXT are accepted.

### Running the backend scripts directly

From `backend/` you can run:

```powershell
python src/create_embeddings.py
python src/chatbot.py
```

## Project structure

```text
Medical_Chatbot_Ollama_Flask/
│
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── .env.example
│   ├── requirements.txt
│   ├── data/
│   │   ├── uploads/
│   │   └── vector_store/
│   └── src/
│       ├── __init__.py
│       ├── create_embeddings.py
│       ├── chatbot.py
│       └── vector_store.py
│
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   └── qa.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           ├── script.js
│           └── qa.js
│
└── README.md
```
