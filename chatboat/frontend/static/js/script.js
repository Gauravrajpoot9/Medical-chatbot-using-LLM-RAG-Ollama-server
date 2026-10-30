// If an old index.html is being used with this script, stop with a clear message
// instead of failing later with a misleading "Server not reachable".
(function checkPageVersion() {
    const required = ["kbPanel", "kbSummary", "progressWrap", "progressBar", "stages", "kbNotice"];
    const missing = required.filter(id => !document.getElementById(id));
    if (document.body.dataset.page !== "home-v2" || missing.length) {
        const msg = "index.html is out of date. Replace frontend/templates/index.html with the new version and restart python app.py.";
        console.error(msg, "Missing elements:", missing);
        const banner = document.createElement("div");
        banner.textContent = msg;
        banner.style.cssText = "background:#ED0007;color:#fff;padding:12px 16px;font:600 15px Arial,sans-serif";
        document.body.prepend(banner);
        throw new Error(msg);
    }
})();

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const trainBtn = document.getElementById("trainBtn");
const askBtn = document.getElementById("askBtn");
const documentList = document.getElementById("documentList");
const documentCount = document.getElementById("documentCount");
const statusEl = document.getElementById("status");
const toast = document.getElementById("toast");

const kbSummary = document.getElementById("kbSummary");
const kbNotice = document.getElementById("kbNotice");
const progressWrap = document.getElementById("progressWrap");
const progressTrack = document.getElementById("progressTrack");
const progressBar = document.getElementById("progressBar");
const progressMessage = document.getElementById("progressMessage");
const progressPercent = document.getElementById("progressPercent");
const progressDetail = document.getElementById("progressDetail");
const progressElapsed = document.getElementById("progressElapsed");
const stageItems = document.querySelectorAll("#stages li");

const STAGE_ORDER = ["reading", "chunking", "embedding", "saving"];

let toastTimer = null;
let pollTimer = null;
let documentsCount = 0;
let training = false;
let trained = false;

// ---------- helpers ----------

function showToast(message) {
    toast.textContent = message;
    toast.style.display = "block";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.style.display = "none",
        Math.max(3500, Math.min(10000, message.length * 60)));
}

function setStatus(text, kind = "") {
    statusEl.textContent = text;
    statusEl.className = "status " + kind;
}

async function readJson(response) {
    try {
        return await response.json();
    } catch {
        return { success: false, message: `Server returned ${response.status}.` };
    }
}

function formatSeconds(seconds) {
    if (seconds == null) return "";
    const s = Math.round(seconds);
    return s < 60 ? `${s}s elapsed` : `${Math.floor(s / 60)}m ${s % 60}s elapsed`;
}

function formatDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function showNotice(text, kind) {
    kbNotice.textContent = text;
    kbNotice.className = "notice " + kind;
    kbNotice.hidden = false;
}

function updateButtons() {
    document.body.classList.toggle("is-training", training);
    uploadBtn.disabled = training;
    trainBtn.disabled = training || documentsCount === 0;
    // Highlight the next step: Train until the knowledge base exists, then Ask.
    trainBtn.classList.toggle("primary", documentsCount > 0 && !training && !trained);
    askBtn.classList.toggle("primary", trained && !training);
}

// ---------- documents ----------

function renderDocuments(documents) {
    documentsCount = documents.length;
    documentCount.textContent = `${documents.length} document(s)`;
    updateButtons();

    if (!documents.length) {
        documentList.innerHTML = '<div class="empty">No documents yet.</div>';
        return;
    }

    documentList.innerHTML = "";
    documents.forEach(name => {
        const item = document.createElement("div");
        item.className = "document";

        const label = document.createElement("span");
        label.textContent = "📄 " + name;

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "delete-doc";
        remove.title = `Delete ${name}`;
        remove.setAttribute("aria-label", `Delete ${name}`);
        remove.textContent = "✕";
        remove.disabled = training;
        remove.addEventListener("click", () => deleteDocument(name, remove));

        item.append(label, remove);
        documentList.appendChild(item);
    });
}

async function loadDocuments() {
    try {
        const data = await readJson(await fetch("/documents"));
        renderDocuments(data.documents || []);
    } catch {
        showToast("Could not load documents.");
    }
}

async function deleteDocument(name, button) {
    if (!confirm(`Delete "${name}"?\n\nClick Train Knowledge Base afterwards to remove it from the AI's answers.`)) return;

    button.disabled = true;
    try {
        const response = await fetch(`/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
        const data = await readJson(response);
        showToast(data.message);
        if (data.success) renderDocuments(data.documents || []);
    } catch {
        showToast("Delete failed. Check the Flask terminal.");
        button.disabled = false;
    }
}

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
    if (!fileInput.files.length) return;

    const formData = new FormData();
    for (const file of fileInput.files) formData.append("files", file);

    setStatus("Uploading…", "busy");
    uploadBtn.disabled = true;

    try {
        const data = await readJson(await fetch("/upload", { method: "POST", body: formData }));
        showToast(data.message);
        if (data.success) await loadDocuments();
    } catch {
        showToast("Upload failed. Check the Flask terminal.");
    } finally {
        fileInput.value = "";
        updateButtons();
        await loadStatus();
    }
});

// ---------- knowledge base summary ----------

async function loadStatus() {
    let data;
    try {
        data = await readJson(await fetch("/status"));
    } catch {
        setStatus("Server not reachable", "error");
        return;
    }

    try {
        if (training) return;
        trained = !!data.trained;
        updateButtons();

        if (!data.trained) {
            kbSummary.innerHTML = '<div class="kb-empty">Not trained yet.</div>';
            setStatus("Not trained", "");
            return;
        }

        kbSummary.innerHTML = "";
        const box = document.createElement("div");
        box.className = "kb-summary";
        [
            [data.documents, "documents"],
            [data.chunks, "embedded chunks"],
            [formatDate(data.created_at), "last trained"],
        ].forEach(([value, label]) => {
            const div = document.createElement("div");
            const b = document.createElement("b");
            b.textContent = value;
            div.append(b, document.createTextNode(label));
            box.appendChild(div);
        });
        kbSummary.appendChild(box);
        setStatus("Knowledge base ready", "ready");
    } catch (error) {
        console.error(error);
        setStatus("Page error, see browser console", "error");
    }
}

// ---------- training + progress ----------

function renderProgress(job) {
    const percent = Math.max(0, Math.min(100, job.percent || 0));
    progressBar.style.width = percent + "%";
    progressTrack.setAttribute("aria-valuenow", String(Math.round(percent)));
    progressPercent.textContent = Math.round(percent) + "%";
    progressMessage.textContent = job.message || "Working…";
    progressElapsed.textContent = formatSeconds(job.elapsed_seconds);

    // Rough time estimate from the progress made so far.
    if (job.state === "running" && job.stage === "embedding" && percent > 15 && job.elapsed_seconds) {
        const remaining = job.elapsed_seconds * (100 - percent) / percent;
        progressDetail.textContent = remaining < 60
            ? `About ${Math.max(1, Math.round(remaining))}s remaining`
            : `About ${Math.round(remaining / 60)} min remaining`;
    } else {
        progressDetail.textContent = "";
    }

    const currentIndex = job.state === "done" ? STAGE_ORDER.length : STAGE_ORDER.indexOf(job.stage);
    stageItems.forEach((li, i) => {
        li.classList.toggle("done", i < currentIndex);
        li.classList.toggle("active", i === currentIndex);
    });
}

function startPolling() {
    clearInterval(pollTimer);
    pollTimer = setInterval(pollTraining, 700);
    pollTraining();
}

async function pollTraining() {
    let job;
    try {
        job = await readJson(await fetch("/train/status"));
    } catch {
        return; // try again on the next tick
    }

    if (job.state === "running") {
        training = true;
        progressWrap.hidden = false;
        kbSummary.hidden = true;
        kbNotice.hidden = true;
        setStatus("Training…", "busy");
        updateButtons();
        renderProgress(job);
        return;
    }

    clearInterval(pollTimer);
    const wasTraining = training;
    training = false;
    updateButtons();
    await loadDocuments();

    if (job.state === "done") {
        renderProgress(job);
        if (wasTraining) {
            showNotice(job.message, "success");
            showToast(job.message);
        }
        setTimeout(() => { progressWrap.hidden = true; }, 1500);
    } else if (job.state === "error") {
        progressWrap.hidden = true;
        if (wasTraining) {
            showNotice("Training failed: " + job.error, "error");
            setStatus("Training failed", "error");
        }
    }

    kbSummary.hidden = false;
    if (job.state !== "error" || !wasTraining) await loadStatus();
}

trainBtn.addEventListener("click", async () => {
    if (!confirm("Training will rebuild the knowledge base using ALL uploaded documents. Continue?")) return;

    kbNotice.hidden = true;
    progressWrap.hidden = false;
    kbSummary.hidden = true;
    renderProgress({ percent: 0, message: "Starting training", stage: "reading", state: "running" });
    training = true;
    updateButtons();
    setStatus("Training…", "busy");

    try {
        const response = await fetch("/train", { method: "POST" });
        const data = await readJson(response);
        if (!data.success && response.status !== 409) {
            throw new Error(data.message);
        }
        startPolling();
    } catch (error) {
        training = false;
        progressWrap.hidden = true;
        kbSummary.hidden = false;
        updateButtons();
        showNotice("Training failed: " + (error.message || "server not reachable"), "error");
        setStatus("Training failed", "error");
    }
});

// ---------- Q&A window ----------

askBtn.addEventListener("click", () => {
    const width = Math.min(980, screen.availWidth - 40);
    const height = Math.min(860, screen.availHeight - 40);
    const left = Math.max(0, (screen.availWidth - width) / 2);
    const top = Math.max(0, (screen.availHeight - height) / 2);

    const win = window.open("/qa", "medicalQA",
        `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`);

    if (win) {
        win.focus();
    } else {
        // Pop-up blocked: open the Q&A page in this tab instead.
        window.location.href = "/qa";
    }
});

// ---------- start ----------

loadDocuments();
loadStatus();
pollTraining(); // resumes the progress bar if training was running before a page refresh