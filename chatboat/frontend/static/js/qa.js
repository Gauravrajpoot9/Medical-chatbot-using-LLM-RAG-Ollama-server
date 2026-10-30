if (document.body.dataset.page !== "qa-v2" || !document.getElementById("chatForm")) {
    const msg = "qa.html is out of date. Replace frontend/templates/qa.html with the new version and restart python app.py.";
    document.body.insertAdjacentHTML("afterbegin",
        '<div style="background:#ED0007;color:#fff;padding:12px 16px;font:600 15px Arial,sans-serif"></div>');
    document.body.firstElementChild.textContent = msg;
    throw new Error(msg);
}

const chatForm = document.getElementById("chatForm");
const question = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const statusEl = document.getElementById("status");

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

function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
}

function addUserMessage(text) {
    const div = document.createElement("div");
    div.className = "msg user";
    div.textContent = text;
    messages.appendChild(div);
    scrollToBottom();
}

function addAssistantMessage(text, isError = false) {
    const div = document.createElement("div");
    div.className = "msg assistant" + (isError ? " error" : "");

    const label = document.createElement("span");
    label.className = "msg-label";
    label.textContent = isError ? "Error" : "Medical AI";
    div.appendChild(label);
    div.appendChild(document.createTextNode(text));

    messages.appendChild(div);
    scrollToBottom();
}

function addTyping() {
    const div = document.createElement("div");
    div.className = "msg assistant";
    div.innerHTML = '<span class="msg-label">Medical AI</span><span class="typing"><i></i><i></i><i></i></span>';
    messages.appendChild(div);
    scrollToBottom();
    return div;
}

async function loadStatus() {
    try {
        const data = await readJson(await fetch("/status"));
        if (data.trained) {
            setStatus(`${data.documents} documents, ${data.chunks} chunks`, "ready");
        } else {
            setStatus("Not trained yet", "error");
        }
    } catch {
        setStatus("Server not reachable", "error");
    }
}

async function ask(text) {
    if (welcome) welcome.remove();
    addUserMessage(text);
    sendBtn.disabled = true;
    const typing = addTyping();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
        });
        const data = await readJson(response);
        typing.remove();

        if (data.success) {
            addAssistantMessage(data.response);
        } else {
            addAssistantMessage(data.message, true);
        }
    } catch {
        typing.remove();
        addAssistantMessage("Unable to connect to the chatbot backend. Check the Flask terminal.", true);
    } finally {
        sendBtn.disabled = false;
        question.focus();
    }
}

chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (!text || sendBtn.disabled) return;
    question.value = "";
    autoResize();
    ask(text);
});

// Enter sends, Shift+Enter adds a new line (ignored while an IME is composing).
question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

function autoResize() {
    question.style.height = "auto";
    question.style.height = Math.min(question.scrollHeight, 160) + "px";
}
question.addEventListener("input", autoResize);

document.querySelectorAll(".example").forEach(button => {
    button.addEventListener("click", () => {
        if (!sendBtn.disabled) ask(button.textContent);
    });
});

loadStatus();
question.focus();