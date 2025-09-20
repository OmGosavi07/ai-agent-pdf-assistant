# AI Agent PDF Assistant — Streamlit + Multi-LLM (OpenAI / Gemini / Ollama)
# Fully updated: better error handling, retries, fallback, UI for API keys.
# ------------------------------------------------------------

import os
import io
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import fitz  # PyMuPDF
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Streamlit UI setup ----------------
st.set_page_config(page_title="AI Agent PDF Assistant", layout="wide")
st.title("🤖 AI Agent for PDFs — Summarize & Ask Anything (Multi-LLM)")
st.caption("Drop a PDF, then chat with an agent. Choose OpenAI, Gemini, or Ollama in the sidebar.")

# Sidebar controls and API keys
with st.sidebar:
    st.subheader("⚙️ Settings")
    provider = st.selectbox("LLM provider", ["openai", "gemini", "ollama"], index=0)
    model = st.text_input(
        "Model name",
        value=(
            "gpt-4o-mini" if provider == "openai"
            else "gemini-2.5" if provider == "gemini"
            else "llama-3.2-3b-it"
        ),
    )
    max_tokens = st.slider("Max tokens (per response)", 256, 8192, 512, step=64)
    temp = st.slider("Temperature", 0.0, 1.2, 0.2, step=0.1)

    st.markdown("---")
    st.write("**Set API keys (optional — will be used for this session)**")
    openai_key_in = st.text_input("OPENAI_API_KEY", type="password", placeholder="sk-...", help="Paste OpenAI API key")
    gemini_key_in = st.text_input("GEMINI_API_KEY", type="password", placeholder="ya29-... or API key", help="Paste Gemini/Google Generative API key or OAuth token")
    # We don't set Ollama key (runs locally)
    if openai_key_in:
        os.environ["OPENAI_API_KEY"] = openai_key_in.strip()
    if gemini_key_in:
        os.environ["GEMINI_API_KEY"] = gemini_key_in.strip()

    st.markdown("---")
    st.write("You can still use Ollama locally at http://localhost:11434 (no key required).")
    if st.button("Test LLM"):
        test_prompt = "Say 'hello' and identify yourself briefly."
        st.info("Testing... (this will call the selected provider once)")
        res = None
        try:
            res = None
            # call llm_generate (we define it later) via a placeholder sentinel; we'll call a tiny helper below after it's defined.
            st.session_state["_test_llm_requested"] = True
            st.session_state["_test_prompt"] = test_prompt
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Test failed: {e}")

    st.markdown("---")
    st.write("**Export**")
    export_summary_btn_placeholder = st.empty()
    export_chat_btn_placeholder = st.empty()

# ---------------- LLM client wrapper (robust) ----------------
def _request_ollama(prompt: str, model_name: str, temperature: float, max_tokens: int, timeout=120) -> Tuple[bool, str]:
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama variants:
        return True, data.get("response") or data.get("generated") or data.get("text") or json.dumps(data)
    except requests.HTTPError as he:
        body = he.response.text if he.response is not None else str(he)
        return False, f"HTTP {he.response.status_code} - Ollama error: {body}"
    except Exception as e:
        return False, f"Ollama request failed: {e}"

def _request_openai(prompt: str, model_name: str, temperature: float, max_tokens: int, timeout=120) -> Tuple[bool, str, Optional[int]]:
    """
    Returns (success, text_or_error, status_code_or_none)
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return False, "OPENAI_API_KEY not found in environment.", None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "n": 1,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        # Capture status for special cases (429)
        status = resp.status_code
        if resp.status_code >= 400:
            # Try to parse JSON for better error display
            try:
                err = resp.json()
                return False, f"HTTP {status} - OpenAI error: {json.dumps(err)}", status
            except Exception:
                return False, f"HTTP {status} - OpenAI error: {resp.text}", status
        data = resp.json()
        # Extract answer
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content")
            if content:
                return True, content, status
            text = data["choices"][0].get("text")
            if text:
                return True, text, status
        return False, "OpenAI: Unexpected response structure: " + json.dumps(data), status
    except requests.HTTPError as he:
        try:
            body = he.response.text
            status = he.response.status_code
        except Exception:
            body = str(he)
            status = None
        return False, f"HTTP error: {body}", status
    except Exception as e:
        return False, f"OpenAI request failed: {e}", None

def _request_gemini(prompt: str, model_name: str, temperature: float, max_tokens: int, timeout=120) -> Tuple[bool, str, Optional[int]]:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return False, "GEMINI_API_KEY not found in environment.", None

    base = f"https://generativelanguage.googleapis.com/v1beta2/models/{model_name}:generateText"
    headers = {"Content-Type": "application/json"}
    params = {}
    # Decide whether to use bearer or key param
    if gemini_key.startswith("ya29-") or gemini_key.startswith("1/") or gemini_key.startswith("ya29."):
        headers["Authorization"] = f"Bearer {gemini_key}"
    else:
        params["key"] = gemini_key

    body = {"prompt": {"text": prompt}, "maxOutputTokens": int(max_tokens), "temperature": float(temperature)}
    try:
        resp = requests.post(base, headers=headers, params=params, json=body, timeout=timeout)
        status = resp.status_code
        if resp.status_code >= 400:
            try:
                err = resp.json()
                return False, f"HTTP {status} - Gemini error: {json.dumps(err)}", status
            except Exception:
                return False, f"HTTP {status} - Gemini error: {resp.text}", status
        data = resp.json()
        # Attempt to extract text robustly
        candidates = data.get("candidates") or data.get("outputs") or []
        if candidates:
            c0 = candidates[0]
            # candidate may be string or dict
            if isinstance(c0, str):
                return True, c0, status
            if isinstance(c0, dict):
                for k in ("output", "content", "text"):
                    if k in c0 and isinstance(c0[k], str):
                        return True, c0[k], status
                cont = c0.get("content") or c0.get("message", {}).get("content", [])
                if isinstance(cont, list) and cont:
                    parts = []
                    for part in cont:
                        if isinstance(part, dict):
                            txt = part.get("text") or part.get("output")
                            if txt:
                                parts.append(txt)
                        elif isinstance(part, str):
                            parts.append(part)
                    if parts:
                        return True, "\n".join(parts), status
        # fallback fields
        for key in ("response", "output", "text"):
            if key in data and isinstance(data[key], str):
                return True, data[key], status
        return False, "Gemini: Unexpected response structure: " + json.dumps(data), status
    except requests.HTTPError as he:
        try:
            body = he.response.text
            status = he.response.status_code
        except Exception:
            body = str(he)
            status = None
        return False, f"HTTP error: {body}", status
    except Exception as e:
        return False, f"Gemini request failed: {e}", None

def llm_generate(prompt: str, model_name: str, provider_name: str, temperature: float, max_tokens: int) -> str:
    """
    Robust orchestration:
     - tries selected provider with retry/backoff on transient errors (429, 5xx)
     - when definitive client errors occur (insufficient_quota / model not found), provides clear messages
     - attempts fallbacks (other providers) if configured and available
    """
    # small nested helper to attempt provider with retries
    def attempt_with_retries(req_fn, label: str, retries=3, backoff_base=1.5):
        attempt = 0
        last_error = None
        while attempt < retries:
            attempt += 1
            ok, content, status = req_fn(prompt, model_name, temperature, max_tokens)
            if ok:
                return True, content, status
            last_error = (content, status)
            # transient retry criteria: status 429 or 500-599 or network errors (status is None)
            is_transient = (status == 429) or (status is not None and 500 <= status < 600) or (status is None)
            if not is_transient:
                # Non-transient (e.g., 400, 401, 403, 404) — break early
                break
            # else wait and retry
            wait = backoff_base ** attempt
            time.sleep(wait)
        return False, last_error[0] if last_error else "Unknown error", last_error[1] if last_error else None

    # Order of tries: primary provider -> other configured providers -> final Ollama (if available)
    tried_providers = []
    provider_order = [provider_name]
    # add other options as fallback candidates
    for p in ("openai", "gemini", "ollama"):
        if p not in provider_order:
            provider_order.append(p)

    final_error_messages = []
    for p in provider_order:
        tried_providers.append(p)
        if p == "openai":
            ok, content, status = attempt_with_retries(_request_openai, "openai")
            if ok:
                return content
            # If OpenAI returned 429 with a clear insufficient_quota message, show that explicitly
            if status == 429 and "insufficient_quota" in content:
                final_error_messages.append(f"OpenAI 429: quota likely exceeded. Message: {content}")
                # try fallbacks
            else:
                final_error_messages.append(f"OpenAI error: {content}")
            continue

        if p == "gemini":
            ok, content, status = attempt_with_retries(_request_gemini, "gemini")
            if ok:
                return content
            if status == 404:
                final_error_messages.append(
                    "Gemini 404: model not found. Check the model name for the Generative API and/or ensure your key has access. "
                    f"Server message: {content}"
                )
            else:
                final_error_messages.append(f"Gemini error: {content}")
            continue

        if p == "ollama":
            ok, content = _request_ollama(prompt, model_name, temperature, max_tokens)
            if ok:
                return content
            final_error_messages.append(f"Ollama error: {content}")
            continue

    # after all attempted
    joined = "\n\n".join(final_error_messages)
    return f"__LLM error__: All providers failed (tried: {tried_providers}). Details:\n\n{joined}"

# ---------------- PDF ingestion ----------------
@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes) -> Tuple[str, List[str]]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(text.strip())
    full_text = "\n\n".join(pages)
    return full_text, pages

@st.cache_resource(show_spinner=False)
def build_tfidf_index(chunks: List[str]):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix

# ---------------- Agent Tools ----------------
class PDFTools:
    def __init__(self):
        self.pdf_text: str = ""
        self.pages: List[str] = []
        self.vectorizer = None
        self.matrix = None
        self.summary_cache: str = ""

    def load(self, file_bytes: bytes):
        self.pdf_text, self.pages = extract_pdf_text(file_bytes)
        self.vectorizer, self.matrix = build_tfidf_index(self.pages)
        self.summary_cache = ""

    def summarize_pdf(self) -> str:
        if not self.pdf_text:
            return "No PDF loaded."
        prompt = (
            "Summarize this PDF in clear bullet points. "
            "Keep important names, numbers, and key findings.\n\n"
            f"{self.pdf_text}"
        )
        final = llm_generate(prompt, model, provider, temp, max_tokens)
        self.summary_cache = final
        return final

    def answer_from_pdf(self, question: str, top_k: int = 5) -> str:
        if not self.pdf_text:
            return "No PDF loaded."
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        retrieved = [self.pages[i] for i in idxs]
        context = "\n\n".join(retrieved)
        prompt = (
            "You are a truthful assistant. Answer ONLY from the provided PDF context. "
            "If the answer isn't in the context, say 'I cannot find this in the PDF.'\n\n"
            f"Question: {question}\n\nPDF context:\n{context}"
        )
        return llm_generate(prompt, model, provider, temp, max_tokens)

    def list_sections(self) -> str:
        if not self.pages:
            return "No PDF loaded."
        headers = []
        for p_i, page in enumerate(self.pages, start=1):
            for line in page.splitlines():
                ln = line.strip()
                if 4 <= len(ln) <= 80 and (ln.isupper() or ln.endswith(":")):
                    headers.append(f"p{p_i}: {ln}")
        if not headers:
            return "Couldn't detect headings reliably. Try the summary."
        return "\n".join(headers)

    def export_summary(self) -> bytes:
        txt = self.summary_cache or "No summary generated yet."
        return txt.encode("utf-8")

pdf = PDFTools()

# ---------------- Agent Orchestrator ----------------
TOOLS_SPEC = {
    "summarize_pdf": {"description": "Summarize the entire loaded PDF.", "args_schema": {}},
    "answer_from_pdf": {"description": "Answer a user question using only the PDF contents.", "args_schema": {"question": "string"}},
    "list_sections": {"description": "List likely section headings detected across pages.", "args_schema": {}},
    "export_summary": {"description": "Return the last computed summary as bytes for download.", "args_schema": {}},
}

AGENT_SYSTEM_PROMPT = (
    "You are an AI agent that chooses tools to help with PDFs. "
    "Decide the next action and respond ONLY as compact JSON.\n\n"
    "Available tools: " + ", ".join(TOOLS_SPEC.keys()) + ".\n"
    "Rules:\n"
    "- If the user asks for a summary, use summarize_pdf.\n"
    "- If the user asks a question, use answer_from_pdf.\n"
    "- If the user wants structure/TOC, use list_sections.\n"
    "- When done, return a final answer.\n"
    "- Always return one of:\n"
    "  {\"action\": \"tool\", \"tool\": <tool_name>, \"args\": {..}}\n"
    "  {\"action\": \"final\", \"answer\": <markdown string>}\n"
)

def agent_step(user_msg: str) -> Dict[str, Any]:
    messages = f"System: {AGENT_SYSTEM_PROMPT}\nUser: {user_msg}"
    raw = llm_generate(messages, model, provider, temp, max_tokens)
    try:
        plan = json.loads(raw)
    except Exception:
        plan = {"action": "final", "answer": raw}
    return plan

# ---------------- UI Flow ----------------
# If Test LLM was requested in sidebar before llm_generate was defined, run it now:
if st.session_state.get("_test_llm_requested"):
    # only run once
    test_prompt = st.session_state.pop("_test_prompt", "Hello")
    st.session_state.pop("_test_llm_requested", None)
    st.sidebar.info("Running quick test...")
    out = llm_generate(test_prompt, model, provider, temp, max_tokens)
    if out.startswith("__LLM error__"):
        st.sidebar.error(out)
    else:
        st.sidebar.success("LLM test success: " + (out[:250] + ("..." if len(out) > 250 else "")))
    # continue rendering main app

uploaded = st.file_uploader("📎 Drop a PDF", type=["pdf"], accept_multiple_files=False)
if uploaded is not None:
    with st.spinner("Reading PDF..."):
        file_bytes = uploaded.read()
        pdf.load(file_bytes)
    st.success("PDF loaded.")
    st.write(f"Pages: **{len(pdf.pages)}**")

    if st.button("🧠 Generate Executive Summary"):
        with st.spinner("Summarizing..."):
            summary = pdf.summarize_pdf()
            st.subheader("Executive Summary")
            st.write(summary)
            export_summary_btn_placeholder.download_button(
                "📥 Download Summary (.txt)",
                data=pdf.export_summary(),
                file_name="summary.txt",
                mime="text/plain",
            )

# Chat section
st.markdown("---")
st.subheader("💬 Chat with the Agent")
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Type a question or instruction")

if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            plan = agent_step(prompt)
            if plan.get("action") == "tool":
                tool = plan.get("tool")
                args = plan.get("args", {})
                result = ""
                if tool == "summarize_pdf":
                    result = pdf.summarize_pdf()
                elif tool == "answer_from_pdf":
                    q = args.get("question") or prompt
                    result = pdf.answer_from_pdf(q)
                elif tool == "list_sections":
                    result = pdf.list_sections()
                elif tool == "export_summary":
                    data = pdf.export_summary()
                    st.download_button("📥 Download Summary (.txt)", data=data, file_name="summary.txt", mime="text/plain")
                    result = "Summary ready to download above."
                else:
                    result = f"Unknown tool: {tool}"

                st.markdown(result)
                st.session_state.chat.append(("assistant", result))
            else:
                answer = plan.get("answer", "I couldn't determine an action.")
                st.markdown(answer)
                st.session_state.chat.append(("assistant", answer))

# Export chat transcript
if st.session_state.chat:
    transcript = "\n\n".join([f"{r.upper()}: {c}" for r, c in st.session_state.chat])
    export_chat_btn_placeholder.download_button(
        "💾 Download Chat (.txt)", data=transcript.encode("utf-8"), file_name="chat_transcript.txt", mime="text/plain"
    )
