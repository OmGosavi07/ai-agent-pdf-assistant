# AI Agent PDF Assistant

Streamlit app that extracts text from PDFs and runs a local LLM (Ollama) or OpenAI to summarize and answer questions.

Run locally:
```bash
python -m venv .venv
# activate .venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
# or (macOS/Linux)
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
