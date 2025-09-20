# AI Agent PDF Assistant

Streamlit app: upload a PDF, ask questions or get executive summary using a local Ollama model or OpenAI.

## Quick start
1. Create virtual env, install packages:
   - Windows PowerShell:
     ```
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     streamlit run app.py
     ```
2. To use Ollama, ensure `ollama serve` is running locally and the model is pulled.

## Config
- Use `.streamlit/secrets.toml` or environment variables for keys:
  - `OPENAI_API_KEY`
  - `OLLAMA_URL` (optional; default http://localhost:11434)

## Notes
- This app sends the entire PDF to the model (no chunking). For large PDFs, prefer OpenAI or enable chunking fallback.
