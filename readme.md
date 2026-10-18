
# AI PDF Chatbot using RAG

A PDF question-answering application built with Python,
Streamlit, ChromaDB, and Sentence Transformers.

## Features
- PDF upload and text extraction
- Text chunking and vector indexing
- Semantic retrieval from uploaded documents
- Local answer generation with Ollama
- Optional Groq API integration
- Source chunks and page references
- Pytest error-handling tests

## Local Setup
1. Install Python and Ollama.
2. Download a model, for example `ollama pull llama3.2:3b`.
3. Create and activate a Python virtual environment.
4. Install dependencies using `pip install -r requirements.txt`.
5. Configure the root `.env` file.
6. Run `python -m streamlit run documents/app.py`.

## Tests
Run `python -m pytest tests -v`.

## Security
Never commit `.env`, API keys, private PDFs, or local database files.
