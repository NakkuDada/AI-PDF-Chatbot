
import os
import re

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import os
import requests
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()


def generate_answer_ollama(question, retrieved_chunks):
    if not retrieved_chunks:
        return (
            "I couldn't find relevant information "
            "in the uploaded PDF."
        )
    model = os.getenv("OLLAMA_MODEL", "").strip()
    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ).rstrip("/")

    if not OLLAMA_MODEL:
        raise RuntimeError(
            "OLLAMA_MODEL is missing from your .env file."
        )

    context = "\n\n".join(
        f"Page {chunk['page']}: {chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = (
        "Answer using only the supplied PDF context. "
        "If the answer is not present, say you could not "
        "find it in the PDF.\n\n"
        f"PDF context:\n{context}\n\n"
        f"Question: {question}"
    )

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code == 404:
            raise RuntimeError(
                "Ollama endpoint or model not found. "
                "Check the URL and installed model name."
            )

        response.raise_for_status()
        data = response.json()

        answer = data.get("message", {}).get("content")

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        return answer

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Start Ollama "
            "and check OLLAMA_BASE_URL."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Ollama took too long to respond."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama HTTP error: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "Ollama returned an invalid JSON response."
        ) from exc


try:
    from .vector_store import (
        index_document,
        retrieve_chunks,
    )
except ImportError:
    from vector_store import (
        index_document,
        retrieve_chunks,
    )



def get_groq_settings():
    import streamlit as st

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")

    try:
        api_key = api_key or st.secrets.get("GROQ_API_KEY")
        model = model or st.secrets.get("GROQ_MODEL")
    except Exception:
        # Streamlit secrets may not be configured locally.
        pass

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    if not model:
        raise RuntimeError("GROQ_MODEL is not configured.")

    return api_key, model


def validate_pdf(pdf_bytes):
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF must be 10 MB or smaller.")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")

load_dotenv()


def extract_pdf_pages(pdf_bytes):
    """Extract selectable text from each PDF page."""

    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No selectable text was found. This PDF may require OCR."
        )

    return pages


def create_chunks(pages, chunk_size=180, overlap=40):
    """Split PDF text into overlapping word chunks."""

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    chunks = []

    for page_data in pages:
        words = re.findall(r"\S+", page_data["text"])
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))

            chunks.append({
                "page": page_data["page"],
                "text": " ".join(words[start:end])
            })

            if end == len(words):
                break

            start = end - overlap

    if not chunks:
        raise ValueError("No usable text chunks were created.")

    return chunks


def generate_answer(question, retrieved_chunks):
    """Generate an answer using a local Ollama model."""

    if not retrieved_chunks:
        return (
            "I could not find relevant information in the indexed PDF.",
            []
        )

    context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in retrieved_chunks
    )

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    local_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    response = local_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied PDF passages. "
                    "If the answer is not supported by the passages, "
                    "say that you could not find it. Cite pages using "
                    "[Page N]. Do not invent facts or citations."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PDF PASSAGES:\n{context}\n\n"
                    f"QUESTION:\n{question}"
                )
            }
        ],
        temperature=0.1
    )

    answer = response.choices[0].message.content

    if not answer:
        answer = "No text answer was returned."

    return answer, retrieved_chunks

MAX_PDF_BYTES = 10 * 1024 * 1024





