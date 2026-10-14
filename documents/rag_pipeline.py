
import os
import re

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

from vector_store import index_document, retrieve_chunks

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


