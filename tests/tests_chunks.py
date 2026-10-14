import sys
from pathlib import Path

import pytest

# Get the AI-PDF-Chatbot project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Get the documents folder
DOCUMENTS_DIR = PROJECT_ROOT / "documents"



# Check whether rag_pipeline.py exists
RAG_PIPELINE_FILE = DOCUMENTS_DIR / "rag_pipeline.py"

if not RAG_PIPELINE_FILE.is_file():
    raise FileNotFoundError(
        f"rag_pipeline.py was not found at: {RAG_PIPELINE_FILE}"
    )

# Allow Python to import rag_pipeline.py
sys.path.insert(0, str(DOCUMENTS_DIR))

from rag_pipeline import create_chunks


def test_chunks_preserve_page_number():
    pages = [{
        "page": 2,
        "text": "one two three four five six seven eight"
    }]

    chunks = create_chunks(
        pages,
        chunk_size=4,
        overlap=1
    )

    assert chunks[0]["page"] == 2
    assert chunks[0]["text"] == "one two three four"
    assert chunks[1]["text"] == "four five six seven"


def test_invalid_chunk_settings():
    with pytest.raises(ValueError):
        create_chunks(
            [{"page": 1, "text": "example text"}],
            chunk_size=5,
            overlap=5
        )
