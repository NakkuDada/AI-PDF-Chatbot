
import json
import os
import secrets
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=True)

# --------------------------------------------------
# Read API token configuration AFTER loading .env
# --------------------------------------------------
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_QUESTION_CHARS = 2000
DATABASE_PATH = PROJECT_ROOT / "api_metadata.sqlite3"

API_TOKENS = json.loads(os.getenv("API_TOKENS_JSON", "{}"))

if not API_TOKENS:
    raise RuntimeError(
        "Configure API_TOKENS_JSON in your .env before starting the API."
    )

for token, user_id in API_TOKENS.items():
    if len(token) < 32 or not user_id:
        raise RuntimeError(
            "Each API token must be at least 32 characters "
            "and have a non-empty user ID."
        )
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag_pipeline import (
    extract_pdf_pages,
    create_chunks,
    index_document,
    retrieve_chunks,
    generate_answer,
)

load_dotenv()

# ---------- Configuration ----------

MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_QUESTION_CHARS = 2000
DATABASE_PATH = Path("./api_metadata.sqlite3")

# Each API token maps to a user ID.
# Example: {"long-secret-token": "user1"}
try:
    API_TOKENS = json.loads(
        os.getenv("API_TOKENS_JSON", "{}")
    )
except json.JSONDecodeError:
    raise RuntimeError("API_TOKENS_JSON must contain valid JSON.")

if not isinstance(API_TOKENS, dict) or not API_TOKENS:
    raise RuntimeError(
        "Configure API_TOKENS_JSON in your .env before starting the API."
    )

if any(
    not isinstance(token, str)
    or len(token) < 32
    or not isinstance(user_id, str)
    or not user_id.strip()
    for token, user_id in API_TOKENS.items()
):
    raise RuntimeError(
        "Each API token must be at least 32 characters "
        "and map to a non-empty user ID."
    )

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AI PDF Chatbot API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

security = HTTPBearer(auto_error=False)


# ---------- Document ownership database ----------

def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


initialize_database()


def save_document_owner(document_id, owner_id, filename):
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            INSERT INTO documents (document_id, owner_id, filename)
            VALUES (?, ?, ?)
            """,
            (document_id, owner_id, filename),
        )
        conn.commit()


def user_owns_document(document_id, owner_id):
    with sqlite3.connect(DATABASE_PATH) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM documents
            WHERE document_id = ? AND owner_id = ?
            """,
            (document_id, owner_id),
        ).fetchone()

    return row is not None


# ---------- Authentication ----------

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Valid bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supplied_token = credentials.credentials

    # Constant-time comparison; tokens are provisioned via .env.
    for expected_token, user_id in API_TOKENS.items():
        if secrets.compare_digest(supplied_token, expected_token):
            return user_id

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------- Request models ----------

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS
    )


# ---------- Health check ----------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- PDF upload endpoint ----------

@app.post("/documents", status_code=201)
@limiter.limit("5/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are accepted."
        )

    if file.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file content type."
        )

    try:
        # Read at most MAX_PDF_BYTES + 1 to detect oversized files.
        pdf_bytes = await file.read(MAX_PDF_BYTES + 1)

        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 10 MB upload limit."
            )

        # Check the PDF signature; MIME type alone is not trustworthy.
        if not pdf_bytes.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=415,
                detail="The uploaded file is not a valid PDF."
            )

        # Reuse the existing RAG pipeline.
        pages = extract_pdf_pages(pdf_bytes)
        chunks = create_chunks(pages)

        document_id = str(uuid.uuid4())

        index_document(document_id, chunks)

        try:
            save_document_owner(
                document_id,
                current_user,
                filename
            )
        except Exception:
            # Best-effort rollback if ownership metadata cannot be saved.
            from vector_store import collection

            existing = collection.get(
                where={"document_id": document_id}
            )
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            raise

        return {
            "document_id": document_id,
            "filename": filename,
            "pages_with_text": len(pages),
            "chunks_indexed": len(chunks),
            "message": "PDF indexed successfully."
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc
    except Exception:
        # Log detailed exceptions privately in production.
        raise HTTPException(
            status_code=500,
            detail="PDF processing failed. Check server logs."
        )
    finally:
        await file.close()


# ---------- Question endpoint ----------

@app.post("/documents/{document_id}/questions")
@limiter.limit("20/minute")
def ask_question(
    request: Request,
    document_id: str,
    body: QuestionRequest,
    current_user: str = Depends(get_current_user),
):
    # Do not let users query another user's document.
    if not user_owns_document(document_id, current_user):
        # Same response for unknown and unauthorized document IDs.
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        chunks = retrieve_chunks(
            document_id,
            body.question,
            top_k=4
        )

        answer, sources = generate_answer(
            body.question,
            chunks
        )

        source_pages = sorted({
            int(source["page"])
            for source in sources
        })

        return {
            "document_id": document_id,
            "question": body.question,
            "answer": answer,
            "source_pages": source_pages,
            "sources": [
                {
                    "page": int(source["page"]),
                    "text": source["text"]
                }
                for source in sources
            ]
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail=(
                "Answer generation failed. "
                "Check that Ollama is running and the model is available."
            )
        )

