
import io

import pymupdf
import pdfplumber
import pytesseract

from PIL import Image


def extract_pdf_pages(pdf_bytes):
    """Extract text from ordinary and scanned PDF pages."""
    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                )

                text = pytesseract.image_to_string(
                    image
                ).strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No readable text found. Check the PDF or OCR setup."
        )

    return pages


def extract_pdf_tables(pdf_bytes):
    """Extract tables separately, preserving page numbers."""
    tables = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if table:
                    tables.append({
                        "page": page_number,
                        "rows": table
                    })

    return tables
