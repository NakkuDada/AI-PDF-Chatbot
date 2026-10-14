
import json
from pathlib import Path
import hashlib

from retrieval import retrieve_chunks

# Replace these examples with questions
# whose answers are present in your test PDF.
test_queries = [
    "What are the technical skills mentioned in this document?",
    "What is the main topic of this document?"
]

for query in test_queries:
    print("\nQuestion:", query)

    try:
        results = retrieve_chunks(query, n_results=5)

        if not results:
            print("No relevant chunks found.")
            continue

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}:")

            if isinstance(result, dict):
                print(result.get("text", result))
            else:
                print(result)

    except Exception as error:
        print("Retrieval failed:", error)


from rag_pipeline import (
    extract_pdf_pages,
    create_chunks,
    index_document,
    retrieve_chunks
)

pdf_path = Path("documents/sample.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(
        "Add your test PDF as documents/sample.pdf"
    )

pdf_bytes = pdf_path.read_bytes()

document_id = hashlib.sha256(
    pdf_bytes
).hexdigest()

pages = extract_pdf_pages(pdf_bytes)
chunks = create_chunks(pages)
index_document(document_id, chunks)

with open(
    "evaluation_questions.json",
    encoding="utf-8"
) as file:
    questions = json.load(file)

hits = 0

for item in questions:
    results = retrieve_chunks(
        document_id,
        item["question"],
        top_k=4
    )

    expected_page = item["expected_page"]

    hit = any(
        result["page"] == expected_page
        for result in results
    )

    hits += int(hit)

    print("Question:", item["question"])
    print("Expected page:", expected_page)
    print("Retrieved pages:", [
        result["page"] for result in results
    ])
    print("Hit:", hit)
    print("-" * 30)

if questions:
    print(
        "Retrieval hit rate:",
        f"{hits / len(questions):.2%}"
    )
else:
    print("No evaluation questions available.")
