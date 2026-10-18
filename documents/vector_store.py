from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"}
)


def index_document(document_id, chunks):
    if not chunks:
        raise ValueError("No text chunks were created.")

    # Replace any existing chunks for this document.
    collection.delete(
        where={"document_id": document_id}
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[
            f"{document_id}_{i}"
            for i in range(len(chunks))
        ],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "page": chunk["page"]
            }
            for chunk in chunks
        ]
    )


def retrieve_chunks(document_id, question, top_k=4):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"document_id": document_id}
    )

    retrieved = []

    if results["documents"] and results["documents"][0]:
        for i, content in enumerate(results["documents"][0]):
            retrieved.append({
                "text": content,
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i]
            })

    return retrieved