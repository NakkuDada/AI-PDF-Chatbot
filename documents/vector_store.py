
import chromadb
from sentence_transformers import SentenceTransformer

# Load the model that converts text into embeddings.
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Create a persistent vector database.
# Run Streamlit from the main project folder.
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_chunks"
)


def index_document(document_id, chunks):
    """Save PDF text chunks and their embeddings."""

    if not chunks:
        raise ValueError("No PDF chunks were provided.")

    # Remove old entries for this document before re-indexing.
    existing = collection.get(
        where={"document_id": document_id}
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    ids = [
        f"{document_id}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": document_id,
            "page": int(chunk["page"])
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def retrieve_chunks(document_id, question, top_k=4):
    """Search the vector database for relevant PDF passages."""

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"document_id": document_id},
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    if not results["ids"] or not results["ids"][0]:
        return retrieved

    for i, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        retrieved.append({
            "text": text,
            "page": metadata["page"],
            "distance": results["distances"][0][i]
        })

    return retrieved

def search_vectors(query, n_results=5, document_id=None):
    """Search the vector database using a query."""

    if not document_id:
        raise ValueError("document_id is required.")

    return retrieve_chunks(
        document_id=document_id,
        question=query,
        top_k=n_results
    )