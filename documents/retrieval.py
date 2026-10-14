
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from vector_store import search_vectors
from vector_store import retrieve_chunks


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def hybrid_search(document_id, question, chunks, top_k=4):
    """Combine vector and keyword search, then rerank."""

    if not chunks:
        return []

    # Step 1: Semantic vector retrieval
    vector_results = search_vectors(
        query=question,
        n_results=min(20, len(chunks)),
        document_id=document_id
    )

    # Step 2: Keyword retrieval using BM25
    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    keyword_scores = bm25.get_scores(
        question.lower().split()
    )

    keyword_order = sorted(
        range(len(chunks)),
        key=lambda i: keyword_scores[i],
        reverse=True
    )[:min(20, len(chunks))]

    # Step 3: Match vector results to original chunks
    lookup = {}

    for i, chunk in enumerate(chunks):
        lookup.setdefault(
            (chunk.get("page", 1), chunk["text"]),
            i
        )

    # Step 4: Reciprocal Rank Fusion (RRF)
    fused_scores = {}

    for rank, item in enumerate(vector_results, start=1):
        idx = lookup.get(
            (item.get("page", 1), item["text"])
        )

        if idx is not None:
            fused_scores[idx] = (
                fused_scores.get(idx, 0)
                + 1 / (60 + rank)
            )

    for rank, idx in enumerate(keyword_order, start=1):
        fused_scores[idx] = (
            fused_scores.get(idx, 0)
            + 1 / (60 + rank)
        )

    candidate_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )[:20]

    candidates = [chunks[i] for i in candidate_ids]

    if not candidates:
        return []

    # Step 5: Rerank question and passage pairs
    reranker = get_reranker()

    scores = reranker.predict([
        (question, item["text"])
        for item in candidates
    ])

    ranked = sorted(
        zip(candidates, scores),
        key=lambda pair: float(pair[1]),
        reverse=True
    )

    return [
        {
            **chunk,
            "rerank_score": float(score)
        }
        for chunk, score in ranked[:top_k]
    ]

