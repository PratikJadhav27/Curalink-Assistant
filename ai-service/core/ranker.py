"""
Ranker - Combines semantic similarity + recency scoring to pick top documents
"""
import re
import numpy as np
from datetime import datetime
from core.embeddings import EmbeddingService
# NOTE: No sklearn needed — sentence-transformers returns L2-normalized vectors,
# so dot product already equals cosine similarity.


CURRENT_YEAR = datetime.now().year


def rank_documents(query: str, documents: list[dict], top_k: int = 8) -> list[dict]:
    """
    Rank documents by a combined score:
      60% semantic similarity (embedding cosine)
      30% recency (publication year normalized)
      10% source credibility bonus
    Returns top_k documents with scores attached.
    """
    if not documents:
        return []

    # Build texts for embedding
    texts = []
    for doc in documents:
        snippet = f"{doc.get('title', '')} {doc.get('abstract', '')}"
        texts.append(snippet[:512])  # truncate to reasonable size

    # Embed query + all docs in one batch
    all_texts = [query] + texts
    vectors = EmbeddingService.embed(all_texts)
    query_vec = vectors[0]
    doc_vecs = vectors[1:]

    semantic_scores = EmbeddingService.cosine_similarity(query_vec, doc_vecs)

    # Recency score
    years = []
    for doc in documents:
        year = _extract_year(doc)
        years.append(year)

    min_year = min(years) if years else CURRENT_YEAR - 10
    max_year = max(years) if years else CURRENT_YEAR
    year_range = max(max_year - min_year, 1)

    recency_scores = np.array([(y - min_year) / year_range for y in years])

    # Source credibility bonus
    credibility_scores = np.array([
        _credibility_bonus(doc.get("source", "")) for doc in documents
    ])

    # Combined score
    combined = 0.60 * semantic_scores + 0.30 * recency_scores + 0.10 * credibility_scores

    # Attach scores and sort
    for i, doc in enumerate(documents):
        doc["relevanceScore"] = float(combined[i])
        doc["year"] = years[i]

    ranked = sorted(documents, key=lambda d: d["relevanceScore"], reverse=True)
    return ranked[:top_k]


def _extract_year(doc: dict) -> int:
    raw = doc.get("year") or doc.get("publicationDate", "")
    if isinstance(raw, int):
        return raw
    match = re.search(r"\b(19|20)\d{2}\b", str(raw))
    if match:
        return int(match.group())
    return CURRENT_YEAR - 5  # neutral default


def _credibility_bonus(source: str) -> float:
    src = source.lower()
    if "pubmed" in src:
        return 1.0
    if "openalex" in src:
        return 0.8
    return 0.5
