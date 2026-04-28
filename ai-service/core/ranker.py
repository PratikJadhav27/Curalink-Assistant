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

    # Disease relevance filter — applied to BOTH publications and clinical trials
    # Penalizes documents that don't mention the queried disease anywhere in their text.
    # This prevents methodology papers and off-topic results from surfacing.
    disease_keywords = _extract_disease_keywords(query)
    if disease_keywords:
        for i, doc in enumerate(documents):
            is_trial = bool(doc.get("nctId") or doc.get("type") == "clinical_trial")
            doc_text = (
                doc.get("title", "") + " " +
                doc.get("abstract", "") + " " +
                " ".join(doc.get("conditions", [])) +
                " " + doc.get("description", "")
            ).lower()
            if not any(kw.lower() in doc_text for kw in disease_keywords):
                # Clinical trials: heavier penalty (must be disease-specific)
                # Publications: lighter penalty (methodology papers sometimes mention disease)
                combined[i] *= 0.30 if is_trial else 0.45

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


def _extract_disease_keywords(query: str) -> list[str]:
    """
    Extract disease-related keywords from the query to use as a relevance filter
    for clinical trials. Returns lowercase keyword strings.
    """
    # Common disease aliases for robust matching
    disease_map = {
        "diabetes":    ["diabetes", "diabetic", "glycem", "insulin", "glucose", "t2dm", "t1dm", "hba1c"],
        "alzheimer":   ["alzheimer", "dementia", "cognitive decline", "amyloid", "tau"],
        "cancer":      ["cancer", "tumor", "tumour", "oncol", "carcinoma", "malignant"],
        "heart":       ["heart", "cardiac", "cardiovascular", "coronary", "myocardial"],
        "hypertension":["hypertension", "blood pressure", "antihypertens"],
        "asthma":      ["asthma", "bronchial", "airway"],
        "covid":       ["covid", "sars-cov", "coronavirus"],
        "depression":  ["depression", "depressive", "antidepressant"],
        "parkinson":   ["parkinson", "dopamin", "lewy"],
        "obesity":     ["obesity", "obese", "overweight", "bmi", "weight loss"],
    }
    q_lower = query.lower()
    for disease, keywords in disease_map.items():
        if disease in q_lower or any(kw in q_lower for kw in keywords):
            return keywords
    # Fallback: use significant words from the query itself
    stop = {"the", "a", "an", "of", "for", "in", "on", "and", "or", "with", "latest", "recent", "new", "treatment", "study", "research"}
    words = [w for w in q_lower.split() if w not in stop and len(w) > 3]
    return words[:3]
