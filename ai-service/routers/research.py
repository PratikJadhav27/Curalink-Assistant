"""
Research Router - Main endpoint orchestrating all retrievers + ranking + LLM
"""
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from retrievers.pubmed import fetch_pubmed
from retrievers.openalex import fetch_openalex
from retrievers.clinical_trials import fetch_clinical_trials
from core.ranker import rank_documents
from core.llm_service import expand_query, synthesize_response

router = APIRouter()


class PatientContext(BaseModel):
    name: str = ""
    disease: str = ""
    location: str = ""
    additionalInfo: str = ""


class ResearchRequest(BaseModel):
    query: str
    patient_context: PatientContext = PatientContext()
    conversation_history: list[dict] = []


@router.post("/research")
async def research(req: ResearchRequest):
    """
    Main research endpoint:
    1. Expand user query using LLM
    2. Retrieve 50-300 candidates concurrently from 3 APIs
    3. Rank and filter to top 6-8
    4. Synthesize structured answer with LLM
    """
    disease = req.patient_context.disease or _infer_disease(req.query)
    location = req.patient_context.location

    # Step 1: Expand query with LLM
    try:
        expanded_query = expand_query(
            disease=disease,
            query=req.query,
            patient_context=req.patient_context.dict(),
        )
    except Exception:
        expanded_query = f"{disease} {req.query}".strip()

    print(f"🔍 Expanded query: '{expanded_query}'")

    # Step 2: Concurrent retrieval from all 3 sources
    pub_task_pm = fetch_pubmed(expanded_query, max_results=50)
    pub_task_oa = fetch_openalex(expanded_query, max_results=50)
    trial_task = fetch_clinical_trials(
        disease=disease,
        query=req.query,
        location=location or None,
        max_results=30,
    )
    # Also fetch recruiting-specific trials
    trial_task_recruiting = fetch_clinical_trials(
        disease=disease,
        query=req.query,
        location=location or None,
        max_results=20,
        status_filter="RECRUITING",
    )

    pubmed_results, openalex_results, trial_results, recruiting_trials = await asyncio.gather(
        pub_task_pm,
        pub_task_oa,
        trial_task,
        trial_task_recruiting,
    )

    # Merge + deduplicate publications
    all_publications = _deduplicate(pubmed_results + openalex_results)
    print(f"📚 Retrieved {len(all_publications)} total publications before ranking")

    # Merge + deduplicate trials
    all_trials = _deduplicate_trials(trial_results + recruiting_trials)
    print(f"🧪 Retrieved {len(all_trials)} clinical trials before ranking")

    # Step 3: Rank publications (use original query for disease filtering, expanded for semantics)
    if all_publications:
        top_publications = rank_documents(req.query, all_publications, top_k=8)
    else:
        top_publications = []

    # Rank trials by semantic similarity only (simpler)
    if all_trials:
        trial_texts = [f"{t.get('title', '')} {t.get('eligibility', '')}" for t in all_trials]
        from core.embeddings import EmbeddingService
        import numpy as np

        if len(all_trials) > 5:
            query_vec = EmbeddingService.embed([expanded_query])[0]
            trial_vecs = EmbeddingService.embed(trial_texts)
            scores = EmbeddingService.cosine_similarity(query_vec, trial_vecs)
            for i, t in enumerate(all_trials):
                t["_score"] = float(scores[i])
            all_trials.sort(key=lambda t: t.get("_score", 0), reverse=True)

        top_trials = all_trials[:6]
    else:
        top_trials = []

    # Step 4: LLM synthesis
    synthesis = synthesize_response(
        query=req.query,
        patient_context=req.patient_context.dict(),
        publications=top_publications,
        clinical_trials=top_trials,
        conversation_history=req.conversation_history,
    )

    return {
        "queryExpanded": expanded_query,
        "conditionOverview": synthesis.get("conditionOverview", ""),
        "answer": synthesis.get("answer", ""),
        "publications": top_publications,
        "clinicalTrials": top_trials,
        "stats": {
            "totalPublicationsRetrieved": len(all_publications),
            "totalTrialsRetrieved": len(all_trials),
            "topPublicationsShown": len(top_publications),
            "topTrialsShown": len(top_trials),
        },
    }


def _infer_disease(query: str) -> str:
    """
    Extract the disease/condition name from the user's query.
    Uses a keyword map first, then strips common stopwords as fallback.
    """
    DISEASE_KEYWORDS = [
        "diabetes", "alzheimer", "cancer", "hypertension", "asthma", "covid",
        "depression", "parkinson", "obesity", "stroke", "arthritis", "dementia",
        "heart disease", "lung cancer", "breast cancer", "leukemia", "lymphoma",
        "multiple sclerosis", "epilepsy", "tuberculosis", "hepatitis", "hiv",
        "fibromyalgia", "lupus", "crohn", "colitis", "psoriasis", "eczema",
        "anxiety", "schizophrenia", "bipolar",
    ]
    q_lower = query.lower()
    for kw in DISEASE_KEYWORDS:
        if kw in q_lower:
            return kw
    # Fallback: strip medical/common stopwords and return first meaningful word
    STOP = {
        "latest", "recent", "new", "treatment", "for", "of", "the", "a", "an",
        "in", "and", "or", "with", "clinical", "trials", "study", "research",
        "what", "is", "are", "how", "best", "top", "find", "show", "get",
        "papers", "about", "causes", "symptoms", "diagnosis"
    }
    words = [w for w in q_lower.split() if w not in STOP and len(w) > 3]
    return words[0] if words else query


def _deduplicate(docs: list[dict]) -> list[dict]:
    """Remove duplicate publications by title (lowercased)."""
    seen = set()
    unique = []
    for doc in docs:
        key = doc.get("title", "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _deduplicate_trials(trials: list[dict]) -> list[dict]:
    """Remove duplicate trials by NCT ID."""
    seen = set()
    unique = []
    for t in trials:
        nct = t.get("nctId", "")
        if nct and nct not in seen:
            seen.add(nct)
            unique.append(t)
        elif not nct:
            unique.append(t)
    return unique
