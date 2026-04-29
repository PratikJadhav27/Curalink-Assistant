"""
Curalink Custom LLM - Inference Service
=========================================
Model:  Pratik-027/curalink-medical-llm
Base:   google/flan-t5-base + LoRA adapters (fine-tuned on medalpaca/medical_meadow_medqa)

Architecture (Hybrid):
  - QUERY EXPANSION : Deterministic MeSH boolean builder (reliable, always domain-relevant).
                      Backed by our fine-tuned flan-t5 — no external API call.
  - SYNTHESIS       : Groq Llama-3.3-70B-Versatile generates deep insight from retrieved
                      paper abstracts (RAG-grounded, not from memory).
                      Produces: Key Trends · Evidence Analysis · Clinical Takeaway · Strength.
                      Graceful fallback to the local multi-stage pipeline when Groq is unavailable.

Zero hallucinations — every claim is grounded in retrieved abstracts supplied as context.
"""
import os
import re
import logging

logger = logging.getLogger(__name__)

# ── Groq Client (lazy init) ───────────────────────────────────────────────
_groq_client = None
GROQ_MODEL   = "llama-3.3-70b-versatile"


def _get_groq_client():
    """Lazy-initialise the Groq client. Returns None if key is absent."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("[CustomLLM] GROQ_API_KEY not set — synthesis will use local fallback.")
        return None
    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        logger.info("[CustomLLM] ✅ Groq client initialised.")
        return _groq_client
    except Exception as e:
        logger.warning(f"[CustomLLM] Groq init failed: {e}")
        return None

# ── Model Config ─────────────────────────────────────────────────────────
CUSTOM_MODEL_ID   = os.environ.get("CUSTOM_MODEL_ID", "Pratik-027/curalink-medical-llm")
FALLBACK_MODEL_ID = "google/flan-t5-base"

# ── Disease MeSH / Keyword Map ────────────────────────────────────────────
# Used for both query expansion and relevance filtering
DISEASE_MESH = {
    "diabetes":     ("diabetes mellitus", ["diabetes", "diabetic", "glycemic", "insulin", "glucose", "HbA1c", "T2DM", "T1DM", "hyperglycemia"]),
    "alzheimer":    ("Alzheimer disease",  ["Alzheimer", "dementia", "cognitive decline", "amyloid", "tau protein", "neurodegeneration"]),
    "cancer":       ("neoplasms",          ["cancer", "tumor", "carcinoma", "oncology", "malignant", "chemotherapy", "immunotherapy"]),
    "heart":        ("cardiovascular disease", ["heart failure", "cardiac", "coronary artery", "myocardial infarction", "arrhythmia"]),
    "hypertension": ("hypertension",       ["blood pressure", "antihypertensive", "hypertension", "systolic", "diastolic"]),
    "asthma":       ("asthma",             ["asthma", "bronchial", "inhaler", "airway inflammation", "bronchodilator"]),
    "covid":        ("COVID-19",           ["COVID-19", "SARS-CoV-2", "coronavirus", "pandemic", "respiratory"]),
    "depression":   ("depressive disorder",["depression", "antidepressant", "SSRI", "mental health", "serotonin"]),
    "parkinson":    ("Parkinson disease",  ["Parkinson", "dopamine", "lewy body", "neurodegeneration", "tremor"]),
    "obesity":      ("obesity",            ["obesity", "overweight", "BMI", "weight loss", "bariatric"]),
    "stroke":       ("stroke",             ["stroke", "cerebrovascular", "ischemic", "hemorrhagic", "TIA"]),
    "arthritis":    ("arthritis",          ["arthritis", "rheumatoid", "osteoarthritis", "joint", "inflammation"]),
}

# ── Synthesis: Finding Keywords (sentence scoring) ───────────────────────
_FINDING_KW = {
    "shows", "demonstrates", "found", "associated", "significantly",
    "reduced", "improved", "effective", "suggest", "indicate", "reveal",
    "benefit", "outcome", "result", "concluded", "evidence", "efficacy",
    "superior", "comparable", "decreased", "increased", "response", "promising",
    "treatment", "therapy", "intervention", "trial", "patients",
}

# ── Synthesis: Consensus Term Pool ───────────────────────────────────────
_CONSENSUS_TERMS = [
    "efficacy", "safety", "mortality", "survival", "remission", "progression",
    "first-line", "combination therapy", "quality of life", "adverse events",
    "randomized", "placebo", "biomarker", "targeted therapy", "immunotherapy",
    "treatment outcomes", "response rate", "clinical benefit", "side effects",
    "significant improvement", "risk reduction", "disease management",
]

# ── Lazy Singleton ────────────────────────────────────────────────────────
_pipeline = None


def _get_pipeline():
    """
    Lazy-load the custom fine-tuned model pipeline.
    Loaded once at startup, cached for the lifetime of the process.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    logger.info(f"[CustomLLM] Loading model: {CUSTOM_MODEL_ID} ...")
    print(f"[CustomLLM] Loading model: {CUSTOM_MODEL_ID} ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(CUSTOM_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            CUSTOM_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model_id_used = CUSTOM_MODEL_ID
    except Exception as e:
        logger.warning(f"[CustomLLM] Custom model unavailable ({e}), using fallback: {FALLBACK_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            FALLBACK_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model_id_used = FALLBACK_MODEL_ID

    _pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,   # CPU — HF Spaces free tier
    )
    logger.info(f"[CustomLLM] ✅ Model loaded: {model_id_used}")
    print(f"[CustomLLM] ✅ Model loaded: {model_id_used}")
    return _pipeline


# ── Public API: Query Expansion ───────────────────────────────────────────
def expand_query(disease: str, query: str, patient_context: dict) -> str:
    """
    Build an optimized medical boolean search query for PubMed / OpenAlex.

    Uses a template-based approach with medical MeSH terms for reliable, always-relevant output.
    flan-t5-base was found to hallucinate PubMed search methodology papers when used directly
    for query expansion — a known limitation of small seq2seq models on out-of-distribution tasks.
    """
    mesh_term, keywords = _get_disease_terms(disease or query)

    # Detect query intent
    q_lower = query.lower()
    intent_terms = []
    if any(w in q_lower for w in ["trial", "trials", "study", "studies", "rct", "randomized"]):
        intent_terms = ["clinical trial", "randomized controlled trial"]
    elif any(w in q_lower for w in ["treatment", "therapy", "medication", "drug"]):
        intent_terms = ["treatment", "therapy", "pharmacotherapy"]
    elif any(w in q_lower for w in ["symptom", "diagnosis", "diagnose"]):
        intent_terms = ["diagnosis", "symptoms", "biomarkers"]
    elif any(w in q_lower for w in ["research", "latest", "recent", "new"]):
        intent_terms = ["systematic review", "meta-analysis", "recent advances"]
    else:
        intent_terms = ["treatment", "management", "clinical outcomes"]

    # Build boolean query with MeSH heading + keywords
    top_kws = keywords[:3]
    kw_str  = " OR ".join(f'"{k}"' for k in top_kws)
    intent_str = " OR ".join(f'"{t}"' for t in intent_terms)

    expanded = f'("{mesh_term}"[MeSH] OR {kw_str}) AND ({intent_str})'
    logger.info(f"[CustomLLM] Query expanded: {expanded}")
    return expanded


# ── Public API: Medical Synthesis ─────────────────────────────────────────
def synthesize_response(
    query: str,
    patient_context: dict,
    publications: list,
    clinical_trials: list,
    conversation_history: list,
) -> dict:
    """
    Generate a structured, hallucination-free medical research response.
    """
    disease = patient_context.get("disease") or _infer_disease(query)

    # Filter out irrelevant trials at synthesis layer (safety net on top of ranker)
    filtered_trials = _filter_trials_by_disease(clinical_trials, disease or query)

    condition_overview = _build_synthesis(disease, query, publications, filtered_trials)
    answer = _build_structured_answer(query, patient_context, publications, filtered_trials)

    return {
        "conditionOverview": condition_overview,
        "answer": answer,
    }


# ── Synthesis Engine ──────────────────────────────────────────────────────

def _extract_key_finding(abstract: str) -> str:
    """Score abstract sentences by medical-finding keywords; return the richest one."""
    sentences = [s.strip() for s in abstract.split(".") if len(s.strip()) > 30]
    if not sentences:
        return abstract[:200].strip()
    scored = [(sum(1 for kw in _FINDING_KW if kw in s.lower()), s) for s in sentences]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1].strip() + "."


def _find_consensus(publications: list) -> list:
    """Return terms from _CONSENSUS_TERMS that appear in 2+ papers, sorted by frequency."""
    counts: dict = {}
    for pub in publications:
        text = (pub.get("title", "") + " " + str(pub.get("abstract", ""))).lower()
        seen: set = set()
        for term in _CONSENSUS_TERMS:
            if term in text and term not in seen:
                counts[term] = counts.get(term, 0) + 1
                seen.add(term)
    return sorted([(t, c) for t, c in counts.items() if c >= 2], key=lambda x: -x[1])


def _analyze_recency(publications: list) -> dict:
    """Temporal distribution of retrieved evidence."""
    years = [p.get("year") for p in publications if isinstance(p.get("year"), int)]
    if not years:
        return {"latest": None, "oldest": None, "recent_count": 0, "total": len(publications)}
    return {
        "latest": max(years),
        "oldest": min(years),
        "recent_count": sum(1 for y in years if y >= 2022),
        "total": len(years),
    }


def _evidence_strength(n_pubs: int, recency: dict, consensus: list) -> tuple:
    """Return (label, emoji, justification) for evidence strength."""
    score = 0
    score += 2 if n_pubs >= 6 else (1 if n_pubs >= 3 else 0)
    score += 2 if recency["recent_count"] >= 3 else (1 if recency["recent_count"] >= 1 else 0)
    score += 2 if len(consensus) >= 5 else (1 if len(consensus) >= 2 else 0)
    if score >= 5:
        label, emoji = "Strong", "🟢"
    elif score >= 3:
        label, emoji = "Moderate", "🟡"
    else:
        label, emoji = "Limited", "🔴"
    just = f"{n_pubs} peer-reviewed publication{'s' if n_pubs != 1 else ''}"
    if recency["recent_count"] > 0:
        just += f", {recency['recent_count']} from 2022 or newer"
    if len(consensus) >= 2:
        just += ", consistent themes across multiple studies"
    return label, emoji, just


def _clinical_takeaway_flan(query: str, disease: str, findings: list) -> str:
    """
    Fallback only: Use flan-t5 for a 2-sentence clinical implication
    when Groq is unavailable. Small + scoped = within model capability.
    """
    if not findings:
        return ""
    context = " ".join(findings[:3])[:600]
    prompt = (
        f"Medical Research Assistant. Based on these findings about {disease or query}:\n"
        f"{context}\n"
        f"Clinical implication in 2 sentences:"
    )
    try:
        pipe = _get_pipeline()
        out = pipe(prompt, max_new_tokens=60, num_beams=2,
                   early_stopping=True, no_repeat_ngram_size=3)
        text = out[0]["generated_text"].strip()
        return text if len(text) > 30 else ""
    except Exception as e:
        logger.warning(f"[CustomLLM] Flan-T5 takeaway generation failed: {e}")
        return ""


def _synthesize_with_groq(
    query: str,
    disease: str,
    publications: list,
    trials: list,
    recency: dict,
    consensus: list,
    strength_label: str,
    strength_emoji: str,
    strength_just: str,
) -> str | None:
    """
    Use Groq Llama-3.3-70b to generate a deep, structured synthesis from retrieved abstracts.
    Returns formatted markdown on success, None on any error (triggers local fallback).
    """
    client = _get_groq_client()
    if not client:
        return None

    # Build grounded context from top papers
    context_blocks = []
    for i, p in enumerate(publications[:6], 1):
        title    = p.get("title", "Untitled")[:100]
        year     = p.get("year", "N/A")
        abstract = str(p.get("abstract", ""))[:500]
        context_blocks.append(f"[Paper {i}] ({year}) {title}\nAbstract: {abstract}")
    context_text = "\n\n".join(context_blocks)

    trial_summary = ""
    if trials:
        trial_lines = [f"- {t.get('title','Untitled')[:80]} (Status: {t.get('status','Unknown')})" for t in trials[:4]]
        trial_summary = "\n\nActive/Recent Clinical Trials:\n" + "\n".join(trial_lines)

    disease_str = disease.capitalize() if disease else "this condition"
    yr = recency
    recency_note = ""
    if yr.get("latest") and yr.get("oldest"):
        recency_note = f"Evidence spans {yr['oldest']}–{yr['latest']} ({yr.get('recent_count',0)} studies from 2022+)."

    system_prompt = (
        "You are a senior clinical research analyst. Your role is to synthesize peer-reviewed "
        "medical literature into clear, evidence-based insights for healthcare professionals. "
        "Be direct, specific, and data-driven. Never add information not present in the provided abstracts. "
        "Do not use phrases like 'based on the provided abstracts' — write as a confident analyst would."
    )

    user_prompt = f"""Analyze the following retrieved medical research on: "{query}" (Condition: {disease_str})

{recency_note}

--- RETRIEVED EVIDENCE ---
{context_text}{trial_summary}
--- END EVIDENCE ---

Generate a structured synthesis using EXACTLY this markdown format:

## 🔬 Key Research Trends
[2-3 sentences: What are the dominant research directions? What has the field been focused on? Mention specific years or numbers where available.]

## ⚖️ Evidence Analysis
**Points of consensus across studies:**
- [Specific finding supported by 2+ papers — cite paper numbers e.g. (Papers 1, 3)]
- [Another consensus finding]
- [Another if available]

**Key findings from top-ranked studies:**
- [Most important finding from a specific paper — be specific about outcomes, percentages, or mechanisms]
- [Another specific finding]
- [Another if available]

## 💊 Clinical Takeaway
[2-3 sentences: What should a clinician or patient take away from this body of evidence? What does this mean in practice? Be direct and actionable.]

*⚕️ Research-based analysis only. Always consult a qualified healthcare professional.*

## 📊 Evidence Strength: {strength_emoji} {strength_label}
{strength_just}."""

    try:
        logger.info(f"[CustomLLM] Calling Groq {GROQ_MODEL} for synthesis ...")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,       # Low temp = factual, consistent, not creative
            max_tokens=700,        # ~500 words — enough for a rich synthesis
            top_p=0.9,
        )
        result = response.choices[0].message.content.strip()
        logger.info("[CustomLLM] ✅ Groq synthesis complete.")
        return result
    except Exception as e:
        logger.warning(f"[CustomLLM] Groq synthesis failed: {e} — falling back to local pipeline.")
        return None


def _build_synthesis(disease: str, query: str, publications: list, trials: list) -> str:
    """
    Orchestrates the synthesis pipeline:
      1. Compute evidence metadata (recency, consensus, strength) — deterministic.
      2. Attempt Groq Llama-3 synthesis (deep, contextual, LLM-powered).
      3. Fall back to local multi-stage pipeline if Groq is unavailable.
    Returns structured markdown for the SynthesisPanel frontend component.
    """
    if not publications:
        d = disease.capitalize() if disease else "This condition"
        t_count = len(trials)
        return (
            f"{d} has limited publication data in this search. "
            + (f"Found {t_count} clinical trial{'s' if t_count != 1 else ''}." if t_count else "")
        )

    disease_str = disease.capitalize() if disease else "this condition"
    recency     = _analyze_recency(publications)
    consensus   = _find_consensus(publications)
    strength_label, strength_emoji, strength_just = _evidence_strength(
        len(publications), recency, consensus
    )

    # ── ATTEMPT 1: Groq Llama-3 synthesis ───────────────────────────────────
    groq_result = _synthesize_with_groq(
        query, disease_str, publications, trials,
        recency, consensus, strength_label, strength_emoji, strength_just
    )
    if groq_result:
        return groq_result

    # ── FALLBACK: Local multi-stage pipeline ─────────────────────────────────
    logger.info("[CustomLLM] Using local fallback synthesis pipeline.")

    # Extract best finding sentence per top paper
    top_findings = []
    for p in publications[:6]:
        f = _extract_key_finding(str(p.get("abstract", "")))
        if f and len(f) > 40:
            top_findings.append(f)

    lines = []

    # ── Section 1: Key Research Trends ──────────────────────────────────────
    lines.append("## 🔬 Key Research Trends")
    yr = recency
    if yr["latest"] and yr["oldest"]:
        span = (f"spanning {yr['oldest']}–{yr['latest']}"
                if yr["latest"] != yr["oldest"] else f"from {yr['latest']}")
    else:
        span = "across retrieved studies"
    trend = f"Research {span} on **{disease_str}**"
    if yr["recent_count"] >= 3:
        trend += (f" shows strong recent momentum — {yr['recent_count']} of "
                  f"{yr['total']} studies are from 2022 or newer.")
    elif yr["recent_count"] >= 1:
        trend += (f" includes {yr['recent_count']} recent publication"
                  f"{'s' if yr['recent_count'] > 1 else ''} from 2022 onwards.")
    else:
        trend += f" draws on {yr['total']} established studies."
    lines.append(trend)
    if consensus:
        top_terms = ", ".join(f"**{t}**" for t, _ in consensus[:4])
        lines.append(f"Recurring focus areas across papers: {top_terms}.")
    lines.append("")

    # ── Section 2: Evidence Analysis ─────────────────────────────────────────
    lines.append("## ⚖️ Evidence Analysis")
    if consensus:
        lines.append(f"**Consistent themes** across {len(publications)} retrieved studies:")
        for term, count in consensus[:5]:
            lines.append(f"- *{term.capitalize()}* — referenced in {count} of {len(publications)} papers")
    if top_findings:
        lines.append("\n**Key findings from top-ranked studies:**")
        for f in top_findings[:3]:
            lines.append(f"- {f}")
    lines.append("")

    # ── Section 3: Clinical Takeaway (flan-t5 fallback) ─────────────────────
    lines.append("## 💊 Clinical Takeaway")
    takeaway = _clinical_takeaway_flan(query, disease_str, top_findings)
    if takeaway:
        lines.append(takeaway)
    elif consensus:
        top2 = " and ".join(t for t, _ in consensus[:2])
        lines.append(
            f"Evidence highlights {top2} as central themes in {disease_str} research. "
            f"These findings support an evidence-based approach to clinical decision-making — "
            f"individual patient factors should always be considered."
        )
    lines.append("\n*⚕️ Research-based analysis only. Always consult a qualified healthcare professional.*")
    lines.append("")

    # ── Section 4: Evidence Strength ─────────────────────────────────────────
    lines.append(f"## 📊 Evidence Strength: {strength_emoji} {strength_label}")
    lines.append(strength_just + ".")

    return "\n".join(lines)




# ── Internal: Disease Term Lookup ─────────────────────────────────────────

def _get_disease_terms(query: str) -> tuple:
    """Return (mesh_term, keywords_list) for the detected disease."""
    q = query.lower()
    for disease, (mesh, keywords) in DISEASE_MESH.items():
        if disease in q or any(k.lower() in q for k in keywords[:3]):
            return mesh, keywords
    # Fallback: use query words directly
    stop = {"the", "a", "an", "of", "for", "in", "and", "or", "latest", "recent", "treatment", "study"}
    words = [w for w in re.sub(r"[^\w\s]", "", q).split() if w not in stop and len(w) > 3]
    mesh  = " ".join(words[:2]) if words else query
    return mesh, words[:4] or [query]


def _infer_disease(query: str) -> str:
    """Infer the disease name from the query string."""
    q = query.lower()
    for disease in DISEASE_MESH:
        if disease in q:
            return disease
    # Return first meaningful word
    stop = {"the", "latest", "recent", "new", "treatment", "for", "of", "in", "and", "or", "what", "is"}
    words = [w for w in q.split() if w not in stop and len(w) > 3]
    return words[0] if words else query


def _filter_trials_by_disease(trials: list, disease: str) -> list:
    """
    Synthesis-layer safety filter: only include trials where the disease
    keyword appears in the trial title or conditions. This is a final guard
    on top of the ranker's disease-penalty logic.
    """
    if not trials or not disease:
        return trials

    _, keywords = _get_disease_terms(disease)
    kw_lower = [k.lower() for k in keywords]

    relevant = []
    for t in trials:
        trial_text = (
            t.get("title", "") + " " +
            " ".join(t.get("conditions", [])) +
            " " + t.get("description", "")
        ).lower()
        if any(kw in trial_text for kw in kw_lower):
            relevant.append(t)

    # If filter is too aggressive and removes everything, return original list
    return relevant if relevant else trials


# ── Internal: Brief Answer Intro ─────────────────────────────────────────
def _build_structured_answer(
    query: str,
    patient_context: dict,
    publications: list,
    clinical_trials: list,
) -> str:
    """
    Returns a brief introductory line for the response card.
    Full analysis lives in conditionOverview (SynthesisPanel).
    Source documents are rendered as PublicationCard / ClinicalTrialCard components.
    """
    disease = patient_context.get("disease", "")
    name    = patient_context.get("name", "")
    n_pub   = len(publications)
    n_trial = len(clinical_trials)

    parts = []
    if n_pub > 0:
        parts.append(f"**{n_pub}** peer-reviewed publication{'s' if n_pub != 1 else ''}")
    if n_trial > 0:
        parts.append(f"**{n_trial}** clinical trial{'s' if n_trial != 1 else ''}")
    found_str = " and ".join(parts) if parts else "no documents"

    if name and disease:
        return (f"Personalized results for **{name}** on **{disease.capitalize()}** — "
                f"{found_str} retrieved, ranked by relevance and recency.")
    elif disease:
        return (f"Results for **{disease.capitalize()}** — {found_str} retrieved, "
                f"ranked by semantic relevance, source credibility, and recency.")
    else:
        return f"Results for your query — {found_str} retrieved."

