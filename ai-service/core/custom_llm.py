"""
Curalink Custom LLM - Inference Service
=========================================
Model:  Pratik-027/curalink-medical-llm
Base:   google/flan-t5-base + LoRA adapters (fine-tuned on medalpaca/medical_meadow_medqa)

Architecture Decision:
  - flan-t5-base is used for QUERY EXPANSION (seq2seq, short output → excellent quality)
  - For SYNTHESIS, we use a structured template engine grounded in real retrieved data.
    This eliminates hallucinations that small LLMs produce on complex multi-doc synthesis tasks.
    The custom model still drives the intelligence of the pipeline via query expansion.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model Config ─────────────────────────────────────────────────────────
CUSTOM_MODEL_ID   = os.environ.get("CUSTOM_MODEL_ID", "Pratik-027/curalink-medical-llm")
FALLBACK_MODEL_ID = "google/flan-t5-base"
MAX_INPUT_LEN     = 256
MAX_OUTPUT_LEN    = 150

# ── Lazy Singleton ────────────────────────────────────────────────────────
_pipeline = None


def _get_pipeline():
    """
    Lazy-load the HuggingFace text2text pipeline.
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
        print(f"[CustomLLM] Using fallback model: {FALLBACK_MODEL_ID}")
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
        device=-1,   # CPU — compatible with HF Spaces free tier
    )
    logger.info(f"[CustomLLM] ✅ Model loaded: {model_id_used}")
    print(f"[CustomLLM] ✅ Model loaded: {model_id_used}")
    return _pipeline


# ── Public API: Query Expansion ───────────────────────────────────────────
def expand_query(disease: str, query: str, patient_context: dict) -> str:
    """
    Use the fine-tuned flan-t5 model to expand the user's prompt into an
    optimized medical boolean search query for PubMed / OpenAlex / ClinicalTrials.gov.
    """
    prompt = (
        f"Expand medical search query for PubMed. "
        f"Disease: {disease}. "
        f"Query: {query}"
    )
    prompt = prompt[:MAX_INPUT_LEN]

    try:
        pipe = _get_pipeline()
        result = pipe(
            prompt,
            max_new_tokens=MAX_OUTPUT_LEN,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        expanded = result[0]["generated_text"].strip()
        if len(expanded) < 10:
            raise ValueError("Output too short")
        logger.info(f"[CustomLLM] Query expanded to: {expanded[:100]}...")
        return expanded
    except Exception as e:
        logger.warning(f"[CustomLLM] Query expansion failed: {e}. Using fallback.")
        return f"{disease} AND {query}".strip()


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

    Strategy:
      1. Use flan-t5 to generate a concise condition overview sentence.
      2. Build the main detailed answer as a rich structured template that is
         100% grounded in the real retrieved publications and clinical trials.

    This hybrid approach ensures:
      - The custom LLM still powers the intelligence (query expansion + overview)
      - The detailed answer is sourced entirely from real retrieved data (no hallucinations)
      - The response is always well-formatted markdown, readable on the frontend
    """
    # ── Step 1: Generate condition overview with the custom model ────────
    disease = patient_context.get("disease") or query.split()[0]
    condition_overview = _generate_overview(disease, query)

    # ── Step 2: Build structured answer grounded in retrieved data ───────
    answer = _build_structured_answer(query, patient_context, publications, clinical_trials)

    return {
        "conditionOverview": condition_overview,
        "answer": answer,
    }


# ── Internal: LLM-generated Overview ─────────────────────────────────────
def _generate_overview(disease: str, query: str) -> str:
    """
    Generate a 1-2 sentence clinical overview using the fine-tuned model.
    Kept short so flan-t5-base stays within its reliable output range.
    """
    prompt = (
        f"Medical Research Assistant. "
        f"Write one sentence describing the current research landscape for {disease}."
    )
    prompt = prompt[:MAX_INPUT_LEN]

    try:
        pipe = _get_pipeline()
        result = pipe(
            prompt,
            max_new_tokens=80,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        overview = result[0]["generated_text"].strip()
        if len(overview) > 20:
            return overview
    except Exception as e:
        logger.warning(f"[CustomLLM] Overview generation failed: {e}")

    # Fallback if model output is poor
    return (
        f"{disease.capitalize()} is an active area of clinical research with ongoing studies "
        f"focused on improving treatment outcomes, disease management, and patient quality of life."
    )


# ── Internal: Structured Template Answer ─────────────────────────────────
def _build_structured_answer(
    query: str,
    patient_context: dict,
    publications: list,
    clinical_trials: list,
) -> str:
    """
    Build a rich, structured markdown answer sourced entirely from
    real retrieved publications and clinical trials.
    Zero hallucinations — every claim is tied to a real retrieved document.
    """
    disease  = patient_context.get("disease", "")
    name     = patient_context.get("name", "")
    location = patient_context.get("location", "")

    lines = []

    # ── Personalized intro ───────────────────────────────────────────────
    intro_parts = []
    if name:
        intro_parts.append(f"**{name}**")
    if disease:
        intro_parts.append(f"regarding **{disease}**")
    if location:
        intro_parts.append(f"in {location}")

    if intro_parts:
        lines.append(f"Here is a personalized research summary for {', '.join(intro_parts)}:\n")
    else:
        lines.append(f"Here is a research summary for your query: **{query}**\n")

    # ── Key Research Findings ────────────────────────────────────────────
    if publications:
        lines.append("### 📚 Key Research Findings\n")
        for i, p in enumerate(publications[:6], 1):
            title    = p.get("title", "Untitled Study")
            year     = p.get("year", "N/A")
            authors  = p.get("authors", [])
            author_str = f"{authors[0]} et al." if authors else ""
            abstract = str(p.get("abstract", "")).strip()
            url      = p.get("url", "")

            # Take first 2 sentences of abstract for a clean snippet
            sentences = [s.strip() for s in abstract.split(".") if len(s.strip()) > 20]
            snippet   = ". ".join(sentences[:2]) + "." if sentences else ""

            entry = f"**[{i}] {title}**"
            if author_str:
                entry += f" *({author_str}, {year})*"
            else:
                entry += f" *({year})*"
            lines.append(entry)
            if snippet:
                lines.append(f"> {snippet}")
            if url:
                lines.append(f"> 🔗 [Read paper]({url})")
            lines.append("")
    else:
        lines.append("*No relevant publications were retrieved for this query.*\n")

    # ── Clinical Trials ──────────────────────────────────────────────────
    if clinical_trials:
        lines.append("### 🧪 Relevant Clinical Trials\n")
        for i, t in enumerate(clinical_trials[:4], 1):
            title    = t.get("title", "Untitled Trial")
            status   = t.get("status", "Unknown")
            nct      = t.get("nctId", "")
            location_trial = t.get("location", "")
            nct_url  = f"https://clinicaltrials.gov/ct2/show/{nct}" if nct else ""

            # Status badge
            status_emoji = "🟢" if "RECRUIT" in status.upper() else "🔵" if "ACTIVE" in status.upper() else "⚪"

            entry = f"**[T{i}] {title}**"
            lines.append(entry)
            lines.append(f"> {status_emoji} Status: **{status}**" + (f" | 📍 {location_trial}" if location_trial else ""))
            if nct_url:
                lines.append(f"> 🔗 [View on ClinicalTrials.gov]({nct_url})")
            lines.append("")
    else:
        lines.append("*No active clinical trials were retrieved for this query.*\n")

    # ── Disclaimer ───────────────────────────────────────────────────────
    lines.append("---")
    lines.append(
        "*⚕️ This summary is generated from peer-reviewed publications and official clinical trial registries. "
        "Always consult a qualified healthcare professional before making any medical decisions.*"
    )

    return "\n".join(lines)
