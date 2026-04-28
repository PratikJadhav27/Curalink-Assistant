"""
Curalink Custom LLM - Inference Service
=========================================
Loads the fine-tuned flan-t5 model from HuggingFace Hub and exposes
the same two functions as the old Groq service:
  - expand_query(disease, query, patient_context) -> str
  - synthesize_response(query, patient_context, publications, trials, history) -> dict

The model is loaded ONCE at startup (lazy singleton) and cached in memory.
All inference runs locally inside the HF Space container — no external API calls.
"""
import os
import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model Config ─────────────────────────────────────────────────────────
# After you run fine_tune.py and push to HF Hub, update this to:
# "Pratik-027/curalink-medical-llm"
# Until then, it falls back to the base model so the app still works.
CUSTOM_MODEL_ID = os.environ.get(
    "CUSTOM_MODEL_ID",
    "Pratik-027/curalink-medical-llm"   # <-- your HF Hub model repo
)
FALLBACK_MODEL_ID = "google/flan-t5-base"   # used if custom model not yet uploaded

MAX_INPUT_LEN  = 512
MAX_OUTPUT_LEN = 256   # for query expansion
MAX_SYNTH_LEN  = 512   # for synthesis

# ── Lazy Singleton ────────────────────────────────────────────────────────
_pipeline = None


def _get_pipeline():
    """
    Lazy-load the HuggingFace pipeline.
    Loads once on first call, then cached for the lifetime of the process.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        logger.info(f"[CustomLLM] Loading model: {CUSTOM_MODEL_ID} ...")
        print(f"[CustomLLM] Loading model: {CUSTOM_MODEL_ID} ...")

        # Try to load the fine-tuned custom model first
        try:
            tokenizer = AutoTokenizer.from_pretrained(CUSTOM_MODEL_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                CUSTOM_MODEL_ID,
                torch_dtype=torch.float32,   # CPU-safe (HF Spaces free tier)
                low_cpu_mem_usage=True,
            )
            model_id_used = CUSTOM_MODEL_ID
        except Exception as e:
            logger.warning(f"[CustomLLM] Custom model not found ({e}), falling back to {FALLBACK_MODEL_ID}")
            print(f"[CustomLLM] Custom model not found, falling back to {FALLBACK_MODEL_ID}")
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
            device=-1,                 # -1 = CPU (compatible with free HF Spaces)
            max_new_tokens=MAX_OUTPUT_LEN,
        )
        logger.info(f"[CustomLLM] ✅ Model loaded: {model_id_used}")
        print(f"[CustomLLM] ✅ Model loaded: {model_id_used}")
        return _pipeline

    except Exception as e:
        logger.error(f"[CustomLLM] ❌ Failed to load model: {e}")
        raise RuntimeError(f"Custom LLM failed to load: {e}")


# ── Public API: Query Expansion ───────────────────────────────────────────
def expand_query(disease: str, query: str, patient_context: dict) -> str:
    """
    Use the fine-tuned flan-t5 model to expand a user query into an optimized
    medical boolean search query suitable for PubMed, OpenAlex, and ClinicalTrials.gov.
    """
    prompt = (
        f"Expand medical search query for PubMed. "
        f"Disease: {disease}. "
        f"Query: {query}"
    )
    # Truncate prompt to avoid exceeding model context window
    prompt = prompt[:MAX_INPUT_LEN]

    try:
        pipe = _get_pipeline()
        result = pipe(
            prompt,
            max_new_tokens=MAX_OUTPUT_LEN,
            num_beams=4,            # Beam search for higher quality output
            early_stopping=True,
            no_repeat_ngram_size=3, # Prevent repetition
        )
        expanded = result[0]["generated_text"].strip()

        # Sanity check: if output is too short or suspiciously empty, fall back
        if len(expanded) < 10:
            raise ValueError("Output too short, using fallback")

        logger.info(f"[CustomLLM] Expanded query: {expanded[:120]}...")
        return expanded

    except Exception as e:
        logger.warning(f"[CustomLLM] Query expansion failed: {e}. Using fallback.")
        # Fallback: basic concatenation of disease + query terms
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
    Use the fine-tuned flan-t5 to generate a medical research summary.
    Returns a dict with 'conditionOverview' and 'answer' keys.

    Strategy: Because flan-t5-base has a smaller context window than Llama-3,
    we break synthesis into two targeted prompts:
      1. Generate a condition overview (2-3 sentences)
      2. Generate a detailed answer grounded in the retrieved publications
    Then we assemble the final dict in Python code (not relying on the LLM for JSON).
    """
    # Build compact context (truncated to fit in model context window)
    pub_snippets = _build_publication_context(publications)
    trial_snippets = _build_trial_context(clinical_trials)
    patient_str = _build_patient_string(patient_context)

    # ── Prompt 1: Condition Overview ────────────────────────────────────
    overview_prompt = (
        f"Medical Research Assistant. "
        f"Patient: {patient_str}. "
        f"Write a 2-sentence clinical overview of the condition and current research landscape "
        f"for this question: {query}"
    )
    overview_prompt = overview_prompt[:MAX_INPUT_LEN]

    # ── Prompt 2: Detailed Research Answer ──────────────────────────────
    answer_prompt = (
        f"Medical Research Assistant. "
        f"Patient context: {patient_str}. "
        f"Question: {query}. "
        f"Research publications: {pub_snippets}. "
        f"Clinical trials: {trial_snippets}. "
        f"Provide a detailed, evidence-based, personalized answer referencing the publications above."
    )
    answer_prompt = answer_prompt[:MAX_INPUT_LEN]

    try:
        pipe = _get_pipeline()

        # Run both prompts
        overview_result = pipe(
            overview_prompt,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        answer_result = pipe(
            answer_prompt,
            max_new_tokens=MAX_SYNTH_LEN,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        condition_overview = overview_result[0]["generated_text"].strip()
        answer = answer_result[0]["generated_text"].strip()

        # Enrich the answer with publication references (ground in real data)
        answer = _enrich_with_citations(answer, publications, clinical_trials)

        return {
            "conditionOverview": condition_overview,
            "answer": answer,
        }

    except Exception as e:
        logger.error(f"[CustomLLM] Synthesis failed: {e}")
        # Graceful fallback: return a rule-based summary from the retrieved data
        return _fallback_synthesis(query, patient_context, publications, clinical_trials)


# ── Helpers ───────────────────────────────────────────────────────────────
def _build_publication_context(publications: list) -> str:
    if not publications:
        return "No publications retrieved."
    lines = []
    for i, p in enumerate(publications[:4], 1):   # Use top 4 to stay within context
        title = p.get("title", "Untitled")
        year = p.get("year", "N/A")
        abstract = str(p.get("abstract", ""))[:200]
        lines.append(f"[{i}] {title} ({year}): {abstract}")
    return " | ".join(lines)


def _build_trial_context(trials: list) -> str:
    if not trials:
        return "No clinical trials retrieved."
    lines = []
    for i, t in enumerate(trials[:3], 1):   # Top 3 trials
        title = t.get("title", "Untitled")
        status = t.get("status", "Unknown")
        nct = t.get("nctId", "N/A")
        lines.append(f"[{i}] {title} | Status: {status} | NCT: {nct}")
    return " | ".join(lines)


def _build_patient_string(ctx: dict) -> str:
    parts = []
    if ctx.get("name"):
        parts.append(f"Name: {ctx['name']}")
    if ctx.get("disease"):
        parts.append(f"Condition: {ctx['disease']}")
    if ctx.get("location"):
        parts.append(f"Location: {ctx['location']}")
    if ctx.get("additionalInfo"):
        parts.append(f"Notes: {ctx['additionalInfo']}")
    return ", ".join(parts) if parts else "General patient"


def _enrich_with_citations(answer: str, publications: list, trials: list) -> str:
    """
    Append a clean 'Sources' section so the answer is always grounded
    in real retrieved papers — regardless of what the model generated.
    """
    if not publications and not trials:
        return answer

    lines = [answer, "\n\n**Sources Retrieved:**"]
    for i, p in enumerate(publications[:5], 1):
        title = p.get("title", "Untitled")
        year = p.get("year", "N/A")
        url = p.get("url", "")
        lines.append(f"[{i}] {title} ({year})" + (f" — {url}" if url else ""))

    if trials:
        lines.append("\n**Relevant Clinical Trials:**")
        for i, t in enumerate(trials[:3], 1):
            title = t.get("title", "Untitled")
            status = t.get("status", "Unknown")
            nct = t.get("nctId", "")
            nct_link = f"https://clinicaltrials.gov/ct2/show/{nct}" if nct else ""
            lines.append(f"[T{i}] {title} | {status}" + (f" | {nct_link}" if nct_link else ""))

    return "\n".join(lines)


def _fallback_synthesis(query: str, ctx: dict, publications: list, trials: list) -> dict:
    """
    Rule-based fallback if the model completely fails.
    Generates a readable summary from retrieved data without LLM involvement.
    """
    disease = ctx.get("disease", "the condition")
    patient = ctx.get("name", "the patient")

    overview = (
        f"{disease.capitalize()} is an active area of medical research with numerous ongoing studies. "
        f"Current research focuses on improving treatment options, patient outcomes, and understanding disease mechanisms."
    )

    answer_lines = [
        f"Based on the latest retrieved research for {patient}'s query about **{query}**:\n"
    ]
    if publications:
        answer_lines.append("**Key Research Findings:**")
        for i, p in enumerate(publications[:5], 1):
            title = p.get("title", "Untitled")
            year = p.get("year", "N/A")
            abstract = str(p.get("abstract", ""))[:200]
            answer_lines.append(f"- [{i}] **{title}** ({year}): {abstract}...")
    if trials:
        answer_lines.append("\n**Relevant Clinical Trials:**")
        for t in trials[:3]:
            title = t.get("title", "Untitled")
            status = t.get("status", "Unknown")
            nct = t.get("nctId", "N/A")
            answer_lines.append(f"- {title} | Status: {status} | NCT: {nct}")

    return {
        "conditionOverview": overview,
        "answer": "\n".join(answer_lines),
    }
