"""
LLM Service - Routes to Custom Fine-tuned Flan-T5 Medical Model
================================================================
This module is the single entry point for all LLM calls in Curalink.
On the custom-llm branch, all calls are routed to our fine-tuned
google/flan-t5-base model hosted on HuggingFace Hub (Pratik-027/curalink-medical-llm).

The Groq/Llama-3 API has been completely removed.
All inference is local — no external API keys required at runtime.
"""
import json
import re
import logging

# ── Import from our custom fine-tuned model service ──────────────────────
from core.custom_llm import (
    expand_query as _custom_expand_query,
    synthesize_response as _custom_synthesize,
)

logger = logging.getLogger(__name__)


# ── Public API: Query Expansion ───────────────────────────────────────────
def expand_query(disease: str, query: str, patient_context: dict) -> str:
    """
    Expand and optimize the user's search query for medical databases.
    Uses our fine-tuned flan-t5 model (no external API call).
    """
    logger.info(f"[LLMService] Expanding query for disease='{disease}', query='{query}'")
    return _custom_expand_query(
        disease=disease,
        query=query,
        patient_context=patient_context,
    )


# ── Public API: Synthesis ────────────────────────────────────────────────
def synthesize_response(
    query: str,
    patient_context: dict,
    publications: list[dict],
    clinical_trials: list[dict],
    conversation_history: list[dict],
) -> dict:
    """
    Core reasoning: given top-ranked results, generate a structured medical response.
    Uses our fine-tuned flan-t5 model (no external API call).
    Returns dict with: conditionOverview, answer.
    """
    logger.info(f"[LLMService] Synthesizing response for query='{query}'")
    return _custom_synthesize(
        query=query,
        patient_context=patient_context,
        publications=publications,
        clinical_trials=clinical_trials,
        conversation_history=conversation_history,
    )


# ── Formatting Helpers (kept for compatibility) ───────────────────────────
def _format_publications(publications: list[dict]) -> str:
    if not publications:
        return "No publications retrieved."
    lines = []
    for i, p in enumerate(publications[:8], 1):
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        lines.append(
            f"[{i}] {p.get('title', 'Untitled')} | {authors} | {p.get('year', 'N/A')} | {p.get('source', '')} | {p.get('url', '')}\n"
            f"    Abstract: {str(p.get('abstract', ''))[:400]}..."
        )
    return "\n\n".join(lines)


def _format_trials(trials: list[dict]) -> str:
    if not trials:
        return "No clinical trials retrieved."
    lines = []
    for i, t in enumerate(trials[:5], 1):
        lines.append(
            f"[{i}] {t.get('title', 'Untitled')} | Status: {t.get('status', 'Unknown')} | "
            f"Location: {t.get('location', 'N/A')} | NCT: {t.get('nctId', 'N/A')}"
        )
    return "\n".join(lines)


def _format_patient(ctx: dict) -> str:
    parts = []
    if ctx.get("name"):
        parts.append(f"Patient Name: {ctx['name']}")
    if ctx.get("disease"):
        parts.append(f"Condition: {ctx['disease']}")
    if ctx.get("location"):
        parts.append(f"Location: {ctx['location']}")
    if ctx.get("additionalInfo"):
        parts.append(f"Additional Context: {ctx['additionalInfo']}")
    return "\n".join(parts) if parts else "No specific patient context provided."


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response robustly (kept for compatibility)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "conditionOverview": "",
        "answer": raw,
    }
