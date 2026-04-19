"""
LLM Service - Groq-powered Llama-3 for synthesis and query expansion
"""
import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

# Load .env BEFORE reading env vars (fixes module-level import order issue)
load_dotenv()

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to ai-service/.env")
        _client = Groq(api_key=api_key)
    return _client

MODEL = "llama-3.1-8b-instant"


SYSTEM_PROMPT = """You are Curalink, an elite AI Medical Research Assistant.
Your role is to synthesize peer-reviewed research publications and clinical trials into structured, 
personalized, and source-backed medical insights for healthcare professionals and patients.

STRICT RULES:
- Never fabricate citations, statistics, or medical claims not present in the provided context.
- Always ground your responses in the retrieved publications and clinical trials provided.
- Use compassionate but precise clinical language.
- Structure every response using the exact JSON format requested.
- If insufficient data exists for a section, explicitly state "Insufficient data available."
"""


def expand_query(disease: str, query: str, patient_context: dict) -> str:
    """Use LLM to expand and optimize search query for medical databases."""
    prompt = f"""You are a medical search query optimizer.
Given:
- Disease/Condition: {disease}
- User Query: {query}
- Patient Context: {json.dumps(patient_context)}

Generate an optimized, expanded medical search query string suitable for PubMed, OpenAlex, and ClinicalTrials.gov.
Combine the disease with specific medical terms from the query.
Return ONLY the expanded query string. No explanations. No quotes. Just the query."""

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
        )
        expanded = response.choices[0].message.content.strip()
        return expanded
    except Exception:
        # Fallback - simple concatenation
        return f"{disease} {query}".strip()


def synthesize_response(
    query: str,
    patient_context: dict,
    publications: list[dict],
    clinical_trials: list[dict],
    conversation_history: list[dict],
) -> dict:
    """
    Core reasoning: given top results, generate structured medical response.
    Returns dict with: answer, conditionOverview, queryExpanded.
    """
    # Build context chunks
    pub_context = _format_publications(publications)
    trial_context = _format_trials(clinical_trials)
    patient_str = _format_patient(patient_context)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject conversation history (last 6 turns max)
    for msg in conversation_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_prompt = f"""PATIENT CONTEXT:
{patient_str}

USER QUESTION: {query}

RETRIEVED RESEARCH PUBLICATIONS:
{pub_context}

RETRIEVED CLINICAL TRIALS:
{trial_context}

---
Based on the above publications and clinical trials, provide a comprehensive, personalized response.

Your response must be a valid JSON object with this exact structure:
{{
  "conditionOverview": "<2-3 sentence overview of the condition and its current research landscape>",
  "answer": "<Detailed, personalized, research-backed answer tailored to the patient's context. Use markdown for formatting. Reference specific studies and their findings. Explain clinical trial relevance if applicable.>"
}}

Be thorough. Be specific. Be human. Never hallucinate citations."""

    messages.append({"role": "user", "content": user_prompt})

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1800,
            temperature=0.4,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_json_response(raw)
    except Exception as e:
        return {
            "conditionOverview": "Unable to generate overview at this time.",
            "answer": f"I encountered an error generating your research summary. Please try again. Error: {str(e)}",
        }


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response robustly."""
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to extract JSON block
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback
    return {
        "conditionOverview": "",
        "answer": raw,
    }


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
