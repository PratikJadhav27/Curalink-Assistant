"""
ClinicalTrials.gov API v2 Retrieval Client
"""
import asyncio
import requests

CLINICAL_TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"


async def fetch_clinical_trials(
    disease: str,
    query: str,
    location: str = None,
    max_results: int = 50,
    status_filter: str = None,
) -> list[dict]:
    """
    Fetch clinical trials from ClinicalTrials.gov API v2.
    Returns list of normalized trial objects.
    """
    search_term = f"{disease} {query}".strip()

    params = {
        "query.cond": disease,
        "query.term": query,
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": "NCTId,BriefTitle,OverallStatus,EligibilityCriteria,LocationCountry,LocationCity,CentralContactName,CentralContactPhone,CentralContactEMail,StudyType,Phase",
    }

    if location:
        params["query.locn"] = location

    if status_filter:
        params["filter.overallStatus"] = status_filter

    headers = {
        "User-Agent": "curl/7.81.0",
        "Accept": "application/json"
    }

    try:
        def _fetch():
            resp = requests.get(CLINICAL_TRIALS_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()

        data = await asyncio.to_thread(_fetch)
        studies = data.get("studies", [])
        return [_normalize_trial(s) for s in studies]
    except Exception as e:
        print(f"[ClinicalTrials] Error: {e}")
        return []


def _normalize_trial(study: dict) -> dict:
    """Normalize a ClinicalTrials v2 study record."""
    protocol = study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    eligibility_module = protocol.get("eligibilityModule", {})
    contacts_module = protocol.get("contactsLocationsModule", {})
    design_module = protocol.get("designModule", {})

    nct_id = id_module.get("nctId", "")
    title = id_module.get("briefTitle", "Untitled Trial")
    status = status_module.get("overallStatus", "Unknown")

    eligibility = eligibility_module.get("eligibilityCriteria", "")
    # Truncate long eligibility text
    if len(eligibility) > 600:
        eligibility = eligibility[:600] + "..."

    # Location
    locations = contacts_module.get("locations", [])
    location_str = ""
    if locations:
        loc = locations[0]
        city = loc.get("city", "")
        country = loc.get("country", "")
        location_str = f"{city}, {country}".strip(", ")

    # Contact
    contacts = contacts_module.get("centralContacts", [])
    contact_str = ""
    if contacts:
        c = contacts[0]
        name = c.get("name", "")
        email = c.get("email", "")
        phone = c.get("phone", "")
        contact_str = f"{name} | {email} | {phone}".strip(" |")

    phase = ""
    phases = design_module.get("phases", [])
    if phases:
        phase = ", ".join(phases)

    return {
        "title": title,
        "status": status,
        "eligibility": eligibility,
        "location": location_str,
        "contact": contact_str,
        "nctId": nct_id,
        "phase": phase,
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
    }
