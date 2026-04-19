"""
OpenAlex Retrieval Client
Free, open scholarly graph API - no auth required
"""
import httpx


OPENALEX_URL = "https://api.openalex.org/works"


async def fetch_openalex(query: str, max_results: int = 100) -> list[dict]:
    """Fetch research publications from OpenAlex API."""
    docs = []
    per_page = min(max_results, 50)  # OpenAlex max per page in free tier
    pages_needed = (max_results + per_page - 1) // per_page

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for page in range(1, pages_needed + 1):
                resp = await client.get(OPENALEX_URL, params={
                    "search": query,
                    "per-page": per_page,
                    "page": page,
                    "sort": "relevance_score:desc",
                    "select": "id,title,abstract_inverted_index,authorships,publication_year,primary_location,doi",
                    "mailto": "research@curalink.ai",  # polite pool
                })
                if resp.status_code != 200:
                    break

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for work in results:
                    docs.append(_normalize(work))

                if len(results) < per_page:
                    break  # no more pages

    except Exception as e:
        print(f"[OpenAlex] Error: {e}")

    return docs


def _normalize(work: dict) -> dict:
    """Convert OpenAlex work object to universal document schema."""
    # Reconstruct abstract from inverted index
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index", {}))

    # Authors
    authors = []
    for authorship in work.get("authorships", [])[:10]:
        name = authorship.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    # URL
    doi = work.get("doi", "")
    primary_loc = work.get("primary_location", {}) or {}
    landing_url = primary_loc.get("landing_page_url", "")
    url = doi if doi else landing_url

    # Source name
    source_info = primary_loc.get("source", {}) or {}
    journal = source_info.get("display_name", "OpenAlex")

    return {
        "title": work.get("title", "Untitled"),
        "abstract": abstract,
        "authors": authors,
        "year": work.get("publication_year"),
        "source": f"OpenAlex / {journal}",
        "url": url,
        "openAlexId": work.get("id", ""),
    }


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex's inverted index format."""
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words.keys()))
