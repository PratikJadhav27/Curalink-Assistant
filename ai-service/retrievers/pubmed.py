"""
PubMed Retrieval Client
Uses NCBI E-utilities: esearch → efetch (XML parse)
"""
import asyncio
import httpx
from xml.etree import ElementTree as ET


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


async def fetch_pubmed(query: str, max_results: int = 100) -> list[dict]:
    """Fetch publications from PubMed. Returns list of normalized documents."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Get IDs
            search_resp = await client.get(ESEARCH_URL, params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "sort": "pub date",
                "retmode": "json",
            })
            search_resp.raise_for_status()
            id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            # Step 2: Fetch details (batch of ids)
            ids = ",".join(id_list[:100])
            fetch_resp = await client.get(EFETCH_URL, params={
                "db": "pubmed",
                "id": ids,
                "retmode": "xml",
            })
            fetch_resp.raise_for_status()
            return _parse_pubmed_xml(fetch_resp.text)
    except Exception as e:
        print(f"[PubMed] Error: {e}")
        return []


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML response into universal document schema."""
    docs = []
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            try:
                medline = article.find("MedlineCitation")
                if medline is None:
                    continue
                art = medline.find("Article")
                if art is None:
                    continue

                title_el = art.find(".//ArticleTitle")
                title = "".join(title_el.itertext()) if title_el is not None else ""

                abstract_el = art.find(".//AbstractText")
                abstract = "".join(abstract_el.itertext()) if abstract_el is not None else ""

                # Authors
                authors = []
                for author in art.findall(".//Author"):
                    last = author.findtext("LastName", "")
                    fore = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {fore}".strip())

                # Year
                year_el = art.find(".//PubDate/Year")
                year = int(year_el.text) if year_el is not None and year_el.text else None

                # PMID
                pmid_el = medline.find("PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                docs.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "source": "PubMed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "pmid": pmid,
                })
            except Exception:
                continue
    except ET.ParseError as e:
        print(f"[PubMed] XML parse error: {e}")
    return docs
