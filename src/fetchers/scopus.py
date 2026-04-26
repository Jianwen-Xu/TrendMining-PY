import requests
import pandas as pd
from src.cleaning.text_cleaner import clean_scopus

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


def _safe_int(value) -> int:
    try:
        return int(value or 0)
    except (ValueError, TypeError):
        return 0


def parse_entry(entry: dict) -> dict:
    authors_raw = entry.get("author", [])
    authors = authors_raw if isinstance(authors_raw, list) else ([authors_raw] if authors_raw else [])
    affiliations_raw = entry.get("affiliation", [])
    affiliations = affiliations_raw if isinstance(affiliations_raw, list) else ([affiliations_raw] if affiliations_raw else [])
    return {
        "Id": entry.get("dc:identifier", ""),
        "Title": entry.get("dc:title", ""),
        "Abstract": entry.get("dc:description", ""),
        "Date": entry.get("prism:coverDate", ""),
        "Cites": _safe_int(entry.get("citedby-count", 0)),
        "PubName": entry.get("prism:publicationName", ""),
        "DOI": entry.get("prism:doi", ""),
        "PubType1": entry.get("subtypeDescription", ""),
        "AuthorName": entry.get("dc:creator", ""),
        "AuthorCount": len(authors),
        "Authors": "|".join(a.get("authname", "") for a in authors),
        "AuthorIds": "|".join(a.get("authid", "") for a in authors),
        "AffiliationCount": len(affiliations),
        "Affiliations": "|".join(a.get("affilname", "") for a in affiliations),
        "AffiliationCountries": "|".join(a.get("affiliation-country", "") for a in affiliations),
        "PageRange": entry.get("prism:pageRange", ""),
    }


class ScopusClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}

    def fetch(self, query: str, max_results: int = 5000) -> pd.DataFrame:
        rows = []
        cursor = "*"
        while len(rows) < max_results:
            params = {
                "query": f'TITLE-ABS-KEY("{query}")',
                "count": 200,
                "cursor": cursor,
                "field": "dc:identifier,dc:title,dc:creator,dc:description,prism:publicationName,prism:coverDate,prism:doi,citedby-count,subtype,subtypeDescription,prism:pageRange,author,affiliation",
            }
            resp = requests.get(SCOPUS_SEARCH_URL, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("search-results", {})
            entries = data.get("entry", [])
            if not entries:
                break
            for entry in entries:
                rows.append(parse_entry(entry))
            cursor = data.get("cursor", {}).get("@next", "")
            if not cursor:
                break
        rows = rows[:max_results]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=list(parse_entry({}).keys()) + ["Abstract_clean"])
        df["Abstract_clean"] = df["Abstract"].apply(clean_scopus)
        return df
