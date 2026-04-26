import requests
import pandas as pd
from datetime import datetime
from src.cleaning.text_cleaner import clean_stackoverflow

SO_API_URL = "https://api.stackexchange.com/2.3/search/advanced"


def _safe_int(value) -> int:
    try:
        return int(value or 0)
    except (ValueError, TypeError):
        return 0


def parse_so_item(item: dict) -> dict:
    ts = item.get("creation_date", 0)
    la_ts = item.get("last_activity_date", 0)
    body = item.get("body", "")
    abstract_clean = clean_stackoverflow(body)
    return {
        "Q_id": item.get("question_id", ""),
        "AuthorId": item.get("owner", {}).get("user_id", ""),
        "Title": item.get("title", ""),
        "Abstract": body,
        "Abstract_clean": abstract_clean,
        "Views": _safe_int(item.get("view_count", 0)),
        "Answers": _safe_int(item.get("answer_count", 0)),
        "Cites": _safe_int(item.get("score", 0)),
        "Tags": "|".join(item.get("tags", [])),
        "Date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "",
        "CR_Date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "",
        "LA_Date": datetime.utcfromtimestamp(la_ts).strftime("%Y-%m-%d") if la_ts else "",
    }


class StackOverflowClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, query: str, page_size: int = 100) -> pd.DataFrame:
        rows = []
        page = 1
        while True:
            params = {
                "q": query,
                "tagged": query,
                "pagesize": page_size,
                "page": page,
                "site": "stackoverflow",
                "filter": "withbody",
                "key": self.api_key,
            }
            resp = requests.get(SO_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            for item in items:
                rows.append(parse_so_item(item))
            if not data.get("has_more") or not items:
                break
            page += 1
        if not rows:
            return pd.DataFrame(columns=list(parse_so_item({}).keys()))
        return pd.DataFrame(rows)
