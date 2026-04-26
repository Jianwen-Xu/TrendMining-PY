import re
import requests
import pandas as pd
from datetime import date
from bs4 import BeautifulSoup
from src.cleaning.text_cleaner import normalize_text

TRENDING_URL = "https://github.com/trending"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TrendMining/1.0)"}


def _parse_stars(text: str) -> int:
    parts = text.split("stars")
    if not parts:
        return 0
    digits = re.sub(r"[^\d]", "", parts[0])
    return int(digits) if digits else 0


def parse_repo_row(article) -> dict:
    title_tag = article.find("h2")
    # Clean up "owner / repo" -> "owner/repo"
    title = re.sub(r"\s*/\s*", "/", title_tag.get_text()) if title_tag else ""
    title = title.strip()

    desc_tag = article.find("p")
    description = desc_tag.get_text(strip=True) if desc_tag else ""

    lang_tag = article.find(attrs={"itemprop": "programmingLanguage"})
    language = lang_tag.get_text(strip=True) if lang_tag else ""

    stars_today_tag = article.find("span", class_=lambda c: c and "float-sm-right" in c)
    stars_today = _parse_stars(stars_today_tag.get_text()) if stars_today_tag else 0

    return {
        "Id": f"github:{title}",
        "Title": title,
        "Abstract": description,
        "Abstract_clean": normalize_text(description),
        "Date": str(date.today()),
        "Cites": stars_today,
        "Tags": language,
        "AuthorName": title.split("/")[0].strip() if "/" in title else "",
    }


def fetch_github_trending(query: str = "", period: str = "daily", language: str = "") -> pd.DataFrame:
    if query:
        url = f"https://github.com/topics/{query}"
    elif language:
        url = f"{TRENDING_URL}/{language}"
    else:
        url = TRENDING_URL
    params = {"since": period}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    articles = soup.find_all("article", class_=lambda c: c and "Box-row" in c)
    rows = [parse_repo_row(a) for a in articles]
    if not rows:
        return pd.DataFrame(columns=list(parse_repo_row(
            BeautifulSoup("<article class='Box-row'></article>", "html.parser").find("article")
        ).keys()))
    return pd.DataFrame(rows)
