import re
import requests
import pandas as pd
from datetime import date
from bs4 import BeautifulSoup
from src.cleaning.text_cleaner import normalize_text

TRENDING_URL = "https://github.com/trending"
TOPICS_URL = "https://github.com/topics"
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


def _parse_compact_stars(text: str) -> int:
    """Parse '31.9k' or '1,234' style star counts to int."""
    text = text.strip().replace(",", "")
    m = re.match(r"([\d.]+)k", text, re.IGNORECASE)
    if m:
        return int(float(m.group(1)) * 1000)
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else 0


def parse_topic_repo(article) -> dict:
    h3 = article.find("h3")
    links = h3.find_all("a") if h3 else []
    owner = links[0].get_text(strip=True) if len(links) > 0 else ""
    repo = links[1].get_text(strip=True) if len(links) > 1 else ""
    title = f"{owner}/{repo}" if owner and repo else (owner or repo)

    desc_tag = article.find("p")
    description = desc_tag.get_text(strip=True) if desc_tag else ""

    lang_tag = article.find(attrs={"itemprop": "programmingLanguage"})
    language = lang_tag.get_text(strip=True) if lang_tag else ""

    # Total star count shown as "31.9k" in a span
    star_span = article.find("span", string=re.compile(r"[\d,.]+k?", re.IGNORECASE))
    stars = _parse_compact_stars(star_span.get_text()) if star_span else 0

    return {
        "Id": f"github:{title}",
        "Title": title,
        "Abstract": description,
        "Abstract_clean": normalize_text(description),
        "Date": str(date.today()),
        "Cites": stars,
        "Tags": language,
        "AuthorName": owner,
    }


def fetch_github_topics(topic: str, language: str = "", pages: int = 3) -> pd.DataFrame:
    """Scrape github.com/topics/{topic} — supports keyword + language filter."""
    rows = []
    for page in range(1, pages + 1):
        params = {"page": page}
        if language:
            params["l"] = language
        url = f"{TOPICS_URL}/{topic}"
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.find_all("article", class_=lambda c: c and "rounded" in (c or []))
        if not articles:
            break
        for a in articles:
            rows.append(parse_topic_repo(a))
    if not rows:
        return pd.DataFrame(columns=list(parse_topic_repo(
            BeautifulSoup("<article></article>", "html.parser").find("article")
        ).keys()))
    return pd.DataFrame(rows)


def fetch_github_trending(query: str = "", period: str = "daily", language: str = "") -> pd.DataFrame:
    # GitHub Trending has no keyword filter — query param is ignored.
    # Language filter uses path: /trending/{language}
    if language:
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
