import re
import unicodedata


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_scopus(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"copyright\s*©.*?(reserved|elsevier[^.]*\.)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"all rights reserved\.?", "", text, flags=re.IGNORECASE)
    return normalize_text(text)


def clean_twitter(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return normalize_text(text)


def clean_stackoverflow(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<pre>.*?</pre>", "", text, flags=re.DOTALL)
    text = re.sub(r"<code>.*?</code>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^}]*\}", "", text)
    return normalize_text(text)
