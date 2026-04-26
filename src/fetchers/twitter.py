import pandas as pd
from src.cleaning.text_cleaner import clean_twitter

_COLUMNS = ["Id", "AuthorName", "Date", "Title", "Title_clean", "Abstract", "Abstract_clean", "Cites"]


def parse_tweet(tweet) -> dict:
    hashtags = tweet.hashtags or []
    title = tweet.rawContent
    abstract = " ".join(f"#{h}" for h in hashtags)
    return {
        "Id": str(tweet.id),
        "AuthorName": tweet.user.username,
        "Date": tweet.date.strftime("%Y-%m-%d"),
        "Title": title,
        "Title_clean": clean_twitter(title),
        "Abstract": abstract,
        "Abstract_clean": clean_twitter(abstract),
        "Cites": tweet.retweetCount,
    }


def fetch_tweets(query: str, max_tweets: int = 6000) -> pd.DataFrame:
    try:
        import snscrape.modules.twitter as sntwitter
    except Exception:
        return pd.DataFrame(columns=_COLUMNS)

    rows = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(rows) >= max_tweets:
            break
        rows.append(parse_tweet(tweet))
    if not rows:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(rows)
