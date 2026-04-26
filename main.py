import argparse
import os
import pandas as pd

from config.settings import SCOPUS_API_KEY, SO_API_KEY, DATA_DIR, OUTPUT_DIR, DEFAULT_QUERY, MAX_TWEETS
from src.fetchers.scopus import ScopusClient
from src.fetchers.stackoverflow import StackOverflowClient
from src.fetchers.twitter import fetch_tweets
from src.fetchers.github_trending import fetch_github_trending
from src.analysis.dtm import build_dtm
from src.analysis.lda_model import build_vectorizer, build_lda, optimize_lda, get_top_words
from src.analysis.trend_analysis import compute_topic_trends, classify_topics
from src.visualization.wordcloud_viz import plot_wordcloud
from src.visualization.dendrogram import plot_dendrogram
from src.visualization.timeline import plot_publications_per_year, plot_hot_cold_topics
from src.visualization.lda_vis import save_interactive_lda
from src.utils.data_store import save, load, exists


def parse_args():
    parser = argparse.ArgumentParser(description="TrendMining Python Pipeline")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--source", choices=["scopus", "twitter", "stackoverflow", "github", "all"], default="scopus")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--optimize-iter", type=int, default=10)
    return parser.parse_args()


def fetch_data(source, query, skip_fetch):
    cache_name = f"{source}_{query}"
    if skip_fetch and exists(cache_name, DATA_DIR):
        print(f"Loading cached {source} data for '{query}'")
        return load(cache_name, DATA_DIR)

    print(f"Fetching {source} data for '{query}'")
    if source == "scopus":
        df = ScopusClient(SCOPUS_API_KEY).fetch(query)
    elif source == "stackoverflow":
        df = StackOverflowClient(SO_API_KEY).fetch(query)
    elif source == "twitter":
        df = fetch_tweets(f"#{query}", max_tweets=MAX_TWEETS)
    else:
        df = fetch_github_trending(query=query, period="daily")
    save(df, cache_name, DATA_DIR)
    print(f"Saved {len(df)} records to {cache_name}.parquet")
    return df


def main():
    args = parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    query = args.query

    if args.source == "all":
        dfs = [fetch_data(s, query, args.skip_fetch) for s in ["scopus", "stackoverflow", "twitter", "github"]]
        df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    else:
        df = fetch_data(args.source, query, args.skip_fetch)

    texts = df["Abstract_clean"].dropna().tolist()
    if not texts:
        print("No text data found. Check data source and query.")
        return

    print(f"Building DTM from {len(texts)} documents")
    dtm, vectorizer_dtm = build_dtm(texts)
    plot_wordcloud(texts, output_path=os.path.join(OUTPUT_DIR, f"wordcloud_{query}.png"))
    plot_dendrogram(dtm, output_path=os.path.join(OUTPUT_DIR, f"dendrogram_{query}.pdf"))

    lda_dtm, lda_vectorizer = build_vectorizer(texts)
    if args.k is None:
        print("Optimizing LDA hyperparameters (this may take several minutes)")
        best = optimize_lda(lda_dtm, maxiter=args.optimize_iter)
        print(f"Optimal: k={best['k']}, alpha={best['alpha']:.4f}, beta={best['beta']:.4f}")
        k, alpha, beta = best["k"], best["alpha"], best["beta"]
    else:
        k, alpha, beta = args.k, 0.1, 0.01

    print(f"Building LDA model with k={k}")
    model, doc_topic_matrix = build_lda(lda_dtm, k=k, alpha=alpha, beta=beta)
    top_words = get_top_words(model, lda_vectorizer)

    years_series = pd.to_datetime(df["Date"], errors="coerce").dt.year.dropna().astype(int)
    years = years_series.tolist()[:len(doc_topic_matrix)]
    if len(years) < len(doc_topic_matrix):
        years = (years * ((len(doc_topic_matrix) // len(years)) + 1))[:len(doc_topic_matrix)]

    trends = compute_topic_trends(doc_topic_matrix, years)
    classified = classify_topics(trends)
    result = classified.merge(top_words, on="topic_id")

    print("\nHot topics:")
    hot = result[result["trend_class"] == "hot"].nlargest(5, "slope")
    print(hot[["topic_id", "slope", "top_words"]].to_string(index=False))
    print("\nCold topics:")
    cold = result[result["trend_class"] == "cold"].nsmallest(5, "slope")
    print(cold[["topic_id", "slope", "top_words"]].to_string(index=False))

    plot_publications_per_year(df, output_path=os.path.join(OUTPUT_DIR, f"timeline_{query}.png"))
    plot_hot_cold_topics(classified, top_words, output_path=os.path.join(OUTPUT_DIR, f"hot_cold_{query}.png"))
    save_interactive_lda(model, lda_dtm, lda_vectorizer, output_path=os.path.join(OUTPUT_DIR, f"lda_{query}.html"))
    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
