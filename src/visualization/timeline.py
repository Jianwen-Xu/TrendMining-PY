import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def plot_publications_per_year(df: pd.DataFrame, date_col: str = "Date", output_path: str = "output/timeline_count.png"):
    df = df.copy()
    df["Year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    yearly = df.groupby("Year").size().reset_index(name="Count")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=yearly, x="Year", y="Count", ax=ax, marker="o")
    ax.set_title("Publications per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_citations_per_year(df: pd.DataFrame, date_col: str = "Date", cite_col: str = "Cites", output_path: str = "output/timeline_cites.png"):
    df = df.copy()
    df["Year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    yearly = df.groupby("Year")[cite_col].sum().reset_index(name="TotalCites")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=yearly, x="Year", y="TotalCites", ax=ax, marker="o", color="orange")
    ax.set_title("Total Citations per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Citations")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_hot_cold_topics(trends_df: pd.DataFrame, topic_words_df: pd.DataFrame, output_path: str = "output/hot_cold_topics.png"):
    merged = trends_df.merge(topic_words_df, on="topic_id")
    hot = merged[merged["trend_class"] == "hot"].nlargest(10, "slope")
    cold = merged[merged["trend_class"] == "cold"].nsmallest(10, "slope")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    if not hot.empty:
        axes[0].barh(hot["top_words"].str[:30], hot["slope"], color="tomato")
        axes[0].set_title("Hot Topics (Increasing Trend)")
    if not cold.empty:
        axes[1].barh(cold["top_words"].str[:30], cold["slope"].abs(), color="steelblue")
        axes[1].set_title("Cold Topics (Decreasing Trend)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
