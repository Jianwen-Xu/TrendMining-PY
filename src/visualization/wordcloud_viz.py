import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordcloud(texts: list, output_path: str = "output/wordcloud.png", max_words: int = 50, title: str = "Word Frequency"):
    all_words = " ".join(texts)
    wc = WordCloud(
        width=1600,
        height=800,
        max_words=max_words,
        background_color="white",
        colormap="viridis",
    ).generate(all_words)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_comparison_cloud(texts_by_group: dict, output_path: str = "output/comparison_cloud.png"):
    n = len(texts_by_group)
    fig, axes = plt.subplots(1, n, figsize=(16, 8))
    if n == 1:
        axes = [axes]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    for ax, (label, texts) in zip(axes, texts_by_group.items()):
        all_words = " ".join(texts)
        wc = WordCloud(width=800, height=600, max_words=50, background_color="white").generate(all_words)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label, fontsize=14)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
