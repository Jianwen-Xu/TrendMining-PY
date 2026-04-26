import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.sparse import issparse


def plot_dendrogram(dtm, labels: list = None, output_path: str = "output/dendrogram.pdf", max_docs: int = 500):
    X = dtm[:max_docs].toarray() if issparse(dtm) else dtm[:max_docs]
    linkage_matrix = ward(X)
    fig, ax = plt.subplots(figsize=(40, 15))
    dendrogram(
        linkage_matrix,
        labels=labels[:max_docs] if labels else None,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    ax.set_title("Document Clustering Dendrogram (Ward)")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
