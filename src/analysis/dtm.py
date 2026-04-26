import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import issparse


DOMAIN_STOPWORDS = {"devops", "software", "system", "using", "based", "paper", "study"}


def build_dtm(docs: list, min_word_length: int = 3, min_df: int = 5, max_df: float = 1.0):
    token_pattern = rf"\b[a-z]{{{min_word_length},}}\b"
    effective_min_df = min_df if len(docs) >= min_df * 2 else max(1, len(docs) // 5)
    vectorizer = CountVectorizer(
        lowercase=True,
        min_df=effective_min_df,
        max_df=max_df,
        token_pattern=token_pattern,
        stop_words=list(DOMAIN_STOPWORDS),
    )
    dtm = vectorizer.fit_transform(docs)
    return dtm, vectorizer


def cluster_documents(dtm, n_clusters: int = 10) -> np.ndarray:
    X = dtm.toarray() if issparse(dtm) else dtm
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(X)
