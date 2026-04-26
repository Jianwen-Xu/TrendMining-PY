import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


EXTRA_STOPWORDS = {
    "devops", "software", "system", "use", "using", "result",
    "paper", "study", "propose", "approach", "method",
}


def build_vectorizer(docs: list, min_word_length: int = 3, min_df: int = 2, max_df: float = 0.9):
    token_pattern = rf"\b[a-z]{{{min_word_length},}}\b"
    vectorizer = CountVectorizer(
        lowercase=True,
        min_df=min(min_df, max(1, len(docs) // 5)),
        max_df=max_df,
        token_pattern=token_pattern,
        stop_words="english",
    )
    dtm = vectorizer.fit_transform(docs)
    return dtm, vectorizer


def build_lda(dtm, k: int, alpha: float = 0.1, beta: float = 0.01, passes: int = 20):
    model = LatentDirichletAllocation(
        n_components=k,
        doc_topic_prior=alpha,
        topic_word_prior=beta,
        max_iter=passes,
        random_state=42,
        n_jobs=-1,
    )
    raw_matrix = model.fit_transform(dtm)
    # normalize rows to sum to 1
    row_sums = raw_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    doc_topic_matrix = raw_matrix / row_sums
    return model, doc_topic_matrix


def compute_perplexity(model: LatentDirichletAllocation, dtm) -> float:
    return float(model.perplexity(dtm))


def get_top_words(model: LatentDirichletAllocation, vectorizer, n_words: int = 10):
    import pandas as pd
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for t in range(model.n_components):
        top_indices = model.components_[t].argsort()[-n_words:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        rows.append({"topic_id": t, "top_words": ", ".join(top_terms)})
    return pd.DataFrame(rows)
