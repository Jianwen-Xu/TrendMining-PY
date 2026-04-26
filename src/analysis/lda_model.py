import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.optimize import differential_evolution


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


def optimize_lda(
    dtm,
    k_min: int = 10,
    k_max: int = 200,
    maxiter: int = 10,
    popsize: int = 10,
    test_ratio: float = 0.2,
    passes: int = 10,
) -> dict:
    n_docs = dtm.shape[0]
    split = max(1, int(n_docs * (1 - test_ratio)))
    train_dtm = dtm[:split]
    test_dtm = dtm[split:] if split < n_docs else dtm[:max(1, split // 5)]

    def objective(params):
        k = max(2, int(round(params[0])))
        alpha = float(params[1])
        beta = float(params[2])
        model = LatentDirichletAllocation(
            n_components=k,
            doc_topic_prior=alpha,
            topic_word_prior=beta,
            max_iter=passes,
            random_state=42,
        )
        model.fit(train_dtm)
        return model.perplexity(test_dtm)

    bounds = [(k_min, k_max), (0.001, 1.0), (0.001, 0.3)]
    result = differential_evolution(
        objective, bounds, maxiter=maxiter, popsize=popsize, seed=42, tol=0.01
    )
    return {
        "k": max(2, int(round(result.x[0]))),
        "alpha": float(result.x[1]),
        "beta": float(result.x[2]),
        "perplexity": float(result.fun),
    }
