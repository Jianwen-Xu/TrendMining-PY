import pytest
import numpy as np
from src.analysis.lda_model import build_vectorizer, build_lda, compute_perplexity

DOCS = [
    "docker container deployment kubernetes cluster",
    "docker image build container registry",
    "continuous integration jenkins pipeline automation",
    "jenkins build deployment automation",
    "kubernetes orchestration cluster container pod",
    "devops agile scrum team collaboration",
    "monitoring logging metrics dashboard alert",
    "security vulnerability patch compliance audit",
    "microservices api gateway service mesh",
    "cloud aws azure infrastructure terraform",
]


def test_build_vectorizer_shape():
    dtm, vectorizer = build_vectorizer(DOCS)
    assert dtm.shape[0] == len(DOCS)
    assert dtm.shape[1] > 0


def test_build_lda_returns_model_and_matrix():
    dtm, vectorizer = build_vectorizer(DOCS)
    model, doc_topic_matrix = build_lda(dtm, k=3, alpha=0.1, beta=0.01, passes=5)
    assert model.n_components == 3
    assert doc_topic_matrix.shape == (len(DOCS), 3)
    # rows should sum to ~1 (normalized doc-topic distribution)
    assert np.allclose(doc_topic_matrix.sum(axis=1), 1.0, atol=0.01)


def test_compute_perplexity_is_positive_float():
    dtm, vectorizer = build_vectorizer(DOCS)
    model, _ = build_lda(dtm, k=3, alpha=0.1, beta=0.01, passes=5)
    perplexity = compute_perplexity(model, dtm)
    assert isinstance(perplexity, float)
    assert perplexity > 0
