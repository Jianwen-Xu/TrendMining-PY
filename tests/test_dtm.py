import pytest
import numpy as np
from src.analysis.dtm import build_dtm, cluster_documents

DOCS = [
    "docker container deployment pipeline",
    "docker image build container",
    "continuous integration jenkins pipeline",
    "jenkins build automation deployment",
    "kubernetes cluster orchestration docker",
]

def test_build_dtm_shape():
    dtm, vectorizer = build_dtm(DOCS)
    assert dtm.shape[0] == len(DOCS)
    assert dtm.shape[1] > 0

def test_build_dtm_min_word_length():
    dtm, vectorizer = build_dtm(DOCS, min_word_length=4)
    vocab = vectorizer.get_feature_names_out()
    assert all(len(w) >= 4 for w in vocab)

def test_cluster_documents_returns_labels():
    dtm, _ = build_dtm(DOCS)
    labels = cluster_documents(dtm, n_clusters=2)
    assert len(labels) == len(DOCS)
    assert set(labels).issubset({0, 1})
