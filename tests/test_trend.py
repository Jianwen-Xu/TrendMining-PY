import pytest
import numpy as np
import pandas as pd
from src.analysis.trend_analysis import (
    compute_topic_trends,
    classify_topics,
)
from src.analysis.lda_model import build_vectorizer, build_lda


@pytest.fixture
def small_lda():
    docs = [
        "docker container deployment",
        "docker image build",
        "jenkins pipeline build",
        "kubernetes cluster container",
        "monitoring alert metrics",
    ] * 4
    dtm, vectorizer = build_vectorizer(docs)
    model, doc_topic_matrix = build_lda(dtm, k=3, alpha=0.1, beta=0.01, passes=5)
    return model, doc_topic_matrix, vectorizer


def test_compute_topic_trends(small_lda):
    _, doc_topic_matrix, _ = small_lda
    years = [2018, 2019, 2020, 2021] * 5
    trends = compute_topic_trends(doc_topic_matrix, years)
    assert "slope" in trends.columns
    assert "p_value" in trends.columns
    assert len(trends) == doc_topic_matrix.shape[1]


def test_classify_topics(small_lda):
    _, doc_topic_matrix, _ = small_lda
    years = [2018, 2019, 2020, 2021] * 5
    trends = compute_topic_trends(doc_topic_matrix, years)
    classified = classify_topics(trends)
    assert "trend_class" in classified.columns
    assert set(classified["trend_class"]).issubset({"hot", "cold", "stable"})
