import numpy as np
import pandas as pd
from scipy import stats


def compute_topic_trends(doc_topic_matrix: np.ndarray, years: list) -> pd.DataFrame:
    years_arr = np.array(years)
    unique_years = sorted(set(years))
    k = doc_topic_matrix.shape[1]

    year_means = np.zeros((len(unique_years), k))
    for i, yr in enumerate(unique_years):
        mask = years_arr == yr
        year_means[i] = doc_topic_matrix[mask].mean(axis=0)

    year_indices = np.arange(len(unique_years), dtype=float)
    rows = []
    for t in range(k):
        slope, intercept, r, p_value, stderr = stats.linregress(year_indices, year_means[:, t])
        rows.append({
            "topic_id": t,
            "slope": slope,
            "p_value": p_value,
            "r_squared": r ** 2,
        })
    return pd.DataFrame(rows)


def classify_topics(trends: pd.DataFrame, p_threshold: float = 0.05) -> pd.DataFrame:
    def _classify(row):
        if row["p_value"] >= p_threshold:
            return "stable"
        return "hot" if row["slope"] > 0 else "cold"
    df = trends.copy()
    df["trend_class"] = df.apply(_classify, axis=1)
    return df
