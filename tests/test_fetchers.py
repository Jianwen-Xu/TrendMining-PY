import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.fetchers.scopus import ScopusClient, parse_entry


def test_parse_entry(sample_scopus_entry):
    row = parse_entry(sample_scopus_entry)
    assert row["Title"] == "DevOps in Practice"
    assert row["Cites"] == 42
    assert row["Date"] == "2020-03-01"
    assert row["AuthorName"] == "Smith, J."


def test_parse_entry_missing_fields():
    row = parse_entry({})
    assert row["Title"] == ""
    assert row["Cites"] == 0


def test_scopus_client_returns_dataframe():
    client = ScopusClient(api_key="test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "search-results": {
            "opensearch:totalResults": "1",
            "entry": [{
                "dc:identifier": "1",
                "dc:title": "Test",
                "dc:creator": "Doe",
                "prism:publicationName": "Nature",
                "prism:coverDate": "2021-01-01",
                "prism:doi": "10.1/test",
                "citedby-count": "5",
                "dc:description": "Abstract.",
            }],
        }
    }
    mock_response.status_code = 200
    with patch("src.fetchers.scopus.requests.get", return_value=mock_response):
        df = client.fetch("devops", max_results=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Abstract_clean" in df.columns


def test_parse_entry_single_author_as_dict():
    entry = {
        "dc:title": "Solo Paper",
        "citedby-count": "1",
        "author": {"authid": "999", "authname": "Solo, A."},
        "affiliation": {"affilname": "Cambridge", "affiliation-country": "UK"},
    }
    row = parse_entry(entry)
    assert row["AuthorCount"] == 1
    assert row["Authors"] == "Solo, A."
    assert row["Affiliations"] == "Cambridge"
