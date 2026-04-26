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


# ---------------------------------------------------------------------------
# Task 4: StackOverflow fetcher tests
# ---------------------------------------------------------------------------
from src.fetchers.stackoverflow import StackOverflowClient, parse_so_item


def test_parse_so_item():
    item = {
        "question_id": 999,
        "title": "How to use Docker?",
        "body": "<p>I want to <code>run docker</code> locally.</p>",
        "score": 15,
        "view_count": 500,
        "answer_count": 3,
        "tags": ["docker", "devops"],
        "creation_date": 1609459200,
        "last_activity_date": 1609545600,
        "owner": {"user_id": 42},
    }
    row = parse_so_item(item)
    assert row["Q_id"] == 999
    assert row["Title"] == "How to use Docker?"
    assert "<" not in row["Abstract_clean"]
    assert row["Cites"] == 15
    assert row["Tags"] == "docker|devops"


def test_parse_so_item_missing_fields():
    row = parse_so_item({})
    assert row["Q_id"] == ""
    assert row["Cites"] == 0
    assert row["Tags"] == ""


def test_so_client_returns_dataframe():
    client = StackOverflowClient(api_key="test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "items": [{
            "question_id": 1,
            "title": "Docker Q",
            "body": "<p>body</p>",
            "score": 5,
            "view_count": 100,
            "answer_count": 2,
            "tags": ["docker"],
            "creation_date": 1609459200,
            "last_activity_date": 1609459200,
            "owner": {"user_id": 7},
        }],
        "has_more": False,
    }
    mock_response.raise_for_status.return_value = None
    mock_response.status_code = 200
    with patch("src.fetchers.stackoverflow.requests.get", return_value=mock_response):
        df = client.fetch("devops")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Abstract_clean" in df.columns
