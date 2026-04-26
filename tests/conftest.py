import pytest
import pandas as pd

@pytest.fixture
def sample_scopus_entry():
    return {
        "dc:identifier": "SCOPUS_ID:12345",
        "dc:title": "DevOps in Practice",
        "dc:creator": "Smith, J.",
        "prism:publicationName": "JSS",
        "prism:coverDate": "2020-03-01",
        "prism:doi": "10.1234/jss.2020",
        "citedby-count": "42",
        "dc:description": "Abstract text here.",
        "author": [{"authid": "111", "authname": "Smith, J."}],
        "affiliation": [{"affilname": "MIT", "affiliation-country": "USA"}],
        "prism:pageRange": "1-10",
        "subtype": "ar",
        "subtypeDescription": "Article",
    }

@pytest.fixture
def sample_scopus_response(sample_scopus_entry):
    return {
        "search-results": {
            "opensearch:totalResults": "1",
            "cursor": {"@next": "*"},
            "entry": [sample_scopus_entry],
        }
    }
