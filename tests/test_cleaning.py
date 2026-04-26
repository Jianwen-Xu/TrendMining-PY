import pytest
from src.cleaning.text_cleaner import clean_scopus, clean_twitter, clean_stackoverflow, normalize_text

def test_normalize_text_lowercase():
    assert normalize_text("Hello WORLD") == "hello world"

def test_normalize_text_removes_numbers():
    assert normalize_text("test 123 foo") == "test foo"

def test_normalize_text_removes_punctuation():
    assert normalize_text("hello, world!") == "hello world"

def test_clean_scopus_removes_copyright():
    text = "Some abstract. Copyright © 2020 Elsevier. All rights reserved."
    result = clean_scopus(text)
    assert "copyright" not in result.lower()
    assert "elsevier" not in result.lower()

def test_clean_twitter_removes_urls():
    text = "Check this https://t.co/abc123 out"
    result = clean_twitter(text)
    assert "http" not in result

def test_clean_twitter_removes_mentions():
    result = clean_twitter("Hello @user how are you")
    assert "@user" not in result

def test_clean_twitter_removes_hashtags():
    result = clean_twitter("Loving #devops today")
    assert "#devops" not in result

def test_clean_stackoverflow_removes_html():
    result = clean_stackoverflow("<p>Hello <code>world</code></p>")
    assert "<" not in result
    assert "hello" in result
