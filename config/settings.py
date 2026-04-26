import os

SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY", "")
SO_API_KEY = os.getenv("SO_API_KEY", "")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

DEFAULT_QUERY = "devops"
MAX_TWEETS = 6000
