import os

SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY", "5cd1321fa87640a65e146086a5cffa2b")
SO_API_KEY = os.getenv("SO_API_KEY", "Zt5gfc9eSaW68FsAKjHXhg((")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

DEFAULT_QUERY = "devops"
MAX_TWEETS = 6000
