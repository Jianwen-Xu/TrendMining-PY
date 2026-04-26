# TrendMining-PY

Python rewrite of [TrendMining](../TrendMining). Mines emerging trends in software engineering from Scopus, StackOverflow, Twitter, and GitHub Trending.

## Quick Start

### Dashboard (recommended)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501` — select source, enter query, click Fetch & Analyze.

### CLI
```bash
python main.py --query devops --source scopus
python main.py --query devops --source scopus --k 20 --skip-fetch
```

## Data Sources

| Source | Auth | Notes |
|---|---|---|
| Scopus | API key required | Set `SCOPUS_API_KEY` in `.env` or sidebar; sorted by citation count for historical spread |
| StackOverflow | Optional | Set `SO_API_KEY` in `.env` or leave blank (300 req/day unauthenticated); max 2500 results |
| Twitter | None | Uses snscrape (no API key needed) |
| GitHub Topics | None | Scrapes `github.com/topics/{query}?l={language}` — keyword searchable |
| GitHub Trending | None | Scrapes `github.com/trending/{language}?since={period}` — no keyword filter |

## Pipeline

```
Fetch → Clean → DTM → LDA (optimized) → Trend Analysis → Visualizations
```

## Tech Stack

- Python 3.11+ (tested on 3.14)
- pandas, requests, beautifulsoup4, snscrape
- scikit-learn (LDA, DTM, clustering) — replaces gensim for Python 3.14 compatibility
- scipy (differential evolution optimizer, Ward dendrogram)
- matplotlib, seaborn, wordcloud, pyLDAvis
- streamlit, pytest, pyarrow

## Dashboard Tabs

| Tab | Content |
|---|---|
| Data | Fetched records table + CSV download |
| Text Mining | Word cloud + dendrogram |
| Timeline | Publications/year + citations/year |
| LDA | Topic word table + embedded pyLDAvis |
| Trends | Hot/cold topic tables + bar chart |

## Outputs

- `output/wordcloud_<query>.png` — word frequency cloud
- `output/dendrogram_<query>.pdf` — hierarchical clustering (40×15 inches)
- `output/timeline_<query>.png` — publications per year
- `output/hot_cold_<query>.png` — hot/cold topic bar chart
- `output/lda_<query>.html` — interactive pyLDAvis

## Notebooks

Run `jupyter notebook` and open `notebooks/`:
1. `01_data_collection.ipynb` — fetch and cache data
2. `02_text_mining.ipynb` — DTM, word clouds, dendrogram
3. `03_lda_modeling.ipynb` — LDA building and optimization
4. `04_trend_analysis.ipynb` — hot/cold topics, visualizations

## Tests

```bash
python -m pytest tests/ -v
```
