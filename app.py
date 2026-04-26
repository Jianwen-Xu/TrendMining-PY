import os
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from config.settings import SCOPUS_API_KEY, SO_API_KEY, DATA_DIR, OUTPUT_DIR
from src.fetchers.scopus import ScopusClient
from src.fetchers.stackoverflow import StackOverflowClient
from src.fetchers.twitter import fetch_tweets
from src.fetchers.github_trending import fetch_github_trending, fetch_github_topics
from src.analysis.dtm import build_dtm
from src.analysis.lda_model import build_vectorizer, build_lda, optimize_lda, get_top_words
from src.analysis.trend_analysis import compute_topic_trends, classify_topics
from src.visualization.wordcloud_viz import plot_wordcloud, plot_comparison_cloud
from src.visualization.dendrogram import plot_dendrogram
from src.visualization.timeline import plot_publications_per_year, plot_citations_per_year, plot_hot_cold_topics
from src.visualization.lda_vis import save_interactive_lda
from src.utils.data_store import save, load, exists

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="TrendMining", layout="wide")
st.title("TrendMining — Software Engineering Trend Analysis")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    source = st.selectbox("Data Source", ["scopus", "stackoverflow", "twitter", "github"])
    query = st.text_input("Search Query", value="devops")
    if source == "github":
        gh_mode = st.radio("GitHub mode", ["Topics (keyword search)", "Trending (today's hot repos)"], index=0)
        gh_language = st.text_input("Language filter (e.g. python, typescript, shell)", value="")
        if "Trending" in gh_mode:
            gh_period = st.selectbox("Period", ["daily", "weekly", "monthly"], index=0)
        else:
            gh_period = "daily"
    else:
        gh_mode = ""
        gh_language = ""
        gh_period = "daily"
    use_cache = st.checkbox("Use cached data", value=True)
    st.divider()
    st.subheader("API Keys")
    if source == "scopus":
        scopus_key = st.text_input("Scopus API Key", value=SCOPUS_API_KEY, type="password")
    else:
        scopus_key = SCOPUS_API_KEY
    if source == "stackoverflow":
        so_key = st.text_input("StackOverflow API Key", value=SO_API_KEY, type="password")
    else:
        so_key = SO_API_KEY
    st.divider()
    st.subheader("LDA Settings")
    auto_optimize = st.checkbox("Auto-optimize hyperparameters (slow)", value=False)
    k = st.slider("Topics (k)", min_value=5, max_value=100, value=20, disabled=auto_optimize)
    alpha = st.slider("Alpha (doc-topic prior)", 0.001, 1.0, 0.1, disabled=auto_optimize)
    beta = st.slider("Beta (topic-word prior)", 0.001, 0.3, 0.01, disabled=auto_optimize)
    fetch_btn = st.button("Fetch & Analyze", type="primary")

# --- Session state init ---
for key in ["df", "texts", "model", "lda_dtm", "lda_vectorizer", "classified", "top_words"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Fetch & analyze ---
if fetch_btn:
    if source == "github":
        if "Topics" in gh_mode:
            cache_name = f"github_topics_{query}_{gh_language or 'all'}"
        else:
            cache_name = f"github_trending_{gh_language or 'all'}_{gh_period}"
    else:
        cache_name = f"{source}_{query}"
    with st.spinner(f"Fetching {source} data for '{query}'..."):
        if source == "scopus" and not scopus_key:
            st.error("Scopus API Key required. Enter it in the sidebar or set SCOPUS_API_KEY in .env")
            st.stop()
        if use_cache and exists(cache_name, DATA_DIR):
            df = load(cache_name, DATA_DIR)
            st.success(f"Loaded {len(df)} cached records.")
        else:
            if source == "scopus":
                df = ScopusClient(scopus_key).fetch(query, max_results=500)
            elif source == "stackoverflow":
                df = StackOverflowClient(so_key).fetch(query)
            elif source == "twitter":
                df = fetch_tweets(f"#{query}", max_tweets=2000)
            else:
                if "Topics" in gh_mode:
                    df = fetch_github_topics(topic=query, language=gh_language)
                else:
                    df = fetch_github_trending(language=gh_language, period=gh_period)
            save(df, cache_name, DATA_DIR)
            st.success(f"Fetched and cached {len(df)} records.")
    st.session_state.df = df

    from src.cleaning.text_cleaner import clean_scopus
    abstract_texts = [t for t in df["Abstract_clean"].dropna().tolist() if t.strip()]
    if len(abstract_texts) < len(df) * 0.1:
        # Fewer than 10% have abstracts — fall back to titles
        title_texts = [clean_scopus(str(t)) for t in df["Title"].dropna().tolist() if str(t).strip()]
        texts = abstract_texts + title_texts
        st.warning(f"Only {len(abstract_texts)}/{len(df)} records have abstracts. Using titles as fallback.")
    else:
        texts = abstract_texts
    texts = [t for t in texts if t.strip()]
    if not texts:
        st.error("No usable text found. Abstracts and titles are empty.")
        st.stop()
    st.info(f"Analyzing {len(texts)} text documents.")
    st.session_state.texts = texts

    lda_dtm, lda_vectorizer = build_vectorizer(texts)

    if auto_optimize:
        with st.spinner("Optimizing LDA (this takes several minutes)..."):
            best = optimize_lda(lda_dtm, maxiter=5)
        lda_k, lda_alpha, lda_beta = best["k"], best["alpha"], best["beta"]
        st.sidebar.info(f"Optimal: k={lda_k}, α={lda_alpha:.3f}, β={lda_beta:.4f}")
    else:
        lda_k, lda_alpha, lda_beta = k, alpha, beta

    with st.spinner(f"Building LDA model (k={lda_k})..."):
        model, doc_topic_matrix = build_lda(lda_dtm, k=lda_k, alpha=lda_alpha, beta=lda_beta)

    st.session_state.model = model
    st.session_state.lda_dtm = lda_dtm
    st.session_state.lda_vectorizer = lda_vectorizer

    years_series = pd.to_datetime(df["Date"], errors="coerce").dt.year.dropna().astype(int)
    years = years_series.tolist()[:len(doc_topic_matrix)]
    if len(years) < len(doc_topic_matrix):
        years = (years * ((len(doc_topic_matrix) // len(years)) + 1))[:len(doc_topic_matrix)]

    trends = compute_topic_trends(doc_topic_matrix, years)
    classified = classify_topics(trends)
    top_words_df = get_top_words(model, lda_vectorizer)
    st.session_state.classified = classified
    st.session_state.top_words = top_words_df

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Text Mining", "Timeline", "LDA", "Trends"])

with tab1:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.metric("Total Records", len(df))
        cols = [c for c in ["Title", "Date", "Cites", "Tags", "AuthorName"] if c in df.columns]
        st.dataframe(df[cols].head(100), use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{source}_{query}.csv")
    else:
        st.info("Configure and click 'Fetch & Analyze' in the sidebar.")

with tab2:
    if st.session_state.df is not None:
        texts = st.session_state.texts or []
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Word Cloud")
            wc_path = plot_wordcloud(texts, output_path=os.path.join(OUTPUT_DIR, f"wc_{query}.png"))
            st.image(wc_path)
        with col2:
            st.subheader("Dendrogram (top 200 docs)")
            dtm, _ = build_dtm(texts, min_df=2)
            dendro_path = plot_dendrogram(dtm, output_path=os.path.join(OUTPUT_DIR, f"dendro_{query}.pdf"), max_docs=200)
            st.success(f"Saved to `{dendro_path}` (open to view)")
    else:
        st.info("Fetch data first.")

with tab3:
    if st.session_state.df is not None:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Publications per Year")
            pub_path = plot_publications_per_year(df, output_path=os.path.join(OUTPUT_DIR, f"pub_{query}.png"))
            st.image(pub_path)
        with col2:
            if "Cites" in df.columns:
                st.subheader("Citations per Year")
                cite_path = plot_citations_per_year(df, output_path=os.path.join(OUTPUT_DIR, f"cites_{query}.png"))
                st.image(cite_path)
    else:
        st.info("Fetch data first.")

with tab4:
    if st.session_state.model is not None:
        model = st.session_state.model
        lda_dtm = st.session_state.lda_dtm
        lda_vectorizer = st.session_state.lda_vectorizer
        top_words_df = st.session_state.top_words

        st.subheader("Top Words per Topic")
        st.dataframe(top_words_df, use_container_width=True)

        st.subheader("Interactive LDA Visualization")
        lda_html_path = os.path.join(OUTPUT_DIR, f"lda_{query}.html")
        save_interactive_lda(model, lda_dtm, lda_vectorizer, output_path=lda_html_path)
        with open(lda_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.info("Fetch data first.")

with tab5:
    if st.session_state.classified is not None:
        classified = st.session_state.classified
        top_words_df = st.session_state.top_words
        merged = classified.merge(top_words_df, on="topic_id")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hot Topics")
            hot = merged[merged["trend_class"] == "hot"].nlargest(10, "slope")
            st.dataframe(hot[["slope", "p_value", "top_words"]].reset_index(drop=True), use_container_width=True)
        with col2:
            st.subheader("Cold Topics")
            cold = merged[merged["trend_class"] == "cold"].nsmallest(10, "slope")
            st.dataframe(cold[["slope", "p_value", "top_words"]].reset_index(drop=True), use_container_width=True)

        hot_cold_path = plot_hot_cold_topics(classified, top_words_df, output_path=os.path.join(OUTPUT_DIR, f"hc_{query}.png"))
        st.image(hot_cold_path)
    else:
        st.info("Fetch data first.")
