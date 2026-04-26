import os
import pyLDAvis
import pyLDAvis.lda_model  # sklearn adapter (pyLDAvis 3.4 ships as lda_model, NOT sklearn)


def save_interactive_lda(model, dtm, vectorizer, output_path: str = "output/lda_interactive.html"):
    """Save pyLDAvis interactive HTML for a fitted sklearn LDA model.

    Parameters
    ----------
    model : sklearn.decomposition.LatentDirichletAllocation
        Fitted LDA model.
    dtm : scipy sparse matrix
        Document-term matrix produced by CountVectorizer.fit_transform().
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        Fitted vectorizer used to produce dtm.
    output_path : str
        Path for the output HTML file.

    Returns
    -------
    str
        The output_path that was written.

    Notes
    -----
    Uses pyLDAvis.lda_model (the sklearn adapter). In pyLDAvis 3.4 the module
    is named ``lda_model``, not ``sklearn`` — ``pyLDAvis.sklearn`` does not exist
    in this release.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    vis_data = pyLDAvis.lda_model.prepare(model, dtm, vectorizer, sort_topics=False)
    pyLDAvis.save_html(vis_data, output_path)
    return output_path
