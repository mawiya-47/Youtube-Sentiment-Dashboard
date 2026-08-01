"""
feature_engineering.py
-----------------------
Phase 5: Feature extraction for the classical ML pipeline.

We implement Bag-of-Words and TF-IDF here (production baseline vectorizers).
Word2Vec / FastText / Sentence-Transformer / BERT embeddings are documented
in the project README as a future-scope extension (Phase 5 comparison table)
rather than implemented here, since they require either large pretrained
downloads or GPU training that isn't a good fit for a lean production
baseline — TF-IDF + a strong linear/ensemble model is what most real
production text-classification systems ship first, precisely because it's
fast, interpretable, and doesn't require a model registry for embeddings.

Comparison of techniques (informational, for the report):

| Technique              | Pros                                        | Cons                                      |
|-------------------------|---------------------------------------------|--------------------------------------------|
| Bag of Words            | Simple, fast, interpretable                  | No word order/semantics, huge sparse dim   |
| TF-IDF                  | Down-weights common words, strong baseline   | Still no semantics, sparse                 |
| Word2Vec / FastText     | Captures semantic similarity                 | Needs large corpus, loses interpretability |
| Sentence Transformers    | Strong semantic/contextual embeddings        | Slow, needs GPU for scale, heavy dependency|
| BERT / DistilBERT        | State-of-the-art contextual understanding    | Expensive to train/serve, needs GPU        |

For this dataset size (~17.6k rows after cleaning), TF-IDF + gradient-boosted
trees or linear SVM is the right complexity/accuracy/cost tradeoff for a
first production release.
"""

from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib


def build_tfidf_vectorizer(max_features: int = 15000, ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """
    TF-IDF vectorizer configuration.

    - max_features=15000: caps vocabulary size to control dimensionality and
      overfitting risk on a ~17k-row dataset.
    - ngram_range=(1,2): unigrams alone miss negation/intensifier patterns
      like "not good" or "really bad"; bigrams capture short local context
      cheaply without the cost of full embeddings.
    - min_df=2: drops words that appear in only one document (noise/typos).
    - sublinear_tf=True: dampens the effect of very high raw term counts
      (log-scaling), which helps with long comments dominating the signal.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
    )


def build_bow_vectorizer(max_features: int = 15000, ngram_range: Tuple[int, int] = (1, 2)) -> CountVectorizer:
    """Bag-of-Words vectorizer, kept for comparison against TF-IDF."""
    return CountVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)


def save_vectorizer(vectorizer, path: str) -> None:
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str):
    return joblib.load(path)
