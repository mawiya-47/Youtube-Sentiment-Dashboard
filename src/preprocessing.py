"""
preprocessing.py
-----------------
Industrial-grade text preprocessing pipeline for YouTube comment sentiment analysis.

Why this module exists:
Raw social-media text is noisy: URLs, HTML fragments, emojis, contractions,
inconsistent casing, and punctuation all add variance that hurts a classical
ML model's ability to find real signal. This module turns raw comments into
clean, normalized tokens ready for feature extraction (TF-IDF / BoW).
"""

import re
import string
import unicodedata
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Ensure required NLTK corpora are present. Downloaded once, cached after.
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

# Minimal contraction map covering common conversational English.
# Why: contractions ("don't" -> "do not") fragment vocabulary if left as-is;
# expanding them merges variants into a single, more frequent token.
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "shouldn't": "should not", "wouldn't": "would not",
    "couldn't": "could not", "mightn't": "might not", "mustn't": "must not",
}

URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
HTML_RE = re.compile(r"<.*?>")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
NUMBER_RE = re.compile(r"\d+")
MULTI_SPACE_RE = re.compile(r"\s+")


def expand_contractions(text: str) -> str:
    """Expand common English contractions to their full form."""
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_emails(text: str) -> str:
    return EMAIL_RE.sub(" ", text)


def remove_html(text: str) -> str:
    return HTML_RE.sub(" ", text)


def remove_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", text)


def remove_numbers(text: str) -> str:
    return NUMBER_RE.sub(" ", text)


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_special_characters(text: str) -> str:
    """Keep only alphabetic characters and whitespace."""
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def normalize_unicode(text: str) -> str:
    """Normalize accented/unicode characters to their closest ASCII form."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")


def remove_multiple_spaces(text: str) -> str:
    return MULTI_SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def lemmatize(tokens: List[str]) -> List[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def stem(tokens: List[str]) -> List[str]:
    """Optional stemming (more aggressive than lemmatization). Off by default
    in the main pipeline because lemmatization preserves more meaning, which
    matters for a 3-class sentiment task."""
    return [STEMMER.stem(t) for t in tokens]


def clean_text(text: str, use_stemming: bool = False) -> str:
    """
    Full cleaning pipeline applied to a single raw comment.

    Order matters: URL/email/HTML/emoji removal must happen before
    punctuation stripping (otherwise stripped punctuation breaks their
    regex patterns), and lowercasing happens early so every subsequent
    regex/dictionary lookup is case-insensitive.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_html(text)
    text = remove_emojis(text)
    text = expand_contractions(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = remove_punctuation(text)
    text = remove_multiple_spaces(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens) if use_stemming else lemmatize(tokens)

    return " ".join(tokens)


def clean_series(texts, use_stemming: bool = False) -> List[str]:
    """Vectorized-friendly wrapper for cleaning a pandas Series / list of comments."""
    return [clean_text(t, use_stemming=use_stemming) for t in texts]
