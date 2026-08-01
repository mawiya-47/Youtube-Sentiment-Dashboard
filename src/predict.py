"""
predict.py
----------
Phase 13: Prediction system.

Loads the saved vectorizer, label encoder, and best model, then exposes a
single `predict_sentiment` function used by both the CLI and the FastAPI app.
"""

import os
import joblib
import numpy as np
from typing import Dict

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_vectorizer = None
_label_encoder = None
_model = None


def _load_artifacts():
    """Lazy-load model artifacts once, then cache in module-level globals.
    Avoids re-reading joblib files from disk on every single prediction
    request (important for API latency)."""
    global _vectorizer, _label_encoder, _model
    if _model is None:
        _vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        _label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        _model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    return _vectorizer, _label_encoder, _model


def predict_sentiment(comment: str) -> Dict:
    """
    Predict sentiment for a single raw YouTube comment.

    Returns:
        {
          "comment": original text,
          "cleaned_comment": preprocessed text,
          "predicted_sentiment": "positive" | "negative" | "neutral",
          "confidence": float (probability of predicted class),
          "probabilities": {"positive": .., "negative": .., "neutral": ..}
        }
    """
    vectorizer, label_encoder, model = _load_artifacts()

    cleaned = clean_text(comment)
    if not cleaned:
        return {
            "comment": comment,
            "cleaned_comment": cleaned,
            "predicted_sentiment": "neutral",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in label_encoder.classes_},
            "warning": "Comment was empty after cleaning (e.g. only stopwords/punctuation).",
        }

    X = vectorizer.transform([cleaned])
    pred_idx = model.predict(X)[0]
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        # Fallback for models without predict_proba: uniform except predicted class
        probs = np.zeros(len(label_encoder.classes_))
        probs[pred_idx] = 1.0

    prob_dict = {label_encoder.classes_[i]: float(round(p, 4)) for i, p in enumerate(probs)}

    return {
        "comment": comment,
        "cleaned_comment": cleaned,
        "predicted_sentiment": predicted_label,
        "confidence": float(round(max(probs), 4)),
        "probabilities": prob_dict,
    }


def predict_batch(comments) -> list:
    """Predict sentiment for a list of comments."""
    return [predict_sentiment(c) for c in comments]


if __name__ == "__main__":
    samples = [
        "This video completely changed how I think about investing, amazing work!",
        "This was the worst explanation I've ever seen, total waste of time.",
        "Not sure what to think, it was okay I guess.",
    ]
    for s in samples:
        result = predict_sentiment(s)
        print(f"\nComment: {s}")
        print(f"Prediction: {result['predicted_sentiment']} (confidence: {result['confidence']})")
        print(f"Probabilities: {result['probabilities']}")
