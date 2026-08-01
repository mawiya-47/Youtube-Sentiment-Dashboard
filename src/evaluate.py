"""
evaluate.py
-----------
Phase 9: Standalone evaluation utility — reload saved artifacts and reproduce
the full evaluation report without retraining. Useful for CI checks and for
the /model-info API endpoint.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_comments.csv")


def load_test_report() -> dict:
    """Return the persisted evaluation report for the best model (fast path,
    no retraining needed — used by the API)."""
    with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(MODELS_DIR, "model_comparison.json")) as f:
        comparison = json.load(f)
    return {"metadata": metadata, "model_comparison": comparison}


def recompute_on_holdout() -> dict:
    """
    Re-run the exact same stratified split (same random_state=42) used in
    training and recompute metrics from scratch. This is a sanity check that
    the saved model's reported metrics are reproducible, not just a training
    log artifact.
    """
    df = pd.read_csv(PROCESSED_PATH)
    df = df.dropna(subset=["clean_comment", "Sentiment"])
    df = df[df["clean_comment"].str.len() > 0]

    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))

    y = label_encoder.transform(df["Sentiment"])
    X = vectorizer.transform(df["clean_comment"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    acc = accuracy_score(y_test, y_pred)

    return {"accuracy": acc, "classification_report": report, "confusion_matrix": cm, "labels": label_encoder.classes_.tolist()}


if __name__ == "__main__":
    result = recompute_on_holdout()
    print(f"Recomputed holdout accuracy: {result['accuracy']:.4f}")
    print(json.dumps(result["classification_report"], indent=2))
