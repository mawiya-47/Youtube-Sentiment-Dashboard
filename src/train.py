"""
train.py
--------
Phases 6-9 + 12: Label encoding, train/test split, model building,
evaluation, and model persistence.

Trains and compares a set of classical ML algorithms on TF-IDF features,
picks the best model by macro F1 (appropriate given class imbalance), and
saves the full inference pipeline (vectorizer + label encoder + model).
"""

import os
import sys
import json
import time
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import build_tfidf_vectorizer, save_vectorizer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_comments.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    df = df.dropna(subset=["clean_comment", "Sentiment"])
    df = df[df["clean_comment"].str.len() > 0]
    return df


def get_models() -> Dict[str, object]:
    """
    Phase 8: Model zoo.

    We include every classical algorithm that is a realistic candidate for a
    sparse, ~15k-dim TF-IDF, 3-class, ~17k-row problem. Deep learning
    (LSTM/GRU/BiLSTM/CNN/Transformer/DistilBERT/RoBERTa) is intentionally
    scoped out of this first pass per project instructions — they need
    GPU-scale training and a tokenizer/embedding pipeline that's a separate
    deliverable, documented in the README as a next iteration.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=40, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=40, class_weight="balanced", n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=40, class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=42),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000, random_state=42), cv=3),
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "Passive Aggressive": CalibratedClassifierCV(PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000, random_state=42), cv=3),
        "SGD Classifier": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1,
        )
    return models


def evaluate_model(name, model, X_test, y_test, label_encoder) -> Dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Weighted F1: {w_f1:.4f}")

    return {
        "model_name": name,
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_precision": w_precision,
        "weighted_recall": w_recall,
        "weighted_f1": w_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def main():
    print("=" * 70)
    print("PHASE 6: LABEL ENCODING")
    print("=" * 70)
    df = load_data()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Sentiment"])
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"Label mapping: {mapping}")

    print("\n" + "=" * 70)
    print("PHASE 5: FEATURE ENGINEERING (TF-IDF)")
    print("=" * 70)
    vectorizer = build_tfidf_vectorizer()
    X = vectorizer.fit_transform(df["clean_comment"])
    print(f"TF-IDF matrix shape: {X.shape}")

    print("\n" + "=" * 70)
    print("PHASE 7: TRAIN/TEST SPLIT (80/20, stratified)")
    print("=" * 70)
    # Why 80/20: with ~17.6k rows and 3 imbalanced classes, 20% test (~3.5k)
    # leaves enough samples per class (even the minority "negative" class)
    # to get a statistically meaningful evaluation, while keeping 80% for
    # training so the model sees enough of the minority class.
    # Stratify=y preserves class proportions in both splits — critical given
    # the imbalance (positive 65% / neutral 26% / negative 13%).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    print("\n" + "=" * 70)
    print("PHASE 8 & 9: MODEL BUILDING + EVALUATION")
    print("=" * 70)
    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        res = evaluate_model(name, model, X_test, y_test, label_encoder)
        res["train_time_seconds"] = round(elapsed, 2)
        results.append(res)
        trained_models[name] = model
        print(f"Train time: {elapsed:.2f}s")

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by Macro F1)")
    print("=" * 70)
    print(results_df[["model_name", "accuracy", "macro_f1", "weighted_f1", "train_time_seconds"]].to_string(index=False))

    best_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} (Macro F1: {results_df.iloc[0]['macro_f1']:.4f})")

    print("\n" + "=" * 70)
    print("PHASE 12: SAVE MODEL ARTIFACTS")
    print("=" * 70)
    save_vectorizer(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump({
            "best_model_name": best_name,
            "label_mapping": {str(k): int(v) for k, v in mapping.items()},
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "vocab_size": int(X.shape[1]),
            "metrics": {k: v for k, v in results_df.iloc[0].to_dict().items() if k not in ["classification_report", "confusion_matrix"]},
        }, f, indent=2, default=str)

    # Save full comparison table for the dashboard / report
    comparison_records = []
    for r in results:
        comparison_records.append({
            "model_name": r["model_name"],
            "accuracy": r["accuracy"],
            "macro_precision": r["macro_precision"],
            "macro_recall": r["macro_recall"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "train_time_seconds": r["train_time_seconds"],
        })
    with open(os.path.join(MODELS_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison_records, f, indent=2)

    with open(os.path.join(MODELS_DIR, "best_model_full_report.json"), "w") as f:
        json.dump(results_df.iloc[0].to_dict(), f, indent=2, default=str)

    print(f"\nAll artifacts saved to {MODELS_DIR}")
    print("Files: tfidf_vectorizer.joblib, label_encoder.joblib, best_model.joblib,")
    print("       model_metadata.json, model_comparison.json, best_model_full_report.json")


if __name__ == "__main__":
    main()
