# YouTube Comment Sentiment Analysis

A production-oriented NLP system that classifies YouTube comments as
**Positive**, **Negative**, or **Neutral**, built on the Kaggle
"YouTube Comments Dataset."

> **Scope note:** This is Stage 1 of the project — a complete, real,
> working classical-ML pipeline (data → EDA → preprocessing → features →
> 13 models trained & compared → best model saved → prediction API →
> tests → Docker). Deep learning models (LSTM/BiLSTM/CNN/DistilBERT/RoBERTa),
> a React dashboard, Optuna/SHAP/LIME, and multi-cloud deployment are
> deliberately **not** included in this pass — they're real, substantial
> pieces of work in their own right and are listed under "Future Scope"
> below rather than stubbed out with placeholder code.

---

## 1. Dataset

| | |
|---|---|
| Source | Kaggle "YouTube Comments Dataset" |
| Raw rows | 18,409 |
| Columns | `Comment` (text), `Sentiment` (positive / negative / neutral) |
| Missing values found | 44 (dropped) |
| Duplicate rows found | 531 (dropped) |
| **Final clean dataset** | **17,647 rows** |

Note: the dataset was *not* fully pre-cleaned as originally described — real
data always deserves a direct check rather than trusting a description, so
missing values and duplicates were found and handled explicitly in
`notebooks/01_data_understanding_eda.py`.

**Class distribution** (imbalanced, as is typical for organic comment data):

| Class | Count | % |
|---|---|---|
| Positive | 11,432 | 64.6% |
| Neutral | 4,638 | 26.2% |
| Negative | 2,338 | 13.2% |

**Business understanding:** the goal is to let a content team or platform
monitor audience sentiment at scale — flagging negative sentiment spikes and
quantifying reception — without manually reading thousands of comments. The
imbalance is a real prior (people comment more when praising content), which
is why model selection uses **macro F1**, not raw accuracy.

---

## 2. Project Structure

```
youtube_sentiment_analysis/
├── data/
│   ├── raw/youtube_comments.csv         # original Kaggle file
│   └── processed/cleaned_comments.csv    # after dedup + cleaning + NLP normalization
├── notebooks/
│   ├── 01_data_understanding_eda.py      # Phase 2 & 3
│   └── figures/                            # 11 saved EDA/eval charts
├── src/
│   ├── preprocessing.py                   # Phase 4: NLP cleaning pipeline
│   ├── feature_engineering.py             # Phase 5: TF-IDF / BoW
│   ├── train.py                           # Phase 6-9, 12: encode, split, train, evaluate, save
│   ├── evaluate.py                        # Phase 9: reproducible evaluation
│   ├── predict.py                         # Phase 13: inference
│   └── config.py                          # config loader
├── models/                                 # saved artifacts (joblib + JSON reports)
├── app/
│   └── main.py                            # Phase 14: FastAPI backend
├── tests/
│   └── test_pipeline.py                   # Phase 18: unit + model + API tests (16 tests, all passing)
├── config/config.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 3. Preprocessing Pipeline (Phase 4)

`src/preprocessing.py` applies, in order: lowercasing → Unicode
normalization → URL/email/HTML/emoji removal → contraction expansion →
number removal → special-character/punctuation stripping → whitespace
collapse → tokenization → stopword removal → lemmatization.

Example:
```
Input:  "I don't LOVE this!! Check http://example.com 😀 123 times..."
Output: "love check time"
```

## 4. Feature Engineering (Phase 5)

TF-IDF (unigrams + bigrams, `max_features=15000`, `min_df=2`,
`sublinear_tf=True`) is the production feature set. Bag-of-Words is also
implemented for comparison. Word2Vec/FastText/Sentence-Transformers/BERT
embeddings are documented (with a pros/cons table) in
`src/feature_engineering.py` as the natural next iteration — they need
either a large pretrained download or GPU compute that isn't the right
first move for a ~17.6k-row dataset.

## 5. Label Encoding (Phase 6)

```
{'negative': 0, 'neutral': 1, 'positive': 2}
```

## 6. Train/Test Split (Phase 7)

80% train / 20% test, **stratified** on the target to preserve the
64/26/13 class ratio in both splits — important given the imbalance.

## 7. Model Comparison (Phase 8 & 9)

13 classical algorithms trained on identical TF-IDF features and evaluated
on the same held-out 20% test set (3,530 comments):

| Model | Accuracy | Macro F1 | Weighted F1 | Train Time (s) |
|---|---|---|---|---|
| **SGD Classifier** ⭐ | **0.751** | **0.659** | **0.750** | 0.10 |
| Logistic Regression | 0.728 | 0.654 | 0.740 | 0.37 |
| Extra Trees | 0.714 | 0.637 | 0.723 | 11.64 |
| Linear SVM | 0.746 | 0.630 | 0.732 | 0.41 |
| LightGBM | 0.736 | 0.620 | 0.728 | 18.91 |
| Random Forest | 0.698 | 0.614 | 0.706 | 11.73 |
| Passive Aggressive | 0.730 | 0.609 | 0.710 | 0.23 |
| XGBoost | 0.733 | 0.603 | 0.718 | 68.20 |
| Decision Tree | 0.612 | 0.537 | 0.629 | 1.58 |
| Gradient Boosting | 0.686 | 0.499 | 0.635 | 38.88 |
| Multinomial Naive Bayes | 0.675 | 0.438 | 0.604 | 0.00 |
| AdaBoost | 0.636 | 0.304 | 0.517 | 5.54 |
| KNN | 0.291 | 0.189 | 0.203 | 0.00 |

**Winner: SGD Classifier (log loss)** — best macro F1 despite being one of
the fastest models to train. Linear models generally beat trees here, which
is typical for high-dimensional sparse TF-IDF features (~15,000 dims):
tree splits struggle to exploit sparse one-hot-like features as well as a
linear decision boundary does.

**Per-class performance of the winning model** (from `src/evaluate.py`):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.55 | 0.51 | 0.53 | 463 |
| Neutral | 0.59 | 0.60 | 0.59 | 860 |
| Positive | 0.85 | 0.86 | 0.86 | 2,207 |

**Honest limitation:** the model is noticeably weaker on Negative and
Neutral than on Positive — a direct consequence of class imbalance (only
13% of training data is Negative) and semantic overlap between mild
negativity and neutral text. Confusion matrix: `notebooks/figures/09_confusion_matrix_best_model.png`.
This is the natural target for future work — e.g. class-weighted loss
tuning, SMOTE-style oversampling, or a contextual embedding model that
can better separate "this was okay" (neutral) from "this was bad" (negative).

## 8. Model Interpretability (Phase 11 — partial)

Not yet implemented: SHAP/LIME require additional dependencies and are
listed under Future Scope. Basic interpretability is available today via
linear model coefficients (`model.coef_` on the saved SGD/LogReg models
maps directly to TF-IDF vocabulary terms).

## 9. Saved Artifacts (Phase 12)

All in `models/`:
- `tfidf_vectorizer.joblib`
- `label_encoder.joblib`
- `best_model.joblib` (SGD Classifier)
- `model_metadata.json`, `model_comparison.json`, `best_model_full_report.json`

## 10. Prediction System (Phase 13)

```python
from src.predict import predict_sentiment
predict_sentiment("This tutorial was absolutely fantastic, learned so much!")
# {'predicted_sentiment': 'positive', 'confidence': 0.8167,
#  'probabilities': {'negative': 0.0656, 'neutral': 0.1177, 'positive': 0.8167}}
```

## 11. REST API (Phase 14)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | liveness check |
| `/model-info` | GET | metadata + full model comparison table |
| `/predict` | POST | `{"comment": "..."}` → sentiment + confidence + probabilities |
| `/predict-batch` | POST | `{"comments": [...]}`, max 200 |
| `/retrain` | POST | re-runs `src/train.py` synchronously |
| `/docs` | GET | Swagger UI (auto-generated) |

Verified working end-to-end (see test run): `/health`, `/predict`, and
`/model-info` all return correct, non-mocked responses from the real
trained model.

## 12. Testing (Phase 18)

`tests/test_pipeline.py` — 16 tests, all passing: unit tests for the
preprocessing pipeline, model tests for the prediction schema/consistency,
and API tests using FastAPI's `TestClient` (no live server needed).

```bash
pytest tests/ -v
```

## 13. Installation

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4

# 1. Run data understanding + EDA (produces data/processed/cleaned_comments.csv)
python notebooks/01_data_understanding_eda.py

# 2. Train & compare all models, save the best one
python src/train.py

# 3. Try a prediction from the CLI
python src/predict.py

# 4. Run tests
pytest tests/ -v

# 5. Start the API
uvicorn app.main:app --reload
```

Or with Docker:
```bash
docker build -t yt-sentiment .
docker run -p 8000:8000 yt-sentiment
```

## 14. MLOps (Phase 16 — implemented subset)

- **Logging:** structured logging to `logs/app.log` + stdout (`app/main.py`)
- **Config:** centralized in `config/config.yaml`, loaded via `src/config.py`
- **Docker:** `Dockerfile` included, with healthcheck
- **Retraining:** `/retrain` endpoint re-runs the full pipeline

Not yet implemented (Future Scope): GitHub Actions CI/CD, MLflow experiment
tracking, DVC data versioning, drift monitoring.

## 15. Future Scope

- Deep learning models: LSTM, BiLSTM, GRU, CNN-for-text, DistilBERT, RoBERTa
- Word2Vec / FastText / Sentence-Transformer embeddings
- Hyperparameter tuning: GridSearchCV / RandomizedSearchCV / Optuna
- Model interpretability: SHAP, LIME
- React glassmorphism dashboard (analytics, live prediction, word clouds)
- MLflow + DVC for experiment/data versioning
- GitHub Actions CI/CD, cloud deployment (Render/Railway/AWS/Azure/GCP/HF Spaces)
- Class-imbalance handling (SMOTE, focal loss) to improve Negative/Neutral recall

## 16. License

MIT — for educational/portfolio use.

## 17. Contributors

Built as an end-to-end ML engineering exercise.
