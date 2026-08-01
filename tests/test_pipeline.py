"""
tests/test_pipeline.py
-----------------------
Phase 18: Unit, model, and API tests.

Run with: pytest tests/ -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing import clean_text, remove_urls, remove_emails, expand_contractions
from predict import predict_sentiment


# ---------------------------------------------------------------------------
# Unit tests: preprocessing
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_lowercase(self):
        assert clean_text("AMAZING") == "amaze" or "amaz" in clean_text("AMAZING")

    def test_removes_urls(self):
        assert "http" not in remove_urls("check http://x.com out")

    def test_removes_emails(self):
        assert "@" not in remove_emails("contact me@example.com now")

    def test_expands_contractions(self):
        assert "not" in expand_contractions("don't")

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_removes_numbers(self):
        cleaned = clean_text("I watched this 100 times in 2023")
        assert not any(ch.isdigit() for ch in cleaned)

    def test_removes_stopwords(self):
        cleaned = clean_text("this is a really good video")
        assert "is" not in cleaned.split()
        assert "a" not in cleaned.split()


# ---------------------------------------------------------------------------
# Model tests: prediction system
# ---------------------------------------------------------------------------
class TestPrediction:
    def test_predict_returns_valid_schema(self):
        result = predict_sentiment("I love this so much, best video ever!")
        assert "predicted_sentiment" in result
        assert result["predicted_sentiment"] in ["positive", "negative", "neutral"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert set(result["probabilities"].keys()) == {"positive", "negative", "neutral"}

    def test_probabilities_sum_to_one(self):
        result = predict_sentiment("This is a decent video, nothing special.")
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_empty_comment_handled_gracefully(self):
        result = predict_sentiment("!!! ... ???")
        assert "predicted_sentiment" in result

    def test_strongly_positive_comment(self):
        result = predict_sentiment("Absolutely amazing, incredible, best content ever, love it so much!")
        assert result["predicted_sentiment"] == "positive"


# ---------------------------------------------------------------------------
# API tests (requires fastapi TestClient, no running server needed)
# ---------------------------------------------------------------------------
class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_endpoint(self, client):
        response = client.post("/predict", json={"comment": "great explanation, thank you!"})
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_sentiment"] in ["positive", "negative", "neutral"]

    def test_predict_endpoint_rejects_empty_body(self, client):
        response = client.post("/predict", json={"comment": ""})
        assert response.status_code == 422  # pydantic min_length validation

    def test_model_info_endpoint(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        assert "metadata" in response.json()

    def test_batch_predict_endpoint(self, client):
        response = client.post("/predict-batch", json={"comments": ["good video", "bad video", "meh"]})
        assert response.status_code == 200
        assert response.json()["count"] == 3
