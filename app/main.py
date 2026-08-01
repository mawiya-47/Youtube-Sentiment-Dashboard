"""
app/main.py
-----------
Phase 14: Production REST API for the YouTube Comment Sentiment model.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health        - liveness check
    GET  /model-info     - metadata about the currently loaded model
    POST /predict         - predict sentiment for one or more comments
    POST /retrain          - kick off retraining from data/processed
"""

import os
import sys
import logging
import subprocess
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_sentiment
from evaluate import load_test_report

# ---------------------------------------------------------------------------
# Logging setup (Phase 16 - MLOps: structured logging)
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("sentiment_api")

app = FastAPI(
    title="YouTube Comment Sentiment Analysis API",
    description="Classifies YouTube comments as Positive, Negative, or Neutral.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="Raw YouTube comment text")


class BatchPredictRequest(BaseModel):
    comments: List[str] = Field(..., min_length=1, max_length=200)


class PredictResponse(BaseModel):
    comment: str
    cleaned_comment: str
    predicted_sentiment: str
    confidence: float
    probabilities: dict


@app.get("/health")
def health():
    """Liveness/readiness probe for load balancers and container orchestrators."""
    try:
        # Cheap sanity check that model artifacts exist and are loadable.
        predict_sentiment("health check")
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")


@app.get("/model-info")
def model_info():
    """Returns metadata and evaluation metrics for the currently deployed model."""
    try:
        return load_test_report()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained model found. Run src/train.py first.")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict sentiment for a single comment."""
    try:
        result = predict_sentiment(request.comment)
        logger.info(f"Prediction: '{request.comment[:50]}...' -> {result['predicted_sentiment']}")
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
def predict_batch_endpoint(request: BatchPredictRequest):
    """Predict sentiment for a batch of comments (max 200 per request)."""
    try:
        results = [predict_sentiment(c) for c in request.comments]
        return {"count": len(results), "predictions": results}
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain():
    """
    Trigger a full retraining run (Phase 8/9/12 pipeline) from
    data/processed/cleaned_comments.csv.

    Note: runs synchronously for simplicity here. In a real production
    deployment this should enqueue a background job (Celery/RQ) or trigger
    a CI/CD pipeline rather than block the API worker.
    """
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(src_dir, "train.py")],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            logger.error(f"Retrain failed: {result.stderr}")
            raise HTTPException(status_code=500, detail="Retraining failed. Check logs.")
        logger.info("Retraining completed successfully.")
        return {"status": "retrained", "log_tail": result.stdout[-2000:]}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Retraining timed out.")


@app.get("/")
def root():
    return {
        "message": "YouTube Comment Sentiment Analysis API",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict", "/predict-batch", "/retrain"],
    }
