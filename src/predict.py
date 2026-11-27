"""Reusable prediction interface for the trained churn model."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

try:
    from src.data_preprocessing import preprocess_input
except ModuleNotFoundError:
    from data_preprocessing import preprocess_input

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
COLUMNS_PATH = ROOT / "models" / "columns.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
DEFAULT_THRESHOLD = 0.30


@lru_cache(maxsize=1)
def load_artifacts():
    """Load artifacts once; clear this cache after retraining in long-running processes."""
    if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing. Run: python src/train_model.py")
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    threshold = DEFAULT_THRESHOLD
    if THRESHOLD_PATH.exists():
        threshold = float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Saved decision threshold must be between 0 and 1.")
    return model, columns, threshold


def predict(data_dict: dict, threshold: float | None = None) -> tuple[int, float]:
    model, columns, saved_threshold = load_artifacts()
    cutoff = saved_threshold if threshold is None else float(threshold)
    if not 0 < cutoff < 1:
        raise ValueError("threshold must be between 0 and 1.")
    frame = preprocess_input(pd.DataFrame([data_dict]), columns)
    probability = float(model.predict_proba(frame)[0, 1])
    return int(probability >= cutoff), probability


if __name__ == "__main__":
    sample = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70, "TotalCharges": 70,
    }
    label, probability = predict(sample)
    print({"prediction": "Churn" if label else "No Churn", "probability": round(probability, 4)})
