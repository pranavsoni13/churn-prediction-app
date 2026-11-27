"""Train, validate, calibrate the decision threshold, and save model artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, fbeta_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from src.data_preprocessing import load_and_preprocess
except ModuleNotFoundError:
    from data_preprocessing import load_and_preprocess

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
COLUMNS_PATH = MODEL_DIR / "columns.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
RANDOM_STATE = 42


def choose_threshold(y_true, probabilities) -> float:
    """Choose a validation threshold that maximizes F2 (recall-weighted)."""
    candidates = np.arange(0.10, 0.81, 0.01)
    scores = [
        fbeta_score(y_true, probabilities >= threshold, beta=2, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place the IBM Telco Customer Churn CSV "
            "in the data/ directory before training."
        )

    frame = load_and_preprocess(DATA_PATH)
    x = frame.drop(columns=["Churn"])
    y = frame["Churn"]
    if y.nunique() != 2:
        raise ValueError("Training data must contain both churn classes.")

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_train_val
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    validation_probability = model.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val, validation_probability)

    # Refit on train + validation after selecting the threshold. Test remains untouched.
    model.fit(x_train_val, y_train_val)
    test_probability = model.predict_proba(x_test)[:, 1]
    test_prediction = (test_probability >= threshold).astype(int)

    metrics = {
        "model_name": "Random Forest",
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, test_prediction)),
        "precision": float(precision_score(y_test, test_prediction, zero_division=0)),
        "recall": float(recall_score(y_test, test_prediction, zero_division=0)),
        "f2_score": float(fbeta_score(y_test, test_prediction, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probability)),
        "test_samples": int(len(y_test)),
        "random_state": RANDOM_STATE,
        "features": int(x.shape[1]),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(x.columns), COLUMNS_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2), encoding="utf-8")

    print("Training complete. Holdout test metrics:")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, test_prediction, target_names=["No Churn", "Churn"], zero_division=0))
    print(f"Saved model and metadata under: {MODEL_DIR}")


if __name__ == "__main__":
    main()
