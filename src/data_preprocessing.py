"""Shared, leakage-aware preprocessing for Telco customer churn."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

TARGET_COLUMN = "Churn"
ID_COLUMNS = ("customerID",)
BINARY_COLUMNS = ("Partner", "Dependents", "PhoneService", "PaperlessBilling")
NUMERIC_COLUMNS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
REQUIRED_INPUT_COLUMNS = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
)
YES_NO_MAP = {"Yes": 1, "No": 0}
GENDER_MAP = {"Male": 1, "Female": 0}


def _validate_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def _encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in BINARY_COLUMNS:
        if column in frame:
            frame[column] = frame[column].map(YES_NO_MAP)
    if "gender" in frame:
        frame["gender"] = frame["gender"].map(GENDER_MAP)
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    categorical = frame.select_dtypes(include=["object", "category"]).columns
    return pd.get_dummies(frame, columns=list(categorical), drop_first=True, dtype=int)


def load_and_preprocess(path: str | Path) -> pd.DataFrame:
    """Load raw Telco data, clean it, and return encoded features plus target."""
    frame = pd.read_csv(path)
    _validate_columns(frame, [*REQUIRED_INPUT_COLUMNS, TARGET_COLUMN])
    frame = frame.drop(columns=[c for c in ID_COLUMNS if c in frame.columns])
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.dropna(subset=["TotalCharges", TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    frame = frame.dropna(subset=[TARGET_COLUMN])
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    encoded = _encode_features(frame.drop(columns=[TARGET_COLUMN]))
    encoded[TARGET_COLUMN] = frame[TARGET_COLUMN].to_numpy()
    return encoded


def preprocess_input(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Transform one or more customer records and align to training features."""
    _validate_columns(frame, REQUIRED_INPUT_COLUMNS)
    encoded = _encode_features(frame.loc[:, REQUIRED_INPUT_COLUMNS])
    aligned = encoded.reindex(columns=list(columns), fill_value=0)
    aligned = aligned.apply(pd.to_numeric, errors="coerce").fillna(0)
    return aligned
