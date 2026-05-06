from pathlib import Path

import joblib
import pandas as pd

from data_preprocessing import preprocess_input

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
COLUMNS_PATH = ROOT / "models" / "columns.pkl"
DEFAULT_THRESHOLD = 0.30

model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)


def predict(data_dict, threshold=DEFAULT_THRESHOLD):
    df = pd.DataFrame([data_dict])
    df = preprocess_input(df, columns)

    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= threshold)

    return prediction, probability


if __name__ == "__main__":
    sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70,
        "TotalCharges": 70,
    }

    pred, prob = predict(sample)
    print("Prediction:", pred)
    print("Churn Probability:", prob)