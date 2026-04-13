import joblib
import pandas as pd
from data_preprocessing import preprocess_input

# Load model & columns
model = joblib.load("../models/churn_model.pkl")
columns = joblib.load("../models/columns.pkl")

def predict(data_dict):
    df = pd.DataFrame([data_dict])

    df = preprocess_input(df, columns)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability


# TEST
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
        "TotalCharges": 70
    }

    pred, prob = predict(sample)

    print("Prediction:", pred)
    print("Churn Probability:", prob)