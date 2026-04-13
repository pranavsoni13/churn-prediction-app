import streamlit as st
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_input

# Load model
model = joblib.load("models/churn_model.pkl")
columns = joblib.load("models/columns.pkl")

st.title("Customer Churn Prediction")

# Inputs
st.subheader("Enter Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure", 0, 72)
monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=50.0,
    step=1.0 
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0,
    step=10.0
)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

if st.button("Predict"):

    data = {
        "gender": gender,
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    df = pd.DataFrame([data])
    df = preprocess_input(df, columns)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer likely to churn (Prob: {prob*100:.2f}%)")
        st.write(f"Churn Probability: {prob*100:.2f}%")
    else:
        st.success(f"Customer likely to stay (Prob: {prob*100:.2f}%)")
        