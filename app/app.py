import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import preprocess_input

APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "models" / "churn_model.pkl"
COLUMNS_PATH = APP_ROOT / "models" / "columns.pkl"
THRESHOLD = 0.30
MODEL_NAME = "Random Forest"


@st.cache_resource
def load_artifacts():
    """Load the trained model and expected feature columns once per session."""
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


model, columns = load_artifacts()

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0f172a;
        --panel: #1e293b;
        --panel-soft: #24324a;
        --border: #34445f;
        --muted: #94a3b8;
        --text: #e5e7eb;
        --primary: #6366f1;
        --success: #22c55e;
        --danger: #ef4444;
        --warning: #f59e0b;
        --cyan: #0ea5e9;
    }

    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stSidebar"] { background: #1d293b; border-right: 1px solid #24324a; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }

    .brand-card {
        background: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 1.7rem 1rem;
        text-align: center;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: .03em;
        margin: .2rem 0 1.9rem;
    }
    .nav-card {
        background: var(--primary);
        border-radius: 8px;
        padding: .85rem 1rem;
        margin-bottom: 1.1rem;
        color: white;
        font-size: 1.03rem;
    }
    .nav-muted {
        color: var(--muted);
        font-size: 1rem;
        padding: .55rem 1rem;
        margin: .25rem 0;
    }
    .sidebar-status {
        position: fixed;
        bottom: 1.4rem;
        width: 18rem;
        max-width: calc(100% - 2rem);
        background: #21446f;
        color: #60a5fa;
        border-radius: 8px;
        padding: 1.45rem 1rem;
        text-align: center;
        line-height: 1.8;
    }
    .sidebar-status strong { color: #10b981; }

    h1, h2, h3 { color: #f8fafc !important; }
    .page-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 850;
        margin: .5rem 0 .45rem;
    }
    .page-subtitle {
        text-align: center;
        color: var(--muted);
        font-size: 1.1rem;
        margin-bottom: .2rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin: .65rem 0 .8rem;
        color: #f1f5f9;
    }

    div[data-testid="stTextInput"] label, div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label, div[data-testid="stCheckbox"] label,
    div[data-testid="stSlider"] label {
        color: var(--muted) !important;
        font-size: .98rem !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text) !important;
        min-height: 4.25rem;
    }
    input, textarea { color: var(--text) !important; }
    [data-testid="stNumberInput"] input { font-size: 1.05rem; }
    .stSelectbox [data-baseweb="select"] span { color: var(--text) !important; }

    div[data-testid="stCheckbox"] {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 4.6rem;
    }
    div[data-testid="stCheckbox"] label {
        width: 100%;
        justify-content: space-between;
        flex-direction: row-reverse;
        gap: 1.5rem;
    }
    div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] { flex: 1; }
    div[data-testid="stCheckbox"] span[data-baseweb="checkbox"] {
        background: var(--panel-soft);
        border-radius: 10px;
        padding: 1.05rem 1.35rem;
        border: 0;
    }
    div[data-testid="stCheckbox"] span[data-baseweb="checkbox"][aria-checked="true"] {
        background: var(--primary);
    }

    div.stButton > button {
        width: 100%;
        min-height: 4.7rem;
        border: none;
        border-radius: 8px;
        background: var(--primary);
        color: white;
        font-size: 1.35rem;
        font-weight: 850;
        letter-spacing: .01em;
    }
    div.stButton > button:hover { background: #7476ff; color: white; border: none; }

    .result-banner {
        border-radius: 8px;
        padding: 1.8rem 1rem;
        text-align: center;
        font-size: 1.65rem;
        font-weight: 850;
        margin: .2rem 0 .7rem;
    }
    .danger-banner { background: rgba(127, 29, 29, .86); border: 2px solid var(--danger); color: #fecaca; }
    .success-banner { background: rgba(20, 83, 45, .9); border: 2px solid var(--success); color: #86efac; }
    .metric-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        min-height: 9.3rem;
    }
    .metric-label { color: var(--muted); font-size: 1.08rem; margin-bottom: 1rem; }
    .metric-value { text-align: center; font-size: 1.45rem; font-weight: 850; margin-top: .8rem; }
    .prob-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.6rem 1.5rem 1rem;
        margin: .6rem 0 .2rem;
    }
    .prob-head { display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: 1.15rem; margin: 0 3rem 1.45rem; }
    .prob-value { font-size: 2.35rem; font-weight: 900; }
    .track { height: 3.9rem; background: #334155; border-radius: 9px; overflow: hidden; }
    .fill { height: 100%; border-radius: 9px; }
    .threshold-note { text-align: center; color: #64748b; margin-top: .55rem; }
    .risk-title { font-size: 1.2rem; font-weight: 850; margin: .65rem 0 .25rem; color: #f1f5f9; }
    .bar-row { display: grid; grid-template-columns: 31% 1fr 3rem; gap: 1rem; align-items: center; margin: .55rem 0; }
    .bar-label { color: #cbd5e1; }
    .bar-bg { background: #334155; height: 2.8rem; border-radius: 6px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 6px; }
    .bar-value { font-weight: 800; text-align: right; }
    .recommendation {
        border: 2px solid var(--warning);
        background: rgba(28, 25, 23, .76);
        border-radius: 8px;
        padding: 1.55rem 3rem;
        margin-top: 1rem;
        color: #d1d5db;
    }
    .recommendation h3 { color: #facc15 !important; font-size: 1.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def model_probability(customer_data: dict) -> tuple[int, float, pd.DataFrame]:
    input_df = pd.DataFrame([customer_data])
    processed_df = preprocess_input(input_df, columns)
    probability = float(model.predict_proba(processed_df)[0][1])
    prediction = int(probability >= THRESHOLD)
    return prediction, probability, processed_df


def aggregate_feature_importance() -> list[tuple[str, float]]:
    names = {
        "tenure": "Tenure",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "Contract": "Contract Type",
        "InternetService": "Internet Service",
        "PaymentMethod": "Payment Method",
        "OnlineSecurity": "Online Security",
        "TechSupport": "Tech Support",
        "PaperlessBilling": "Paperless Billing",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "PhoneService": "Phone Service",
    }
    grouped: dict[str, float] = {}
    for col, importance in zip(columns, model.feature_importances_):
        base = col.split("_")[0]
        grouped[base] = grouped.get(base, 0.0) + float(importance)
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [(names.get(name, name), value) for name, value in ranked]


def active_risk_factors(data: dict) -> list[tuple[str, int, str]]:
    factors: list[tuple[str, int, str]] = []
    if data["Contract"] == "Month-to-month":
        factors.append(("Month-to-month contract", 92, "#ef4444"))
    if data["MonthlyCharges"] >= 65:
        factors.append((f"High monthly charges (${data['MonthlyCharges']:.2f})", 78, "#f97316"))
    if data["PaymentMethod"] == "Electronic check":
        factors.append(("Electronic check payment", 65, "#eab308"))
    if data["InternetService"] == "Fiber optic":
        factors.append(("Fiber optic internet", 58, "#6366f1"))
    if data["OnlineSecurity"] == "No" and data["InternetService"] != "No":
        factors.append(("No online security", 47, "#10b981"))
    if data["TechSupport"] == "No" and data["InternetService"] != "No":
        factors.append(("No tech support", 41, "#0ea5e9"))
    if data["tenure"] <= 12:
        factors.append(("New customer tenure", 38, "#a855f7"))
    if data["PaperlessBilling"] == "Yes":
        factors.append(("Paperless billing", 31, "#14b8a6"))
    return factors[:5]


def render_probability_panel(probability: float, churn: bool):
    color = "#ef4444" if churn else "#22c55e"
    note = "Above threshold → Churn Flagged" if churn else "Below threshold → Low Churn Risk"
    st.markdown(
        f"""
        <div class="prob-panel">
            <div class="prob-head">
                <span>Churn Probability</span>
                <span class="prob-value" style="color:{color};">{pct(probability)}</span>
            </div>
            <div class="track"><div class="fill" style="width:{min(probability * 100, 100):.1f}%; background:{color};"></div></div>
            <div class="threshold-note">Threshold: {THRESHOLD:.2f} &nbsp;|&nbsp; {note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bar_rows(rows: list[tuple[str, int, str]]):
    for label, value, color in rows:
        st.markdown(
            f"""
            <div class="bar-row">
                <div class="bar-label">{label}</div>
                <div class="bar-bg"><div class="bar-fill" style="width:{value}%; background:{color};"></div></div>
                <div class="bar-value" style="color:{color};">{value}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown('<div class="brand-card">⚡ CCPS</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-card">📊 Prediction<br><span style="padding-left:3.2rem;">👤 Customer Input</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-muted">📈 Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-muted">🧠 Model Info</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-muted">📄 Documentation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sidebar-status">Model: {MODEL_NAME}<br><strong>Threshold: {THRESHOLD:.2f}</strong></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="page-title">Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Enter customer details to predict churn probability</div>', unsafe_allow_html=True)

left, right = st.columns([1.05, 1], gap="large")

with left:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=24, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.50, step=0.5, format="%.2f")
    suggested_total = round(float(tenure) * float(monthly_charges), 2)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=suggested_total, step=10.0, format="%.2f")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

with right:
    c1, c2 = st.columns([1.25, .75])
    with c1:
        st.write("")
        senior_citizen = st.checkbox("Senior Citizen", value=False)
        partner = st.checkbox("Partner", value=True)
        dependents = st.checkbox("Dependents", value=False)
        phone_service = st.checkbox("Phone Service", value=True)
        online_security = st.checkbox("Online Security", value=False, disabled=internet_service == "No")
        tech_support = st.checkbox("Tech Support", value=True, disabled=internet_service == "No")
        streaming_tv = st.checkbox("Streaming TV", value=True, disabled=internet_service == "No")
        paperless_billing = st.checkbox("Paperless Billing", value=True)

with st.expander("Advanced customer options", expanded=False):
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"] if phone_service else ["No phone service"])
    with adv2:
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"] if internet_service != "No" else ["No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
    with adv3:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])

customer_data = {
    "gender": gender,
    "SeniorCitizen": int(senior_citizen),
    "Partner": yes_no(partner),
    "Dependents": yes_no(dependents),
    "tenure": int(tenure),
    "PhoneService": yes_no(phone_service),
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": yes_no(online_security) if internet_service != "No" else "No internet service",
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": yes_no(tech_support) if internet_service != "No" else "No internet service",
    "StreamingTV": yes_no(streaming_tv) if internet_service != "No" else "No internet service",
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": yes_no(paperless_billing),
    "PaymentMethod": payment_method,
    "MonthlyCharges": float(monthly_charges),
    "TotalCharges": float(total_charges),
}

if st.button("🔮  Predict Churn Probability"):
    prediction, probability, _ = model_probability(customer_data)
    churn = prediction == 1

    if churn:
        st.markdown('<div class="page-title">Prediction Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-banner danger-banner">⚠️ &nbsp; Likely to Churn</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="page-title">Prediction Result — Not Likely to Churn</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-banner success-banner">✅ &nbsp; Not Likely to Churn</div>', unsafe_allow_html=True)

    render_probability_panel(probability, churn)

    if churn:
        cards = st.columns(4)
        card_values = [
            ("Prediction", "🚩 Churn", "#ef4444"),
            ("Confidence", pct(probability), "#f97316"),
            ("Threshold", f"{THRESHOLD:.2f}", "#818cf8"),
            ("Model", MODEL_NAME, "#10b981"),
        ]
        for card, (label, value, color) in zip(cards, card_values):
            with card:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color};">{value}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown('<div class="risk-title">Top Churn Risk Factors Identified:</div>', unsafe_allow_html=True)
        factors = active_risk_factors(customer_data) or [("Model probability above churn threshold", int(probability * 100), "#ef4444")]
        render_bar_rows(factors)
        st.markdown(
            """
            <div class="recommendation">
                <h3>💡 Retention Recommendation:</h3>
                <p>High-risk customer. Consider offering a loyalty discount, online security/support bundle, or a long-term contract upgrade.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-title">📊 Feature Importance (Random Forest — Gini Impurity Reduction)</div>', unsafe_allow_html=True)
        importance_rows = []
        palette = ["#6366f1", "#8b5cf6", "#a78bfa", "#0ea5e9", "#38bdf8", "#10b981", "#34d399", "#6ee7b7"]
        top = aggregate_feature_importance()[:8]
        max_value = top[0][1] if top else 1
        for idx, (label, value) in enumerate(top):
            normalized = max(3, int(value / max_value * 94))
            importance_rows.append((label, normalized, palette[idx % len(palette)]))
        render_bar_rows(importance_rows)
else:
    st.caption("Prediction results will appear here after you submit the customer profile.")