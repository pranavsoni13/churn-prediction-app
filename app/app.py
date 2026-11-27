import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import preprocess_input

def setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


API_URL = setting("CHURN_API_URL").rstrip("/")
API_TOKEN = setting("CHURN_API_TOKEN")

APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "models" / "churn_model.pkl"
COLUMNS_PATH = APP_ROOT / "models" / "columns.pkl"
USER_STORE_PATH = APP_ROOT / ".streamlit_users.json"
THRESHOLD = float(json.loads((APP_ROOT / 'models' / 'threshold.json').read_text(encoding='utf-8')).get('threshold', 0.30)) if (APP_ROOT / 'models' / 'threshold.json').exists() else 0.30
MODEL_NAME = "Random Forest"
METRICS_PATH = APP_ROOT / 'models' / 'metrics.json'
METRICS = json.loads(METRICS_PATH.read_text(encoding='utf-8')) if METRICS_PATH.exists() else {}

MODEL_ACCURACY = f"{METRICS.get('accuracy', 0):.1%}" if METRICS else "Not trained"
MODEL_ROC_AUC = f"{METRICS.get('roc_auc', 0):.1%}" if METRICS else "Not trained"

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    """Load trusted local model artifacts once per Streamlit process."""
    if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing. Train with: python src/train_model.py")
    return joblib.load(MODEL_PATH), joblib.load(COLUMNS_PATH)


try:
    model, columns = load_artifacts()
    ARTIFACT_ERROR = None
except FileNotFoundError as exc:
    model, columns = None, []
    ARTIFACT_ERROR = str(exc)

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

    .stApp { background: #0b1220; color: #e5e7eb; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #101a2d; border-right: 1px solid #26344d; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    .hero { padding: 1.5rem 0 1.1rem; margin-bottom: 1rem; }
    .hero h1 { font-size: clamp(2rem, 3vw, 2.8rem); letter-spacing: -.04em; margin: .3rem 0; }
    .hero p { color: #9fb0c8; font-size: 1.05rem; margin: 0; }
    .eyebrow { color: #8b9cff; font-size: .72rem; font-weight: 800; letter-spacing: .14em; }
    .section-heading { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin: 1.3rem 0 .65rem; }
    .section-heading h3 { margin: .2rem 0 0; font-size: 1.25rem; }
    .section-hint { color: #8fa1bb; font-size: .82rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: rgba(20,31,51,.72); border: 1px solid #293953; border-radius: 16px; padding: 1rem 1.15rem; }
    div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 10px !important; min-height: 2.8rem; }
    div.stButton > button { border-radius: 10px; min-height: 3rem; font-weight: 750; }
    div.stButton > button[kind="primary"] { background: linear-gradient(110deg,#6366f1,#8b5cf6); border: 0; }
    .metric-card { background: #121e33; border: 1px solid #293953; border-radius: 14px; padding: 1rem; min-height: 6.8rem; }
    .metric-label { font-size: .82rem; margin-bottom: .55rem; }
    .metric-value { font-size: 1.2rem; margin-top: .4rem; }
    .prob-panel, .content-card { background: #121e33; border: 1px solid #293953; border-radius: 14px; }
    .result-banner { border-radius: 14px; }
    @media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } .section-heading { align-items: start; flex-direction: column; } .section-hint { display: none; } }
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
        margin: .2rem 0 1.25rem;
    }
    .login-status {
        background: #172554;
        border: 1px solid #1d4ed8;
        color: #bfdbfe;
        border-radius: 8px;
        padding: .85rem 1rem;
        margin-bottom: 1.1rem;
        text-align: center;
    }
    .sidebar-status {
        background: #21446f;
        color: #60a5fa;
        border-radius: 8px;
        padding: 1.25rem 1rem;
        text-align: center;
        line-height: 1.8;
        margin-top: 1.2rem;
    }
    .sidebar-status strong { color: #10b981; }
    .side-note {
        color: var(--muted);
        font-size: .9rem;
        text-align: center;
        margin: .7rem 0 1rem;
    }
    .main-panel {
        padding: .6rem 1.1rem 1rem;
    }

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
        margin-bottom: .9rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin: .65rem 0 .8rem;
        color: #f1f5f9;
    }

    div[data-testid="stRadio"] label { color: var(--muted) !important; }
    div[role="radiogroup"] label {
        background: transparent;
        border-radius: 8px;
        padding: .35rem .45rem;
        margin: .12rem 0;
    }
    div[role="radiogroup"] label:has(input:checked) {
        background: var(--primary);
        color: white !important;
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
        min-height: 3.4rem;
        border: none;
        border-radius: 8px;
        background: var(--primary);
        color: white;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: .01em;
    }
    div.stButton > button:hover { background: #7476ff; color: white; border: none; }
    .predict-button div.stButton > button { min-height: 4.7rem; font-size: 1.35rem; }

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
        min-height: 8.2rem;
    }
    .metric-label { color: var(--muted); font-size: 1.08rem; margin-bottom: 1rem; }
    .metric-value { text-align: center; font-size: 1.45rem; font-weight: 850; margin-top: .8rem; }
    .prob-panel, .content-card {
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


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def current_threshold() -> float:
    return float(st.session_state.get("active_threshold", THRESHOLD))


def hash_password(password: str, salt: str | None = None) -> str:
    """Hash passwords with PBKDF2-HMAC-SHA256 and a per-user random salt."""
    import secrets
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 310_000)
    return "pbkdf2_sha256$" + salt + "$" + digest.hex()


def verify_password(password: str, stored: str) -> tuple[bool, str | None]:
    """Verify PBKDF2 hashes and upgrade legacy SHA-256 hashes at next login."""
    import hmac
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, salt, expected = stored.split("$", 2)
            candidate = hash_password(password, salt)
            return hmac.compare_digest(candidate, stored), None
        except (ValueError, TypeError):
            return False, None
    legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if hmac.compare_digest(legacy, stored):
        return True, hash_password(password)
    return False, None


def load_users() -> dict:
    if not USER_STORE_PATH.exists():
        return {}
    return json.loads(USER_STORE_PATH.read_text())


def save_users(users: dict) -> None:
    USER_STORE_PATH.write_text(json.dumps(users, indent=2, sort_keys=True))


def model_probability(customer_data: dict) -> tuple[int, float, pd.DataFrame]:
    input_df = pd.DataFrame([customer_data])
    if API_URL:
        headers = {"Authorization": "Bearer " + API_TOKEN} if API_TOKEN else {}
        response = requests.post(API_URL + "/v1/predictions", json=customer_data, headers=headers, timeout=20)
        response.raise_for_status()
        result = response.json()
        st.session_state["active_threshold"] = float(result["threshold"])
        return int(result["prediction"]), float(result["probability"]), input_df
    if model is None:
        raise RuntimeError(ARTIFACT_ERROR or "Model artifacts are unavailable. Configure CHURN_API_URL or train a local model.")
    processed_df = preprocess_input(input_df, columns)
    probability = float(model.predict_proba(processed_df)[0][1])
    st.session_state["active_threshold"] = THRESHOLD
    prediction = int(probability >= THRESHOLD)
    return prediction, probability, processed_df


def aggregate_feature_importance() -> list[tuple[str, float]]:
    if model is None or not hasattr(model, "feature_importances_"):
        return []
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
            <div class="threshold-note">Threshold: {current_threshold():.2f} &nbsp;|&nbsp; {note}</div>
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


def render_metric_card(label: str, value: str, color: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color};">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_feature_importance():
    st.markdown('<div class="section-title">📈 Feature Importance (Random Forest — Gini Impurity Reduction)</div>', unsafe_allow_html=True)
    if model is None:
        st.warning(ARTIFACT_ERROR or "Model artifacts are unavailable. Train the model first.")
        return
    palette = ["#6366f1", "#8b5cf6", "#a78bfa", "#0ea5e9", "#38bdf8", "#10b981", "#34d399", "#6ee7b7"]
    top = aggregate_feature_importance()[:8]
    max_value = top[0][1] if top else 1
    rows = []
    for idx, (label, value) in enumerate(top):
        normalized = max(3, int(value / max_value * 94))
        rows.append((label, normalized, palette[idx % len(palette)]))
    render_bar_rows(rows)


def render_prediction_page():
    st.markdown('<div class="hero"><div class="eyebrow">CUSTOMER INTELLIGENCE / CHURN ANALYTICS</div><h1>Customer risk workspace</h1><p>Build a customer profile and estimate their likelihood to churn.</p></div>', unsafe_allow_html=True)
    if not st.session_state.authenticated:
        st.warning("Sign in from the sidebar to run and save predictions.")
        return
    if model is None and not API_URL:
        st.error(ARTIFACT_ERROR or "Model artifacts are unavailable.")
        st.info("Configure the inference API or train a local model.")
        return
    st.markdown('<div class="section-heading"><div><span class="eyebrow">STEP 01</span><h3>Customer profile</h3></div><span class="section-hint">Customer and billing information</span></div>', unsafe_allow_html=True)
    with st.container(border=True):
        primary, billing = st.columns(2, gap="large")
        with primary:
            st.markdown("**Account tenure & charges**")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=24, step=1)
            monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=65.50, step=0.5, format="%.2f")
            suggested_total = round(float(tenure) * float(monthly_charges), 2)
            total_charges = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=suggested_total, step=10.0, format="%.2f")
        with billing:
            st.markdown("**Contract & billing**")
            contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            payment_method = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    st.markdown('<div class="section-heading"><div><span class="eyebrow">STEP 02</span><h3>Services & customer attributes</h3></div><span class="section-hint">Select the services currently used</span></div>', unsafe_allow_html=True)
    with st.container(border=True):
        profile_a, profile_b = st.columns(2, gap="large")
        with profile_a:
            st.markdown("**Customer attributes**")
            senior_citizen = st.checkbox("Senior citizen", value=False)
            partner = st.checkbox("Has a partner", value=True)
            dependents = st.checkbox("Has dependents", value=False)
            paperless_billing = st.checkbox("Paperless billing", value=True)
        with profile_b:
            st.markdown("**Subscribed services**")
            phone_service = st.checkbox("Phone service", value=True)
            online_security = st.checkbox("Online security", value=False, disabled=internet_service == "No")
            tech_support = st.checkbox("Tech support", value=True, disabled=internet_service == "No")
            streaming_tv = st.checkbox("Streaming TV", value=True, disabled=internet_service == "No")
    with st.expander("Advanced customer options", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"] if phone_service else ["No phone service"])
        with adv2:
            online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"] if internet_service != "No" else ["No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
        with adv3:
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
    customer_data = {
        "gender": gender, "SeniorCitizen": int(senior_citizen), "Partner": yes_no(partner),
        "Dependents": yes_no(dependents), "tenure": int(tenure), "PhoneService": yes_no(phone_service),
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": yes_no(online_security) if internet_service != "No" else "No internet service",
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": yes_no(tech_support) if internet_service != "No" else "No internet service",
        "StreamingTV": yes_no(streaming_tv) if internet_service != "No" else "No internet service",
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": yes_no(paperless_billing), "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges), "TotalCharges": float(total_charges),
    }
    action_col, helper_col = st.columns([1, 2], vertical_alignment="center")
    with action_col:
        submitted = st.button("⚡ Analyze churn risk", type="primary", use_container_width=True)
    with helper_col:
        st.caption("Avoid entering personal identifiers. Only customer attributes are needed.")
    if submitted:
        with st.spinner("Analyzing customer profile..."):
            try:
                prediction, probability, _ = model_probability(customer_data)
                st.session_state["last_prediction"] = {"prediction": prediction, "probability": probability, "customer_data": customer_data}
            except requests.RequestException as exc:
                st.error("The prediction API could not be reached. Check its URL, service token and health.")
                st.caption(str(exc))
                return
            except (ValueError, RuntimeError, KeyError) as exc:
                st.error("Prediction could not be completed.")
                st.caption(str(exc))
                return
    result = st.session_state.get("last_prediction")
    if not result:
        st.info("Your prediction summary will appear here after analysis.")
        return
    prediction, probability, customer_data = int(result["prediction"]), float(result["probability"]), result["customer_data"]
    churn = prediction == 1
    st.markdown('<div class="section-heading"><div><span class="eyebrow">STEP 03</span><h3>Prediction insight</h3></div><span class="section-hint">Model output for this customer profile</span></div>', unsafe_allow_html=True)
    if churn:
        st.error("HIGHER CHURN RISK · This profile is above the model decision threshold.")
    else:
        st.success("LOWER CHURN RISK · This profile is below the model decision threshold.")
    render_probability_panel(probability, churn)
    cards = st.columns(4)
    values = [("Prediction", "Likely to churn" if churn else "Not likely", "#ef4444" if churn else "#22c55e"), ("Probability", pct(probability), "#f97316" if churn else "#22c55e"), ("Decision threshold", f"{current_threshold():.2f}", "#818cf8"), ("Model", MODEL_NAME, "#38bdf8")]
    for card, (label, value, color) in zip(cards, values):
        with card:
            render_metric_card(label, value, color)
    if churn:
        st.markdown("#### Suggested retention actions")
        factors = active_risk_factors(customer_data) or [("Model probability above churn threshold", int(probability * 100), "#ef4444")]
        render_bar_rows(factors)
        st.info("Review contract flexibility, service experience and support needs before offering a targeted retention incentive.")
    else:
        st.markdown("#### Customer profile summary")
        st.caption("This profile is below the selected churn threshold. Continue monitoring customer experience and engagement.")

def render_model_info_page():
    st.markdown('<div class="page-title">Model Info</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Current trained model, cutoff, and evaluation summary</div>', unsafe_allow_html=True)
    cards = st.columns(4)
    values = [
        ("Model", MODEL_NAME, "#10b981"),
        ("Threshold", f"{current_threshold():.2f}", "#818cf8"),
        ("Accuracy", MODEL_ACCURACY, "#38bdf8"),
        ("ROC AUC", MODEL_ROC_AUC, "#f59e0b"),
    ]
    for card, (label, value, color) in zip(cards, values):
        with card:
            render_metric_card(label, value, color)
    st.markdown(
        """
        <div class="content-card">
            <h3>How this model is used</h3>
            <p>The app preprocesses the submitted customer profile, aligns it to the training columns, and uses the Random Forest model to estimate churn probability.</p>
            <p>If the churn probability is greater than or equal to 0.30, the customer is flagged for retention action.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_history_page():
    st.markdown('<div class="page-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Recent predictions saved by the FastAPI service</div>', unsafe_allow_html=True)
    if not API_URL:
        st.info("Prediction history requires CHURN_API_URL and the FastAPI backend.")
        return
    headers = {"Authorization": "Bearer " + API_TOKEN} if API_TOKEN else {}
    try:
        response = requests.get(API_URL + "/v1/predictions?limit=100", headers=headers, timeout=20)
        response.raise_for_status()
        records = response.json()
    except requests.RequestException as exc:
        st.error("Could not load prediction history. Check API URL, token, and backend health.")
        st.caption(str(exc))
        return
    if not records:
        st.info("No predictions have been recorded yet.")
        return
    history = pd.DataFrame(records)
    if "created_at" in history:
        history["created_at"] = pd.to_datetime(history["created_at"], errors="coerce")
    st.metric("Recorded predictions", len(history))
    st.dataframe(
        history[["id", "created_at", "label", "probability", "threshold", "model_name"]],
        use_container_width=True,
        hide_index=True,
    )
    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button("Download prediction history (CSV)", csv, "churn_prediction_history.csv", "text/csv")


def render_documentation_page():
    st.markdown('<div class="page-title">Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">How to use, understand, and retrain the churn prediction system</div>', unsafe_allow_html=True)

    overview_col, workflow_col = st.columns(2, gap="large")
    with overview_col:
        st.markdown(
            """
            <div class="content-card">
                <h3>📌 Project Overview</h3>
                <p><strong>CCPS</strong> predicts whether a telecom customer is likely to churn using customer profile, service, contract, billing, and charge information.</p>
                <ul>
                    <li><strong>Model:</strong> Random Forest classifier</li>
                    <li><strong>Churn cutoff:</strong> 0.30 probability</li>
                    <li><strong>Output:</strong> churn probability, prediction label, risk factors, and retention recommendation</li>
                    <li><strong>Authentication:</strong> local demo login/sign-up for app access</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with workflow_col:
        st.markdown(
            """
            <div class="content-card">
                <h3>🧭 App Workflow</h3>
                <ol>
                    <li>Open <strong>Login / Sign Up</strong> and create a demo account.</li>
                    <li>Go to <strong>Prediction</strong>.</li>
                    <li>Enter tenure, charges, contract, service, and billing details.</li>
                    <li>Click <strong>Predict Churn Probability</strong>.</li>
                    <li>Review probability, churn flag, top risk factors, and recommendation.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="content-card">
            <h3>🧾 Input Field Guide</h3>
            <ul>
                <li><strong>Tenure:</strong> Number of months the customer has stayed with the company.</li>
                <li><strong>Monthly Charges:</strong> Customer's current monthly bill amount.</li>
                <li><strong>Total Charges:</strong> Total amount billed over the full customer lifetime.</li>
                <li><strong>Contract Type:</strong> Month-to-month customers usually have higher churn risk than one-year or two-year customers.</li>
                <li><strong>Internet Service:</strong> DSL, fiber optic, or no internet service.</li>
                <li><strong>Payment Method:</strong> Electronic check often appears as a high-risk churn signal in this dataset.</li>
                <li><strong>Service toggles:</strong> Online Security, Tech Support, Streaming TV, Paperless Billing, and other features help the model understand the customer profile.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col, threshold_col, retrain_col = st.columns(3, gap="large")
    with metric_col:
        render_metric_card("Model", MODEL_NAME, "#10b981")
    with threshold_col:
        render_metric_card("Threshold", f"{current_threshold():.2f}", "#818cf8")
    with retrain_col:
        render_metric_card("ROC AUC", MODEL_ROC_AUC, "#f59e0b")

    st.markdown(
        """
        <div class="content-card">
            <h3>🔁 Retraining Instructions</h3>
            <p>To retrain the model, place the IBM Telco Customer Churn dataset at:</p>
            <p><code>data/WA_Fn-UseC_-Telco-Customer-Churn.csv</code></p>
            <p>Then run:</p>
            <p><code>python src/train_model.py</code></p>
            <p>The script will preprocess the data, train the Random Forest model, print evaluation metrics, and save refreshed artifacts to <code>models/churn_model.pkl</code> and <code>models/columns.pkl</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Tip: If Prediction shows a login warning, open Login / Sign Up first, create an account, then return to Prediction.")


def render_auth_page():
    st.markdown('<div class="page-title">Login / Sign Up</div>', unsafe_allow_html=True)
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            users = load_users()
            valid, upgraded_hash = verify_password(password, users.get(username, ""))
            if username in users and valid:
                if upgraded_hash:
                    users[username] = upgraded_hash
                    save_users(users)
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid username or password.")

    with signup_tab:
        new_username = st.text_input("Choose Username", key="signup_username")
        new_password = st.text_input("Choose Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        if st.button("Create Account", key="signup_button"):
            users = load_users()
            if not new_username.strip():
                st.error("Username is required.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif new_username in users:
                st.error("This username already exists.")
            else:
                users[new_username] = hash_password(new_password)
                save_users(users)
                st.session_state.authenticated = True
                st.session_state.username = new_username
                st.success(f"Account created. You are logged in as {new_username}.")

    if st.session_state.authenticated:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.success("You have been logged out.")


NAV_ITEMS = ["📊 Prediction", "🗂️ Prediction History", "📈 Feature Importance", "🧠 Model Info", "📄 Documentation", "🔐 Login / Sign Up"]


def render_navigation_panel() -> str:
    st.markdown('<div class="brand-card">⚡ CCPS</div>', unsafe_allow_html=True)
    status = f"Logged in as <strong>{st.session_state.username}</strong>" if st.session_state.authenticated else "Not logged in"
    st.markdown(f'<div class="login-status">{status}</div>', unsafe_allow_html=True)
    default_index = 0 if st.session_state.authenticated else 5
    selected_page = st.radio(
        "Navigation",
        NAV_ITEMS,
        index=default_index,
    )
    st.markdown(
        f'<div class="sidebar-status">Model: {MODEL_NAME}<br><strong>Threshold: {current_threshold():.2f}</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-note">Use the menu above to move between prediction, model details, docs, and account pages.</div>', unsafe_allow_html=True)
    return selected_page


def render_selected_page(page: str) -> None:
    st.markdown('<div class="main-panel">', unsafe_allow_html=True)
    if page == "📊 Prediction":
        render_prediction_page()
    elif page == "🗂️ Prediction History":
        render_prediction_history_page()
    elif page == "📈 Feature Importance":
        st.markdown('<div class="page-title">Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Global model drivers from the trained Random Forest</div>', unsafe_allow_html=True)
        render_feature_importance()
    elif page == "🧠 Model Info":
        render_model_info_page()
    elif page == "📄 Documentation":
        render_documentation_page()
    else:
        render_auth_page()
    st.markdown('</div>', unsafe_allow_html=True)


with st.sidebar:
    page = render_navigation_panel()

render_selected_page(page)
