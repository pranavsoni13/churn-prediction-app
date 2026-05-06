import pandas as pd

BINARY_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
YES_NO_MAP = {"Yes": 1, "No": 0}
GENDER_MAP = {"Male": 1, "Female": 0}


def load_and_preprocess(path):
    """Load the Telco churn dataset and create model-ready features."""
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    for col in BINARY_COLUMNS:
        df[col] = df[col].map(YES_NO_MAP)

    df["gender"] = df["gender"].map(GENDER_MAP)
    df = pd.get_dummies(df, drop_first=True)

    return df


def preprocess_input(df, columns):
    """Apply the same transformations used during training and align columns."""
    df = df.copy()

    for col in BINARY_COLUMNS:
        df[col] = df[col].map(YES_NO_MAP)

    df["gender"] = df["gender"].map(GENDER_MAP)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=columns, fill_value=0)

    return df