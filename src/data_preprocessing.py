import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df.drop("customerID", axis=1, inplace=True)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1, 'No':0})

    df["gender"] = df["gender"].map({"Male":1, "Female":0})

    df = pd.get_dummies(df, drop_first=True)

    return df

def preprocess_input(df, columns):
    
    binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1, 'No':0})

    df["gender"] = df["gender"].map({"Male":1, "Female":0})

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training data
    df = df.reindex(columns=columns, fill_value=0)

    return df