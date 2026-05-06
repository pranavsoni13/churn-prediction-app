from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_preprocess

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
COLUMNS_PATH = ROOT / "models" / "columns.pkl"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download the IBM Telco Customer Churn CSV "
            "and place it in the data/ directory before training."
        )

    df = load_and_preprocess(DATA_PATH)
    x = df.drop("Churn", axis=1)
    y = df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(x.columns, COLUMNS_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Columns saved to {COLUMNS_PATH}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


if __name__ == "__main__":
    main()