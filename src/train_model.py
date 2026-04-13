from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_and_preprocess

# Load data
df = load_and_preprocess("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

joblib.dump(model, "../models/churn_model.pkl")
joblib.dump(X.columns, "../models/columns.pkl")

print("Model and columns saved!")

# Save model
joblib.dump(model, "../models/churn_model.pkl")

print("Model trained and saved successfully!")