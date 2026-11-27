# Customer Churn Prediction

An end-to-end machine-learning application for estimating customer churn risk using the IBM Telco Customer Churn dataset. The project includes a Streamlit interface, reusable preprocessing, model training, threshold selection, evaluation artifacts, and automated checks.

## Current stack

- Python 3.11
- Streamlit
- Pandas / NumPy
- Scikit-learn
- Joblib
- GitHub Actions

## Project layout

```text
app/                    Streamlit application
src/
  data_preprocessing.py Shared feature cleaning and encoding
  train_model.py        Training, validation threshold selection, evaluation
  predict.py            Reusable prediction interface
models/                 Generated model and metadata artifacts
tests/                  Unit tests
data/                   Local Telco CSV (not committed)
.github/workflows/      CI checks
.streamlit/config.toml  Streamlit theme and server defaults
```

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Place the IBM Telco Customer Churn CSV at:

   ```text
   data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

4. Train the model and generate artifacts:

   ```bash
   python src/train_model.py
   ```

5. Launch Streamlit:

   ```bash
   streamlit run app/app.py
   ```

## Training and evaluation

The training script uses stratified train/validation/test splits. The decision threshold is selected on the validation split by maximizing F2 score, then the model is refit on train + validation and evaluated once on the untouched test split. Metrics and the selected threshold are written to `models/metrics.json` and `models/threshold.json`.

The model artifacts are generated locally and are intentionally not committed. Retrain after changing dependencies or preprocessing logic.

## Tests

Run:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs Python compilation and the unit test suite for pushes and pull requests.

## Important limitations

- This is a portfolio/demo project, not a validated production retention decision system. Evaluate performance on representative, current business data before operational use.
- The public Telco dataset is historical and may not represent current customer populations.
- Model feature importance is global association, not causal explanation. Risk-factor heuristics shown in the interface are illustrative and must not be interpreted as SHAP explanations.
- The demo account store is local-file based and is not suitable for multi-user production deployment. Use a managed identity provider and secure database before exposing accounts to real users.
- Do not upload customer-identifiable or confidential data to a public deployment.
- Scikit-learn/joblib artifacts should only be loaded from trusted sources.

## Author

Pranav Soni — AI & Data Science
