# Customer Churn Prediction

An end-to-end churn analytics application with a Streamlit dashboard, FastAPI inference service, SQLAlchemy persistence, and a reusable scikit-learn model workflow.

## Architecture

- **Streamlit**: interactive dashboard. Deploy on Streamlit Community Cloud.
- **FastAPI**: REST inference API and prediction history. Deploy separately, for example on Render.
- **ML package**: shared preprocessing, training, threshold selection, and inference.
- **Database**: SQLAlchemy supports PostgreSQL in production and SQLite for local development.
- **CI**: GitHub Actions compiles the app/backend and runs unit tests.

## Repository layout

```text
app/                    Streamlit dashboard
backend/
  main.py               FastAPI routes and service authentication
  database.py           SQLAlchemy engine and prediction history
  schemas.py            Pydantic input/output validation
src/
  data_preprocessing.py Shared feature preprocessing
  train_model.py        Model training and holdout evaluation
  predict.py            Shared inference interface
models/                 Generated model artifacts
tests/                  Unit tests
.github/workflows/      CI
render.yaml             Render API service blueprint
```

## 1. Install and train locally

Use Python 3.11.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the IBM Telco Customer Churn CSV and place it at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Train and generate model artifacts:

```bash
python src/train_model.py
```

Artifacts are written under `models/`. CI does not fabricate model artifacts. Keep artifacts from trusted training runs only.

## 2. Run the Streamlit dashboard

```bash
streamlit run app/app.py
```

The dashboard can run local inference when model artifacts exist. For hybrid mode, configure `CHURN_API_URL` and `CHURN_API_TOKEN` so predictions are sent to FastAPI.

## 3. Run the FastAPI backend locally

SQLite is used automatically for local development:

```bash
uvicorn backend.main:app --reload
```

- Health check: `http://127.0.0.1:8000/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`
- Create prediction: `POST /v1/predictions`
- List recent predictions: `GET /v1/predictions?limit=50`

The service creates the prediction-history table on startup. In production, set `APP_ENV=production`, configure a strong `CHURN_API_TOKEN`, and provide a PostgreSQL `DATABASE_URL`. Protected API routes require `Authorization: Bearer <CHURN_API_TOKEN>`.

## 4. Deployment

### Streamlit Community Cloud

1. Deploy this repository from Streamlit Community Cloud.
2. Set the main file path to `app/app.py`.
3. Add these values under App settings → Secrets:

```toml
CHURN_API_URL = "https://YOUR-API-SERVICE.onrender.com"
CHURN_API_TOKEN = "the-same-generated-token-configured-on-the-api"
```

Do not commit actual secrets. If the API is not configured, the dashboard can use local model artifacts as a fallback.

### Render API

1. Create a PostgreSQL database on Render and copy its internal connection URL.
2. Create a Blueprint deployment using `render.yaml`.
3. Set `DATABASE_URL` to the PostgreSQL URL in the API service environment.
4. Keep the generated `CHURN_API_TOKEN` secret private and copy it to Streamlit Cloud secrets.
5. Ensure the API has trusted model artifacts available at `models/churn_model.pkl`, `models/columns.pkl`, and `models/threshold.json`. The Blueprint does not train a model automatically.

The API health endpoint is `/health`. The service is not ready for predictions until model artifacts are provisioned.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q app src backend tests
```

GitHub Actions runs compilation and unit tests on pushes to main/upgrade branches and pull requests.

## Security and operational notes

- The API token is service-to-service protection, not end-user identity or role-based access control.
- The existing Streamlit local-file account system is still demo-only. Do not use it for real customer accounts or multi-instance deployments.
- Use PostgreSQL, TLS, managed identity, secrets management, backups, and migrations before handling real customer information.
- The public IBM Telco dataset is historical. Evaluate model quality and drift on representative current data before operational retention decisions.
- Gini feature importance is a global model statistic, not a causal or individual-level explanation. The dashboard's manually defined risk-factor list is illustrative.
- Only load model artifacts from trusted sources.

## Author

Pranav Soni — AI & Data Science
