"""FastAPI service for customer churn inference and prediction history."""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.database import Base, PredictionRecord, engine, get_db
from backend.schemas import CustomerProfile, PredictionHistoryItem, PredictionResponse
from src.predict import load_artifacts, predict

APP_ENV = os.getenv("APP_ENV", "development").lower()
API_TOKEN = os.getenv("CHURN_API_TOKEN", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Validated inference and prediction history for the churn prediction application.",
    version="1.0.0",
    lifespan=lifespan,
)


def require_service_token(authorization: str | None = Header(default=None)) -> None:
    """Require a service token in production; allow local development without one."""
    if APP_ENV != "production":
        return
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="CHURN_API_TOKEN is not configured.")
    expected = f"Bearer {API_TOKEN}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing service token.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "churn-prediction-api",
        "model_artifacts_available": (load_artifacts_available()),
        "database_configured": bool(os.getenv("DATABASE_URL")),
    }


def load_artifacts_available() -> bool:
    from src.predict import MODEL_PATH, COLUMNS_PATH
    return MODEL_PATH.exists() and COLUMNS_PATH.exists()


@app.post(
    "/v1/predictions",
    response_model=PredictionResponse,
    dependencies=[Depends(require_service_token)],
)
def create_prediction(payload: CustomerProfile, db: Session = Depends(get_db)):
    try:
        label, probability = predict(payload.model_dump())
        _, _, threshold = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record = PredictionRecord(
        prediction=label,
        probability=probability,
        threshold=threshold,
        customer_profile=payload.model_dump(),
    )
    db.add(record)
    db.commit()
    return PredictionResponse(
        prediction=label,
        label="Churn" if label else "No Churn",
        probability=probability,
        threshold=threshold,
        model_name="Random Forest",
    )


@app.get(
    "/v1/predictions",
    response_model=list[PredictionHistoryItem],
    dependencies=[Depends(require_service_token)],
)
def list_predictions(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = db.query(PredictionRecord).order_by(desc(PredictionRecord.created_at)).limit(limit).all()
    return [
        PredictionHistoryItem(
            id=row.id,
            prediction=row.prediction,
            label="Churn" if row.prediction else "No Churn",
            probability=row.probability,
            threshold=row.threshold,
            model_name="Random Forest",
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]
