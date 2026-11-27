"""Validated request and response schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CustomerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str = Field(pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str = Field(pattern="^(Yes|No)$")
    Dependents: str = Field(pattern="^(Yes|No)$")
    tenure: int = Field(ge=0, le=72)
    PhoneService: str = Field(pattern="^(Yes|No)$")
    MultipleLines: str = Field(pattern="^(Yes|No|No phone service)$")
    InternetService: str = Field(pattern="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(pattern="^(Yes|No|No internet service)$")
    Contract: str = Field(pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(pattern="^(Yes|No)$")
    PaymentMethod: str = Field(
        pattern="^(Electronic check|Mailed check|Bank transfer \(automatic\)|Credit card \(automatic\))$"
    )
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: float = Field(ge=0, le=100000)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    threshold: float
    model_name: str


class PredictionHistoryItem(PredictionResponse):
    id: int
    created_at: str
