from fastapi import APIRouter

import json

from app.services.forecasting_service import (
    generate_forecast
)


router = APIRouter()


@router.get("/health")
def health_check():

    return {
        "status": "running"
    }


@router.get("/best-model")
def best_model():

    with open(
        "saved_models/model_registry.json",
        "r"
    ) as file:

        registry = json.load(file)

    return registry


@router.get("/forecast/{state}")
def forecast_state(state: str):

    forecast = generate_forecast(
        state_name=state,
        forecast_days=56
    )

    return {
        "state": state,
        "forecast_days": 56,
        "model_used": "LSTM",
        "forecast": forecast
    }


@router.get("/")
def home():

    return {
        "message": (
            "Sales Forecasting API Running"
        )
    }