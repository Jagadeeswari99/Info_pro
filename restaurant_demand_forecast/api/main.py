from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.feature_engineering_v2 import engineer_features_v2, FEATURE_COLS_V2
from src.inventory_optimizer import get_reorder_recommendation
from src.explain import explain_forecast

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "pos_sales_raw.csv"

app = FastAPI(title="FreshFlow API", version="0.1.0")


class ForecastRequest(BaseModel):
    days: int


class ForecastItem(BaseModel):
    date: str
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    item: str
    forecasts: list[ForecastItem]


@app.get("/")
def root():
    return {"message": "FreshFlow — AI Demand & Inventory Intelligence", "docs": "/docs"}


@app.post("/forecast/{item}", response_model=ForecastResponse)
def forecast_item(item: str, request: ForecastRequest):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    weather = pd.read_csv(BASE_DIR / "data" / "weather_real.csv", parse_dates=["date"])
    events = pd.read_csv(BASE_DIR / "data" / "events_calendar.csv", parse_dates=["date"])
    feat_df = engineer_features_v2(df, item, weather=weather, events=events)
    item_key = item.replace(" ", "_")
    with open(MODEL_DIR / f"{item_key}_q10.pkl", "rb") as fh:
        p10_model = pickle.load(fh)
    with open(MODEL_DIR / f"{item_key}_q50.pkl", "rb") as fh:
        p50_model = pickle.load(fh)
    with open(MODEL_DIR / f"{item_key}_q90.pkl", "rb") as fh:
        p90_model = pickle.load(fh)

    latest = feat_df.sort_values("date").tail(max(1, request.days))
    preds = []
    for _, row in latest.iterrows():
        x = pd.DataFrame([row[FEATURE_COLS_V2]])
        preds.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "p10": round(float(p10_model.predict(x)[0]), 2),
            "p50": round(float(p50_model.predict(x)[0]), 2),
            "p90": round(float(p90_model.predict(x)[0]), 2),
        })
    return ForecastResponse(item=item, forecasts=preds)


@app.get("/inventory/reorder/{item}")
def inventory_reorder(item: str, current_stock: float):
    return get_reorder_recommendation(item, current_stock)


@app.get("/explain/{item}/{date}")
def explain_endpoint(item: str, date: str):
    return {"item": item, "date": date, "explanation": explain_forecast(item, date)}


@app.get("/anomalies")
def anomalies():
    anomalies_df = pd.read_csv(BASE_DIR / "outputs" / "anomalies.csv")
    return anomalies_df.tail(10).to_dict(orient="records")


@app.get("/metrics")
def metrics():
    quantile_metrics = pd.read_csv(BASE_DIR / "outputs" / "quantile_metrics.csv")
    avg_pinball = quantile_metrics.groupby("quantile")["pinball_loss"].mean().to_dict()
    return {
        "mae_reference": "Original point-forecast baseline was ~11.7 units MAE; quantile evaluation is reported as pinball loss and calibration.",
        "pinball_loss": avg_pinball,
        "calibration": {
            "p90_under_actual": "See outputs/quantile_metrics.csv and per-item terminal output",
        },
    }
