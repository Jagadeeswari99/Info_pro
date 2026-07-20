import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.feature_engineering import engineer_features, FEATURE_COLS
from src.inventory_optimizer import get_reorder_recommendation

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
    return {"message": "FreshFlow — demand forecasting + inventory recommendation MVP", "docs": "/docs"}


@app.post("/forecast/{item}", response_model=ForecastResponse)
def forecast_item(item: str, request: ForecastRequest):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feat_df = engineer_features(df, item)
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
        x = pd.DataFrame([row[FEATURE_COLS]])
        preds.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "p10": round(float(p10_model.predict(x)[0]), 2),
                "p50": round(float(p50_model.predict(x)[0]), 2),
                "p90": round(float(p90_model.predict(x)[0]), 2),
            }
        )
    return ForecastResponse(item=item, forecasts=preds)


@app.get("/inventory/reorder/{item}")
def inventory_reorder(item: str, current_stock: float):
    return get_reorder_recommendation(item, current_stock)


@app.get("/metrics")
def metrics():
    quantile_metrics = pd.read_csv(BASE_DIR / "outputs" / "quantile_metrics.csv")
    mae_by_item = quantile_metrics.groupby("menu_item")["mae"].mean().to_dict()
    calibration_by_item = quantile_metrics.groupby("menu_item")["calibration_pct"].mean().to_dict()
    return {
        "mae": mae_by_item,
        "calibration_pct": calibration_by_item,
    }
