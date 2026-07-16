from pathlib import Path
from datetime import datetime, timedelta
import os
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "weather_real.csv"


def fetch_weather_daily(city="Bengaluru", latitude=12.9716, longitude=77.5946, days=730, overwrite=False):
    if OUTPUT_PATH.exists() and not overwrite:
        return pd.read_csv(OUTPUT_PATH, parse_dates=["date"])

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    url = (
        "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        "&start_date={start}&end_date={end}&daily=temperature_2m_mean,precipitation_sum,weathercode"
        "&timezone=auto"
    ).format(lat=latitude, lon=longitude, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "temp_mean_c": daily.get("temperature_2m_mean", []),
        "precipitation_sum_mm": daily.get("precipitation_sum", []),
        "weather_code": daily.get("weathercode", []),
        "city": city,
    })
    df = df.dropna().reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = fetch_weather_daily()
    print(df.head())
    print(f"Saved {len(df)} weather rows to {OUTPUT_PATH}")
