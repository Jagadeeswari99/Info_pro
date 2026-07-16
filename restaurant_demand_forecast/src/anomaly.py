from pathlib import Path
import pickle
import pandas as pd
import numpy as np

try:
    from src.feature_engineering_v2 import engineer_features_v2, FEATURE_COLS_V2
    from src.feature_engineering import sequential_split
except ModuleNotFoundError:  # pragma: no cover
    from feature_engineering_v2 import engineer_features_v2, FEATURE_COLS_V2
    from feature_engineering import sequential_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "pos_sales_raw.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_anomalies():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    weather = pd.read_csv(BASE_DIR / "data" / "weather_real.csv", parse_dates=["date"])
    events = pd.read_csv(BASE_DIR / "data" / "events_calendar.csv", parse_dates=["date"])

    rows = []
    for item in df["menu_item"].unique():
        feat_df = engineer_features_v2(df, item, weather=weather, events=events)
        train, test = sequential_split(feat_df, test_months=2)
        X_test = test[FEATURE_COLS_V2]
        y_test = test["units_sold"].values
        item_key = item.replace(" ", "_")
        with open(MODEL_DIR / f"{item_key}_q50.pkl", "rb") as fh:
            p50_model = pickle.load(fh)
        with open(MODEL_DIR / f"{item_key}_q10.pkl", "rb") as fh:
            p10_model = pickle.load(fh)
        with open(MODEL_DIR / f"{item_key}_q90.pkl", "rb") as fh:
            p90_model = pickle.load(fh)

        p10 = p10_model.predict(X_test).clip(min=0)
        p50 = p50_model.predict(X_test).clip(min=0)
        p90 = p90_model.predict(X_test).clip(min=0)
        flagged = (y_test < p10) | (y_test > p90)
        for idx, flag in enumerate(flagged):
            if flag:
                rows.append({
                    "date": test.iloc[idx]["date"],
                    "item": item,
                    "actual": int(y_test[idx]),
                    "p10": int(round(p10[idx])),
                    "p90": int(round(p90[idx])),
                    "flagged": True,
                })

    anomalies_df = pd.DataFrame(rows)
    anomalies_df.to_csv(OUTPUT_DIR / "anomalies.csv", index=False)
    print(anomalies_df.head())
    print(f"Saved {len(anomalies_df)} anomalies to {OUTPUT_DIR / 'anomalies.csv'}")


if __name__ == "__main__":
    detect_anomalies()
