import pickle
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
for path in (str(BASE_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.feature_engineering import engineer_features, FEATURE_COLS
except ModuleNotFoundError:  # pragma: no cover
    from feature_engineering import engineer_features, FEATURE_COLS

DATA_PATH = BASE_DIR / "data" / "pos_sales_raw.csv"
MODEL_DIR = BASE_DIR / "models"


def get_reorder_recommendation(item, current_stock, lead_time_days=2, safety_buffer=0.1):
    item_key = item.replace(" ", "_")
    with open(MODEL_DIR / f"{item_key}_q90.pkl", "rb") as fh:
        model = pickle.load(fh)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feat_df = engineer_features(df, item)
    last_row = feat_df.iloc[[-1]]
    feature_frame = last_row[FEATURE_COLS].copy()
    p90_forecast = float(model.predict(feature_frame).clip(min=0)[0])

    reorder_point = p90_forecast * lead_time_days * (1 + safety_buffer)
    recommended_qty = max(0, int(round(reorder_point - float(current_stock))))
    reasoning = (
        f"Using a P90 forecast of {p90_forecast:.1f} units over {lead_time_days} days, "
        f"the recommended order is {recommended_qty} units."
    )
    return {"recommended_order_qty": recommended_qty, "reasoning": reasoning}


if __name__ == "__main__":
    print(get_reorder_recommendation("Chicken Biryani", 20))
