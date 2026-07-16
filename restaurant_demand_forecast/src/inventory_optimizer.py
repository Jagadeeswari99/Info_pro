import sys
from pathlib import Path
import pickle
import yaml
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
for path in (str(BASE_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.feature_engineering_v2 import engineer_features_v2, FEATURE_COLS_V2
except ModuleNotFoundError:  # pragma: no cover
    from feature_engineering_v2 import engineer_features_v2, FEATURE_COLS_V2
CONFIG_PATH = BASE_DIR / "config.yaml"
DATA_PATH = BASE_DIR / "data" / "pos_sales_raw.csv"
MODEL_DIR = BASE_DIR / "models"

if not CONFIG_PATH.exists():
    CONFIG_PATH.write_text("""items:\n  Chicken Biryani:\n    lead_time_days: 3\n    safety_buffer: 15\n  Paneer Butter Masala:\n    lead_time_days: 2\n    safety_buffer: 10\n  Veg Fried Rice:\n    lead_time_days: 2\n    safety_buffer: 12\n  Masala Dosa:\n    lead_time_days: 2\n    safety_buffer: 10\n  Grilled Fish:\n    lead_time_days: 3\n    safety_buffer: 12\n""")

with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh) or {}


def get_reorder_recommendation(item, current_stock):
    item_key = item.replace(" ", "_")
    with open(MODEL_DIR / f"{item_key}_q90.pkl", "rb") as fh:
        import pickle
        model = pickle.load(fh)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    weather = pd.read_csv(BASE_DIR / "data" / "weather_real.csv", parse_dates=["date"])
    events = pd.read_csv(BASE_DIR / "data" / "events_calendar.csv", parse_dates=["date"])
    feat_df = engineer_features_v2(df, item, weather=weather, events=events)
    last_row = feat_df.iloc[[-1]]
    feature_frame = last_row[FEATURE_COLS_V2].copy()
    p90_forecast = float(model.predict(feature_frame).clip(min=0)[0])
    item_cfg = CONFIG.get("items", {}).get(item, {})
    lead_time_days = item_cfg.get("lead_time_days", 2)
    safety_buffer = item_cfg.get("safety_buffer", 10)
    reorder_point = p90_forecast * lead_time_days + safety_buffer
    recommended_qty = max(0, int(round(reorder_point - float(current_stock))))
    reasoning = (
        f"Using a P90 forecast of {p90_forecast:.1f} units, a {lead_time_days}-day lead time, and a {safety_buffer}-unit safety buffer, "
        f"the recommended order is {recommended_qty} units."
    )
    return {"recommended_order_qty": recommended_qty, "reasoning": reasoning}


if __name__ == "__main__":
    print(get_reorder_recommendation("Chicken Biryani", 20))
