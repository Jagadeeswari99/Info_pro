from pathlib import Path
import pandas as pd
import numpy as np

try:
    from src.feature_engineering import add_chronological_features, add_lag_features, add_rolling_features
except ModuleNotFoundError:  # pragma: no cover
    from feature_engineering import add_chronological_features, add_lag_features, add_rolling_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "pos_sales_raw.csv"
WEATHER_PATH = BASE_DIR / "data" / "weather_real.csv"
EVENTS_PATH = BASE_DIR / "data" / "events_calendar.csv"


def load_weather_and_events():
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    events = pd.read_csv(EVENTS_PATH, parse_dates=["date"])
    return weather, events


def engineer_features_v2(df: pd.DataFrame, item: str, weather: pd.DataFrame | None = None, events: pd.DataFrame | None = None) -> pd.DataFrame:
    if weather is None or events is None:
        weather, events = load_weather_and_events()

    ts = df[df["menu_item"] == item].copy()
    ts = ts.sort_values("date").set_index("date")
    ts = ts.asfreq("D")
    ts["units_sold"] = ts["units_sold"].fillna(ts["units_sold"].rolling(7, min_periods=1).mean())

    ts = add_chronological_features(ts)
    ts = add_lag_features(ts)
    ts = add_rolling_features(ts)
    ts["ewm_7"] = ts["units_sold"].shift(1).ewm(span=7, adjust=False).mean()
    ts["ewm_14"] = ts["units_sold"].shift(1).ewm(span=14, adjust=False).mean()
    ts["is_holiday"] = ts["is_holiday"].fillna(0).astype(int)
    ts["trend"] = (ts.index - ts.index[0]).days

    weather = weather.rename(columns={"temp_mean_c": "temp_mean_c", "precipitation_sum_mm": "precipitation_sum_mm", "weather_code": "weather_code"})
    weather = weather[["date", "temp_mean_c", "precipitation_sum_mm", "weather_code"]].drop_duplicates(subset=["date"])
    ts = ts.join(weather.set_index("date"), how="left")
    ts["temp_mean_c"] = ts["temp_mean_c"].fillna(ts["temp_mean_c"].mean())
    ts["precipitation_sum_mm"] = ts["precipitation_sum_mm"].fillna(0)
    ts["weather_code"] = ts["weather_code"].fillna(ts["weather_code"].mode().iloc[0])

    event_df = events.copy()
    event_df = event_df[["date", "expected_demand_multiplier"]].copy()
    event_df["is_event"] = True
    ts = ts.join(event_df.set_index("date"), how="left")
    ts["is_event"] = ts["is_event"].fillna(False).astype(bool)
    ts["event_multiplier"] = ts["expected_demand_multiplier"].fillna(1.0)
    ts["expected_demand_multiplier"] = ts["event_multiplier"]

    ts = ts.dropna().reset_index()
    ts = ts.rename(columns={"index": "date"})

    corr = ts.select_dtypes(include=[np.number]).corr(numeric_only=True)
    print("\nFeature correlation table:")
    print(corr[["units_sold"]].sort_values("units_sold", ascending=False).head(15).to_string())
    return ts


FEATURE_COLS_V2 = [
    "day_of_week", "day_of_month", "month", "quarter", "week_of_year", "year",
    "is_weekend", "is_month_start", "is_month_end", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "lag_1", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_max_7", "rolling_mean_14", "rolling_std_14",
    "rolling_max_14", "rolling_mean_30", "rolling_std_30", "rolling_max_30",
    "ewm_7", "ewm_14", "is_holiday", "trend", "temp_mean_c", "precipitation_sum_mm",
    "weather_code", "is_event", "event_multiplier",
]


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    weather, events = load_weather_and_events()
    feat_df = engineer_features_v2(df, "Chicken Biryani", weather=weather, events=events)
    print(feat_df[["date", "units_sold"] + FEATURE_COLS_V2[:8]].head())
