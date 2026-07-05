"""
feature_engineering.py
Transforms raw POS data into ML-ready features:
- Lag features (7-day, 14-day)
- Rolling window statistics (7-day, 14-day, 30-day moving averages)
- Chronological features
- Train/test sequential split (no data leakage)
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame, item: str) -> pd.DataFrame:
    """
    Engineer features for a single menu item time series.

    Args:
        df: Full POS dataframe
        item: Menu item name to filter on

    Returns:
        Feature-engineered DataFrame, ready for modeling
    """
    ts = df[df["menu_item"] == item].copy()
    ts = ts.sort_values("date").set_index("date")
    ts = ts.asfreq("D")  # Ensure daily frequency, fill gaps

    # Handle any missing days
    ts["units_sold"] = ts["units_sold"].fillna(ts["units_sold"].rolling(7, min_periods=1).mean())

    # ── Chronological Features ────────────────────────────────────────────────
    ts["day_of_week"]   = ts.index.dayofweek         # 0=Mon, 6=Sun
    ts["day_of_month"]  = ts.index.day
    ts["month"]         = ts.index.month
    ts["quarter"]       = ts.index.quarter
    ts["week_of_year"]  = ts.index.isocalendar().week.astype(int)
    ts["year"]          = ts.index.year
    ts["is_weekend"]    = (ts.index.dayofweek >= 5).astype(int)
    ts["is_month_start"]= ts.index.is_month_start.astype(int)
    ts["is_month_end"]  = ts.index.is_month_end.astype(int)

    # Day-of-week sine/cosine encoding (cyclic)
    ts["dow_sin"] = np.sin(2 * np.pi * ts["day_of_week"] / 7)
    ts["dow_cos"] = np.cos(2 * np.pi * ts["day_of_week"] / 7)
    ts["month_sin"] = np.sin(2 * np.pi * ts["month"] / 12)
    ts["month_cos"] = np.cos(2 * np.pi * ts["month"] / 12)

    # ── Lag Features ─────────────────────────────────────────────────────────
    for lag in [1, 3, 7, 14, 21, 28]:
        ts[f"lag_{lag}"] = ts["units_sold"].shift(lag)

    # ── Rolling Window Statistics ─────────────────────────────────────────────
    for window in [7, 14, 30]:
        ts[f"rolling_mean_{window}"] = (
            ts["units_sold"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        ts[f"rolling_std_{window}"] = (
            ts["units_sold"].shift(1).rolling(window=window, min_periods=1).std()
        )
        ts[f"rolling_max_{window}"] = (
            ts["units_sold"].shift(1).rolling(window=window, min_periods=1).max()
        )

    # ── Exponentially Weighted Mean ───────────────────────────────────────────
    ts["ewm_7"]  = ts["units_sold"].shift(1).ewm(span=7, adjust=False).mean()
    ts["ewm_14"] = ts["units_sold"].shift(1).ewm(span=14, adjust=False).mean()

    # ── Holiday / External Feature ────────────────────────────────────────────
    ts["is_holiday"] = ts["is_holiday"].fillna(0).astype(int)

    # ── Trend (days since start) ───────────────────────────────────────────────
    ts["trend"] = (ts.index - ts.index[0]).days

    # Drop rows where lag/rolling features are NaN (first 28 days)
    ts = ts.dropna()
    ts = ts.reset_index()

    return ts


def sequential_split(df: pd.DataFrame, test_months: int = 2):
    """
    Split data sequentially to prevent data leakage.
    Train on first N months, test on last `test_months`.
    """
    split_date = df["date"].max() - pd.DateOffset(months=test_months)
    train = df[df["date"] <= split_date].copy()
    test  = df[df["date"] >  split_date].copy()
    return train, test


FEATURE_COLS = [
    "day_of_week", "day_of_month", "month", "quarter",
    "week_of_year", "year", "is_weekend", "is_month_start", "is_month_end",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_max_7",
    "rolling_mean_14", "rolling_std_14", "rolling_max_14",
    "rolling_mean_30", "rolling_std_30", "rolling_max_30",
    "ewm_7", "ewm_14", "is_holiday", "trend",
]

TARGET_COL = "units_sold"


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/restaurant_demand_forecast/data/pos_sales_raw.csv",
                     parse_dates=["date"])
    item = "Chicken Biryani"
    feat_df = engineer_features(df, item)
    print(f"Features for '{item}': {feat_df.shape}")
    print(feat_df[["date", "units_sold"] + FEATURE_COLS[:5]].head())
