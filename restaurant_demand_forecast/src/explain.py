from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt

try:
    from src.feature_engineering_v2 import engineer_features_v2, FEATURE_COLS_V2
except ModuleNotFoundError:  # pragma: no cover
    from feature_engineering_v2 import engineer_features_v2, FEATURE_COLS_V2

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "pos_sales_raw.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "shap_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def explain_forecast(item, date):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    weather = pd.read_csv(BASE_DIR / "data" / "weather_real.csv", parse_dates=["date"])
    events = pd.read_csv(BASE_DIR / "data" / "events_calendar.csv", parse_dates=["date"])
    feat_df = engineer_features_v2(df, item, weather=weather, events=events)
    feat_df = feat_df.sort_values("date")
    item_key = item.replace(" ", "_")
    with open(MODEL_DIR / f"{item_key}_q50.pkl", "rb") as fh:
        model = pickle.load(fh)

    row = feat_df[feat_df["date"].astype(str) == str(date)]
    if row.empty:
        raise ValueError(f"No feature row found for {item} on {date}")
    x = row[FEATURE_COLS_V2].iloc[0:1]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    contribution_df = pd.DataFrame({
        "feature": FEATURE_COLS_V2,
        "shap_value": np.asarray(shap_values).reshape(-1),
    })
    contribution_df = contribution_df.sort_values("shap_value", ascending=False).head(5)
    summary = []
    for _, row in contribution_df.iterrows():
        if row["shap_value"] >= 0:
            summary.append(f"{row['feature']} contributed +{row['shap_value']:.1f} units")
        else:
            summary.append(f"{row['feature']} contributed {row['shap_value']:.1f} units")
    return summary


def save_example_plots():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    weather = pd.read_csv(BASE_DIR / "data" / "weather_real.csv", parse_dates=["date"])
    events = pd.read_csv(BASE_DIR / "data" / "events_calendar.csv", parse_dates=["date"])
    for item in df["menu_item"].unique():
        feat_df = engineer_features_v2(df, item, weather=weather, events=events)
        item_key = item.replace(" ", "_")
        with open(MODEL_DIR / f"{item_key}_q50.pkl", "rb") as fh:
            model = pickle.load(fh)
        sample = feat_df[FEATURE_COLS_V2].iloc[[len(feat_df)//2]]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            sample.iloc[0],
            feature_names=FEATURE_COLS_V2,
            matplotlib=True,
            show=False,
        )
        plt.savefig(OUTPUT_DIR / f"{item_key}_shap.png", dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print(explain_forecast("Chicken Biryani", "2023-02-05"))
    save_example_plots()
    print(f"Saved SHAP plots to {OUTPUT_DIR}")
