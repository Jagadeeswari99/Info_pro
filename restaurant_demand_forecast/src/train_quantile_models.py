from importlib import metadata
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "pos_sales_raw.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from src.feature_engineering import engineer_features, FEATURE_COLS, sequential_split
except ModuleNotFoundError:  # pragma: no cover
    from feature_engineering import engineer_features, FEATURE_COLS, sequential_split


def pinball_loss(y_true, y_pred, quantile):
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def build_quantile_models():
    xgb_version = metadata.version("xgboost")
    print(f"xgboost version: {xgb_version}")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    rows = []

    for item in sorted(df["menu_item"].unique()):
        feat_df = engineer_features(df, item)
        train, test = sequential_split(feat_df, test_months=2)
        X_train = train[FEATURE_COLS]
        y_train = train["units_sold"]
        X_test = test[FEATURE_COLS]
        y_test = test["units_sold"]

        model_by_quantile = {}
        preds_by_quantile = {}
        item_calibration = None

        for quantile in [0.1, 0.5, 0.9]:
            model = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                random_state=42,
                tree_method="hist",
                verbosity=0,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test).clip(min=0)
            model_by_quantile[quantile] = model
            preds_by_quantile[quantile] = preds

            item_key = item.replace(" ", "_")
            quantile_suffix = {0.1: "q10", 0.5: "q50", 0.9: "q90"}[quantile]
            with open(MODEL_DIR / f"{item_key}_{quantile_suffix}.pkl", "wb") as fh:
                pickle.dump(model, fh)

        p90_pred = preds_by_quantile[0.9]
        item_calibration = np.mean(y_test.values < p90_pred) * 100

        for quantile in [0.1, 0.5, 0.9]:
            preds = preds_by_quantile[quantile]
            mae = mean_absolute_error(y_test, preds)
            pinball = pinball_loss(y_test, preds, quantile)
            rows.append(
                {
                    "menu_item": item,
                    "quantile": quantile,
                    "pinball_loss": round(float(pinball), 4),
                    "mae": round(float(mae), 3),
                    "calibration_pct": round(float(item_calibration), 2),
                }
            )

        print(f"{item}: P50 MAE = {mae:.2f} | {item_calibration:.1f}% of actuals below P90")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "quantile_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    print(f"Saved quantile metrics to {OUTPUT_DIR / 'quantile_metrics.csv'}")


if __name__ == "__main__":
    build_quantile_models()
