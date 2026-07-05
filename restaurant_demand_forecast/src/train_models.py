"""
train_models.py
Week 3: Model Training and Selection

Trains 3 models per menu item:
1. Baseline: Linear Regression
2. XGBoost Regressor with TimeSeriesSplit hyperparameter tuning
3. Prophet (additive time-series model)

Saves results and metrics to /outputs/
"""

import pandas as pd
import numpy as np
import pickle, json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from prophet import Prophet

import sys
sys.path.insert(0, "/home/claude/restaurant_demand_forecast/src")
from feature_engineering import engineer_features, sequential_split, FEATURE_COLS, TARGET_COL

OUTPUT_DIR = "/home/claude/restaurant_demand_forecast/outputs"
MODEL_DIR  = "/home/claude/restaurant_demand_forecast/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred, model_name):
    return {
        "model": model_name,
        "MAE":   round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE":  round(rmse(y_true, y_pred), 3),
        "R2":    round(r2_score(y_true, y_pred), 4),
    }


def tune_xgboost(X_train, y_train):
    """TimeSeriesSplit cross-validation for XGBoost hyperparameter tuning."""
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = [
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.02, "subsample": 0.9},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9},
    ]

    best_mae   = float("inf")
    best_params = param_grid[0]

    for params in param_grid:
        fold_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            xf_tr = X_train.iloc[train_idx]
            yf_tr = y_train.iloc[train_idx]
            xf_val = X_train.iloc[val_idx]
            yf_val = y_train.iloc[val_idx]

            m = XGBRegressor(**params, random_state=42,
                             tree_method="hist", verbosity=0)
            m.fit(xf_tr, yf_tr,
                  eval_set=[(xf_val, yf_val)],
                  verbose=False)
            preds = m.predict(xf_val)
            fold_maes.append(mean_absolute_error(yf_val, preds))

        avg_mae = np.mean(fold_maes)
        if avg_mae < best_mae:
            best_mae    = avg_mae
            best_params = params

    print(f"      Best XGBoost params: {best_params} (CV MAE={best_mae:.2f})")
    return best_params


def train_prophet(train_df, test_df):
    """Train Facebook Prophet model."""
    prophet_train = train_df[["date", TARGET_COL]].rename(
        columns={"date": "ds", TARGET_COL: "y"}
    )
    prophet_test = test_df[["date", TARGET_COL]].rename(
        columns={"date": "ds", TARGET_COL: "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=len(prophet_test))
    forecast = model.predict(future)

    test_forecast = forecast.set_index("ds").loc[prophet_test["ds"]]["yhat"]
    y_pred = test_forecast.values.clip(min=0)
    y_true = prophet_test["y"].values

    return model, y_true, y_pred


def train_all_items():
    df = pd.read_csv("/home/claude/restaurant_demand_forecast/data/pos_sales_raw.csv",
                     parse_dates=["date"])

    all_results   = []
    all_forecasts = {}

    for item in df["menu_item"].unique():
        print(f"\n{'='*55}")
        print(f"  Training models for: {item}")
        print(f"{'='*55}")

        feat_df = engineer_features(df, item)
        train, test = sequential_split(feat_df, test_months=2)

        X_train = train[FEATURE_COLS]
        y_train = train[TARGET_COL]
        X_test  = test[FEATURE_COLS]
        y_test  = test[TARGET_COL]

        # ── 1. Baseline: Linear Regression ────────────────────────────────
        print("  [1/3] Training Linear Regression (baseline)...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test).clip(min=0)
        lr_metrics = evaluate(y_test, lr_preds, "Linear Regression")
        print(f"      MAE={lr_metrics['MAE']:.2f}  RMSE={lr_metrics['RMSE']:.2f}  R²={lr_metrics['R2']:.4f}")

        # ── 2. XGBoost ────────────────────────────────────────────────────
        print("  [2/3] Tuning + Training XGBoost...")
        best_params = tune_xgboost(X_train, y_train)
        xgb = XGBRegressor(**best_params, random_state=42,
                           tree_method="hist", verbosity=0)
        xgb.fit(X_train, y_train, verbose=False)
        xgb_preds = xgb.predict(X_test).clip(min=0)
        xgb_metrics = evaluate(y_test, xgb_preds, "XGBoost")
        print(f"      MAE={xgb_metrics['MAE']:.2f}  RMSE={xgb_metrics['RMSE']:.2f}  R²={xgb_metrics['R2']:.4f}")

        # ── 3. Prophet ────────────────────────────────────────────────────
        print("  [3/3] Training Prophet...")
        prop_model, prop_ytrue, prop_preds = train_prophet(train, test)
        prop_metrics = evaluate(prop_ytrue, prop_preds, "Prophet")
        print(f"      MAE={prop_metrics['MAE']:.2f}  RMSE={prop_metrics['RMSE']:.2f}  R²={prop_metrics['R2']:.4f}")

        # ── Feature Importance ────────────────────────────────────────────
        feat_imp = pd.Series(xgb.feature_importances_, index=FEATURE_COLS)
        feat_imp = feat_imp.sort_values(ascending=False)

        # ── Save models ────────────────────────────────────────────────────
        item_key = item.replace(" ", "_")
        with open(f"{MODEL_DIR}/{item_key}_xgb.pkl", "wb") as f:
            pickle.dump(xgb, f)
        with open(f"{MODEL_DIR}/{item_key}_lr.pkl", "wb") as f:
            pickle.dump(lr, f)

        # ── Store results ──────────────────────────────────────────────────
        for m in [lr_metrics, xgb_metrics, prop_metrics]:
            m["menu_item"] = item
        all_results.extend([lr_metrics, xgb_metrics, prop_metrics])

        all_forecasts[item] = {
            "dates":       test["date"].dt.strftime("%Y-%m-%d").tolist(),
            "actual":      y_test.tolist(),
            "lr_pred":     lr_preds.tolist(),
            "xgb_pred":    xgb_preds.tolist(),
            "prop_pred":   prop_preds.tolist(),
            "feat_imp":    feat_imp.head(15).to_dict(),
        }

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv", index=False)

    with open(f"{OUTPUT_DIR}/forecasts.json", "w") as f:
        json.dump(all_forecasts, f, indent=2)

    print(f"\n\n{'='*55}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    print(results_df.to_string(index=False))
    print(f"\nSaved metrics → {OUTPUT_DIR}/model_metrics.csv")
    print(f"Saved forecasts → {OUTPUT_DIR}/forecasts.json")
    return results_df, all_forecasts


if __name__ == "__main__":
    results_df, forecasts = train_all_items()
