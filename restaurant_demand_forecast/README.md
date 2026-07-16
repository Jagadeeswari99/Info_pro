# FreshFlow — AI Demand & Inventory Intelligence

FreshFlow is a prototype restaurant-demand forecasting and inventory-planning system. It starts from synthetic daily POS sales data, adds weather and event signals, trains quantile forecasting models, explains predictions with SHAP, flags anomalies, and produces inventory reorder suggestions.

## What this project is actually doing

This repository is doing five jobs at once:

1. Synthetic data generation
   - The project creates realistic daily restaurant sales records for five menu items using the script in [src/data_generator.py](src/data_generator.py).
   - The generated data includes date, menu item, units sold, revenue, holiday flags, weekday/weekend indicators, and calendar features.

2. Feature engineering
   - The original pipeline in [src/feature_engineering.py](src/feature_engineering.py) builds lag features, rolling averages, and calendar features.
   - The upgraded pipeline in [src/feature_engineering_v2.py](src/feature_engineering_v2.py) extends this with real weather data, event flags, and event multipliers.

3. Forecasting
   - The original point-forecast training in [src/train_models.py](src/train_models.py) trains baseline linear regression and XGBoost models.
   - The new quantile workflow in [src/train_quantile_models.py](src/train_quantile_models.py) trains XGBoost quantile models for the 10th, 50th, and 90th percentiles.
   - Those quantiles allow the system to predict a range instead of one single number.

4. Explainability and monitoring
   - [src/explain.py](src/explain.py) uses SHAP to explain why a P50 prediction was made for a selected date.
   - [src/anomaly.py](src/anomaly.py) compares actual values against the P10–P90 band and flags days that fall outside it.

5. Inventory decisions
   - [src/inventory_optimizer.py](src/inventory_optimizer.py) uses a simple safety-stock formula to recommend how much stock to reorder.
   - The configuration file [config.yaml](config.yaml) stores lead times and safety buffers per item.

## Architecture

```text
+-------------------+        +---------------------+
| Synthetic POS     |        | Real weather data   |
| data generator    |        | + events calendar   |
+--------+----------+        +----------+------+
         |                               |
         v                               v
+----------------------+      +------------------------+
| Feature engineering |      | Quantile model training|
| lag / rolling /     |      | P10 / P50 / P90        |
| weather / events    |      +-----------+------------+
+----------+-------------------------------+            |
           |                                           |
           v                                           v
+--------------------+                      +---------------------+
| FastAPI service    |                      | SHAP / anomaly /   |
| /forecast /metrics|                      | inventory modules  |
+---------+----------+                      +----------+----------+
          |                                           |
          v                                           v
+--------------------+                      +---------------------+
| Streamlit dashboard|                      | Outputs / artifacts |
| forecast / reorder |                      | CSVs, PNGs, models |
+--------------------+                      +---------------------+
```

## Current scope and honesty note

This is an offline prototype, not a live operational pilot.

- The base sales data is synthetic.
- The weather data is real from Open-Meteo, but the event calendar is manually curated.
- The evaluation is offline and should be interpreted as a demonstration of the workflow rather than a deployed business system.
- Calibration percentages are reported as measured from the available test window, not as a guarantee of live accuracy.

## What is produced

The project writes the following outputs:

- [data/weather_real.csv](data/weather_real.csv): real weather data fetched from Open-Meteo
- [data/events_calendar.csv](data/events_calendar.csv): curated event multipliers
- [outputs/quantile_metrics.csv](outputs/quantile_metrics.csv): pinball loss by item and quantile
- [outputs/anomalies.csv](outputs/anomalies.csv): flagged anomalous days
- [outputs/shap_plots](outputs/shap_plots): example SHAP plots
- [models](models): saved quantile models and legacy point-forecast models

## How to run in Codespaces

### Option 1: one-command startup
From the project folder:

```bash
chmod +x run_freshflow.sh
./run_freshflow.sh
```

This will:

1. install requirements
2. generate synthetic sales data
3. build engineered features
4. train quantile models
5. generate explainability and anomaly outputs
6. start the API and dashboard

Then open:

- API docs: http://127.0.0.1:8000/docs
- Dashboard: http://127.0.0.1:8501

### Option 2: manual steps

```bash
pip install -r requirements.txt
python src/data_generator.py
python src/feature_engineering_v2.py
python src/train_quantile_models.py
python src/explain.py
python src/anomaly.py
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
python -m streamlit run dashboard/app.py --server.headless true --server.port 8501
```

## Project layout

```text
restaurant_demand_forecast/
├── api/                  # FastAPI application
├── dashboard/            # Streamlit UI
├── data/                 # input CSV files and fetched weather data
├── models/               # trained model artifacts
├── notebooks/            # notebook workspace
├── outputs/              # quantile metrics, anomalies, SHAP plots
├── src/                  # data generation, feature engineering, training, explainability, anomaly, inventory logic
├── config.yaml           # per-item reorder parameters
├── requirements.txt      # Python dependencies
├── run_freshflow.sh      # one-command startup helper
└── README.md             # this document
```

## Portfolio framing

This project is a strong portfolio piece because it demonstrates a full machine-learning workflow:

- data generation
- feature engineering
- forecasting under uncertainty
- explainable AI
- anomaly detection
- operational decision support

It is especially suitable for showing how a forecasting system can move from a single-point prediction to a decision-support tool for inventory planning.
