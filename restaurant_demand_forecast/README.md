# Project 3: AI Demand Forecasting & Inventory Optimization
**Infotact Technical Internship Program — Food & Restaurant Services**

---

## Overview
AI-powered demand forecasting for a restaurant chain using 2 years of synthetic POS data.
Models predict daily units sold per menu item, enabling proactive inventory management and
reduction of food waste.

**Tech Stack:** Python · Pandas · NumPy · Scikit-Learn · XGBoost · Prophet · Matplotlib · Plotly

---

## Project Structure
```
restaurant_demand_forecast/
├── data/
│   └── pos_sales_raw.csv          # Synthetic POS data (not tracked in git — see .gitignore)
├── notebooks/
│   └── demand_forecasting_project.ipynb   # Full EDA + modeling notebook (4 weeks)
├── src/
│   ├── data_generator.py          # Synthetic POS data generator
│   ├── feature_engineering.py     # Lag features, rolling stats, cyclic encoding
│   └── train_models.py            # LR + XGBoost + Prophet training pipeline
├── models/                        # Saved model weights (not tracked in git)
├── outputs/
│   ├── model_metrics.csv          # MAE / RMSE comparison
│   └── forecasts.json             # Per-item predictions for visualization
└── README.md
```

---

## Results

| Menu Item | LR MAE | XGBoost MAE | Improvement |
|---|---|---|---|
| Chicken Biryani | 65.5 | 16.1 | **75%** |
| Masala Dosa | 61.1 | 13.0 | **79%** |
| Paneer Butter Masala | 57.2 | 7.7 | **87%** |
| Veg Fried Rice | 61.6 | 14.0 | **77%** |
| Grilled Fish | 33.3 | 7.5 | **77%** |
| **Average** | **55.7** | **11.7** | **~79%** |

**Top predictive features (XGBoost):** `day_of_week` (19–31%), `is_holiday` (12–20%), `lag_7`, `rolling_mean_7`, `ewm_14`

---

## Weekly Roadmap & GitHub Commit Guide

### Week 1 — Data Ingestion & Time-Series EDA
```bash
git checkout -b week1/data-eda
# Day 1
git add src/data_generator.py
git commit -m "data-clean: add synthetic POS data generator with weekly/seasonal patterns"
# Day 2
git add notebooks/demand_forecasting_project.ipynb
git commit -m "eda: plot overall sales trend and 30-day rolling mean"
# Day 3
git commit -m "eda: weekly seasonality bar chart — fri/sat spike confirmed"
# Day 4
git commit -m "eda: monthly heatmap showing oct-dec festive uplift"
# Day 5
git commit -m "eda: seasonal decomposition and autocorrelation analysis (lag-7 pattern)"
git checkout main && git merge week1/data-eda
```

### Week 2 — Advanced Feature Engineering
```bash
git checkout -b week2/feature-engineering
git commit -m "feature-eng: add lag features (1,3,7,14,21,28 days)"
git commit -m "feature-eng: rolling window stats — mean/std/max at 7,14,30 day windows"
git commit -m "feature-eng: cyclic encoding for day_of_week and month"
git commit -m "feature-eng: sequential train/test split (no data leakage) — 10mo/2mo"
git commit -m "eda: feature-target correlation plot — lag_7 and rolling_mean_7 strongest"
git checkout main && git merge week2/feature-engineering
```

### Week 3 — Model Training & Selection
```bash
git checkout -b week3/modeling
git commit -m "model-tuning: baseline linear regression — avg MAE 55.7 units"
git commit -m "model-tuning: XGBoost with TimeSeriesSplit CV hyperparameter search"
git commit -m "model-tuning: best XGBoost params n_estimators=300 depth=5 lr=0.03"
git commit -m "model-tuning: XGBoost avg MAE 11.7 — 79% improvement over baseline"
git commit -m "model-tuning: Prophet additive model with multiplicative seasonality"
git checkout main && git merge week3/modeling
```

### Week 4 — Evaluation & Business Reporting
```bash
git checkout -b week4/evaluation
git commit -m "eval: MAE/RMSE comparison table across all 5 menu items"
git commit -m "eval: forecast vs actual line chart for test period"
git commit -m "eval: residual distribution and scatter analysis"
git commit -m "eval: XGBoost feature importance — day_of_week top driver (19-31%)"
git commit -m "eval: business summary report with inventory recommendations"
git checkout main && git merge week4/evaluation
```

---

## .gitignore
```
data/pos_sales_raw.csv
models/*.pkl
models/*.h5
outputs/forecasts.json
*.env
.env
__pycache__/
.ipynb_checkpoints/
```

---

## Key Business Insights
1. **day_of_week** is the single strongest predictor — Friday/Saturday demand is 30–60% above weekday average.
2. **is_holiday** explains sudden spikes — holiday uplift of 40–80% captured accurately.
3. **lag_7 and rolling_mean_7** capture weekly demand cycles better than raw date features.
4. XGBoost reduces average MAE from **55.7 → 11.7 units** (79% improvement).
5. With 11.7 unit average error, restaurant managers can confidently plan weekly orders.

---

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python src/data_generator.py

# 3. Train all models
python src/train_models.py

# 4. Open notebook for full analysis
jupyter notebook notebooks/demand_forecasting_project.ipynb
```
