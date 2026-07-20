# FreshFlow — demand forecasting + inventory recommendation MVP

FreshFlow is a simple demo that trains quantile forecasting models, exposes them through a FastAPI service, and shows a reorder recommendation in a Streamlit dashboard.

## Results

- P50 MAE and calibration are produced by running the training script.
- MVP built on synthetic POS data; weather, SHAP, and anomaly detection are the next layer, not yet included.

## How to run

```bash
python src/train_quantile_models.py
python -m uvicorn api.main:app --reload
```

In another terminal:

```bash
python -m streamlit run dashboard/app.py
```

Then open:

- API docs: http://127.0.0.1:8000/docs
- Dashboard: http://127.0.0.1:8501
