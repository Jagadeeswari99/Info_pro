# FreshFlow — demand forecasting + inventory recommendation MVP

FreshFlow is a simple demo that trains quantile forecasting models, exposes them through a FastAPI service, and shows a reorder recommendation in a Streamlit dashboard.

## Results

- P50 MAE and calibration are produced by running the training script.
- MVP built on synthetic POS data; weather, SHAP, and anomaly detection are the next layer, not yet included.
## How to Run

### 1. Activate the environment
```bash
cd restaurant_demand_forecast
source ../.venv/bin/activate
```

### 2. Train the models
```bash
python src/train_quantile_models.py
```
This creates model files in `models/` and writes metrics to `outputs/`.

### 3. Start the API
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Docs available at: `http://127.0.0.1:8000/docs`

### 4. Start the dashboard
In a **separate terminal**:
```bash
python -m streamlit run dashboard/app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```
Dashboard available at: `http://127.0.0.1:8501`

### 5. One-command option
```bash
chmod +x run_freshflow.sh
./run_freshflow.sh
```

### Quick health check
```bash
curl -s 'http://127.0.0.1:8000/forecast/Chicken%20Biryani' -H 'Content-Type: application/json' -d '{"days":2}'
```

---

### Troubleshooting: "address already in use"

If you see `Errno 98: address already in use` (API) or `Port 8501 is not available` (dashboard), a previous run is still holding that port — usually from a terminal tab that wasn't properly closed, or a Codespaces session reload.

**Fix — kill the stuck process:**
```bash
kill -9 $(lsof -t -i:8000) 2>/dev/null   # for the API
kill -9 $(lsof -t -i:8501) 2>/dev/null   # for the dashboard
```
Then rerun the start commands above.

**Alternative — use different ports:**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
python -m streamlit run dashboard/app.py --server.headless true --server.address 0.0.0.0 --server.port 8502
```
⚠️ If you change the API port, update the API base URL inside `dashboard/app.py` to match (it defaults to `http://127.0.0.1:8000`), or the dashboard won't be able to reach the API.
