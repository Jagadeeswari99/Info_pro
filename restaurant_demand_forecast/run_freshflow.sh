#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements.txt >/tmp/freshflow-pip.log 2>&1 || { cat /tmp/freshflow-pip.log; exit 1; }
python src/data_generator.py >/tmp/freshflow-data.log 2>&1 || { cat /tmp/freshflow-data.log; exit 1; }
python src/feature_engineering_v2.py >/tmp/freshflow-features.log 2>&1 || { cat /tmp/freshflow-features.log; exit 1; }
python src/train_quantile_models.py >/tmp/freshflow-train.log 2>&1 || { cat /tmp/freshflow-train.log; exit 1; }
python src/explain.py >/tmp/freshflow-explain.log 2>&1 || { cat /tmp/freshflow-explain.log; exit 1; }
python src/anomaly.py >/tmp/freshflow-anomaly.log 2>&1 || { cat /tmp/freshflow-anomaly.log; exit 1; }
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > /tmp/freshflow-api.log 2>&1 &
python -m streamlit run dashboard/app.py --server.headless true --server.port 8501 > /tmp/freshflow-dashboard.log 2>&1 &
echo "FreshFlow is starting..."
echo "API: http://127.0.0.1:8000/docs"
echo "Dashboard: http://127.0.0.1:8501"
