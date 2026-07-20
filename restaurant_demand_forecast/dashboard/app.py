import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("FRESHFLOW_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FreshFlow MVP", layout="wide")
st.title("FreshFlow — demand forecasting + inventory recommendation MVP")
st.caption("A simple demo that shows a demand forecast band and a reorder recommendation from the same API.")

items = ["Chicken Biryani", "Paneer Butter Masala", "Veg Fried Rice", "Masala Dosa", "Grilled Fish"]
selected_item = st.selectbox("Select menu item", items, key="menu_item")
days = st.slider("Forecast horizon (days)", 1, 14, 7)
current_stock = st.number_input("Current stock", min_value=0, value=20, step=1)

if st.button("Get recommendation"):
    try:
        reorder_resp = requests.get(
            f"{API_URL}/inventory/reorder/{selected_item}",
            params={"current_stock": current_stock},
            timeout=20,
        )
        if reorder_resp.ok:
            reorder_data = reorder_resp.json()
            st.metric("Suggested order", reorder_data["recommended_order_qty"])
            st.info(reorder_data["reasoning"])
        else:
            st.warning("Inventory API is unavailable.")
    except requests.RequestException as exc:
        st.error(f"Inventory API connection failed: {exc}")

try:
    forecast_resp = requests.post(
        f"{API_URL}/forecast/{selected_item}",
        json={"days": days},
        timeout=20,
    )
    metrics_resp = requests.get(f"{API_URL}/metrics", timeout=20)
except requests.RequestException as exc:
    st.error(f"API connection failed: {exc}")
    st.stop()

if forecast_resp.ok:
    payload = forecast_resp.json()
    forecasts = pd.DataFrame(payload["forecasts"])
else:
    st.error("Forecast endpoint returned an error.")
    st.stop()

if metrics_resp.ok:
    metrics_data = metrics_resp.json()
    mae_value = metrics_data.get("mae", {}).get(selected_item, "n/a")
    calibration_value = metrics_data.get("calibration_pct", {}).get(selected_item, "n/a")
else:
    mae_value = "n/a"
    calibration_value = "n/a"

col1, col2, col3 = st.columns(3)
col1.metric("Selected item", selected_item)
col2.metric("Forecast horizon", f"{days} days")
col3.metric("P50 MAE", f"{mae_value}" if mae_value != "n/a" else "n/a")

st.markdown("---")
st.subheader("Forecast band")
chart_df = forecasts.set_index("date")[["p10", "p50", "p90"]]
st.line_chart(chart_df, width="stretch")

st.markdown("---")
st.caption(f"Calibration for {selected_item}: {calibration_value}% of actuals below P90")
