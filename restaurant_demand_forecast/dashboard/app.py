import os
import streamlit as st
import pandas as pd
import requests

API_URL = os.getenv("FRESHFLOW_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FreshFlow Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    .stApp { background: linear-gradient(135deg, #f8fbff 0%, #eef7ff 100%); }
    div[data-testid="stMetric"] { background: white; border: 1px solid #dce8f5; border-radius: 12px; padding: 0.8rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("FreshFlow — AI Demand & Inventory Intelligence")
st.caption("A polished decision-support view for demand forecasting, explainability, and reorder planning.")

items = ["Chicken Biryani", "Paneer Butter Masala", "Veg Fried Rice", "Masala Dosa", "Grilled Fish"]
selected_item = st.selectbox("Select menu item", items, key="menu_item")

if st.button("Refresh data", width="stretch"):
    st.rerun()

with st.sidebar:
    st.header("Decision controls")
    days = st.slider("Forecast horizon (days)", 3, 14, 7)
    current_stock = st.number_input("Current stock available", min_value=0, value=20, step=1)

    st.markdown("---")
    st.subheader("Reorder recommendation")
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

try:
    forecast_resp = requests.post(
        f"{API_URL}/forecast/{selected_item}",
        json={"days": days},
        timeout=20,
    )
    metrics_resp = requests.get(f"{API_URL}/metrics", timeout=20)
    anomalies_resp = requests.get(f"{API_URL}/anomalies", timeout=20)
    explain_resp = requests.get(
        f"{API_URL}/explain/{selected_item}/2023-02-05",
        timeout=20,
    )
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
    pinball_loss = metrics_data.get("pinball_loss", {})
else:
    metrics_data = {}
    pinball_loss = {}

col1, col2, col3 = st.columns(3)
col1.metric("Selected item", selected_item)
col2.metric("Forecast horizon", f"{days} days")
col3.metric("P50 pinball loss", f"{pinball_loss.get('0.5', 'n/a'):.2f}" if isinstance(pinball_loss.get('0.5'), (int, float)) else "n/a")

st.markdown("---")
st.subheader("Forecast band")
chart_df = forecasts.set_index("date")[["p10", "p50", "p90"]]
st.line_chart(chart_df, width="stretch")

st.markdown("---")
left_col, right_col = st.columns([1.2, 0.8])

with left_col:
    st.subheader("Why this forecast matters")
    if explain_resp.ok:
        explanation = explain_resp.json().get("explanation", [])
        for point in explanation:
            st.write(f"• {point}")
    else:
        st.caption("SHAP explanation is unavailable right now.")

with right_col:
    st.subheader("Anomaly watchlist")
    if anomalies_resp.ok:
        anomalies = pd.DataFrame(anomalies_resp.json())
        if not anomalies.empty:
            st.dataframe(anomalies.head(8), width="stretch", hide_index=True)
        else:
            st.success("No anomalies were flagged in the current window.")
    else:
        st.caption("Anomaly data is unavailable.")

st.markdown("---")
st.subheader("Executive summary")
st.write(
    "FreshFlow combines uncertainty-aware forecasts with actionable inventory guidance so a restaurant manager can see both expected demand and the range of risk."
)
