"""
data_generator.py
Generates synthetic restaurant POS sales data for demand forecasting.
Simulates realistic patterns: weekday/weekend spikes, seasonality, holidays, trends.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_pos_data(
    start_date="2023-01-01",
    end_date="2024-12-31",
    menu_items=None,
    seed=42
):
    """
    Generate synthetic daily POS sales data for multiple menu items.

    Returns:
        pd.DataFrame with columns: date, menu_item, units_sold, revenue
    """
    np.random.seed(seed)

    if menu_items is None:
        menu_items = {
            "Chicken Biryani": {"base": 80, "price": 220},
            "Paneer Butter Masala": {"base": 55, "price": 180},
            "Veg Fried Rice": {"base": 70, "price": 130},
            "Masala Dosa": {"base": 110, "price": 90},
            "Grilled Fish": {"base": 40, "price": 280},
        }

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Indian holidays (approximate)
    holidays = set(pd.to_datetime([
        "2023-01-26", "2023-03-08", "2023-04-14", "2023-08-15",
        "2023-10-02", "2023-10-24", "2023-11-12", "2023-12-25",
        "2024-01-26", "2024-03-25", "2024-04-14", "2024-08-15",
        "2024-10-02", "2024-10-13", "2024-11-01", "2024-12-25",
    ]))

    records = []

    for item_name, config in menu_items.items():
        base = config["base"]
        price = config["price"]

        for date in dates:
            # === Trend: slight growth over time ===
            day_index = (date - dates[0]).days
            trend_factor = 1 + 0.0002 * day_index

            # === Weekly seasonality: Fri/Sat/Sun are busier ===
            weekday = date.dayofweek  # 0=Mon, 6=Sun
            if weekday in [4, 5]:   # Fri, Sat
                weekly_factor = np.random.uniform(1.3, 1.6)
            elif weekday == 6:       # Sun
                weekly_factor = np.random.uniform(1.15, 1.35)
            else:
                weekly_factor = np.random.uniform(0.85, 1.05)

            # === Monthly seasonality (festive months Oct-Dec, summer dip May-Jun) ===
            month = date.month
            if month in [10, 11, 12]:
                seasonal_factor = np.random.uniform(1.15, 1.3)
            elif month in [5, 6]:
                seasonal_factor = np.random.uniform(0.8, 0.95)
            elif month in [1, 3, 4]:   # New Year, Holi, Tamil New Year
                seasonal_factor = np.random.uniform(1.05, 1.2)
            else:
                seasonal_factor = np.random.uniform(0.95, 1.1)

            # === Holiday spike ===
            holiday_factor = np.random.uniform(1.4, 1.8) if date in holidays else 1.0

            # === Random noise ===
            noise = np.random.normal(1.0, 0.08)

            # === Compute units sold ===
            units = int(base * trend_factor * weekly_factor * seasonal_factor * holiday_factor * noise)
            units = max(units, 0)

            revenue = round(units * price * np.random.uniform(0.97, 1.03), 2)

            records.append({
                "date": date,
                "menu_item": item_name,
                "units_sold": units,
                "revenue": revenue,
                "price_per_unit": price,
                "is_holiday": date in holidays,
                "day_of_week": date.dayofweek,
                "day_name": date.strftime("%A"),
                "month": date.month,
                "week_of_year": date.isocalendar().week,
                "is_weekend": weekday >= 5,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["menu_item", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_pos_data()
    df.to_csv("/home/claude/restaurant_demand_forecast/data/pos_sales_raw.csv", index=False)
    print(f"Generated {len(df)} records across {df['menu_item'].nunique()} menu items.")
    print(df.head(10))
