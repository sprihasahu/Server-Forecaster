import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Server CPU Forecaster", layout="wide")
st.title("Server Resource Utilization Forecaster")
st.write(
    "Predicts hourly server CPU usage for the next 24 hours, "
    "so servers can be scaled up before a traffic peak."
)

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
days = st.sidebar.slider("Days of data to generate", 30, 120, 60)
threshold = st.sidebar.slider("Scale-up alert threshold (CPU %)", 40, 95, 70)
n_trees = st.sidebar.slider("Number of trees in the model", 50, 400, 200, step=50)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

FEATURES = ["hour", "day_of_week", "lag_24", "lag_48", "lag_168", "rolling_mean_24"]


@st.cache_data
def make_data(days, seed):
    """Create synthetic hourly CPU data."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    is_weekend = (hours.day_of_week >= 5).astype(int)
    daily = 30 * np.sin((hours.hour.values - 8) * np.pi / 12).clip(min=0)
    cpu = 35 + daily - 10 * is_weekend + rng.normal(0, 3, len(hours))
    return pd.DataFrame({"cpu": np.clip(cpu, 5, 100)}, index=hours)


@st.cache_data
def add_features(raw):
    """Create the hints (features) the model learns from."""
    df = raw.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.day_of_week
    df["lag_24"] = df["cpu"].shift(24)
    df["lag_48"] = df["cpu"].shift(48)
    df["lag_168"] = df["cpu"].shift(168)
    df["rolling_mean_24"] = df["cpu"].shift(24).rolling(24).mean()
    return df.dropna()


@st.cache_resource
def train_model(train_df, n_trees, seed):
    model = GradientBoostingRegressor(
        n_estimators=n_trees, max_depth=3, random_state=int(seed)
    )
    model.fit(train_df[FEATURES], train_df["cpu"])
    return model


# ---------- Pipeline ----------
raw = make_data(days, seed)
df = add_features(raw)
train, test = df.iloc[:-24 * 7], df.iloc[-24 * 7:]   # last 7 days = test
model = train_model(train, n_trees, seed)

pred = model.predict(test[FEATURES])
mae = mean_absolute_error(test["cpu"], pred)
rmse = mean_squared_error(test["cpu"], pred) ** 0.5

# Forecast for the last 24 hours
last_day = test.iloc[-24:]
forecast = pd.Series(model.predict(last_day[FEATURES]), index=last_day.index)
peak, peak_time = forecast.max(), forecast.idxmax()

# ---------- Results ----------
c1, c2, c3 = st.columns(3)
c1.metric("MAE (average error)", f"{mae:.2f} CPU%")
c2.metric("RMSE", f"{rmse:.2f} CPU%")
c3.metric("Predicted peak (next 24h)", f"{peak:.1f}%", f"at {peak_time:%H:%M}")

if peak > threshold:
    st.error(f"ALERT: predicted peak {peak:.1f}% is above {threshold}%. Scale up before {peak_time:%H:%M}!")
else:
    st.success(f"All good: predicted peak {peak:.1f}% is below the {threshold}% threshold.")

# ---------- Chart 1: Actual vs predicted (7-day test) ----------
st.subheader("Actual vs Predicted CPU (last 7 days)")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(test.index, test["cpu"], label="Actual CPU %", alpha=0.7)
ax.plot(test.index, pred, label="Predicted CPU %", linestyle="--")
ax.axhline(threshold, color="red", linewidth=0.8, label=f"Threshold ({threshold}%)")
ax.set_xlabel("Time")
ax.set_ylabel("CPU %")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

# ---------- Chart 2: Next 24 hours ----------
st.subheader("Forecast for the next 24 hours")
fig2, ax2 = plt.subplots(figsize=(11, 3.5))
ax2.plot(forecast.index, forecast.values, marker="o", color="green")
ax2.axhline(threshold, color="red", linewidth=0.8)
ax2.set_xlabel("Time")
ax2.set_ylabel("Predicted CPU %")
fig2.tight_layout()
st.pyplot(fig2)

# ---------- Data table ----------
with st.expander("View data and download"):
    st.dataframe(raw.tail(48))
    st.download_button(
        "Download full data as CSV",
        raw.to_csv().encode("utf-8"),
        file_name="server_metrics.csv",
        mime="text/csv",
    )