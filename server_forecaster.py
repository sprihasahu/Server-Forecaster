"""
Server Resource Utilization Forecaster
--------------------------------------
Predicts the next 24 hours of server CPU usage from past usage logs,
so infrastructure can be scaled BEFORE a traffic peak causes a crash.

Steps: 1) generate synthetic server logs  2) feature engineering
       3) train model  4) evaluate  5) forecast next 24h + plot
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)

# ---------- 1. Synthetic data: 60 days of hourly CPU usage ----------
hours = pd.date_range("2026-01-01", periods=60 * 24, freq="h")
hour_of_day = hours.hour.values
is_weekend = (hours.dayofweek >= 5).astype(int)

daily_pattern = 30 * np.sin((hour_of_day - 8) * np.pi / 12).clip(min=0)  # busy in daytime
cpu = 35 + daily_pattern - 10 * is_weekend + np.random.normal(0, 3, len(hours))
cpu = np.clip(cpu, 5, 100)

df = pd.DataFrame({"timestamp": hours, "cpu": cpu}).set_index("timestamp")
df.to_csv("server_metrics.csv")

# ---------- 2. Feature engineering ----------
# Only lags >= 24h are used, so we can forecast a full day ahead without
# needing values we wouldn't know yet.
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["lag_24"] = df["cpu"].shift(24)                      # same hour yesterday
df["lag_48"] = df["cpu"].shift(48)                      # same hour 2 days ago
df["lag_168"] = df["cpu"].shift(168)                    # same hour last week
df["rolling_mean_24"] = df["cpu"].shift(24).rolling(24).mean()  # yesterday's average
df = df.dropna()

FEATURES = ["hour", "day_of_week", "lag_24", "lag_48", "lag_168", "rolling_mean_24"]

# ---------- 3. Time-based train/test split (never shuffle time series!) ----------
train, test = df.iloc[:-24 * 7], df.iloc[-24 * 7:]       # last 7 days = test
model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
model.fit(train[FEATURES], train["cpu"])

# ---------- 4. Evaluate ----------
pred = model.predict(test[FEATURES])
mae = mean_absolute_error(test["cpu"], pred)
rmse = mean_squared_error(test["cpu"], pred) ** 0.5
print(f"MAE : {mae:.2f} CPU%")
print(f"RMSE: {rmse:.2f} CPU%")

# ---------- 5. Forecast the last 24 hours + flag scaling alert ----------
last_day = test.iloc[-24:]
forecast = pd.Series(model.predict(last_day[FEATURES]), index=last_day.index)
peak = forecast.max()
print(f"Predicted peak CPU in next 24h: {peak:.1f}% at {forecast.idxmax():%H:%M}")
if peak > 70:
    print("ALERT: scale up before the peak!")

plt.figure(figsize=(11, 5))
plt.plot(test.index, test["cpu"], label="Actual CPU %", alpha=0.7)
plt.plot(test.index, pred, label="Predicted CPU %", linestyle="--")
plt.axhline(70, color="red", linewidth=0.8, label="Scale-up threshold (70%)")
plt.title("Server CPU Utilization: Actual vs Predicted (last 7 days)")
plt.xlabel("Time")
plt.ylabel("CPU %")
plt.legend()
plt.tight_layout()
plt.savefig("forecast.png", dpi=150)
print("Saved forecast.png and server_metrics.csv")
