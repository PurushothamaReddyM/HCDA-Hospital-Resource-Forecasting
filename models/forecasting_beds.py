import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

print("🛏️ Training Bed Occupancy Forecasting Model")

# -------------------------------
# 1. Load dataset
# -------------------------------
data = pd.read_csv("../data/h_data.csv")
data["date"] = pd.to_datetime(data["date"])

# -------------------------------
# 2. Prepare data for Prophet
# -------------------------------
df = data.rename(columns={
    "date": "ds",
    "beds_occupied": "y"
})

# Safety check
df["y"] = df["y"].clip(lower=0)

# -------------------------------
# 3. Initialize Prophet model
# -------------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add external regressors
model.add_regressor("temperature")
model.add_regressor("month")

# -------------------------------
# 4. Train model
# -------------------------------
model.fit(df[["ds", "y", "temperature", "month"]])

print("✅ Bed Occupancy Model Trained Successfully")

# -------------------------------
# 5. Create future dataframe
# -------------------------------
future = model.make_future_dataframe(periods=30)

future["month"] = future["ds"].dt.month
future["temperature"] = data["temperature"].iloc[-7:].mean()

# -------------------------------
# 6. Forecast
# -------------------------------
forecast = model.predict(future)

forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
    ["yhat", "yhat_lower", "yhat_upper"]
].clip(lower=0)

# -------------------------------
# 7. Save forecast
# -------------------------------
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
    "../data/bed_forecast_output.csv",
    index=False
)

print("📁 Bed forecast saved to data/bed_forecast_output.csv")

# -------------------------------
# 8. Plot
# -------------------------------
model.plot(forecast)
plt.title("Bed Occupancy Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Beds Occupied")
plt.show()
