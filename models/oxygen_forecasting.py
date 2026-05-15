import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

print("🔵 Training Oxygen Usage Forecasting Model...")

# ===============================
# 1. Load data
# ===============================
data = pd.read_csv("../data/h_data.csv")
data["date"] = pd.to_datetime(data["date"])

# ===============================
# 2. Prepare Prophet dataframe
# ===============================
df = data.rename(columns={
    "date": "ds",
    "oxygen_usage": "y"
})

# Remove negative values
df["y"] = df["y"].clip(lower=0)

# ===============================
# 3. Initialize model
# ===============================
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add regressors
model.add_regressor("patients")
model.add_regressor("icu_occupied")
model.add_regressor("emergencies")
model.add_regressor("temperature")
model.add_regressor("month")

# ===============================
# 4. Train model
# ===============================
model.fit(df[[
    "ds", "y",
    "patients",
    "icu_occupied",
    "emergencies",
    "temperature",
    "month"
]])

# ===============================
# 5. Create future dataframe
# ===============================
future = model.make_future_dataframe(periods=30)

# Add future regressor values (realistic assumptions)
future["patients"] = data["patients"].iloc[-7:].mean()
future["icu_occupied"] = data["icu_occupied"].iloc[-7:].mean()
future["emergencies"] = data["emergencies"].iloc[-7:].mean()
future["temperature"] = data["temperature"].iloc[-7:].mean()
future["month"] = future["ds"].dt.month

# ===============================
# 6. Forecast
# ===============================
forecast = model.predict(future)

# Remove negative predictions
forecast["yhat"] = forecast["yhat"].clip(lower=0)
forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

# ===============================
# 7. Plot
# ===============================
model.plot(forecast)
plt.title("Oxygen Usage Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Oxygen Cylinders Used")
plt.show()

# ===============================
# 8. Print results
# ===============================
print("\n📅 Next 30 Days Oxygen Usage Forecast:")
print(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    .tail(30)
)

print("✅ Oxygen forecasting model trained successfully")
