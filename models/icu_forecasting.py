import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


data = pd.read_csv("../data/h_data.csv")
data["date"] = pd.to_datetime(data["date"])


df = data.rename(columns={
    "date": "ds",
    "icu_occupied": "y"
})

# Avoid negative values
df["y"] = df["y"].clip(lower=0)


model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add regressors
model.add_regressor("patients")
model.add_regressor("emergencies")
model.add_regressor("temperature")
model.add_regressor("month")

# ===============================
# 4. Train Model
# ===============================
model.fit(df[["ds", "y", "patients", "emergencies", "temperature", "month"]])

print("✅ ICU forecasting model trained successfully")

# ===============================
# 5. Create Future Data (Next 30 Days)
# ===============================
future = model.make_future_dataframe(periods=30)

future["patients"] = data["patients"].iloc[-7:].mean()
future["emergencies"] = data["emergencies"].iloc[-7:].mean()
future["temperature"] = data["temperature"].iloc[-7:].mean()
future["month"] = future["ds"].dt.month

# ===============================
# 6. Forecast
# ===============================
forecast = model.predict(future)

# Remove negative predictions
forecast["yhat"] = forecast["yhat"].clip(lower=0)

# ===============================
# 7. Plot Forecast
# ===============================
fig = model.plot(forecast)
plt.title("ICU Bed Occupancy Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("ICU Beds Occupied")
plt.show()

# ===============================
# 8. Display Next 30 Days
# ===============================
print("\n📅 Next 30 Days ICU Occupancy Forecast:")
print(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    .tail(30)
)
