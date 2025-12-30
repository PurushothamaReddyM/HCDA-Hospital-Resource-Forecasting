import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Load hospital dataset
# ---------------------------------
data = pd.read_csv("../data/h_data.csv")

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# ---------------------------------
# 2. Prepare data for Prophet
# ---------------------------------
df = data.rename(columns={
    'date': 'ds',
    'patients': 'y'
})

# Ensure no negative values
df['y'] = df['y'].clip(lower=0)

# ---------------------------------
# 3. Initialize Prophet model
# ---------------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add external regressors
model.add_regressor('temperature')
model.add_regressor('month')

# ---------------------------------
# 4. Train the model (ML Training)
# ---------------------------------
model.fit(df[['ds', 'y', 'temperature', 'month']])

print("✅ Forecasting model trained successfully")

# ---------------------------------
# 5. Create future dates (next 30 days)
# ---------------------------------
future = model.make_future_dataframe(periods=30)

# Add future regressor values
future['month'] = future['ds'].dt.month
future['temperature'] = data['temperature'].iloc[-7:].mean()

# ---------------------------------
# 6. Generate forecast
# ---------------------------------
forecast = model.predict(future)

# Clip negative predictions
forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[
    ['yhat', 'yhat_lower', 'yhat_upper']
].clip(lower=0)

# ---------------------------------
# 7. Plot forecast
# ---------------------------------
fig = model.plot(forecast)
plt.title("Hospital Patient Forecast for Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# ---------------------------------
# 8. Print next 30-day predictions
# ---------------------------------
print("\n📅 Next 30 Days Patient Forecast:")
print(
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    .tail(30)
    .to_string(index=False)
)

# ---------------------------------
# 9. Save forecast output (optional but recommended)
# ---------------------------------
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_csv(
    "../data/forecast_output.csv", index=False
)

print("\n📁 Forecast saved to data/forecast_output.csv")
