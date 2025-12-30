import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Date range (1 year)
start_date = datetime(2024, 1, 1)
days = 365
dates = [start_date + timedelta(days=i) for i in range(days)]

data = []

base_patients = 120

for date in dates:
    month = date.month

    # Seasonal effect
    seasonal_factor = 1.0
    if month in [6, 7, 8]:        # summer
        seasonal_factor = 1.15
    elif month in [11, 12, 1]:    # winter
        seasonal_factor = 1.25

    # Random daily variation
    patients = int(base_patients * seasonal_factor + np.random.randint(-15, 20))
    patients = max(patients, 50)

    admissions = int(patients * np.random.uniform(0.25, 0.35))
    discharges = int(admissions * np.random.uniform(0.7, 0.9))

    beds_occupied = int(patients * np.random.uniform(0.75, 0.9))
    icu_occupied = int(patients * np.random.uniform(0.08, 0.15))
    oxygen_usage = int(icu_occupied * np.random.uniform(20, 30))

    emergencies = int(admissions * np.random.uniform(0.2, 0.4))

    # Weather
    temperature = np.random.randint(20, 40)
    rainfall = np.random.randint(0, 20)

    # Inject rare anomaly spikes
    if np.random.rand() < 0.03:
        patients += np.random.randint(40, 80)
        icu_occupied += np.random.randint(5, 10)
        oxygen_usage += np.random.randint(100, 200)
        emergencies += np.random.randint(10, 20)

    data.append([
        date.strftime("%Y-%m-%d"),
        patients,
        admissions,
        discharges,
        beds_occupied,
        icu_occupied,
        oxygen_usage,
        emergencies,
        temperature,
        rainfall,
        month
    ])

# Create DataFrame
columns = [
    "date",
    "patients",
    "admissions",
    "discharges",
    "beds_occupied",
    "icu_occupied",
    "oxygen_usage",
    "emergencies",
    "temperature",
    "rainfall",
    "month"
]

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("h_data.csv", index=False)

print("✅ Realistic hospital dataset generated: h_data.csv")
