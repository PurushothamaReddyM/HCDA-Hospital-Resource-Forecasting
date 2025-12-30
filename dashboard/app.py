import streamlit as st
import pandas as pd
from prophet import Prophet

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="HCDA – Hospital Forecasting",
    layout="wide"
)

# =====================================================
# CUSTOM CSS (FOR POLISHED UI)
# =====================================================
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
}
.sub-title {
    font-size: 16px;
    color: #6c757d;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
}
.card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown('<div class="main-title">🏥 Hospital Resource Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-based patient load forecasting using Machine Learning (Prophet)</div>', unsafe_allow_html=True)
st.write("")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("../data/h_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

data = load_data()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("⚙️ Forecast Controls")

forecast_days = st.sidebar.slider(
    "Forecast Period (Days)",
    min_value=7,
    max_value=30,
    value=30
)

run_forecast = st.sidebar.button("🚀 Run Forecast")

# =====================================================
# KPI CARDS
# =====================================================
avg_patients = int(data["patients"].mean())
max_patients = int(data["patients"].max())
latest_patients = int(data["patients"].iloc[-1])

k1, k2, k3 = st.columns(3)

with k1:
    st.metric("📊 Average Patients", avg_patients)

with k2:
    st.metric("📈 Peak Patients", max_patients)

with k3:
    st.metric("🕒 Latest Count", latest_patients)

st.divider()

# =====================================================
# MAIN DASHBOARD LAYOUT
# =====================================================
col1, col2 = st.columns([2, 1])

# -----------------------------
# Historical Trend
# -----------------------------
with col1:
    st.markdown('<div class="section-title">📊 Historical Patient Trend</div>', unsafe_allow_html=True)
    st.line_chart(data.set_index("date")["patients"])

# -----------------------------
# Statistics Panel
# -----------------------------
with col2:
    st.markdown('<div class="section-title">📈 Current Statistics</div>', unsafe_allow_html=True)
    st.metric("Average Patients", avg_patients)
    st.metric("Maximum Patients", max_patients)
    st.metric("Latest Patient Count", latest_patients)

# =====================================================
# FORECASTING SECTION
# =====================================================
if run_forecast:
    st.divider()
    st.markdown(
        f'<div class="section-title">🔮 Patient Forecast for Next {forecast_days} Days</div>',
        unsafe_allow_html=True
    )

    # -----------------------------
    # Prepare data for Prophet
    # -----------------------------
    df_prophet = data.rename(columns={
        "date": "ds",
        "patients": "y"
    })

    df_prophet["y"] = df_prophet["y"].clip(lower=0)

    # -----------------------------
    # Initialize ML Model
    # -----------------------------
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    # Add regressors
    model.add_regressor("temperature")
    model.add_regressor("month")

    # -----------------------------
    # Train ML Model
    # -----------------------------
    model.fit(df_prophet[["ds", "y", "temperature", "month"]])

    # -----------------------------
    # Future Data
    # -----------------------------
    future = model.make_future_dataframe(periods=forecast_days)
    future["month"] = future["ds"].dt.month
    future["temperature"] = data["temperature"].iloc[-7:].mean()

    # -----------------------------
    # Forecast
    # -----------------------------
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    # -----------------------------
    # Forecast Graph
    # -----------------------------
    st.markdown('<div class="section-title">📈 Forecast Trend</div>', unsafe_allow_html=True)
    st.line_chart(forecast.set_index("ds")["yhat"])

    # -----------------------------
    # Forecast Table
    # -----------------------------
    st.markdown('<div class="section-title">📋 Forecast Table</div>', unsafe_allow_html=True)
    st.dataframe(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .tail(forecast_days)
        .rename(columns={
            "ds": "Date",
            "yhat": "Predicted Patients",
            "yhat_lower": "Lower Estimate",
            "yhat_upper": "Upper Estimate"
        })
    )

    # -----------------------------
    # ALERT SYSTEM
    # -----------------------------
    peak_prediction = forecast["yhat"].max()
    threshold = avg_patients * 1.2

    if peak_prediction > threshold:
        st.error(
            f"🚨 **High patient load expected!**\n\n"
            f"Peak prediction: **{int(peak_prediction)} patients**.\n\n"
            f"Recommended actions:\n"
            f"- Increase bed availability\n"
            f"- Prepare ICU capacity\n"
            f"- Ensure oxygen stock\n"
            f"- Adjust staff scheduling"
        )
    else:
        st.success(
            "✅ **Patient load expected to remain normal.**\n\n"
            "No emergency resource expansion required."
        )

else:
    st.info(
        "👈 Use the **sidebar controls** to select forecast duration "
        "and click **Run Forecast** to generate predictions."
    )
