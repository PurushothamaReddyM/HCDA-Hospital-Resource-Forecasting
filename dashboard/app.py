import sys
from xgboost import XGBRegressor
import os
import smtplib
from email.mime.text import MIMEText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# =====================================================
# PROJECT PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.append(PROJECT_ROOT)

# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from PIL import Image
from streamlit_autorefresh import st_autorefresh
import time
from models.anomaly_detection import detect_anomalies

# =====================================================
# EMAIL ALERT FUNCTION
# =====================================================
def send_email_alert(subject, body):
    sender_email = st.secrets["EMAIL_USER"]
    sender_password = st.secrets["EMAIL_PASSWORD"]
    receiver_email = st.secrets["RECEIVER_EMAIL"]
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        st.success("✅ Email sent successfully")

    except Exception as e:
        st.warning(f"Email alert failed: {e}")

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="CareCast – Hospital Resource Forecasting",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# REAL-TIME AUTO REFRESH
# =====================================================
st_autorefresh(interval=5000, key="realtime_refresh")

# =====================================================
# SESSION STATE
# =====================================================
if "last_alert_date" not in st.session_state:
    st.session_state.last_alert_date = None

if "run_forecast" not in st.session_state:
    st.session_state.run_forecast = False

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

/* VERY LIGHT GREEN BACKGROUND */
.stApp {
    background-color: #eefaf7;
}

/* prevent scroll leaking to page */
.stPlotlyChart {
    overscroll-behavior: contain !important;
}

/* BORDER + SHADOW style */
.border-shadow {
    border: 2px solid black;
    box-shadow: 5px 5px 0px black;
}

/* Text Styles */
.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #000000;
}
.sub-title {
    font-size: 16px;
    color: #6c757d;
    margin-bottom: 25px;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 30px;
    color: #000000;
}

.card {
    background-color: #ffffff;
    padding: 18px;
    border: 2px solid black;
    box-shadow: 5px 5px 0px black;
}

.hero-title {
    font-size: 34px;
    font-weight: 700;
    color: #000000;
}
.hero-text {
    font-size: 17px;
    color: #495057;
    line-height: 1.6;
}
.hero-container {
    padding-top: 40px;
    padding-bottom: 40px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    '<div class="main-title">🏥 CareCast – Hospital Resource Forecasting Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">ML-based forecasting of Patients, Beds, ICU, and Oxygen using Prophet</div>',
    unsafe_allow_html=True
)

st.success("🟢 Real-Time Monitoring Active (Auto refresh every 5 sec)")

# =====================================================
# HERO SECTION
# =====================================================
col1, col2 = st.columns([1, 1.2], gap="large")

hero_path = os.path.join(PROJECT_ROOT, "assets", "hero.png")

with col1:
    if os.path.exists(hero_path):
        st.image(hero_path, width="stretch")
    else:
        st.warning("hero.png not found")

with col2:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">
            AI-Driven Hospital Resource Forecasting
        </div>
        <div class="hero-text">
            Predict patient inflow, bed occupancy, ICU utilization, and oxygen demand
            using machine learning models to enable proactive hospital planning,
            early risk detection, and efficient healthcare resource management.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(ttl=5)
def load_data():
    csv_path = os.path.join(PROJECT_ROOT, "data", "h_data.csv")

    if not os.path.exists(csv_path):
        st.error(f"CSV file not found: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df

data = load_data()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
with st.sidebar:
    st.header("⚙️ Control Panel")

    resource_map = {
        "Patients": "patients",
        "Beds Occupied": "beds_occupied",
        "ICU Occupancy": "icu_occupied",
        "Oxygen Usage": "oxygen_usage"
    }

    resource_name = st.selectbox("Select Resource", list(resource_map.keys()))
    target_column = resource_map[resource_name]

    forecast_days = st.slider("Forecast Period (Days)", 7, 30, 30)

    st.markdown("---")

    if st.button("🚀 Run Forecast", width="stretch"):
        st.session_state.run_forecast = True

    run_forecast = st.session_state.run_forecast

    # =====================================================
    # MANUAL INPUT
    # =====================================================
    st.markdown("---")
    st.subheader("➕ Add New Hospital Record")

    manual_patients = st.number_input("Patients", min_value=0, value=150)
    manual_beds = st.number_input("Beds Occupied", min_value=0, value=120)
    manual_icu = st.number_input("ICU Occupancy", min_value=0, value=18)
    manual_oxygen = st.number_input("Oxygen Usage", min_value=0, value=400)
    manual_temp = st.number_input("Temperature", min_value=0, value=30)
    manual_rain = st.number_input("Rainfall", min_value=0, value=5)
    manual_emergency = st.number_input("Emergencies", min_value=0, value=12)
    manual_admissions = st.number_input("Admissions", min_value=0, value=45)
    manual_discharges = st.number_input("Discharges", min_value=0, value=35)
    manual_readmissions = st.number_input("Readmissions", min_value=0, value=20)
    

    if st.button("✅ Add Record", width="stretch"):
        if manual_readmissions > manual_admissions:
            st.error("❌ Readmissions cannot exceed admissions")
            st.stop()
        csv_path = os.path.join(PROJECT_ROOT, "data", "h_data.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        existing = pd.read_csv(csv_path)

        if existing.empty:
            new_date = pd.Timestamp.today()
        else:
            new_date = pd.to_datetime(existing["date"]).max() + pd.Timedelta(days=1)

        new_row = pd.DataFrame([{
            "date": new_date.strftime("%Y-%m-%d"),
            "patients": manual_patients,
            "admissions": manual_admissions,
            "readmissions": manual_readmissions,
            "discharges": manual_discharges,
            "beds_occupied": manual_beds,
            "icu_occupied": manual_icu,
            "oxygen_usage": manual_oxygen,
            "emergencies": manual_emergency,
            "temperature": manual_temp,
            "rainfall": manual_rain,
            "month": new_date.month
        }])

        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(csv_path, index=False)

        st.success(f"Record added for {new_date.date()}")
        st.rerun()

OXYGEN_CAPACITY = 500

# =====================================================
# KPI SECTION
# =====================================================
avg_value = round(data[target_column].mean(), 2)
max_value = int(data[target_column].max())
latest_value = int(data[target_column].iloc[-1])

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="card">
        <h4>📊 Average</h4>
        <h2>{avg_value}</h2>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="card">
        <h4>📈 Peak</h4>
        <h2>{max_value}</h2>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="card">
        <h4>🕒 Latest</h4>
        <h2>{latest_value}</h2>
    </div>
    """, unsafe_allow_html=True)

if target_column == "oxygen_usage" and latest_value > 0.85 * OXYGEN_CAPACITY:
    st.warning("⚠️ Oxygen usage nearing maximum capacity")

st.divider()

# =====================================================
# HISTORICAL TREND
# =====================================================
st.markdown(
    f'<div class="section-title">📊 Historical {resource_name} Trend</div>',
    unsafe_allow_html=True
)

hist_fig = go.Figure()

hist_fig.add_trace(go.Scatter(
    x=data["date"],
    y=data[target_column],
    mode="lines",
    line=dict(width=3),
    name="Historical"
))

hist_fig.update_layout(
    template="plotly_white",
    height=350,
    hovermode="x unified"
)

st.plotly_chart(
    hist_fig,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
        ]
    }
)
st.markdown(
    '<div class="section-title">📉 Readmission Analysis (HRR)</div>',
    unsafe_allow_html=True
)

# Latest values
latest_readmissions = data["readmissions"].iloc[-1]
latest_admissions = data["admissions"].iloc[-1]

if latest_admissions > 0:
    readmission_rate = latest_readmissions / latest_admissions
else:
    readmission_rate = 0


# Standard expected rate (from theory)
expected_rate = 0.20

expected_readmissions = latest_admissions * expected_rate

# ERR Calculation
if expected_readmissions > 0:
    ERR = latest_readmissions / expected_readmissions
else:
    ERR = 0

# ---- DISPLAY ---- #
c1, c2, c3 = st.columns(3)

c1.metric("Readmission Rate", f"{readmission_rate:.2%}")
c2.metric("Expected Rate", "20%")
c3.metric("ERR", round(ERR, 2))

# ---- STATUS ---- #
if ERR > 1:
    st.error("⚠️ High Readmission Rate → Penalty Risk (HRR Program)")
else:
    st.success("✅ Readmission Rate within acceptable range")

hospital_revenue = 500000   # You can adjust this

if ERR > 1:
    penalty_percentage = min((ERR - 1) * 3, 3)   # max 3%
    penalty_cost = (penalty_percentage / 100) * hospital_revenue

    p1, p2 = st.columns(2)
    p1.metric("Penalty %", f"{penalty_percentage:.2f}%")
    p2.metric("Penalty Cost", f"₹{round(penalty_cost,2)}")

    st.warning(f"💰 Estimated Penalty Cost: ₹{round(penalty_cost,2)}")

else:
    st.success("💰 No financial penalty")


# =====================================================
# 🏥 HVBP (Hospital Value-Based Performance)
# =====================================================

st.markdown(
    '<div class="section-title">🏥 Hospital Performance Score (HVBP)</div>',
    unsafe_allow_html=True
)

# -------------------------------
# NORMALIZED SCORES (0–100)
# -------------------------------

# 1. Readmission Score (lower is better)
readmission_score = max(0, 100 - (readmission_rate * 100))

# 2. Bed Utilization (optimal ~85%)
patients_latest = data["patients"].iloc[-1]

if patients_latest > 0:
    bed_util = data["beds_occupied"].iloc[-1] / patients_latest
    icu_util = data["icu_occupied"].iloc[-1] / patients_latest
    emergency_rate = data["emergencies"].iloc[-1] / patients_latest
else:
    bed_util = icu_util = emergency_rate = 0
bed_score = max(0, 100 - abs(bed_util - 0.85) * 100)

# 3. ICU Utilization (optimal ~10%)

icu_score = max(0, 100 - abs(icu_util - 0.10) * 100)

# 4. Emergency Handling (lower emergencies = better)

emergency_score = max(0, 100 - (emergency_rate * 100))

# -------------------------------
# FINAL HVBP SCORE (Weighted)
# -------------------------------
hvbp_score = (
    0.4 * readmission_score +
    0.2 * bed_score +
    0.2 * icu_score +
    0.2 * emergency_score
)

hvbp_score = round(hvbp_score, 2)

# -------------------------------
# DISPLAY
# -------------------------------
h1, h2 = st.columns(2)

h1.metric("HVBP Score", f"{hvbp_score}/100")

# Rating
if hvbp_score >= 90:
    h2.success("🌟 Excellent Performance")
elif hvbp_score >= 80:
    h2.success("👍 Good Performance")
elif hvbp_score >= 65:
    h2.warning("⚠️ Average Performance")
else:
    h2.error("🚨 Poor Performance")
# =====================================================
# 🦠 HAC (Hospital Acquired Conditions)
# =====================================================

st.markdown(
    '<div class="section-title">🦠 Hospital Acquired Conditions (HAC)</div>',
    unsafe_allow_html=True
)

# HAC Risk based on ICU + Emergencies
hac_rate = (icu_util * 0.6) + (emergency_rate * 0.4)

hac_score = max(0, 100 - (hac_rate * 100))

h3, h4 = st.columns(2)

h3.metric("HAC Score", f"{hac_score:.2f}/100")

if hac_score >= 80:
    h4.success("🟢 Low Risk (Good)")
elif hac_score >= 60:
    h4.warning("⚠️ Moderate Risk")
else:
    h4.error("🚨 High Risk (Poor)")



# =====================================================
# FORECASTING
# =====================================================
if run_forecast:

    with st.spinner("Running AI Forecast Model..."):
        time.sleep(1)

    st.markdown(
        f'<div class="section-title">🔮 {resource_name} Forecast (Next {forecast_days} Days)</div>',
        unsafe_allow_html=True
    )

    df_prophet = data.rename(columns={"date": "ds", target_column: "y"}).copy()
    df_prophet["y"] = df_prophet["y"].clip(lower=0)
    df_prophet["month"] = df_prophet["ds"].dt.month
    
    df_prophet["day"] = df_prophet["ds"].dt.day
    df_prophet["dayofweek"] = df_prophet["ds"].dt.dayofweek
    df_prophet["week"] = df_prophet["ds"].dt.isocalendar().week.astype(int)

    # =====================================================
    # RESOURCE-WISE REGRESSORS
    # =====================================================
    if target_column == "patients":
        regressors = ["temperature", "month"]

    elif target_column == "beds_occupied":
        regressors = ["temperature", "month"]

    elif target_column == "icu_occupied":
        regressors = ["patients", "emergencies", "temperature", "month"]

    elif target_column == "oxygen_usage":
        regressors = ["patients", "icu_occupied", "emergencies", "temperature", "month"]

    # =====================================================
    # TRAIN DATA
    # =====================================================
    train_data = df_prophet[["ds", "y"] + regressors].dropna()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    # =====================================================
    # ADD REGRESSORS DYNAMICALLY
    # =====================================================
    for reg in regressors:
        model.add_regressor(reg)

    model.fit(train_data)

    # =====================================================
    # FUTURE DATAFRAME
    # =====================================================
    future = model.make_future_dataframe(periods=forecast_days)
    future["month"] = future["ds"].dt.month

    if "temperature" in regressors:
        future["temperature"] = data["temperature"].iloc[-7:].mean()

    if "patients" in regressors:
        future["patients"] = data["patients"].iloc[-7:].mean()

    if "emergencies" in regressors:
        future["emergencies"] = data["emergencies"].iloc[-7:].mean()

    if "icu_occupied" in regressors:
        future["icu_occupied"] = data["icu_occupied"].iloc[-7:].mean()

    # =====================================================
    # FORECAST
    # =====================================================
    forecast = model.predict(future)
    
    historical_forecast = forecast.iloc[:len(df_prophet)]
    actual = df_prophet["y"].values
    predicted = historical_forecast["yhat"].values
    # MAE
    mae = mean_absolute_error(actual, predicted)
    # RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # MAPE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    feature_cols = ["month", "day", "dayofweek", "week"]
    X = df_prophet[feature_cols]
    y = df_prophet["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )
    
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    
    xgb_predictions = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    xgb_mape = np.mean(np.abs((y_test - xgb_predictions) / y_test)) * 100
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    next_forecast = forecast.tail(forecast_days)

    # =====================================================
    # PLOT
    # =====================================================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=next_forecast["ds"],
        y=next_forecast["yhat"],
        mode="lines+markers",
        line=dict(width=3),
        name="Prediction"
    ))

    fig.add_trace(go.Scatter(
        x=next_forecast["ds"],
        y=next_forecast["yhat_lower"],
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=next_forecast["ds"],
        y=next_forecast["yhat_upper"],
        fill="tonexty",
        fillcolor="rgba(13,110,253,0.2)",
        line=dict(width=0),
        name="Confidence Interval"
    ))

    fig.update_layout(
        template="plotly_white",
        height=450,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=resource_name
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown(
            '<div class="section-title">📊 Forecast Evaluation Metrics</div>',
            unsafe_allow_html=True
    )
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="card">
            <h4>MAE</h4>
            <h2>{round(mae, 2)}</h2>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="card">
            <h4>RMSE</h4>
            <h2>{round(rmse, 2)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class="card">
            <h4>MAPE</h4>
            <h2>{round(mape, 2)}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🤖 Model Comparison</div>',
        unsafe_allow_html=True
    )
    comparison_df = pd.DataFrame({
        "Model": ["Prophet", "XGBoost"],
        "MAE": [round(mae, 2), round(xgb_mae, 2)],
        "RMSE": [round(rmse, 2), round(xgb_rmse, 2)],
        "MAPE": [round(mape, 2), round(xgb_mape, 2)]
    })
    st.dataframe(comparison_df, width="stretch")

    # =====================================================
    # TABLE + IMAGE
    # =====================================================
    tcol1, tcol2 = st.columns([1, 1], vertical_alignment="center")

    with tcol1:
        st.markdown('<div class="section-title">📋 Forecast Table</div>', unsafe_allow_html=True)

        forecast_table = next_forecast.rename(columns={
            "ds": "Date",
            "yhat": f"Predicted {resource_name}",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound"
        })[["Date", f"Predicted {resource_name}", "Lower Bound", "Upper Bound"]]

        numeric_cols = forecast_table.select_dtypes(include=["number"]).columns
        forecast_table[numeric_cols] = forecast_table[numeric_cols].round(2)

        st.dataframe(forecast_table, width="stretch", height=520)

    with tcol2:
        context_path = os.path.join(PROJECT_ROOT, "assets", "context.png")
        if os.path.exists(context_path):
            context_img = Image.open(context_path)
            st.image(context_img, width="stretch")
        else:
            st.warning("context.png not found")

else:
    st.info("👈 Select a resource and click **Run Forecast** to begin.")

# =====================================================
# ANOMALY DETECTION
# =====================================================
st.divider()

st.markdown(
    f'<div class="section-title">🚨 Anomaly Detection – {resource_name}</div>',
    unsafe_allow_html=True
)

try:
    anomaly_df = detect_anomalies(data, target_column)

    total_anomalies = anomaly_df["is_anomaly"].sum()

    a1, a2 = st.columns(2)

    a1.metric("🚨 Total Anomalies Detected", int(total_anomalies))
    a2.metric("📊 Total Records", len(anomaly_df))

    latest_row = anomaly_df.tail(1)

    if latest_row["is_anomaly"].iloc[0]:
        anomaly_date = latest_row["date"].iloc[0]

        st.error(
            f"🚨 Latest anomaly detected on "
            f"{anomaly_date.strftime('%Y-%m-%d')}"
        )

        if st.session_state.last_alert_date != str(anomaly_date):
            email_body = f"""
Anomaly detected in {resource_name}
Date: {anomaly_date}
Value: {latest_row[target_column].iloc[0]}
"""

            send_email_alert(
                f"HCDA Alert - {resource_name}",
                email_body
            )

            st.session_state.last_alert_date = str(anomaly_date)

        if target_column == "patients":
            st.warning("📌 Recommendation: Increase OP staff and prepare emergency intake units.")

        elif target_column == "beds_occupied":
            st.warning("📌 Recommendation: Prepare additional bed allocation and discharge planning.")

        elif target_column == "icu_occupied":
            st.warning("📌 Recommendation: Alert ICU staff and reserve critical care beds.")

        elif target_column == "oxygen_usage":
            st.warning("📌 Recommendation: Refill oxygen cylinders and verify backup supply.")

    else:
        st.success("✅ No critical anomalies detected")

    st.markdown("### 📡 Latest Incoming Records")
    st.dataframe(anomaly_df.tail(10), width="stretch")

    anom_fig = go.Figure()

    anom_fig.add_trace(go.Scatter(
        x=anomaly_df[~anomaly_df["is_anomaly"]]["date"],
        y=anomaly_df[~anomaly_df["is_anomaly"]][target_column],
        mode="lines",
        name="Normal",
        line=dict(color="blue", width=2)
    ))

    anom_fig.add_trace(go.Scatter(
        x=anomaly_df[anomaly_df["is_anomaly"]]["date"],
        y=anomaly_df[anomaly_df["is_anomaly"]][target_column],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=10, symbol="x")
    ))

    anom_fig.update_layout(
        template="plotly_white",
        height=400,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=resource_name
    )

    st.plotly_chart(
        anom_fig,
        width="stretch",
        config={"displaylogo": False}
    )
except Exception as e:
    st.error(f"Anomaly detection failed: {e}")

st.info(
    "🔍 **Anomalies are detected using Prophet residual analysis.**\n\n"
    "Red ❌ points indicate abnormal spikes or drops compared to historical patterns.\n"
    "These may indicate sudden patient surges, equipment failure, or data irregularities."
)