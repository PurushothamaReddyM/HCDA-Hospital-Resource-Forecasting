import pandas as pd
from prophet import Prophet


def detect_anomalies(
    df: pd.DataFrame,
    value_column: str,
    threshold_std: float = 2.0
):
    """
    Residual-based anomaly detection using Prophet forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - 'date' column (datetime or convertible)
        - value_column (target numeric column)

    value_column : str
        Column on which anomaly detection is performed.

    threshold_std : float, default=3.0
        Number of residual standard deviations used to flag anomalies.

    Returns
    -------
    pd.DataFrame
        Original dataframe +:
        - predicted       : Prophet expected value
        - residual        : actual - predicted
        - upper_bound     : +threshold
        - lower_bound     : -threshold
        - is_anomaly      : True/False
        - label           : "Anomaly" / "Normal"
    """

    # ==================================================
    # 1️⃣ COPY & BASIC CLEANING
    # ==================================================
    data = df.copy()

    # Ensure datetime format
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Remove invalid or missing rows
    data = data.dropna(subset=["date", value_column])

    # Sort by date (VERY IMPORTANT for time series)
    data = data.sort_values("date").reset_index(drop=True)

    # ==================================================
    # 2️⃣ PREPARE DATA FOR PROPHET
    # ==================================================
    prophet_df = data.rename(columns={
        "date": "ds",
        value_column: "y"
    }).copy()

    # Add month regressor (same as forecast model)
    prophet_df["month"] = prophet_df["ds"].dt.month

    # Prevent negative values
    prophet_df["y"] = prophet_df["y"].clip(lower=0)

    # ==================================================
    # 2️⃣ RESOURCE-WISE REGRESSORS
    # ==================================================
    if value_column == "patients":
        regressors = ["temperature", "month"]

    elif value_column == "beds_occupied":
        regressors = ["temperature", "month"]

    elif value_column == "icu_occupied":
        regressors = ["patients", "emergencies", "temperature", "month"]

    elif value_column == "oxygen_usage":
        regressors = ["patients", "icu_occupied", "emergencies", "temperature", "month"]

    # Keep original logic + add regressors
    prophet_df = prophet_df[["ds", "y"] + regressors]

    # ==================================================
    # 3️⃣ TRAIN PROPHET MODEL
    # ==================================================
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    # Same regressors as forecast model
    for reg in regressors:
        model.add_regressor(reg)

    model.fit(prophet_df[["ds", "y"] + regressors])

    # ==================================================
    # 4️⃣ PREDICT ON HISTORICAL DATES
    # ==================================================
    forecast = model.predict(
        prophet_df[["ds"] + regressors]
    )

    # Add predicted values to dataframe
    data["predicted"] = forecast["yhat"].values

    # ==================================================
    # 5️⃣ RESIDUAL CALCULATION
    # ==================================================
    data["residual"] = data[value_column] - data["predicted"]

    # Residual standard deviation
    std_residual = data["residual"][:-1].std()

    # Safety if std becomes zero
    if pd.isna(std_residual) or std_residual == 0:
        std_residual = 1

    # ==================================================
    # 6️⃣ THRESHOLD BOUNDS (± k·σ)
    # ==================================================
    upper_bound = threshold_std * std_residual
    lower_bound = -threshold_std * std_residual

    data["upper_bound"] = upper_bound
    data["lower_bound"] = lower_bound

    # ==================================================
    # 7️⃣ ANOMALY DETECTION
    # ==================================================
    data["is_anomaly"] = (
        (data["residual"] > upper_bound) |
        (data["residual"] < lower_bound)
    )

    # Human-readable labels
    data["label"] = data["is_anomaly"].map({
        True: "Anomaly",
        False: "Normal"
    })

    return data