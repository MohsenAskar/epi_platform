# 20_time_series_analysis.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

########################
# Page Title
########################
st.title("Time Series Analysis")

########################
# Main Selectbox
########################
analysis_type = st.selectbox(
    "Select a Time Series Analysis Topic",
    [
        "Exploratory Analysis & Stationarity",
        "ARIMA Modeling",
        "Seasonal Decomposition"
    ]
)

###############################################################################
# Helper Function: Simulate a Seasonal Time Series with Trend
###############################################################################
def simulate_time_series(n_points=100, trend_slope=0.1, season_period=12, season_amp=5, noise_std=1.0):
    """Simulate a basic time series with optional trend & seasonality."""
    t = np.arange(n_points)
    # Linear trend
    trend = trend_slope * t
    # Seasonal component (sine wave with given amplitude, period)
    season = season_amp * np.sin(2 * np.pi * t / season_period)
    # Noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_std, n_points)
    data = trend + season + noise
    return data

###############################################################################
# 1. Exploratory Analysis & Stationarity
###############################################################################
if analysis_type == "Exploratory Analysis & Stationarity":
    st.header("Exploratory Analysis & Stationarity")

    st.write("""
    **Scenario**: We have a time series with potential trend and seasonality. 
    We can visualize it, then perform a stationarity test like the 
    Augmented Dickey-Fuller (ADF) test.
    """)

    st.subheader("Simulation Setup")
    n_points = st.slider("Number of Time Points", 50, 300, 100)
    trend_slope = st.slider("Trend Slope", 0.0, 1.0, 0.1, 0.05)
    season_period = st.slider("Season Period", 2, 50, 12)
    season_amp = st.slider("Season Amplitude", 0.0, 10.0, 5.0, 0.5)
    noise_std = st.slider("Noise Std Dev", 0.0, 5.0, 1.0, 0.1)

    # Simulate
    data = simulate_time_series(
        n_points=n_points,
        trend_slope=trend_slope,
        season_period=season_period,
        season_amp=season_amp,
        noise_std=noise_std
    )
    time_index = np.arange(1, n_points + 1)

    df = pd.DataFrame({"t": time_index, "value": data})

    st.write("### Time Series Data (first rows)")
    st.dataframe(df.head(10))

    # Plot the data
    fig = px.line(
        df, x="t", y="value",
        title="Simulated Time Series",
        labels={"t": "Time", "value": "Value"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Augmented Dickey-Fuller test
    adf_res = adfuller(data, autolag='AIC')
    adf_stat = adf_res[0]
    p_value = adf_res[1]

    st.write("### Augmented Dickey-Fuller Test")
    st.write(f"**ADF Statistic**: {adf_stat:.4f}")
    st.write(f"**p-value**: {p_value:.4f}")

    conclusion = "Stationary" if p_value < 0.05 else "Non-Stationary"
    st.write(f"**Conclusion**: Series is likely **{conclusion}** (at 5% level).")

    st.markdown("""
    **Interpretation**: If the p-value from ADF is < 0.05, we typically 
    reject the null hypothesis (unit root present) and conclude the 
    series is stationary. Otherwise, we view it as non-stationary 
    (e.g., has a trend, seasonal component).
    """)

###############################################################################
# 2. ARIMA Modeling
###############################################################################
elif analysis_type == "ARIMA Modeling":
    st.header("ARIMA Modeling")

    st.write("""
    **Scenario**: We fit an ARIMA model (p,d,q) to a time series, 
    then forecast future points.
    """)

    st.subheader("Simulation Setup")
    n_points = st.slider("Number of Time Points", 50, 300, 100)
    trend_slope = st.slider("Trend Slope", 0.0, 1.0, 0.0, 0.05)  # default 0 for simpler ARIMA
    season_period = st.slider("Season Period", 2, 50, 12)
    season_amp = st.slider("Season Amplitude", 0.0, 10.0, 0.0, 0.5)  # default 0 for simpler ARIMA
    noise_std = st.slider("Noise Std Dev", 0.0, 5.0, 1.0, 0.1)

    st.write("We’ll keep trend & season amplitude at zero or small for a simpler ARIMA demonstration.")
    
    data = simulate_time_series(
        n_points=n_points,
        trend_slope=trend_slope,
        season_period=season_period,
        season_amp=season_amp,
        noise_std=noise_std
    )
    time_index = np.arange(1, n_points + 1)
    df = pd.DataFrame({"t": time_index, "value": data})

    # ARIMA order
    st.subheader("ARIMA Order (p, d, q)")
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
    with col2:
        d = st.number_input("d (Diff order)", min_value=0, max_value=5, value=0)
    with col3:
        q = st.number_input("q (MA order)", min_value=0, max_value=5, value=0)

    # Fit ARIMA
    model = ARIMA(data, order=(p, d, q))
    results = model.fit()

    st.write("### Model Summary")
    st.text(results.summary())

    # Forecast horizon
    forecast_horizon = st.slider("Forecast Steps", 1, 50, 10)
    forecast_res = results.get_forecast(steps=forecast_horizon)
    forecast_df = forecast_res.summary_frame()

    # Combine original and forecast
    t_future = np.arange(n_points + 1, n_points + forecast_horizon + 1)
    df_forecast = pd.DataFrame({
        "t": t_future,
        "forecast": forecast_df["mean"],
        "lower": forecast_df["mean_ci_lower"],
        "upper": forecast_df["mean_ci_upper"]
    })

    # Plot actual vs forecast
    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["value"], mode='lines', name='Observed'
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=df_forecast["t"], y=df_forecast["forecast"], mode='lines+markers', 
        name='Forecast'
    ))
    # CI band
    fig.add_trace(go.Scatter(
        x=pd.concat([df_forecast["t"], df_forecast["t"][::-1]]),
        y=pd.concat([df_forecast["upper"], df_forecast["lower"][::-1]]),
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title="ARIMA Model Forecast",
        xaxis_title="Time",
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation**: The ARIMA model captures autocorrelation (AR(p)), 
    differencing (d), and moving average (MA(q)) components. We generate 
    a forecast for future steps and display confidence intervals.
    """)

###############################################################################
# 3. Seasonal Decomposition
###############################################################################
elif analysis_type == "Seasonal Decomposition":
    st.header("Seasonal Decomposition")

    st.write("""
    **Scenario**: We want to decompose a time series into trend, 
    seasonal, and residual components (using, e.g., STL or classical 
    decomposition).
    """)

    st.subheader("Simulation Setup")
    n_points = st.slider("Number of Time Points", 50, 300, 120)
    trend_slope = st.slider("Trend Slope", 0.0, 1.0, 0.2, 0.05)
    season_period = st.slider("Season Period", 2, 50, 12)
    season_amp = st.slider("Season Amplitude", 0.0, 10.0, 5.0, 0.5)
    noise_std = st.slider("Noise Std Dev", 0.0, 5.0, 1.0, 0.1)

    data = simulate_time_series(
        n_points=n_points,
        trend_slope=trend_slope,
        season_period=season_period,
        season_amp=season_amp,
        noise_std=noise_std
    )
    df = pd.DataFrame({"value": data})

    st.write("### Decomposition Method")
    decomposition_method = st.selectbox("Select Decomposition Type", ["additive", "multiplicative"])

    # We'll use classical seasonal_decompose from statsmodels
    # (requires freq or period; here we pass in period=season_period if it makes sense)
    result = seasonal_decompose(df["value"], model=decomposition_method, period=season_period)

    # Convert results to dataframes
    trend_comp = result.trend
    seasonal_comp = result.seasonal
    resid_comp = result.resid

    st.write("### Decomposition Plots")
    # Plot each component
    # We'll build a figure with subplots
    time_index = np.arange(1, len(data)+1)

    fig = go.Figure()

    # Original series
    fig.add_trace(go.Scatter(
        x=time_index, y=df["value"],
        mode='lines', name='Original'
    ))

    # Trend
    fig.add_trace(go.Scatter(
        x=time_index, y=trend_comp,
        mode='lines', name='Trend'
    ))

    # Seasonal
    fig.add_trace(go.Scatter(
        x=time_index, y=seasonal_comp,
        mode='lines', name='Seasonal'
    ))

    # Residual
    fig.add_trace(go.Scatter(
        x=time_index, y=resid_comp,
        mode='lines', name='Residual'
    ))

    fig.update_layout(
        title="Seasonal Decomposition (Trend, Seasonality, Residual)",
        xaxis_title="Time",
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation**: Decomposition separates the observed series into a 
    smooth trend, a repeating seasonal pattern, and a residual (random) part. 
    The "period" used here is the approximate seasonal cycle length.
    """)

########################
# Method Description
########################
st.header("Method Descriptions")
if analysis_type == "Exploratory Analysis & Stationarity":
    st.write("""
    Stationarity is a key assumption for many time series methods. 
    Visualizing and applying tests like **ADF** helps assess whether 
    the series is stationary or if transformations (e.g., differencing) 
    are needed.
    """)
elif analysis_type == "ARIMA Modeling":
    st.write("""
    **ARIMA(p,d,q)** models capture autoregression (p), differencing (d), 
    and moving average (q) components. They are commonly used for 
    short-term forecasting of stationary or differenced-stationary series.
    """)
elif analysis_type == "Seasonal Decomposition":
    st.write("""
    Decomposition splits a time series into **trend**, 
    **seasonal**, and **residual** components, aiding interpretation 
    of underlying patterns.
    """)

########################
# References
########################
st.header("References")
st.write("""
1. **Stationarity**: 
   - Fuller WA. Introduction to Statistical Time Series. Wiley.
2. **ARIMA**: 
   - Box, Jenkins, and Reinsel. Time Series Analysis: Forecasting and Control.
3. **Seasonal Decomposition**: 
   - Cleveland RB, Cleveland WS, McRae JE, et al. STL: A seasonal-trend decomposition procedure based on Loess.
4. **Statsmodels**: 
   - [https://www.statsmodels.org/](https://www.statsmodels.org/)
""")

st.header("Check your understanding")

if analysis_type == "Exploratory Analysis & Stationarity":
    q1 = st.radio(
        "Which test is commonly used to check if a time series is stationary?",
        [
            "Dickey-Fuller Test",
            "Chi-Square Test",
            "Kolmogorov-Smirnov Test",
            "Shapiro-Wilk Test"
        ]
    )
    if q1 == "Dickey-Fuller Test":
        st.success("Correct! The **Augmented Dickey-Fuller (ADF) test** is used to check for stationarity.")
    else:
        st.error("Not quite. The correct answer is **Dickey-Fuller Test**.")

    q2 = st.radio(
        "What does a **high p-value** (> 0.05) in the Augmented Dickey-Fuller (ADF) test suggest?",
        [
            "The series is stationary",
            "The series is non-stationary",
            "The series has no autocorrelation",
            "The series has seasonality"
        ]
    )
    if q2 == "The series is non-stationary":
        st.success("Correct! A high p-value suggests failure to reject the null hypothesis, meaning the series is **non-stationary**.")
    else:
        st.error("Not quite. A **high p-value** in ADF means the series is likely **non-stationary**.")

# Quiz for ARIMA Modeling
elif analysis_type == "ARIMA Modeling":
    q3 = st.radio(
        "What does the parameter **d** represent in an ARIMA(p,d,q) model?",
        [
            "Number of autoregressive terms",
            "Number of differencing steps to make the series stationary",
            "Number of moving average terms",
            "Seasonal period"
        ]
    )
    if q3 == "Number of differencing steps to make the series stationary":
        st.success("Correct! The **d** parameter represents how many times the series is differenced to become stationary.")
    else:
        st.error("Not quite. **d** represents the differencing steps required for stationarity.")

    q4 = st.radio(
        "If a time series has strong autocorrelation, which ARIMA parameter should be adjusted?",
        [
            "p (Autoregressive order)",
            "d (Differencing order)",
            "q (Moving average order)",
            "None of the above"
        ]
    )
    if q4 == "p (Autoregressive order)":
        st.success("Correct! The **p** parameter controls how many past observations influence the current value.")
    else:
        st.error("Not quite. **p** (autoregressive order) captures dependency on past values.")

# Quiz for Seasonal Decomposition
elif analysis_type == "Seasonal Decomposition":
    q5 = st.radio(
        "Which component is **not** part of time series decomposition?",
        [
            "Trend",
            "Seasonality",
            "Residuals",
            "Regression Coefficients"
        ]
    )
    if q5 == "Regression Coefficients":
        st.success("Correct! Decomposition consists of **Trend, Seasonality, and Residuals**, but **Regression Coefficients** are not part of this.")
    else:
        st.error("Not quite. Time series decomposition consists of **Trend, Seasonality, and Residuals**.")

    q6 = st.radio(
        "When should a **multiplicative** decomposition be used instead of an additive one?",
        [
            "When seasonal variations are constant over time",
            "When seasonal variations increase proportionally with the trend",
            "When the trend is decreasing",
            "When there is no noise in the data"
        ]
    )
    if q6 == "When seasonal variations increase proportionally with the trend":
        st.success("Correct! A **multiplicative decomposition** is used when seasonality varies **proportionally** to the trend.")
    else:
        st.error("Not quite. Use **multiplicative decomposition** when seasonal effects are proportional to the trend.")

st.write("Great job! Keep practicing to master Time Series Analysis. 🚀")

