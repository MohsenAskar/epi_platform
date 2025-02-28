# 17_environmental_occupational.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

########################
# Page Title
########################
st.title("Environmental and Occupational Epidemiology")

########################
# Main Selectbox
########################
topic = st.selectbox(
    "Select a Subtopic",
    [
        "Exposure Assessment Methods",
        "Dose-Response Relationships",
        "Time-Series Analysis of Environmental Data",
        "Spatial Analysis",
        "Risk Assessment Models",
        "Health Impact Assessment"
    ]
)

###############################################################################
# 1. Exposure Assessment Methods
###############################################################################
if topic == "Exposure Assessment Methods":
    st.header("Exposure Assessment Methods")
    st.write("""
    **Scenario**: Estimating exposure is crucial in environmental and 
    occupational studies. Approaches range from direct measures (personal 
    monitors) to area sampling, biomonitoring, or modeling based on distance 
    from a source.
    """)

    st.subheader("Simple Interactive Example: Mixed Personal & Area Measurements")

    # Fraction of exposure estimated from personal devices vs. from area sampling
    frac_personal = st.slider("Fraction Personal Monitoring", 0.0, 1.0, 0.5)

    # Mean/Std for personal monitor and area-level measures (dummy example)
    personal_mean = st.number_input("Personal Monitor Mean (µg/m³)", 0.0, 100.0, 25.0)
    personal_sd   = st.number_input("Personal Monitor SD (µg/m³)", 0.0, 50.0, 5.0)
    area_mean     = st.number_input("Area Sampling Mean (µg/m³)", 0.0, 100.0, 15.0)
    area_sd       = st.number_input("Area Sampling SD (µg/m³)", 0.0, 50.0, 4.0)

    # Simulate exposure estimates
    n_samples = 200
    rng = np.random.default_rng(seed=42)
    personal_data = rng.normal(personal_mean, personal_sd, n_samples)
    area_data = rng.normal(area_mean, area_sd, n_samples)

    # Weighted average for each "worker"
    combined_exposure = frac_personal * personal_data + (1 - frac_personal)* area_data
    df_exposure = pd.DataFrame({
        "Personal": personal_data,
        "Area": area_data,
        "Combined": combined_exposure
    })

    st.write("### Summary of Simulated Exposures")
    st.dataframe(df_exposure.describe().round(2))

    fig = px.histogram(
        df_exposure,
        x="Combined",
        nbins=30,
        title="Distribution of Combined Exposure Estimates"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 2. Dose-Response Relationships
###############################################################################
elif topic == "Dose-Response Relationships":
    st.header("Dose-Response Relationships")
    st.write("""
    **Scenario**: Dose-response models link exposure levels to health outcomes, 
    often taking forms like linear, log-linear, or threshold models.
    """)

    st.subheader("Parametric Dose-Response Model")
    dose = st.slider("Dose (exposure level)", 0.0, 100.0, 30.0)
    model_type = st.selectbox("Model Type", ["Linear", "Log-Linear", "Threshold"])
    
    # Example parameters
    beta0 = st.number_input("Intercept (beta0)", -10.0, 10.0, 0.0)
    beta1 = st.number_input("Slope (beta1)", -2.0, 2.0, 0.1)
    threshold = st.slider("Threshold (for Threshold Model)", 0.0, 50.0, 10.0)

    def linear_model(x):
        return beta0 + beta1*x

    def loglinear_model(x):
        # Avoid negative or zero in logs
        return np.exp(beta0 + beta1*np.log(x+1))

    def threshold_model(x, thresh):
        # If x < thresh, outcome=0, else outcome= (beta0 + beta1*(x-thresh))
        return 0 if x < thresh else (beta0 + beta1*(x - thresh))

    # Evaluate chosen model at the selected dose
    if model_type == "Linear":
        outcome = linear_model(dose)
    elif model_type == "Log-Linear":
        outcome = loglinear_model(dose)
    else:
        outcome = threshold_model(dose, threshold)

    st.write(f"**Predicted Outcome at Dose {dose:.1f}:** {outcome:.3f}")

    # Plot the chosen model over a range
    x_vals = np.linspace(0, 100, 200)
    if model_type == "Linear":
        y_vals = [linear_model(x) for x in x_vals]
    elif model_type == "Log-Linear":
        y_vals = [loglinear_model(x) for x in x_vals]
    else:
        y_vals = [threshold_model(x, threshold) for x in x_vals]

    fig = px.line(
        x=x_vals,
        y=y_vals,
        labels={"x": "Dose", "y": "Outcome"},
        title=f"{model_type} Dose-Response Curve"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 3. Time-Series Analysis of Environmental Data
###############################################################################
elif topic == "Time-Series Analysis of Environmental Data":
    st.header("Time-Series Analysis of Environmental Data")
    st.write("""
    **Scenario**: Daily air pollution levels or temperature data can be 
    associated with daily counts of hospital admissions or mortality. 
    A time-series approach looks at these data over time, examining lagged effects.
    """)

    st.subheader("Simple Interactive Time-Series Generation")

    n_days = st.slider("Number of Days to Simulate", 30, 365, 90)
    baseline_pollution = st.slider("Mean Pollution Level (PM2.5, µg/m³)", 0.0, 150.0, 30.0)
    baseline_outcome   = st.slider("Baseline Outcome Count (per day)", 0, 500, 50)
    pollution_effect   = st.slider("Pollution Effect on Outcome", 0.0, 1.0, 0.2)

    rng = np.random.default_rng(seed=1)
    # Simple random walk or AR(1) for pollution
    pollution = [baseline_pollution]
    for _ in range(n_days-1):
        next_val = pollution[-1] + rng.normal(0, 2)
        pollution.append(next_val)
    pollution = np.clip(pollution, 0, None)
    pollution = np.array(pollution)

    # Outcome ~ baseline + some fraction of pollution
    outcome = baseline_outcome + pollution_effect*pollution + rng.normal(0, 5, n_days)
    outcome = np.clip(outcome, 0, None)

    df_time = pd.DataFrame({
        "Day": np.arange(1, n_days+1),
        "Pollution": pollution,
        "Outcome": outcome
    })

    st.write("### Time-Series Data")
    st.dataframe(df_time.head(10))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_time["Day"], y=df_time["Pollution"],
                             mode='lines+markers', name='Pollution'))
    fig.add_trace(go.Scatter(x=df_time["Day"], y=df_time["Outcome"],
                             mode='lines+markers', name='Outcome',
                             yaxis='y2'))
    fig.update_layout(
        title="Simulated Pollution vs. Outcome over Time",
        xaxis_title="Day",
        yaxis=dict(title="Pollution (µg/m³)", side='left'),
        yaxis2=dict(title="Outcome Count", overlaying='y', side='right')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick correlation
    corr = np.corrcoef(df_time["Pollution"], df_time["Outcome"])[0,1]
    st.write(f"Correlation between Pollution & Outcome: {corr:.2f}")

###############################################################################
# 4. Spatial Analysis
###############################################################################
elif topic == "Spatial Analysis":
    st.header("Spatial Analysis")
    st.write("""
    **Scenario**: In environmental and occupational epidemiology, the location of 
    exposures (factories, farms, traffic) can affect health outcomes. 
    Spatial methods explore geographic patterns and clusters.
    """)

    st.subheader("Basic Spatial Distribution Visualization")
    n_points = st.slider("Number of Locations", 10, 500, 50)
    rng = np.random.default_rng(123)
    # Generate random lat/lon around some center
    lat_center = 40.0
    lon_center = -100.0
    latitudes  = lat_center + rng.normal(0, 0.5, n_points)
    longitudes = lon_center + rng.normal(0, 0.5, n_points)

    # Simulate an "exposure" measure that is higher near the center
    distance_from_center = np.sqrt((latitudes - lat_center)**2 + (longitudes - lon_center)**2)
    exposure_level = 50 - distance_from_center * 20 + rng.normal(0,2, n_points)

    df_spatial = pd.DataFrame({
        "Latitude": latitudes,
        "Longitude": longitudes,
        "Exposure": exposure_level
    })

    fig = px.scatter(
        df_spatial,
        x="Longitude",
        y="Latitude",
        color="Exposure",
        color_continuous_scale="Viridis",
        title="Spatial Exposure Distribution (Synthetic)",
        hover_data=["Exposure"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    For true spatial analysis, advanced methods (e.g., kriging, cluster detection, 
    spatial regression) are often employed using specialized libraries.
    """)

###############################################################################
# 5. Risk Assessment Models
###############################################################################
elif topic == "Risk Assessment Models":
    st.header("Risk Assessment Models")
    st.write("""
    **Scenario**: Risk assessment often involves dose-response functions, 
    reference doses, uncertainty factors, etc. We'll do a simplified 
    linear risk model demonstration.
    """)

    st.subheader("Linear Low-Dose Risk Model")

    # Let user set slope factor and reference dose
    slope_factor = st.number_input("Slope Factor (per mg/kg-day)", 0.0, 1.0, 0.01)
    dose_val = st.slider("Dose (mg/kg-day)", 0.0, 10.0, 1.0)
    
    # Simplified risk = slope_factor * dose_val
    # Real-world might also have a threshold, or use exponential.
    risk = slope_factor * dose_val
    st.write(f"**Estimated Excess Risk:** {risk:.4f}")

    st.write("""
    In practice, risk = 1 - exp(-slope * dose), or other forms might be used. 
    Regulatory agencies often define acceptable risk levels (e.g. 1 in 100,000).
    """)

    st.write("### Risk Curve")
    dose_range = np.linspace(0, 10, 100)
    risk_vals  = slope_factor * dose_range
    fig = px.line(
        x=dose_range,
        y=risk_vals,
        labels={"x": "Dose", "y": "Excess Risk"},
        title="Linear Dose-Risk Function"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 6. Health Impact Assessment
###############################################################################
elif topic == "Health Impact Assessment":
    st.header("Health Impact Assessment")
    st.write("""
    **Scenario**: Health Impact Assessment (HIA) estimates the overall burden 
    of an exposure in a population, integrating exposure prevalence, 
    dose-response, and population size.
    """)

    st.subheader("Simple HIA Calculation")

    population = st.number_input("Population Size", 1000, 10_000_000, 100_000)
    exposed_fraction = st.slider("Fraction of Population Exposed", 0.0, 1.0, 0.3)
    baseline_incidence = st.number_input("Baseline Incidence (cases per person)", 0.0, 1.0, 0.02)
    relative_risk = st.slider("Relative Risk (exposed vs unexposed)", 1.0, 5.0, 1.5)

    # Attributable fraction among exposed
    af_exposed = (relative_risk - 1) / relative_risk
    # Cases among exposed
    exposed_pop = population * exposed_fraction
    baseline_cases_exposed = exposed_pop * baseline_incidence
    # Additional cases due to exposure
    additional_cases = baseline_cases_exposed * af_exposed

    st.write(f"**Attributable Fraction among Exposed**: {af_exposed:.2f}")
    st.write(f"**Additional Cases in Exposed Group**: {int(additional_cases)}")

    st.write("""
    In full-scale HIA, we might combine multiple sources of data, model 
    uncertain parameters, and explore different exposure scenarios.
    """)

########################
# Method Description
########################
st.header("Method Descriptions")
if topic == "Exposure Assessment Methods":
    st.write("""
    Exposure assessment is key in environmental and occupational 
    epidemiology. Methods include personal sampling, area sampling, 
    biomonitoring, and modeling (distance-based, dispersion models, etc.).
    """)
elif topic == "Dose-Response Relationships":
    st.write("""
    Dose-response relationships describe how changes in exposure level 
    translate into changes in adverse health outcomes. Common forms are 
    linear, log-linear, threshold, and more complex toxicological or 
    epidemiologic models.
    """)
elif topic == "Time-Series Analysis of Environmental Data":
    st.write("""
    Time-series methods are used to analyze daily (or hourly) data on 
    pollution and health outcomes. They account for autocorrelation, 
    trends, and lags in exposure-effect.
    """)
elif topic == "Spatial Analysis":
    st.write("""
    Spatial methods help identify geographic patterns of exposure and disease, 
    such as clustering of cases near industrial sites. Tools range from 
    simple mapping to geostatistical modeling.
    """)
elif topic == "Risk Assessment Models":
    st.write("""
    Risk assessment integrates hazard identification, dose-response, 
    exposure assessment, and risk characterization, often forming the 
    basis for regulations or guidelines.
    """)
elif topic == "Health Impact Assessment":
    st.write("""
    HIA estimates how many health events (e.g., hospital admissions, 
    deaths) are attributable to a given exposure at the population level. 
    It incorporates exposure data, dose-response, and baseline rates.
    """)

########################
# References
########################
st.header("References")
st.write("""
1. **Exposure Assessment**: Lioy PJ, et al. Exposure science and the 
   exposome: an opportunity for coherence in the environmental health sciences.
2. **Dose-Response**: Finney DJ. Statistical method in biological assay.
3. **Time-Series**: Dominici F, et al. Air pollution and health: a 
   time-series analysis approach.
4. **Spatial Analysis**: Lawson AB. Statistical methods in spatial 
   epidemiology.
5. **Risk Assessment**: NRC. Risk Assessment in the Federal Government: 
   Managing the Process.
6. **Health Impact Assessment**: WHO. Health Impact Assessment: Main Concepts 
   and Suggested Approach.
""")

st.header("Check your understanding")

if topic == "Exposure Assessment Methods":
    q1 = st.radio(
        "Which of the following is NOT an exposure assessment method?",
        [
            "Personal monitoring",
            "Area sampling",
            "Epidemiologic modeling",
            "Biomonitoring"
        ]
    )
    if q1 == "Epidemiologic modeling":
        st.success("Correct! Epidemiologic modeling focuses on disease patterns, not direct exposure measurement.")
    else:
        st.error("Not quite. The other methods are used for direct exposure assessment.")

elif topic == "Dose-Response Relationships":
    q2 = st.radio(
        "What does a dose-response relationship describe?",
        [
            "The severity of symptoms experienced by an exposed population",
            "The effect of increasing exposure levels on health outcomes",
            "The number of people affected by an environmental hazard",
            "The duration of exposure required to cause a disease"
        ]
    )
    if q2 == "The effect of increasing exposure levels on health outcomes":
        st.success("Correct! Dose-response models quantify the relationship between exposure levels and health effects.")
    else:
        st.error("Not quite. The correct answer focuses on how increasing exposure levels impact health outcomes.")

elif topic == "Time-Series Analysis of Environmental Data":
    q3 = st.radio(
        "What is the main purpose of time-series analysis in environmental epidemiology?",
        [
            "To identify spatial clusters of disease",
            "To analyze trends in exposure and health outcomes over time",
            "To calculate individual risk scores for exposed individuals",
            "To compare exposure levels between different regions"
        ]
    )
    if q3 == "To analyze trends in exposure and health outcomes over time":
        st.success("Correct! Time-series analysis helps examine how exposure and health effects change over time.")
    else:
        st.error("Not quite. The correct answer focuses on temporal patterns in exposure and outcomes.")

# Quiz for Spatial Analysis
elif topic == "Spatial Analysis":
    q4 = st.radio(
        "Which method is commonly used in spatial epidemiology?",
        [
            "Survival analysis",
            "Time-series regression",
            "Kriging",
            "Randomized controlled trials"
        ]
    )
    if q4 == "Kriging":
        st.success("Correct! Kriging is a geostatistical method used for spatial interpolation in exposure assessment.")
    else:
        st.error("Not quite. Kriging is a key spatial method used to estimate exposures across a region.")

# Quiz for Risk Assessment Models
elif topic == "Risk Assessment Models":
    q5 = st.radio(
        "Which of the following is a key component of risk assessment?",
        [
            "Hazard identification",
            "Predictive modeling",
            "Case-control study design",
            "Biological plausibility testing"
        ]
    )
    if q5 == "Hazard identification":
        st.success("Correct! Risk assessment includes hazard identification, dose-response, exposure assessment, and risk characterization.")
    else:
        st.error("Not quite. The correct answer is a fundamental part of risk assessment.")

# Quiz for Health Impact Assessment
elif topic == "Health Impact Assessment":
    q6 = st.radio(
        "What does a Health Impact Assessment (HIA) estimate?",
        [
            "The long-term financial cost of environmental policies",
            "The number of people exposed to a hazard",
            "The potential health effects of a policy, program, or project",
            "The accuracy of exposure assessment methods"
        ]
    )
    if q6 == "The potential health effects of a policy, program, or project":
        st.success("Correct! HIA evaluates the health consequences of proposed actions before they are implemented.")
    else:
        st.error("Not quite. The correct answer highlights the purpose of HIA in policy evaluation.")
