# 18_time_to_event_analysis.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# For survival analysis
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

########################
# Title
########################
st.title("Time-to-Event (Survival) Analysis")

########################
# Main Selectbox
########################
analysis_type = st.selectbox(
    "Select Time-to-Event Analysis Topic",
    [
        "Kaplan-Meier Estimation",
        "Log-Rank Test",
        "Cox Proportional Hazards Model"
    ]
)

###############################################################################
# 1. Kaplan-Meier Estimation
###############################################################################
if analysis_type == "Kaplan-Meier Estimation":
    st.header("Kaplan-Meier Estimation")

    st.write("""
    **Scenario**: We want to estimate the survival function 
    (probability of not having the event by a certain time) 
    from censored event-time data.
    """)

    # Simulation parameters
    st.subheader("Simulation Setup")
    n_samples = st.slider("Number of Individuals", 20, 500, 100)
    scale_param = st.slider("Scale Parameter (Exponential Dist.)", 0.5, 5.0, 2.0)
    censor_fraction = st.slider("Fraction Censored", 0.0, 0.9, 0.2)

    # Simulate event times from Exponential distribution
    rng = np.random.default_rng(42)
    event_times = rng.exponential(scale=scale_param, size=n_samples)

    # Introduce random censoring
    censor_threshold = np.quantile(event_times, 1 - censor_fraction)
    censored = event_times > censor_threshold
    event_times[censored] = censor_threshold
    # "E" is the event indicator: 1 if event occurred, 0 if censored
    event_occurred = ~censored  # True if event time < threshold

    # Build a DataFrame
    df = pd.DataFrame({
        "time": event_times,
        "event": event_occurred.astype(int)
    })

    st.write("### Preview of Simulated Data")
    st.write(df.head(10))

    # Fit Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["time"], event_observed=df["event"], label="KM Estimate")

    st.write("### Kaplan-Meier Survival Table (first few intervals)")
    st.write(kmf.survival_function_.head())

    # Convert KM data into a form we can plot with Plotly
    # "timeline" -> "KM estimate" step function
    km_data = pd.DataFrame({
        "timeline": kmf.survival_function_.index,
        "survival": kmf.survival_function_["KM Estimate"]
    }).reset_index(drop=True)

    # Plot with plotly
    fig = px.line(
        km_data,
        x="timeline",
        y="survival",
        line_shape='hv',
        title="Kaplan-Meier Survival Curve",
        labels={"timeline": "Time", "survival": "Survival Probability"}
    )
    fig.update_yaxes(range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The **Kaplan-Meier estimator** calculates a stepwise survival function, 
    accommodating right-censored observations (individuals who have not yet 
    experienced the event by end of follow-up).
    """)

###############################################################################
# 2. Log-Rank Test
###############################################################################
elif analysis_type == "Log-Rank Test":
    st.header("Log-Rank Test")

    st.write("""
    **Scenario**: We want to compare survival between **two** groups 
    (e.g., Treatment vs. Control) to see if they differ significantly 
    in time-to-event.
    """)

    st.subheader("Simulation Setup")
    n_each = st.slider("Number of Individuals in Each Group", 20, 500, 100)
    scale_group1 = st.slider("Scale Param (Group 1, Exponential)", 0.1, 5.0, 1.5)
    scale_group2 = st.slider("Scale Param (Group 2, Exponential)", 0.1, 5.0, 2.5)
    censor_fraction = st.slider("Fraction Censored (Both Groups)", 0.0, 0.9, 0.2)

    # Generate group 1 data
    rng = np.random.default_rng(101)
    event_times1 = rng.exponential(scale=scale_group1, size=n_each)
    # Censoring
    threshold1 = np.quantile(event_times1, 1 - censor_fraction)
    censored1 = event_times1 > threshold1
    event_times1[censored1] = threshold1
    event_occurred1 = ~censored1

    # Generate group 2 data
    event_times2 = rng.exponential(scale=scale_group2, size=n_each)
    threshold2 = np.quantile(event_times2, 1 - censor_fraction)
    censored2 = event_times2 > threshold2
    event_times2[censored2] = threshold2
    event_occurred2 = ~censored2

    # Create combined DataFrame
    df1 = pd.DataFrame({
        "time": event_times1,
        "event": event_occurred1.astype(int),
        "group": "Group1"
    })
    df2 = pd.DataFrame({
        "time": event_times2,
        "event": event_occurred2.astype(int),
        "group": "Group2"
    })
    df_both = pd.concat([df1, df2]).reset_index(drop=True)

    st.write("### Simulated Data (first rows)")
    st.write(df_both.head(10))

    # Fit KM for each group
    kmf1 = KaplanMeierFitter()
    kmf1.fit(df1["time"], event_observed=df1["event"], label="Group1")
    kmf2 = KaplanMeierFitter()
    kmf2.fit(df2["time"], event_observed=df2["event"], label="Group2")

    # Log-rank test
    results = logrank_test(
        df1["time"], df2["time"],
        event_observed_A=df1["event"], event_observed_B=df2["event"]
    )

    p_val = results.p_value
    test_stat = results.test_statistic

    st.write(f"**Log-Rank Test Statistic**: {test_stat:.3f}")
    st.write(f"**p-value**: {p_val:.5f}")

    # Prepare data for Plotly step plots
    # Group1
    km_data1 = pd.DataFrame({
        "timeline": kmf1.survival_function_.index,
        "survival": kmf1.survival_function_["Group1"],
        "Group": "Group1"
    }).reset_index(drop=True)
    # Group2
    km_data2 = pd.DataFrame({
        "timeline": kmf2.survival_function_.index,
        "survival": kmf2.survival_function_["Group2"],
        "Group": "Group2"
    }).reset_index(drop=True)
    km_plot_data = pd.concat([km_data1, km_data2])

    fig = px.line(
        km_plot_data,
        x="timeline",
        y="survival",
        color="Group",
        title="Kaplan-Meier Curves by Group",
        labels={"timeline": "Time", "survival": "Survival Probability"}
    )
    fig.update_yaxes(range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The **log-rank test** checks if there's a significant difference 
    between the two survival curves. Here, a low p-value suggests 
    they differ more than would be expected by chance.
    """)

###############################################################################
# 3. Cox Proportional Hazards Model
###############################################################################
elif analysis_type == "Cox Proportional Hazards Model":
    st.header("Cox Proportional Hazards Model")

    st.write("""
    **Scenario**: We want to model the relationship between one or more 
    covariates and the hazard of the event over time. The Cox model 
    is a semi-parametric approach, widely used in survival analysis.
    """)

    st.subheader("Simulation Setup")
    n_obs = st.slider("Number of Individuals", 30, 500, 150)
    true_coef = st.slider("True Coefficient for Covariate", -2.0, 2.0, 0.5)

    rng = np.random.default_rng(seed=999)

    # Simulate a single covariate X ~ Normal(0,1)
    X = rng.normal(0, 1, n_obs)

    # Baseline hazard ~ Exponential with scale=1. We incorporate X via hazard ratio
    # True hazard ~ lambda(t) * exp(beta*X). For simplicity, let's just generate
    # event times from an exponential whose rate depends on X.
    # rate_i = exp(beta * X_i)
    # T_i ~ Exponential(rate=rate_i) => scale_i = 1 / rate_i
    rates = np.exp(true_coef * X)
    event_times = rng.exponential(scale=1 / rates)

    # Add random censoring
    censor_cutoff = np.median(event_times) * 1.5
    is_censored = (event_times > censor_cutoff)
    event_times[is_censored] = censor_cutoff
    event_observed = (~is_censored).astype(int)

    # Build DataFrame
    df_cox = pd.DataFrame({
        "time": event_times,
        "event": event_observed,
        "X": X
    })

    st.write("### Simulated Data (first rows)")
    st.dataframe(df_cox.head(10))

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="time", event_col="event", show_progress=False)
    st.write("### Cox Model Summary")
    st.write(cph.summary)

    # Plot partial effect with plotly
    # We'll predict survival for a few hypothetical covariate values
    st.subheader("Survival Curves for Different Covariate Values")
    cov_values = st.text_input("Enter Covariate Values (comma-separated)", "0, 1")
    try:
        vals = [float(x.strip()) for x in cov_values.split(",")]
    except:
        vals = [0, 1]  # fallback

    # Generate survival function for each specified value of X
    surv_dfs = []
    for v in vals:
        tmp_df = pd.DataFrame({"X": [v]})
        sf = cph.predict_survival_function(tmp_df)
        # sf is an index = times, columns = each row of tmp_df
        surv_df = sf.reset_index()
        surv_df.columns = ["time", f"X={v:.2f}"]
        surv_dfs.append(surv_df)

    # Combine them on 'time'
    merged = surv_dfs[0]
    for extra in surv_dfs[1:]:
        merged = pd.merge(merged, extra, on="time", how="outer")
    merged.sort_values("time", inplace=True)

    # Melt for plotly
    mdf = merged.melt(id_vars="time", var_name="Covariate Value", value_name="Survival")
    fig = px.line(
        mdf,
        x="time",
        y="Survival",
        line_shape='vh',
        color="Covariate Value",
        title="Cox PH Model: Predicted Survival for Different Covariate Values",
        labels={"time": "Time"}
    )
    fig.update_yaxes(range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    The **Cox PH Model** estimates a hazard function 
    \\(\\lambda(t) = \\lambda_0(t) e^{{\\beta X}}\\). 
    Here, the true \\(\\beta\\) was {true_coef}, 
    and the model estimates it from the simulated data.
    """)

########################
# Method Description
########################
st.header("Method Descriptions")
if analysis_type == "Kaplan-Meier Estimation":
    st.write("""
    The **Kaplan-Meier** method estimates the survival function using 
    observed event times and accounts for right-censored observations 
    (those lost to follow-up or event-free at end of study).
    """)
elif analysis_type == "Log-Rank Test":
    st.write("""
    The **log-rank test** compares survival curves between two (or more) groups. 
    It is non-parametric and commonly used to test if survival functions 
    differ significantly over time.
    """)
elif analysis_type == "Cox Proportional Hazards Model":
    st.write("""
    The **Cox PH model** relates covariates (predictors) to the hazard function. 
    It assumes hazards for different values of X are proportional over time.
    """)

########################
# References
########################
st.header("References")
st.write("""
1. **Kaplan-Meier**: Kaplan EL, Meier P. Nonparametric Estimation from Incomplete 
   Observations. J Am Stat Assoc. 1958.
2. **Log-Rank Test**: Mantel N. Evaluation of survival data and two new rank 
   order statistics arising in its consideration. Cancer Chemother Rep. 1966.
3. **Cox PH Model**: Cox DR. Regression Models and Life-Tables. J R Stat Soc B. 1972.
4. **lifelines library**: [https://lifelines.readthedocs.io/](https://lifelines.readthedocs.io/)
""")


st.header("Check your understanding")

# Quiz for Kaplan-Meier Estimation
if analysis_type == "Kaplan-Meier Estimation":
    q1 = st.radio(
        "What does the Kaplan-Meier estimator estimate?",
        [
            "The cumulative hazard function",
            "The survival function over time",
            "The probability of developing a disease",
            "The time-to-event for all individuals"
        ]
    )
    if q1 == "The survival function over time":
        st.success("Correct! The Kaplan-Meier estimator estimates the probability of survival at different time points.")
    else:
        st.error("Not quite. The Kaplan-Meier estimator specifically calculates the survival probability over time.")

    q2 = st.radio(
        "What does right censoring mean in survival analysis?",
        [
            "An event occurs before the observation period ends",
            "An individual is lost to follow-up or still event-free at study end",
            "A person has multiple events during follow-up",
            "A survival function follows a normal distribution"
        ]
    )
    if q2 == "An individual is lost to follow-up or still event-free at study end":
        st.success("Correct! Right censoring occurs when we do not observe the event for some individuals.")
    else:
        st.error("Not quite. Right censoring means we don’t see the event happening for some individuals before the study ends.")

# Quiz for Log-Rank Test
elif analysis_type == "Log-Rank Test":
    q3 = st.radio(
        "What is the primary purpose of the log-rank test?",
        [
            "To compare survival curves between groups",
            "To estimate the hazard ratio",
            "To predict individual survival times",
            "To test for proportional hazards"
        ]
    )
    if q3 == "To compare survival curves between groups":
        st.success("Correct! The log-rank test assesses whether survival distributions are statistically different between groups.")
    else:
        st.error("Not quite. The log-rank test is specifically used to compare survival curves.")

    q4 = st.radio(
        "Which assumption must be met for the log-rank test to be valid?",
        [
            "Survival functions must be proportional over time",
            "Survival times must be normally distributed",
            "Hazard ratios must vary significantly between groups",
            "Only uncensored data should be included"
        ]
    )
    if q4 == "Survival functions must be proportional over time":
        st.success("Correct! The log-rank test assumes proportionality of survival functions over time.")
    else:
        st.error("Not quite. The key assumption is that survival functions remain proportional over time.")

# Quiz for Cox Proportional Hazards Model
elif analysis_type == "Cox Proportional Hazards Model":
    q5 = st.radio(
        "What does the Cox Proportional Hazards (Cox PH) model estimate?",
        [
            "The median survival time",
            "The probability of survival over time",
            "The effect of covariates on the hazard rate",
            "The probability of developing an event within a fixed time"
        ]
    )
    if q5 == "The effect of covariates on the hazard rate":
        st.success("Correct! The Cox PH model estimates how predictor variables influence the hazard rate.")
    else:
        st.error("Not quite. The Cox PH model specifically models how covariates affect the hazard function.")

    q6 = st.radio(
        "Which assumption must hold for the Cox Proportional Hazards model?",
        [
            "Survival times follow an exponential distribution",
            "The hazard ratio remains constant over time",
            "Only continuous variables can be used as predictors",
            "The baseline hazard must be normally distributed"
        ]
    )
    if q6 == "The hazard ratio remains constant over time":
        st.success("Correct! The Cox model assumes proportional hazards, meaning hazard ratios remain the same across time.")
    else:
        st.error("Not quite. The key assumption is proportional hazards—hazard ratios do not change over time.")

st.write("Great job! Keep practicing survival analysis concepts. 🚀")
