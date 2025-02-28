# 19_longitudinal_data_analysis.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf

########################
# Page Title
########################
st.title("Longitudinal Data Analysis")

########################
# Main Selectbox
########################
analysis_type = st.selectbox(
    "Select a Longitudinal Analysis Topic",
    [
        "Linear Mixed Effects Model",
        "Generalized Estimating Equations (GEE)",
        "Repeated Measures ANOVA"
    ]
)

###############################################################################
# 1. Linear Mixed Effects Model
###############################################################################
if analysis_type == "Linear Mixed Effects Model":
    st.header("Linear Mixed Effects Model")

    st.write("""
    **Scenario**: We have repeated measurements on individuals over time, 
    with random variation at the individual level. We can model this using 
    a random intercept (and possibly random slope) approach.
    """)

    st.subheader("Simulation Setup")
    n_subjects = st.slider("Number of Subjects", 5, 200, 30)
    n_timepoints = st.slider("Number of Repeated Measurements per Subject", 2, 10, 5)
    true_beta = st.slider("True Fixed Effect (beta)", -2.0, 2.0, 0.5)
    random_slope = st.checkbox("Include Random Slopes?", value=False)

    rng = np.random.default_rng(1)
    
    # Create a 'time' variable (e.g., 1..n_timepoints)
    time_points = np.arange(n_timepoints)
    
    # For each subject, pick a random intercept and (optional) slope
    subject_ids = []
    times = []
    outcome = []
    
    # Random intercept
    intercept_sd = 1.0
    # Random slope
    slope_sd = 0.5 if random_slope else 0.0
    
    for i in range(n_subjects):
        subj_id = i + 1
        # random intercept
        b_i = rng.normal(0, intercept_sd)
        # random slope
        m_i = rng.normal(0, slope_sd)
        
        for t in time_points:
            x_t = float(t)
            # Y = (fixed intercept=0) + random intercept + (true_beta + random slope)*time + error
            y_t = b_i + (true_beta + m_i)*x_t + rng.normal(0, 1)
            
            subject_ids.append(subj_id)
            times.append(x_t)
            outcome.append(y_t)
    
    df_lmm = pd.DataFrame({
        "subject": subject_ids,
        "time": times,
        "outcome": outcome
    })

    st.write("### Simulated Longitudinal Data (first rows)")
    st.dataframe(df_lmm.head(10))

    # Fit a mixed effects model using statsmodels
    # We'll use a formula Y ~ time, random intercept + random slope if user selected
    if random_slope:
        md = smf.mixedlm("outcome ~ time", df_lmm, groups=df_lmm["subject"], re_formula="~time")
    else:
        md = smf.mixedlm("outcome ~ time", df_lmm, groups=df_lmm["subject"])
    
    mdf = md.fit()
    st.write("### Model Results")
    st.text(mdf.summary())

    # Plot means by time
    fig = px.scatter(
        df_lmm.groupby("time")["outcome"].mean().reset_index(),
        x="time",
        y="outcome",
        title="Average Outcome by Time",
        labels={"outcome": "Mean Outcome"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation**: The random intercept captures each subject's deviation 
    from the overall intercept, while the (optional) random slope captures 
    subject-specific time trends. The fixed effect of `time` (beta) indicates 
    the average rate of change over time.
    """)

###############################################################################
# 2. Generalized Estimating Equations (GEE)
###############################################################################
elif analysis_type == "Generalized Estimating Equations (GEE)":
    st.header("Generalized Estimating Equations (GEE)")

    st.write("""
    **Scenario**: GEE is often used for repeated measures or correlated data, 
    fitting a marginal model (focus on population-averaged effects) rather 
    than subject-specific random effects.
    """)

    st.subheader("Simulation Setup")
    n_subjects = st.slider("Number of Subjects", 10, 200, 40)
    n_timepoints = st.slider("Timepoints per Subject", 2, 10, 4)
    true_beta = st.slider("True Coefficient for 'time'", -1.0, 1.0, 0.4)

    rng = np.random.default_rng(42)
    time_points = np.arange(n_timepoints)

    subject_ids = []
    times = []
    # We'll do a binary outcome, e.g., logistic model
    outcome = []

    for i in range(n_subjects):
        subj_id = i + 1
        for t in time_points:
            # Probability p = logistic(intercept + beta*time)
            # We'll fix intercept at 0
            linear_pred = 0 + true_beta * t
            p = 1.0 / (1 + np.exp(-linear_pred))
            y = rng.binomial(n=1, p=p)
            
            subject_ids.append(subj_id)
            times.append(t)
            outcome.append(y)

    df_gee = pd.DataFrame({
        "subject": subject_ids,
        "time": times,
        "Y": outcome
    })

    st.write("### Simulated Binary Repeated Measures Data")
    st.dataframe(df_gee.head(10))

    # Fit GEE (logistic) with an exchangeable correlation structure
    # "family=sm.families.Binomial()" for logistic
    # "cov_struct=sm.cov_struct.Exchangeable()" is a common GEE correlation assumption
    model_gee = smf.gee("Y ~ time", "subject", data=df_gee,
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable())
    result_gee = model_gee.fit()
    st.write("### GEE Results (Logistic)")
    st.text(result_gee.summary())

    # Quick plot: proportion of Y=1 by time
    df_plot = df_gee.groupby("time")["Y"].mean().reset_index()
    fig = px.line(
        df_plot,
        x="time",
        y="Y",
        markers=True,
        title="Proportion of Y=1 by Time (Empirical)",
        labels={"Y": "Proportion (Outcome=1)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **GEE** focuses on population-averaged inference. Here, we used an 
    exchangeable correlation structure and logistic link. The estimated 
    coefficient of `time` reflects how odds of the outcome change with time.
    """)

###############################################################################
# 3. Repeated Measures ANOVA
###############################################################################
elif analysis_type == "Repeated Measures ANOVA":
    st.header("Repeated Measures ANOVA")

    st.write("""
    **Scenario**: A simpler approach to analyzing repeated measures of a 
    continuous outcome across multiple time points (or conditions) within the 
    same subjects. We'll simulate data for 3 time points (or conditions).
    """)

    st.subheader("Simulation Setup")
    n_subjects = st.slider("Number of Subjects", 5, 100, 20)
    effect_size = st.slider("Time Effect (difference in means)", 0.0, 5.0, 2.0)

    rng = np.random.default_rng(2023)

    # We'll fix 3 repeated measurements
    time_levels = [1, 2, 3]
    subject_ids = []
    time_factors = []
    outcome = []

    for i in range(n_subjects):
        # random intercept per subject
        subj_int = rng.normal(loc=10, scale=2)
        for t in time_levels:
            # outcome = subject intercept + time effect * t + random noise
            y = subj_int + effect_size*(t-1) + rng.normal(0, 1)
            subject_ids.append(i+1)
            time_factors.append(t)
            outcome.append(y)

    df_rm = pd.DataFrame({
        "subject": subject_ids,
        "time": time_factors,
        "outcome": outcome
    })

    st.write("### Simulated Data (first rows)")
    st.dataframe(df_rm.head(10))

    # statsmodels has "AnovaRM" for repeated measures ANOVA
    from statsmodels.stats.anova import AnovaRM
    anova_model = AnovaRM(
        data=df_rm, depvar="outcome", subject="subject", within=["time"]
    ).fit()

    st.write("### Repeated Measures ANOVA Results")
    st.text(anova_model)

    # Show mean outcome by time
    means_df = df_rm.groupby("time")["outcome"].mean().reset_index()
    fig = px.bar(
        means_df,
        x="time",
        y="outcome",
        title="Mean Outcome by Time",
        labels={"outcome": "Mean Outcome"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Repeated Measures ANOVA** tests whether the mean outcome differs 
    across the repeated time points/conditions. Assumes sphericity 
    (or uses corrections like Greenhouse-Geisser if violated).
    """)

########################
# Method Description
########################
st.header("Method Descriptions")
if analysis_type == "Linear Mixed Effects Model":
    st.write("""
    A **linear mixed effects model** (LMM) includes both fixed effects (common 
    to all subjects) and random effects (varying by subject). Commonly used 
    for continuous outcomes measured repeatedly over time.
    """)
elif analysis_type == "Generalized Estimating Equations (GEE)":
    st.write("""
    **GEE** provides population-averaged estimates for correlated data, 
    e.g. repeated measurements or cluster-based designs. Various link 
    functions and correlation structures can be used.
    """)
elif analysis_type == "Repeated Measures ANOVA":
    st.write("""
    **Repeated Measures ANOVA** compares means across multiple time points or 
    conditions in the same subjects. It’s a special case of the general linear model 
    that accounts for within-subject correlation.
    """)

########################
# References
########################
st.header("References")
st.write("""
1. **Mixed Effects Models**: Laird NM, Ware JH. Random-effects models for 
   longitudinal data. Biometrics, 1982.
2. **GEE**: Liang K-Y, Zeger SL. Longitudinal data analysis using generalized 
   linear models. Biometrika, 1986.
3. **Repeated Measures ANOVA**: Maxwell SE, Delaney HD. Designing experiments 
   and analyzing data: A model comparison perspective.  
4. **Statsmodels**: [https://www.statsmodels.org/](https://www.statsmodels.org/)
""")

st.header("Check your understanding")

if analysis_type == "Linear Mixed Effects Model":
    q1 = st.radio(
        "What distinguishes a Linear Mixed Effects Model (LMM) from standard linear regression?",
        [
            "LMM allows for non-linear relationships",
            "LMM includes random effects for individual variations",
            "LMM only works with categorical data",
            "LMM requires equal time intervals between measurements"
        ]
    )
    if q1 == "LMM includes random effects for individual variations":
        st.success("Correct! LMMs allow for random effects to account for subject-specific deviations.")
    else:
        st.error("Not quite. The key feature of LMMs is the inclusion of **random effects**.")

    q2 = st.radio(
        "When should a **random slope** be included in a mixed effects model?",
        [
            "When individuals have the same response trend over time",
            "When individuals exhibit different response trends over time",
            "When time is not included as a variable",
            "When the model includes categorical predictors only"
        ]
    )
    if q2 == "When individuals exhibit different response trends over time":
        st.success("Correct! A random slope models individual differences in the effect of time.")
    else:
        st.error("Not quite. A random slope is used when the effect of time varies across individuals.")

# Quiz for Generalized Estimating Equations (GEE)
elif analysis_type == "Generalized Estimating Equations (GEE)":
    q3 = st.radio(
        "What is a key characteristic of GEE?",
        [
            "It models within-subject variability using random effects",
            "It estimates population-averaged effects rather than subject-specific effects",
            "It requires equal numbers of observations per subject",
            "It assumes no correlation between repeated measurements"
        ]
    )
    if q3 == "It estimates population-averaged effects rather than subject-specific effects":
        st.success("Correct! GEE focuses on **population-averaged** effects rather than individual variations.")
    else:
        st.error("Not quite. GEE estimates marginal (population-averaged) effects.")

    q4 = st.radio(
        "Which correlation structure is commonly used in GEE?",
        [
            "Exchangeable",
            "Autoregressive",
            "Independent",
            "All of the above"
        ]
    )
    if q4 == "All of the above":
        st.success("Correct! GEE allows different correlation structures like exchangeable, autoregressive, and independent.")
    else:
        st.error("Not quite. GEE supports multiple correlation structures.")

# Quiz for Repeated Measures ANOVA
elif analysis_type == "Repeated Measures ANOVA":
    q5 = st.radio(
        "What assumption does **Repeated Measures ANOVA** require?",
        [
            "Observations are independent",
            "Each subject contributes a single measurement",
            "Sphericity (variance of differences is equal across time points)",
            "Data follows a non-parametric distribution"
        ]
    )
    if q5 == "Sphericity (variance of differences is equal across time points)":
        st.success("Correct! Repeated Measures ANOVA assumes **sphericity**, meaning equal variance of differences between time points.")
    else:
        st.error("Not quite. The key assumption is **sphericity**, which ensures correct variance estimates.")

    q6 = st.radio(
        "How does Repeated Measures ANOVA differ from LMM?",
        [
            "It does not allow for missing data",
            "It models individual-level random effects",
            "It estimates subject-specific slopes",
            "It assumes uncorrelated residuals"
        ]
    )
    if q6 == "It does not allow for missing data":
        st.success("Correct! Repeated Measures ANOVA struggles with missing data, whereas LMM handles it better.")
    else:
        st.error("Not quite. A key limitation of Repeated Measures ANOVA is its handling of **missing data**.")

st.write("Great job! Keep practicing to master longitudinal data analysis. 🚀")
