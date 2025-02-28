# 17_clinical_epidemiology.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

########################
# Page Title
########################
st.title("Clinical Epidemiology")

########################
# Main Selectbox
########################
method = st.selectbox(
    "Select a Clinical Epidemiology Topic",
    [
        "Diagnostic Decision Trees",
        "Clinical Prediction Rules",
        "Treatment Effect Heterogeneity",
        "Competing Risks Analysis",
        "Quality of Life Measures",
        "Cost-Effectiveness Analysis"
    ]
)

###############################################################################
# 1. Diagnostic Decision Trees
###############################################################################
if method == "Diagnostic Decision Trees":
    st.header("Diagnostic Decision Trees")

    st.write("""
    **Scenario**: Suppose we have a test for a disease with certain accuracy and 
    associated costs (or utilities). We can construct a simplified decision tree 
    to estimate the expected cost (or payoff) of testing vs. not testing.
    """)

    st.subheader("Parameters")
    prev = st.slider("Disease Prevalence", 0.0, 1.0, 0.2)
    sens = st.slider("Test Sensitivity", 0.0, 1.0, 0.90)
    spec = st.slider("Test Specificity", 0.0, 1.0, 0.95)

    cost_test = st.number_input("Cost of Test (per patient)", 0.0, 10000.0, 100.0)
    cost_false_neg = st.number_input("Cost of Missed Disease (False Negative)", 0.0, 100000.0, 2000.0)
    cost_false_pos = st.number_input("Additional Cost for False Positive (unnecessary treatment, follow-up)", 0.0, 100000.0, 500.0)

    st.write("### Expected Cost Calculation")

    # Probability computations
    p_disease = prev
    p_no_disease = 1 - prev
    # Prob of test positive if disease present
    p_pos_disease = sens
    # Prob of test negative if disease absent
    p_neg_no_disease = spec

    # Decision 1: "Test everyone"
    #   True Positive: cost_test, no missed disease cost
    #   False Negative: cost_test + missed disease
    #   False Positive: cost_test + false positive cost
    #   True Negative: cost_test, no extra cost

    # Weighted expected cost (Test strategy)
    exp_cost_test = (
        p_disease * p_pos_disease * cost_test                # true positive
        + p_disease * (1 - p_pos_disease) * (cost_test + cost_false_neg)  # false negative
        + p_no_disease * (1 - spec) * (cost_test + cost_false_pos)        # false positive
        + p_no_disease * spec * cost_test                   # true negative
    )

    # Decision 2: "Do not test"
    #   If we do not test at all, we skip cost_test, but 
    #   all diseased patients remain undiagnosed -> cost_false_neg for each diseased
    exp_cost_no_test = p_disease * cost_false_neg

    st.write(f"**Expected Cost if Test:** {exp_cost_test:,.2f}")
    st.write(f"**Expected Cost if No Test:** {exp_cost_no_test:,.2f}")

    decision = "Test" if exp_cost_test < exp_cost_no_test else "Do Not Test"
    st.success(f"**Recommended Decision:** {decision}")

    # Optional visualization
    labels = ["Test", "No Test"]
    costs = [exp_cost_test, exp_cost_no_test]
    fig = px.bar(
        x=labels,
        y=costs,
        labels={"x": "Strategy", "y": "Expected Cost"},
        title="Decision Tree: Expected Cost Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 2. Clinical Prediction Rules
###############################################################################
elif method == "Clinical Prediction Rules":
    st.header("Clinical Prediction Rules")

    st.write("""
    **Scenario**: Clinical prediction rules combine multiple predictors 
    (e.g., signs, symptoms, tests) into a score that helps estimate 
    patient risk. Below is a simplified example using logistic regression 
    to predict the probability of an outcome.
    """)

    st.subheader("Input Your Coefficients and Patient Features")

    # Suppose we have a logistic model: logit(p) = beta0 + beta1*X1 + beta2*X2 + ...
    beta0 = st.number_input("Intercept (beta0)", -10.0, 10.0, 0.0, step=0.1)
    beta1 = st.number_input("Coefficient for Predictor 1 (beta1)", -5.0, 5.0, 1.0, step=0.1)
    beta2 = st.number_input("Coefficient for Predictor 2 (beta2)", -5.0, 5.0, 0.5, step=0.1)

    X1 = st.number_input("Value of Predictor 1 (X1)", -100.0, 100.0, 10.0)
    X2 = st.number_input("Value of Predictor 2 (X2)", -100.0, 100.0, 20.0)

    # Compute predicted probability
    linear_pred = beta0 + beta1 * X1 + beta2 * X2
    pred_prob = 1 / (1 + np.exp(-linear_pred))

    st.write(f"**Predicted Probability of Outcome:** {pred_prob:.2%}")

    # Visualization: vary X1 and see how predicted probability changes
    st.write("### Sensitivity to Predictor 1")
    x1_range = np.linspace(X1-20, X1+20, 50)
    x2_fixed = X2
    prob_vals = 1 / (1 + np.exp(-(beta0 + beta1*x1_range + beta2*x2_fixed)))
    fig = px.line(
        x=x1_range,
        y=prob_vals,
        labels={"x": "Predictor 1 (X1)", "y": "Predicted Probability"},
        title="Clinical Prediction Rule: Sensitivity to X1"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 3. Treatment Effect Heterogeneity
###############################################################################
elif method == "Treatment Effect Heterogeneity":
    st.header("Treatment Effect Heterogeneity")

    st.write("""
    **Scenario**: Different subgroups of patients may respond differently to 
    the same treatment. We'll simulate a small dataset of patients with 
    various risk profiles, and show how the treatment effect can differ 
    across subgroups.
    """)

    st.subheader("Simulation Parameters")
    n_patients = st.slider("Number of Simulated Patients", 50, 500, 100)
    baseline_risk = st.slider("Baseline Risk (control group)", 0.0, 1.0, 0.2)
    treat_effect = st.slider("Overall Treatment Effect (risk difference)", -0.5, 0.5, -0.1)
    subgroup_factor = st.slider("Subgroup Effect Modifier", -0.5, 0.5, 0.1)

    # Simulate: each patient has a 'risk_score' from 0-1
    rng = np.random.default_rng(123)
    risk_score = rng.random(n_patients)

    # Probability of outcome if untreated, depends on baseline + portion of risk_score
    p_control = baseline_risk + 0.3*(risk_score - 0.5)
    # Probability of outcome if treated
    # we say that effect is treat_effect + (subgroup_factor * risk_score)
    p_treat = p_control + treat_effect + subgroup_factor*(risk_score - 0.5)

    # Clip probabilities to [0,1]
    p_control = np.clip(p_control, 0, 1)
    p_treat = np.clip(p_treat, 0, 1)

    # We could do a forest plot-like breakdown: split patients into quartiles
    df = pd.DataFrame({
        "risk_score": risk_score,
        "p_control": p_control,
        "p_treat": p_treat
    })

    # Create subgroups by quartiles of risk_score
    df["subgroup"] = pd.qcut(df["risk_score"], q=4, labels=["Q1","Q2","Q3","Q4"])
    subgroup_summary = df.groupby("subgroup").agg(
        mean_risk_score=("risk_score","mean"),
        outcome_control=("p_control","mean"),
        outcome_treat=("p_treat","mean")
    ).reset_index()

    st.write("### Subgroup Averages")
    st.dataframe(subgroup_summary)

    # Show difference in each subgroup
    subgroup_summary["risk_diff"] = subgroup_summary["outcome_treat"] - subgroup_summary["outcome_control"]

    # Plot a small forest-like figure
    fig = go.Figure()
    for idx, row in subgroup_summary.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["risk_diff"]],
            y=[row["subgroup"]],
            mode="markers",
            marker=dict(size=12),
            name=f"{row['subgroup']}"
        ))
        # Add a line at x=0
    fig.add_vline(x=0, line_color="red", line_dash="dash")
    fig.update_layout(
        title="Treatment Effect Heterogeneity (Risk Difference by Subgroup)",
        xaxis_title="Treatment Effect (Risk Difference)",
        yaxis_title="Subgroup"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 4. Competing Risks Analysis
###############################################################################
elif method == "Competing Risks Analysis":
    st.header("Competing Risks Analysis")

    st.write("""
    **Scenario**: In time-to-event analysis, patients can experience different 
    types of 'events', and one type of event can preclude the occurrence of 
    another (competing risks). We'll do a simple demonstration of cumulative 
    incidence functions.
    """)

    n = st.slider("Number of Individuals", 50, 1000, 200)
    hazard_eventA = st.slider("Hazard Rate for Event A", 0.0, 0.2, 0.05, 0.01)
    hazard_eventB = st.slider("Hazard Rate for Event B", 0.0, 0.2, 0.03, 0.01)
    max_time = st.slider("Max Follow-up Time", 1, 10, 5)

    # Simple discrete time approximation
    times = np.arange(0, max_time, 0.1)
    cum_inc_A = []
    cum_inc_B = []
    surv = 1.0  # Probability still event-free

    for t in times:
        # instantaneous hazards for each event
        hazardA = hazard_eventA
        hazardB = hazard_eventB
        # Probability that event A occurs in small interval
        pA = surv * hazardA
        # Probability that event B occurs in small interval
        pB = surv * hazardB
        # Update survival
        surv = surv - pA - pB
        # Cumulative incidence so far is 1 - surv for both events combined,
        # but let's separately track A and B
        incA = 1 - (surv + sum(cum_inc_B))
        incB = 1 - (surv + sum(cum_inc_A))
        # This approach is a bit simplistic, so we'll do a simpler method:
        # track them cumulatively
        # We track partial sums in arrays instead
        cum_inc_A.append(pA)
        cum_inc_B.append(pB)

    # Convert from 'incremental' to actual cumulative
    cumA = np.cumsum(cum_inc_A)
    cumB = np.cumsum(cum_inc_B)

    df_compet = pd.DataFrame({
        "time": times,
        "Cumulative Incidence A": cumA,
        "Cumulative Incidence B": cumB
    })

    fig = px.line(
        df_compet,
        x="time",
        y=["Cumulative Incidence A","Cumulative Incidence B"],
        title="Competing Risks: Cumulative Incidence Functions",
        labels={"value":"Cumulative Incidence","time":"Time"}
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 5. Quality of Life Measures
###############################################################################
elif method == "Quality of Life Measures":
    st.header("Quality of Life Measures")

    st.write("""
    **Scenario**: Quality-adjusted life years (QALYs) incorporate both the 
    quantity and quality of life. We'll compute a simple estimate of QALYs 
    based on user-defined survival time and utility weights.
    """)

    st.subheader("Parameters")
    survival_years = st.slider("Expected Survival (years)", 0.0, 30.0, 10.0)
    qol_weight = st.slider("Average Quality of Life Weight (0=worst, 1=perfect)", 0.0, 1.0, 0.75)

    qalys = survival_years * qol_weight
    st.metric("Estimated QALYs", f"{qalys:.2f}")

    st.write("""
    QALYs = Survival Time × QoL Weight. 
    In practice, these weights can vary year to year or across health states.
    """)

    # Simple scenario analysis: vary QoL weight from 0.1 to 1.0
    q_range = np.linspace(0.1, 1.0, 50)
    qaly_values = survival_years * q_range

    fig = px.line(
        x=q_range,
        y=qaly_values,
        labels={"x":"Quality of Life Weight", "y":"QALYs"},
        title="Sensitivity of QALYs to Different Utility Weights"
    )
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 6. Cost-Effectiveness Analysis
###############################################################################
elif method == "Cost-Effectiveness Analysis":
    st.header("Cost-Effectiveness Analysis")

    st.write("""
    **Scenario**: Compare two interventions (A and B) in terms of cost and 
    effectiveness (e.g., QALYs). We'll compute the Incremental Cost-Effectiveness 
    Ratio (ICER).
    """)

    st.subheader("Costs and Effects of Two Interventions")

    col1, col2 = st.columns(2)
    with col1:
        costA = st.number_input("Cost of Intervention A", 0.0, 1e6, 20000.0)
        effectA = st.number_input("Effect (QALYs) of A", 0.0, 50.0, 10.0)
    with col2:
        costB = st.number_input("Cost of Intervention B", 0.0, 1e6, 25000.0)
        effectB = st.number_input("Effect (QALYs) of B", 0.0, 50.0, 12.0)

    # Incremental cost and incremental effect
    dC = costB - costA
    dE = effectB - effectA
    if abs(dE) < 1e-15:
        icer = np.nan
    else:
        icer = dC / dE

    st.subheader("Results")
    st.write(f"**Incremental Cost** (B - A): {dC:,.2f}")
    st.write(f"**Incremental Effect** (B - A): {dE:.2f}")
    st.write(f"**ICER** = ΔCost / ΔEffect = {icer:,.2f}")

    # Simple CE plane
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[effectA, effectB],
        y=[costA, costB],
        mode='markers+text',
        text=["A","B"],
        textposition="bottom center",
        marker=dict(size=12)
    ))
    fig.update_layout(
        title="Cost-Effectiveness (CE) Plane",
        xaxis_title="Effect (QALYs)",
        yaxis_title="Cost",
    )
    st.plotly_chart(fig, use_container_width=True)


########################
# Method Description
########################
st.header("Method Descriptions")
if method == "Diagnostic Decision Trees":
    st.write("""
    Decision trees break down decisions into a tree-like structure of events 
    (test positive/negative, disease present/absent) with assigned probabilities 
    and costs/utilities. Summing expected values helps guide decisions.
    """)
elif method == "Clinical Prediction Rules":
    st.write("""
    Clinical prediction rules quantify patient risk using several predictors. 
    Logistic regression is common; the resulting probability helps guide care decisions.
    """)
elif method == "Treatment Effect Heterogeneity":
    st.write("""
    Treatment effects can vary across subgroups (effect modification). 
    Identifying which subgroups benefit most (or least) can optimize care.
    """)
elif method == "Competing Risks Analysis":
    st.write("""
    In survival analysis with multiple possible events, a competing risk is an event 
    that precludes the occurrence of the primary event of interest. We often use 
    cumulative incidence functions to describe the probability of each event over time.
    """)
elif method == "Quality of Life Measures":
    st.write("""
    Quality-adjusted life years (QALYs) integrate length of survival with the 
    quality (or utility) of those years, enabling more patient-centered evaluations.
    """)
elif method == "Cost-Effectiveness Analysis":
    st.write("""
    Compares interventions' costs and outcomes (often QALYs) to determine 
    the incremental cost-effectiveness ratio (ICER). Useful in resource allocation.
    """)

########################
# References
########################
st.header("References")
st.write("""
1. **Diagnostic Decision Trees**: Pauker SG, Kassirer JP. The Threshold 
   Approach to Clinical Decision Making. N Engl J Med 1980.
2. **Clinical Prediction Rules**: Reilly BM, Evans AT. Translating clinical 
   research into clinical practice: impact of using prediction rules to make 
   decisions. Ann Intern Med. 2006.
3. **Treatment Effect Heterogeneity**: Kent DM, et al. Assessing and reporting 
   heterogeneity in treatment effects in clinical trials. Ann Intern Med. 2010.
4. **Competing Risks**: Fine JP, Gray RJ. A proportional hazards model for the 
   subdistribution of a competing risk. JASA 1999.
5. **Quality of Life Measures**: Torrance GW. Measurement of health state utilities 
   for economic appraisal. J Health Econ 1986.
6. **Cost-Effectiveness**: Drummond MF, et al. Methods for the Economic Evaluation 
   of Health Care Programmes. Oxford Univ Press.
""")

st.header("Check your understanding")
if method == "Diagnostic Decision Trees":
    q1 = st.radio(
        "What is the primary purpose of a diagnostic decision tree?",
        [
            "To identify the best statistical model for diagnosis",
            "To calculate the probability of a disease given a test result and associated costs",
            "To determine the accuracy of a new diagnostic test",
            "To estimate the prevalence of a disease in a population"
        ]
    )
    if q1 == "To calculate the probability of a disease given a test result and associated costs":
        st.success("Correct! Decision trees help balance costs, test performance, and disease prevalence.")
    else:
        st.error("Not quite. The key goal is to compare expected costs and outcomes based on testing.")

elif method == "Clinical Prediction Rules":
    q2 = st.radio(
        "What is a key advantage of clinical prediction rules?",
        [
            "They always provide 100% accurate predictions",
            "They integrate multiple patient factors to improve risk estimation",
            "They eliminate the need for laboratory testing",
            "They are only applicable to rare diseases"
        ]
    )
    if q2 == "They integrate multiple patient factors to improve risk estimation":
        st.success("Correct! Clinical prediction rules combine multiple predictors to refine risk estimation.")
    else:
        st.error("Not quite. The main benefit is improving risk prediction using multiple factors.")

elif method == "Treatment Effect Heterogeneity":
    q3 = st.radio(
        "What does treatment effect heterogeneity refer to?",
        [
            "The variation in treatment effects across different patient subgroups",
            "The overall effectiveness of a treatment in a randomized trial",
            "The differences in drug formulations used in clinical practice",
            "The process of standardizing treatment protocols"
        ]
    )
    if q3 == "The variation in treatment effects across different patient subgroups":
        st.success("Correct! Treatment effect heterogeneity refers to how treatment effects vary across patient groups.")
    else:
        st.error("Not quite. The key point is that some patients may benefit more (or less) than others.")
        
elif method == "Competing Risks Analysis":
    q4 = st.radio(
        "In competing risks analysis, what is a competing risk?",
        [
            "A factor that confounds the primary outcome",
            "An event that prevents the occurrence of the primary event of interest",
            "A risk factor that influences disease incidence",
            "A statistical test for comparing survival curves"
        ]
    )
    if q4 == "An event that prevents the occurrence of the primary event of interest":
        st.success("Correct! Competing risks refer to events that make the primary outcome impossible.")
    else:
        st.error("Not quite. A competing risk occurs when one event precludes another.")

elif method == "Quality of Life Measures":
    q5 = st.radio(
        "What does a quality-adjusted life year (QALY) measure?",
        [
            "Only the length of survival",
            "Only the quality of life in a population",
            "A combination of survival duration and quality of life",
            "The cost-effectiveness of a treatment"
        ]
    )
    if q5 == "A combination of survival duration and quality of life":
        st.success("Correct! QALYs integrate both length of life and quality of life into a single measure.")
    else:
        st.error("Not quite. QALYs adjust survival time based on quality of life.")

elif method == "Cost-Effectiveness Analysis":
    q6 = st.radio(
        "What does the Incremental Cost-Effectiveness Ratio (ICER) represent?",
        [
            "The absolute cost of a new intervention",
            "The ratio of cost savings to patient survival",
            "The additional cost per unit of health benefit (e.g., per QALY) compared to another intervention",
            "The amount of funding needed to implement a new health intervention"
        ]
    )
    if q6 == "The additional cost per unit of health benefit (e.g., per QALY) compared to another intervention":
        st.success("Correct! The ICER represents the additional cost per unit of effectiveness.")
    else:
        st.error("Not quite. ICER tells us how much extra we pay for an additional unit of health benefit.")

