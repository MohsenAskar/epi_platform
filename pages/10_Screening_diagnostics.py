# pages/10_screening_diagnostics.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats

st.title("Screening and Diagnostic Tests Analysis")

# Method selector
method = st.selectbox(
    "Select Analysis Method",
    ["Basic Test Metrics", "ROC Analysis", "Predictive Values", 
     "Likelihood Ratios", "Screening Program Evaluation"]
)

if method == "Basic Test Metrics":
    st.header("Sensitivity and Specificity Calculator")
    
    # Input for confusion matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease Present")
        true_positive = st.number_input("True Positive (TP)", min_value=0, value=85)
        false_negative = st.number_input("False Negative (FN)", min_value=0, value=15)
    
    with col2:
        st.subheader("Disease Absent")
        false_positive = st.number_input("False Positive (FP)", min_value=0, value=90)
        true_negative = st.number_input("True Negative (TN)", min_value=0, value=810)
    
    # Calculate metrics
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    ppv = true_positive / (true_positive + false_positive)
    npv = true_negative / (true_negative + false_negative)
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sensitivity", f"{sensitivity:.2%}")
    with col2:
        st.metric("Specificity", f"{specificity:.2%}")
    with col3:
        st.metric("PPV", f"{ppv:.2%}")
    with col4:
        st.metric("NPV", f"{npv:.2%}")
    
    # Visualization of confusion matrix
    confusion_matrix = np.array([[true_positive, false_negative],
                               [false_positive, true_negative]])
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=['Disease +', 'Disease -'],
        y=['Test +', 'Test -'],
        text=confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale='Blues'
    ))
    
    fig.update_layout(title='Confusion Matrix Visualization')
    st.plotly_chart(fig, use_container_width=True)

elif method == "ROC Analysis":
    st.header("ROC Curve Analysis")
    
    # Parameters for simulated data
    n_samples = st.slider("Sample Size", 100, 1000, 500)
    auc_target = st.slider("Target AUC", 0.5, 1.0, 0.8)
    
    # Generate simulated test data
    def generate_test_data(n, auc_target):
        # Generate disease status
        disease = np.random.binomial(1, 0.3, n)
        
        # Generate test scores
        mu_healthy = 0
        mu_diseased = stats.norm.ppf(auc_target) * np.sqrt(2)
        
        scores = np.where(
            disease == 1,
            np.random.normal(mu_diseased, 1, n),
            np.random.normal(mu_healthy, 1, n)
        )
        
        return disease, scores
    
    disease_status, test_scores = generate_test_data(n_samples, auc_target)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(disease_status, test_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})'
    ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Random Classifier'
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate (1 - Specificity)',
        yaxis_title='True Positive Rate (Sensitivity)',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive threshold selection
    threshold_percentile = st.slider(
        "Select threshold percentile", 
        0.0, 100.0, 50.0
    )
    
    threshold = np.percentile(test_scores, threshold_percentile)
    predictions = (test_scores >= threshold).astype(int)
    
    # Calculate metrics at selected threshold
    tp = np.sum((predictions == 1) & (disease_status == 1))
    tn = np.sum((predictions == 0) & (disease_status == 0))
    fp = np.sum((predictions == 1) & (disease_status == 0))
    fn = np.sum((predictions == 0) & (disease_status == 1))
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sensitivity at threshold", f"{sens:.2%}")
    with col2:
        st.metric("Specificity at threshold", f"{spec:.2%}")

elif method == "Predictive Values":
    st.header("Predictive Values Analysis")
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        sensitivity = st.slider("Test Sensitivity", 0.0, 1.0, 0.85)
        specificity = st.slider("Test Specificity", 0.0, 1.0, 0.95)
    
    with col2:
        prevalence = st.slider("Disease Prevalence", 0.0, 1.0, 0.10)
    
    # Calculate predictive values
    def calculate_predictive_values(sens, spec, prev):
        # Positive Predictive Value
        ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
        
        # Negative Predictive Value
        npv = (spec * (1 - prev)) / ((1 - sens) * prev + spec * (1 - prev))
        
        return ppv, npv
    
    ppv, npv = calculate_predictive_values(sensitivity, specificity, prevalence)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive Predictive Value", f"{ppv:.2%}")
    with col2:
        st.metric("Negative Predictive Value", f"{npv:.2%}")
    
    # Visualization of how PPV changes with prevalence
    prevalence_range = np.linspace(0.01, 0.99, 100)
    ppv_values = [
        calculate_predictive_values(sensitivity, specificity, p)[0]
        for p in prevalence_range
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prevalence_range,
        y=ppv_values,
        mode='lines',
        name='PPV'
    ))
    
    fig.add_vline(x=prevalence, line_dash="dash")
    
    fig.update_layout(
        title='PPV vs. Disease Prevalence',
        xaxis_title='Disease Prevalence',
        yaxis_title='Positive Predictive Value',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Likelihood Ratios":
    st.header("Likelihood Ratios Calculator")
    
    # Inputs for test characteristics
    sensitivity = st.slider("Test Sensitivity", 0.0, 1.0, 0.85)
    specificity = st.slider("Test Specificity", 0.0, 1.0, 0.95)
    
    # Calculate likelihood ratios
    lr_positive = sensitivity / (1 - specificity)
    lr_negative = (1 - sensitivity) / specificity
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive Likelihood Ratio", f"{lr_positive:.2f}")
    with col2:
        st.metric("Negative Likelihood Ratio", f"{lr_negative:.2f}")
    
    # Pre-test probability input
    pretest_prob = st.slider("Pre-test Probability", 0.0, 1.0, 0.30)
    
    # Calculate post-test probabilities
    def prob_to_odds(p):
        return p / (1 - p)
    
    def odds_to_prob(o):
        return o / (1 + o)
    
    pretest_odds = prob_to_odds(pretest_prob)
    posttest_odds_pos = pretest_odds * lr_positive
    posttest_odds_neg = pretest_odds * lr_negative
    
    posttest_prob_pos = odds_to_prob(posttest_odds_pos)
    posttest_prob_neg = odds_to_prob(posttest_odds_neg)
    
    # Display post-test probabilities
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Post-test Probability (Positive Test)", f"{posttest_prob_pos:.2%}")
    with col2:
        st.metric("Post-test Probability (Negative Test)", f"{posttest_prob_neg:.2%}")
    
    # Visualization
    prob_range = np.linspace(0.01, 0.99, 100)
    odds_range = prob_to_odds(prob_range)
    
    post_odds_pos = odds_range * lr_positive
    post_odds_neg = odds_range * lr_negative
    
    post_prob_pos = odds_to_prob(post_odds_pos)
    post_prob_neg = odds_to_prob(post_odds_neg)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prob_range,
        y=post_prob_pos,
        name='Positive Test'
    ))
    
    fig.add_trace(go.Scatter(
        x=prob_range,
        y=post_prob_neg,
        name='Negative Test'
    ))
    
    fig.add_trace(go.Scatter(
        x=prob_range,
        y=prob_range,
        line=dict(dash='dash'),
        name='No Change'
    ))
    
    fig.update_layout(
        title='Pre-test vs. Post-test Probability',
        xaxis_title='Pre-test Probability',
        yaxis_title='Post-test Probability'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Screening Program Evaluation":
    st.header("Screening Program Evaluation")
    
    # Program parameters
    population_size = st.number_input("Population Size", min_value=1000, value=100000)
    disease_prevalence = st.slider("Disease Prevalence", 0.0, 0.1, 0.01)
    screening_coverage = st.slider("Screening Coverage", 0.0, 1.0, 0.7)
    test_sensitivity = st.slider("Test Sensitivity", 0.0, 1.0, 0.85)
    test_specificity = st.slider("Test Specificity", 0.0, 1.0, 0.95)
    
    # Cost parameters
    cost_per_test = st.number_input("Cost per Test ($)", value=10)
    cost_per_followup = st.number_input("Cost per Follow-up ($)", value=100)
    
    # Calculate program outcomes
    n_screened = int(population_size * screening_coverage)
    n_diseased = int(n_screened * disease_prevalence)
    n_healthy = n_screened - n_diseased
    
    # Test results
    true_positives = int(n_diseased * test_sensitivity)
    false_negatives = n_diseased - true_positives
    false_positives = int(n_healthy * (1 - test_specificity))
    true_negatives = n_healthy - false_positives
    
    # Program costs
    total_test_cost = n_screened * cost_per_test
    total_followup_cost = (true_positives + false_positives) * cost_per_followup
    total_cost = total_test_cost + total_followup_cost
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number Screened", f"{n_screened:,}")
        st.metric("Cases Detected", f"{true_positives:,}")
    with col2:
        st.metric("False Positives", f"{false_positives:,}")
        st.metric("Cases Missed", f"{false_negatives:,}")
    with col3:
        st.metric("Total Cost", f"${total_cost:,.2f}")
        st.metric("Cost per Case Detected", f"${total_cost/true_positives:,.2f}")
    
    # Visualization of program outcomes
    outcomes_data = pd.DataFrame([
        {"Category": "True Positives", "Count": true_positives},
        {"Category": "False Positives", "Count": false_positives},
        {"Category": "True Negatives", "Count": true_negatives},
        {"Category": "False Negatives", "Count": false_negatives}
    ])
    
    fig = px.pie(
        outcomes_data,
        values='Count',
        names='Category',
        title='Screening Program Outcomes'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Educational content for each method
st.header("Technical Details")

if method == "Basic Test Metrics":
    st.write("""
    **Key test characteristics:**
    
    1. **Sensitivity (True Positive Rate)**
    - Proportion of true disease cases correctly identified
    - Se = TP / (TP + FN)
    
    2. **Specificity (True Negative Rate)**
    - Proportion of true non-disease cases correctly identified
    - Sp = TN / (TN + FP)
    
    3. **Applications:**
    - Test evaluation
    - Comparison of diagnostic methods
    - Quality control
    """)

# Continuing from previous code...

elif method == "ROC Analysis":
    st.write("""
    **ROC Curve Analysis:**
    
    1. **Purpose**
    - Evaluates test performance across all possible thresholds
    - Provides visual and quantitative assessment of test accuracy
    - Helps select optimal cutoff points
    
    2. **Area Under the Curve (AUC)**
    - Ranges from 0.5 (random guess) to 1.0 (perfect test)
    - 0.7-0.8: Acceptable discrimination
    - 0.8-0.9: Excellent discrimination
    - >0.9: Outstanding discrimination
    
    3. **Optimal Threshold Selection**
    - Consider clinical consequences of FP vs FN
    - May use Youden's Index (maximizes sensitivity + specificity - 1)
    - Cost-based approaches available
    
    4. **Limitations**
    - Assumes valid gold standard
    - May not reflect clinical utility
    - Affected by spectrum bias
    """)

elif method == "Predictive Values":
    st.write("""
    **Predictive Values:**
    
    1. **Positive Predictive Value (PPV)**
    - Probability of disease given a positive test
    - Strongly affected by disease prevalence
    - PPV = (Se × Prev) / (Se × Prev + (1-Sp) × (1-Prev))
    
    2. **Negative Predictive Value (NPV)**
    - Probability of no disease given a negative test
    - Also affected by prevalence
    - NPV = (Sp × (1-Prev)) / ((1-Se) × Prev + Sp × (1-Prev))
    
    3. **Clinical Application**
    - Use for patient counseling
    - Important for screening program planning
    - Must consider local disease prevalence
    
    4. **Limitations**
    - Values from one setting may not apply to another
    - Need accurate prevalence estimates
    - May vary in different patient subgroups
    """)

elif method == "Likelihood Ratios":
    st.write("""
    **Likelihood Ratios:**
    
    1. **Positive Likelihood Ratio (LR+)**
    - Ratio of true positive rate to false positive rate
    - LR+ = Sensitivity / (1 - Specificity)
    - Interpretation:
      * >10: Large increase in probability
      * 5-10: Moderate increase
      * 2-5: Small increase
    
    2. **Negative Likelihood Ratio (LR-)**
    - Ratio of false negative rate to true negative rate
    - LR- = (1 - Sensitivity) / Specificity
    - Interpretation:
      * <0.1: Large decrease in probability
      * 0.1-0.2: Moderate decrease
      * 0.2-0.5: Small decrease
    
    3. **Advantages**
    - Independent of prevalence
    - Can be used with multiple sequential tests
    - Directly applicable to individual patients
    
    4. **Clinical Application**
    - Convert pre-test to post-test probability
    - Use with probability nomogram
    - Useful for test sequencing decisions
    """)

elif method == "Screening Program Evaluation":
    st.write("""
    **Screening Program Evaluation:**
    
    1. **Key Metrics**
    - Coverage rate
    - Yield (cases detected)
    - False positive rate
    - Cost per case detected
    - Program costs
    
    2. **Economic Considerations**
    - Direct costs (tests, staff, facilities)
    - Indirect costs (patient time, anxiety)
    - Cost of follow-up investigations
    - Long-term healthcare savings
    
    3. **Quality Assurance**
    - Test performance monitoring
    - Provider training and certification
    - Quality control procedures
    - Data management and analysis
    
    4. **Program Planning**
    - Target population definition
    - Screening interval determination
    - Resource allocation
    - Implementation strategy
    
    5. **Evaluation Framework**
    - Process measures
    - Outcome measures
    - Impact assessment
    - Cost-effectiveness analysis
    """)

st.header("Best Practices")

if method == "Basic Test Metrics":
    st.write("""
    1. **Sample Size Considerations**
    - Ensure adequate cases and controls
    - Calculate confidence intervals
    - Consider verification bias
    
    2. **Study Design**
    - Representative population
    - Appropriate reference standard
    - Blinded assessment
    
    3. **Reporting**
    - Follow STARD guidelines
    - Report all relevant subgroups
    - Document missing data
    """)

elif method == "ROC Analysis":
    st.write("""
    1. **Data Collection**
    - Continuous or ordinal measurements
    - Complete verification of disease status
    - Representative spectrum of patients
    
    2. **Analysis**
    - Bootstrap for confidence intervals
    - Compare AUCs appropriately
    - Consider partial AUC when relevant
    
    3. **Interpretation**
    - Context-specific thresholds
    - Balance sensitivity/specificity
    - Consider clinical utility
    """)

elif method == "Predictive Values":
    st.write("""
    1. **Calculation**
    - Use local prevalence data
    - Account for verification bias
    - Consider subgroup variations
    
    2. **Application**
    - Update with new evidence
    - Consider population characteristics
    - Account for test dependencies
    
    3. **Communication**
    - Use natural frequencies
    - Provide visual aids
    - Explain uncertainties
    """)

elif method == "Likelihood Ratios":
    st.write("""
    1. **Calculation**
    - Use appropriate confidence intervals
    - Consider continuous test results
    - Account for multiple cutoffs
    
    2. **Application**
    - Use with pre-test probability
    - Consider sequential testing
    - Document assumptions
    
    3. **Interpretation**
    - Consider clinical context
    - Use standardized categories
    - Account for test independence
    """)

elif method == "Screening Program Evaluation":
    st.write("""
    1. **Program Design**
    - Clear objectives
    - Defined target population
    - Evidence-based protocol
    - Quality assurance plan
    
    2. **Implementation**
    - Pilot testing
    - Staff training
    - Data management
    - Regular monitoring
    
    3. **Evaluation**
    - Process indicators
    - Outcome measures
    - Cost analysis
    - Impact assessment
    
    4. **Reporting**
    - Standardized metrics
    - Regular updates
    - Stakeholder communication
    """)

# Add references
st.header("Further Reading")
st.write("""
1. Zhou XH, et al. Statistical Methods in Diagnostic Medicine.
2. Pepe MS. The Statistical Evaluation of Medical Tests for Classification and Prediction.
3. Zweig MH, Campbell G. Receiver-operating characteristic (ROC) plots.
4. Bossuyt PM, et al. STARD 2015: Updated guidelines for reporting diagnostic accuracy studies.
""")

st.header("Check your understanding")

if method == "Basic Test Metrics":
    quiz_basic = st.radio(
        "Which statement best describes 'specificity'?",
        [
            "The proportion of true disease cases that test positive",
            "The proportion of true non-disease cases that test negative",
            "The probability that a test is correct"
        ]
    )
    if quiz_basic == "The proportion of true non-disease cases that test negative":
        st.success("Correct! Specificity is TN / (TN + FP).")
    else:
        st.error("Not quite. Remember, specificity focuses on those *without* the disease.")
        
elif method == "ROC Analysis":
    quiz_roc = st.radio(
        "In ROC analysis, which metric is plotted on the x-axis?",
        [
            "True Positive Rate (Sensitivity)",
            "False Positive Rate (1 - Specificity)",
            "Positive Predictive Value"
        ]
    )
    if quiz_roc == "False Positive Rate (1 - Specificity)":
        st.success("Correct! The ROC curve typically plots FPR on the x-axis and TPR on the y-axis.")
    else:
        st.error("Remember: The x-axis is the False Positive Rate (1 - specificity).")

elif method == "Predictive Values":
    quiz_pv = st.radio(
        "Which factor has the greatest impact on Positive Predictive Value (PPV)?",
        [
            "Sensitivity of the test",
            "Specificity of the test",
            "Disease prevalence in the population"
        ]
    )
    if quiz_pv == "Disease prevalence in the population":
        st.success("Correct! PPV is heavily influenced by how common the disease is (the prevalence).")
    else:
        st.error("Although sensitivity and specificity matter, prevalence typically drives the biggest change in PPV.")

elif method == "Likelihood Ratios":
    quiz_lr = st.radio(
        "Which statement best describes the Positive Likelihood Ratio (LR+)?",
        [
            "Probability of a negative test given the disease is present",
            "Ratio of the true positive rate to the false positive rate",
            "Equivalent to the specificity of a test"
        ]
    )
    if quiz_lr == "Ratio of the true positive rate to the false positive rate":
        st.success("Correct! LR+ = sensitivity / (1 - specificity).")
    else:
        st.error("Not quite. LR+ is indeed the ratio of how often diseased individuals test positive vs. how often healthy individuals test positive.")

elif method == "Screening Program Evaluation":
    quiz_screening = st.radio(
        "Which metric indicates the average expenditure required to find one true case?",
        [
            "Total Cost",
            "Cost per Case Detected",
            "False Positive Rate"
        ]
    )
    if quiz_screening == "Cost per Case Detected":
        st.success("Correct! This measures how much money is spent to identify a single true positive case.")
    else:
        st.error("Not quite. 'Cost per Case Detected' specifically captures the average cost of each true positive found.")
