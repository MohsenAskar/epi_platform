# pages/16_bias_analysis.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.title("Quantitative Bias Analysis")

method = st.selectbox(
    "Select Analysis Method",
    ["Unmeasured Confounding", "Selection Bias", 
     "Measurement Error", "Multiple Bias Analysis"]
)

if method == "Unmeasured Confounding":
    st.header("Unmeasured Confounding Analysis")
    
    # Input parameters for observed data
    st.subheader("Observed Study Results")
    observed_rr = st.number_input("Observed Risk Ratio", 0.1, 10.0, 2.0)
    
    # Parameters for unmeasured confounder
    st.subheader("Unmeasured Confounder Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        pc1 = st.slider("Prevalence in Exposed", 0.0, 1.0, 0.6)
        rrud = st.slider("Risk Ratio (Confounder-Disease)", 1.0, 10.0, 2.0)
    with col2:
        pc0 = st.slider("Prevalence in Unexposed", 0.0, 1.0, 0.3)
    
    # Calculate bias-adjusted estimate
    def calculate_bias_adjustment(rr_obs, p1, p0, rr_ud):
        bias_factor = (p1 * rr_ud + (1 - p1)) / (p0 * rr_ud + (1 - p0))
        return rr_obs / bias_factor
    
    adjusted_rr = calculate_bias_adjustment(observed_rr, pc1, pc0, rrud)
    
    # Display results
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Observed RR", f"{observed_rr:.2f}")
    with col2:
        st.metric("Adjusted RR", f"{adjusted_rr:.2f}")
    
    # Generate contour plot for sensitivity analysis
    rr_ud_range = np.linspace(1, 10, 50)
    pc1_range = np.linspace(0, 1, 50)
    
    RR_UD, PC1 = np.meshgrid(rr_ud_range, pc1_range)
    ADJUSTED_RR = calculate_bias_adjustment(observed_rr, PC1, pc0, RR_UD)
    
    fig = go.Figure(data=
        go.Contour(
            z=ADJUSTED_RR,
            x=rr_ud_range,
            y=pc1_range,
            colorscale='Viridis',
            colorbar=dict(title='Adjusted RR')
        )
    )
    
    fig.update_layout(
        title='Sensitivity Analysis for Unmeasured Confounding',
        xaxis_title='RR (Confounder-Disease)',
        yaxis_title='Prevalence in Exposed'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Selection Bias":
    st.header("Selection Bias Analysis")
    
    # Input parameters
    st.subheader("Study Parameters")
    
    # Selection probabilities
    col1, col2 = st.columns(2)
    with col1:
        s11 = st.slider("Selection Prob (Exposed, Case)", 0.0, 1.0, 0.8)
        s10 = st.slider("Selection Prob (Exposed, Control)", 0.0, 1.0, 0.6)
    with col2:
        s01 = st.slider("Selection Prob (Unexposed, Case)", 0.0, 1.0, 0.7)
        s00 = st.slider("Selection Prob (Unexposed, Control)", 0.0, 1.0, 0.5)
    
    observed_or = st.number_input("Observed Odds Ratio", 0.1, 10.0, 2.0)
    
    # Calculate bias-adjusted estimate
    selection_bias_factor = (s11 * s00) / (s10 * s01)
    adjusted_or = observed_or / selection_bias_factor
    
    # Display results
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Observed OR", f"{observed_or:.2f}")
    with col2:
        st.metric("Selection Bias Factor", f"{selection_bias_factor:.2f}")
    with col3:
        st.metric("Adjusted OR", f"{adjusted_or:.2f}")
    
    # Sensitivity analysis
    s1_range = np.linspace(0.1, 1, 50)
    s0_range = np.linspace(0.1, 1, 50)
    
    S1, S0 = np.meshgrid(s1_range, s0_range)
    BIAS_FACTOR = (S1 * s00) / (s10 * S0)
    ADJUSTED_OR = observed_or / BIAS_FACTOR
    
    fig = go.Figure(data=
        go.Contour(
            z=ADJUSTED_OR,
            x=s1_range,
            y=s0_range,
            colorscale='Viridis',
            colorbar=dict(title='Adjusted OR')
        )
    )
    
    fig.update_layout(
        title='Sensitivity Analysis for Selection Bias',
        xaxis_title='Selection Probability (Exposed, Case)',
        yaxis_title='Selection Probability (Unexposed, Case)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Measurement Error":
    st.header("Measurement Error Analysis")
    
    # Input parameters
    st.subheader("Measurement Parameters")
    
    error_type = st.selectbox(
        "Type of Measurement Error",
        ["Non-differential", "Differential"]
    )
    
    # =============================
    # NON-DIFFERENTIAL MISCLASSIFICATION
    # =============================
    if error_type == "Non-differential":
        sensitivity = st.slider("Sensitivity", 0.0, 1.0, 0.8)
        specificity = st.slider("Specificity", 0.0, 1.0, 0.9)
        
        # Matrix: rows = [Exposed, Unexposed], columns = [Cases, Controls]
        observed_data = np.array([
            [st.number_input("Exposed Cases", 0, 1000, 100),
             st.number_input("Exposed Controls", 0, 1000, 200)],
            [st.number_input("Unexposed Cases", 0, 1000, 50),
             st.number_input("Unexposed Controls", 0, 1000, 250)]
        ])
        
        # --- Function to correct single row (cases, controls) ---
        def correct_counts(cases_observed, controls_observed, sens, spec):
            """
            Corrects the observed cases/controls for non-differential misclassification.
            Returns (true_cases, true_controls).
            """
            total = cases_observed + controls_observed
            # Avoid dividing by zero if sens + spec = 1
            denom = sens + spec - 1.0
            if abs(denom) < 1e-15:
                # If sensitivity + specificity == 1, cannot correct
                return (np.nan, np.nan)
            true_cases = (cases_observed - (1 - spec)*total) / denom
            true_controls = total - true_cases
            return (true_cases, true_controls)
        
        # Correct the first row (Exposed)
        a_obs, b_obs = observed_data[0, 0], observed_data[0, 1]
        a_true, b_true = correct_counts(a_obs, b_obs, sensitivity, specificity)
        
        # Correct the second row (Unexposed)
        c_obs, d_obs = observed_data[1, 0], observed_data[1, 1]
        c_true, d_true = correct_counts(c_obs, d_obs, sensitivity, specificity)
        
        # Calculate observed OR and corrected OR
        # observed_data: a=Exposed Cases, b=Exposed Controls,
        #               c=Unexposed Cases, d=Unexposed Controls
        observed_or = (a_obs * d_obs) / (b_obs * c_obs) if (b_obs * c_obs) != 0 else np.nan
        true_or = (a_true * d_true) / (b_true * c_true) if (b_true * c_true) != 0 else np.nan
    
    # =============================
    # DIFFERENTIAL MISCLASSIFICATION
    # =============================
    else:
        # Different sensitivity/specificity for each group
        col1, col2 = st.columns(2)
        with col1:
            sens_cases = st.slider("Sensitivity (Cases)", 0.0, 1.0, 0.8)
            spec_cases = st.slider("Specificity (Cases)", 0.0, 1.0, 0.9)
        with col2:
            sens_controls = st.slider("Sensitivity (Controls)", 0.0, 1.0, 0.7)
            spec_controls = st.slider("Specificity (Controls)", 0.0, 1.0, 0.95)
        
        observed_data = np.array([
            [st.number_input("Observed Exposed Cases", 0, 1000, 100),
             st.number_input("Observed Exposed Controls", 0, 1000, 200)],
            [st.number_input("Observed Unexposed Cases", 0, 1000, 50),
             st.number_input("Observed Unexposed Controls", 0, 1000, 250)]
        ])
        
        # Correct with differential error
        def correct_differential(observed_2x2, sens_case, spec_case, sens_ctrl, spec_ctrl):
            """
            observed_2x2 is 2x2: 
              [ [Exposed Cases, Exposed Controls],
                [Unexposed Cases, Unexposed Controls] ]
            We use different (sensitivity, specificity) for cases vs controls.
            
            Returns a 2x2 matrix of the 'true' counts.
            """
            import numpy as np
            
            # For cases column: apply (sens_case, spec_case)
            #   matrix_case = [[sens_case,      1 - spec_case],
            #                  [1 - sens_case,  spec_case    ]]
            
            # For controls column: apply (sens_ctrl, spec_ctrl)
            #   matrix_ctrl = [[sens_ctrl,      1 - spec_ctrl],
            #                  [1 - sens_ctrl,  spec_ctrl    ]]
            
            matrix_case = np.array([
                [sens_case,       1. - spec_case],
                [1. - sens_case,  spec_case]
            ])
            matrix_ctrl = np.array([
                [sens_ctrl,       1. - spec_ctrl],
                [1. - sens_ctrl,  spec_ctrl]
            ])
            
            # Observed "cases" = [Exposed Cases, Unexposed Cases]
            obs_cases = observed_2x2[:, 0]
            # Observed "controls" = [Exposed Controls, Unexposed Controls]
            obs_controls = observed_2x2[:, 1]
            
            # Solve linear systems to get the true [Exposed Cases, Unexposed Cases]
            true_cases = np.linalg.solve(matrix_case, obs_cases)
            # Solve for [Exposed Controls, Unexposed Controls]
            true_controls = np.linalg.solve(matrix_ctrl, obs_controls)
            
            # Put them back into 2x2 shape
            true_2x2 = np.column_stack((true_cases, true_controls))
            return true_2x2
        
        true_counts = correct_differential(
            observed_data, sens_cases, spec_cases,
            sens_controls, spec_controls
        )
        
        # Calculate observed OR from original 2x2
        a_obs, b_obs = observed_data[0,0], observed_data[0,1]
        c_obs, d_obs = observed_data[1,0], observed_data[1,1]
        
        observed_or = (a_obs * d_obs)/(b_obs * c_obs) if (b_obs * c_obs) != 0 else np.nan
        
        # Calculate corrected OR from true_counts
        a_true, b_true = true_counts[0,0], true_counts[0,1]
        c_true, d_true = true_counts[1,0], true_counts[1,1]
        
        true_or = (a_true * d_true)/(b_true * c_true) if (b_true * c_true) != 0 else np.nan
    
    # =============================
    # Display results
    # =============================
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Observed OR", f"{float(observed_or):.2f}")
    with col2:
        st.metric("Corrected OR", f"{float(true_or):.2f}")
    
    # =============================
    # Sensitivity Analysis Visualization (Non-differential Only)
    # =============================
    if error_type == "Non-differential":
        sens_range = np.linspace(0.5, 1, 50)
        spec_range = np.linspace(0.5, 1, 50)
        
        # Vectorize a function that returns the corrected OR given sens, spec
        def calculate_corrected_or(sens, spec):
            # Recompute corrected counts for Exposed
            a_true, b_true = correct_counts(a_obs, b_obs, sens, spec)
            # Recompute corrected counts for Unexposed
            c_true, d_true = correct_counts(c_obs, d_obs, sens, spec)
            
            denom1 = (b_true * c_true)
            if abs(denom1) < 1e-15:
                return np.nan
            return (a_true * d_true)/denom1
        
        calc_corrected_vectorized = np.vectorize(calculate_corrected_or)
        
        SENS, SPEC = np.meshgrid(sens_range, spec_range)
        CORRECTED_OR = calc_corrected_vectorized(SENS, SPEC)
        
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Contour(
            z=CORRECTED_OR,
            x=sens_range,
            y=spec_range,
            colorscale='Viridis',
            colorbar=dict(title='Corrected OR')
        ))
        
        fig.update_layout(
            title='Sensitivity Analysis for Non-differential Measurement Error',
            xaxis_title='Sensitivity',
            yaxis_title='Specificity'
        )
        
        st.plotly_chart(fig, use_container_width=True)


elif method == "Multiple Bias Analysis":
    st.header("Multiple Bias Analysis")
    
    # Parameters for multiple bias sources
    st.subheader("Observed Data")
    observed_effect = st.number_input("Observed Effect Estimate", 0.1, 10.0, 2.0)
    
    # Unmeasured confounding parameters
    st.subheader("Unmeasured Confounding")
    conf_strength = st.slider("Confounding Strength", 1.0, 5.0, 2.0)
    conf_imbalance = st.slider("Confounder Imbalance", 0.0, 1.0, 0.3)
    
    # Selection bias parameters
    st.subheader("Selection Bias")
    selection_ratio = st.slider("Selection Ratio", 0.1, 2.0, 1.0)
    
    # Measurement error parameters
    st.subheader("Measurement Error")
    sensitivity = st.slider("Sensitivity", 0.5, 1.0, 0.8)
    specificity = st.slider("Specificity", 0.5, 1.0, 0.9)
    
    # Monte Carlo simulation for multiple bias analysis
    n_simulations = 1000
    
    def simulate_multiple_bias(n_sims, obs_effect, conf_str, conf_imb, 
                             sel_ratio, sens, spec):
        # Initialize results
        adjusted_effects = np.zeros(n_sims)
        
        for i in range(n_sims):
            # Add random variation to bias parameters
            conf_effect = conf_str * np.random.lognormal(0, 0.1)
            conf_diff = conf_imb * np.random.normal(0, 0.1)
            sel_effect = sel_ratio * np.random.lognormal(0, 0.1)
            sens_i = sens * np.random.normal(1, 0.05)
            spec_i = spec * np.random.normal(1, 0.05)
            
            # Apply successive bias adjustments
            effect = obs_effect
            # Confounding adjustment
            effect = effect / conf_effect ** conf_diff
            # Selection bias adjustment
            effect = effect / sel_effect
            # Measurement error adjustment
            effect = effect * (sens_i + spec_i - 1)
            
            adjusted_effects[i] = effect
        
        return adjusted_effects
    
    results = simulate_multiple_bias(
        n_simulations, observed_effect, conf_strength, conf_imbalance,
        selection_ratio, sensitivity, specificity
    )
    
    # Display results
    st.subheader("Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Median Adjusted Effect", f"{np.median(results):.2f}")
    with col2:
        st.metric("95% CI Lower", f"{np.percentile(results, 2.5):.2f}")
    with col3:
        st.metric("95% CI Upper", f"{np.percentile(results, 97.5):.2f}")
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=results,
        nbinsx=30,
        name='Adjusted Effects'
    ))
    
    fig.add_vline(
        x=observed_effect,
        line_dash="dash",
        line_color='red',
                annotation=dict(
                    text="Observed Effect",
                    yref="paper",
                    y=1.0
                    )
                )

    fig.update_layout(
        title='Distribution of Bias-Adjusted Effect Estimates',
        xaxis_title='Effect Estimate',
        yaxis_title='Frequency',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Tornado plot for sensitivity to different bias sources
    base_effect = np.median(results)
    
    # Calculate effect of varying each bias parameter
    def calculate_parameter_impact(parameter_name, base_value, range_mult=0.5):
        low_value = base_value * (1 - range_mult)
        high_value = base_value * (1 + range_mult)
        
        if parameter_name == 'conf_strength':
            low_results = simulate_multiple_bias(100, observed_effect, low_value, 
                conf_imbalance, selection_ratio, sensitivity, specificity)
            high_results = simulate_multiple_bias(100, observed_effect, high_value, 
                conf_imbalance, selection_ratio, sensitivity, specificity)
        elif parameter_name == 'conf_imbalance':
            low_results = simulate_multiple_bias(100, observed_effect, conf_strength, 
                low_value, selection_ratio, sensitivity, specificity)
            high_results = simulate_multiple_bias(100, observed_effect, conf_strength, 
                high_value, selection_ratio, sensitivity, specificity)
        elif parameter_name == 'selection_ratio':
            low_results = simulate_multiple_bias(100, observed_effect, conf_strength, 
                conf_imbalance, low_value, sensitivity, specificity)
            high_results = simulate_multiple_bias(100, observed_effect, conf_strength, 
                conf_imbalance, high_value, sensitivity, specificity)
        else:  # measurement parameters
            low_results = simulate_multiple_bias(100, observed_effect, conf_strength, 
                conf_imbalance, selection_ratio, low_value, low_value)
            high_results = simulate_multiple_bias(100, observed_effect, conf_strength, 
                conf_imbalance, selection_ratio, high_value, high_value)
            
        return np.median(low_results), np.median(high_results)
    
    parameters = {
        'Confounding Strength': (conf_strength, 'conf_strength'),
        'Confounder Imbalance': (conf_imbalance, 'conf_imbalance'),
        'Selection Ratio': (selection_ratio, 'selection_ratio'),
        'Measurement Accuracy': (sensitivity, 'measurement')
    }
    
    tornado_data = []
    for param_name, (base_value, param_key) in parameters.items():
        low, high = calculate_parameter_impact(param_key, base_value)
        tornado_data.append({
            'Parameter': param_name,
            'Low': low,
            'High': high,
            'Range': high - low
        })
    
    tornado_df = pd.DataFrame(tornado_data)
    tornado_df = tornado_df.sort_values('Range', ascending=True)
    
    # Create tornado plot
    fig = go.Figure()
    
    for idx, row in tornado_df.iterrows():
        fig.add_trace(go.Bar(
            y=[row['Parameter']],
            x=[row['High'] - base_effect],
            orientation='h',
            name=f"{row['Parameter']} High",
            showlegend=False,
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            y=[row['Parameter']],
            x=[row['Low'] - base_effect],
            orientation='h',
            name=f"{row['Parameter']} Low",
            showlegend=False,
            marker_color='blue'
        ))
    
    fig.update_layout(
        title='Tornado Plot: Sensitivity to Bias Parameters',
        xaxis_title='Change in Effect Estimate',
        yaxis_title='Bias Parameter',
        barmode='overlay',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add educational content
st.header("Method Description")

if method == "Unmeasured Confounding":
    st.write("""
    Unmeasured confounding analysis examines how strong an unmeasured confounder 
    would need to be to explain away the observed association. Key components:
    
    1. Confounder prevalence in exposed/unexposed groups
    2. Strength of confounder-outcome association
    3. Impact on observed effect estimate
    """)

elif method == "Selection Bias":
    st.write("""
    Selection bias analysis quantifies how differential selection into the study 
    may have affected the results. Considerations include:
    
    1. Selection probabilities by exposure and outcome
    2. Impact on odds ratio estimation
    3. Bounds on bias from selection
    """)

elif method == "Measurement Error":
    st.write("""
    Measurement error analysis examines the impact of misclassification:
    
    1. Differential vs non-differential error
    2. Sensitivity and specificity of measurement
    3. Impact on effect estimation
    4. Correction methods
    """)

elif method == "Multiple Bias Analysis":
    st.write("""
    Multiple bias analysis combines various sources of bias:
    
    1. Joint impact of multiple bias sources
    2. Probabilistic analysis of uncertainty
    3. Relative importance of different biases
    4. Overall bounds on true effect
    """)

# Add references
st.header("Further Reading")
st.write("""
1. Lash TL, et al. Good Practices for Quantitative Bias Analysis
2. VanderWeele TJ, Ding P. Sensitivity Analysis in Observational Research
3. Greenland S. Multiple-bias Modelling for Analysis of Observational Data
""")       

st.header("Check your understanding")
if method == "Unmeasured Confounding":
    quiz_unmeasured = st.radio(
        "What is the main concern with unmeasured confounding?",
        [
            "It can lead to an overestimate or underestimate of the true effect",
            "It only affects case-control studies",
            "It is not a problem if the sample size is large",
            "It can be completely eliminated with statistical adjustment"
        ]
    )
    if quiz_unmeasured == "It can lead to an overestimate or underestimate of the true effect":
        st.success("Correct! Unmeasured confounding can distort the true effect size in either direction.")
    else:
        st.error("Not quite. The primary concern is that unmeasured confounding can bias the estimated effect.")
        
elif method == "Selection Bias":
    quiz_selection = st.radio(
        "Which of the following scenarios best describes selection bias?",
        [
            "Participants are randomly assigned to treatment and control groups",
            "The selection of participants into the study is related to both exposure and outcome",
            "All confounding variables have been measured and adjusted for",
            "A measurement tool is not accurately capturing the exposure"
        ]
    )
    if quiz_selection == "The selection of participants into the study is related to both exposure and outcome":
        st.success("Correct! Selection bias occurs when participation in the study depends on both exposure and outcome.")
    else:
        st.error("Not quite. Selection bias arises when the way participants enter a study affects the results.")

elif method == "Measurement Error":
    quiz_measurement = st.radio(
        "What is the main difference between differential and non-differential measurement error?",
        [
            "Non-differential error occurs equally across groups, while differential error depends on exposure or outcome status",
            "Non-differential error does not affect study results",
            "Differential error can only occur in case-control studies",
            "Non-differential error always leads to overestimation of the effect"
        ]
    )
    if quiz_measurement == "Non-differential error occurs equally across groups, while differential error depends on exposure or outcome status":
        st.success("Correct! Non-differential error is independent of exposure and outcome, whereas differential error is related to one or both.")
    else:
        st.error("Not quite. Non-differential misclassification is independent of exposure or outcome, but differential misclassification is not.")

elif method == "Multiple Bias Analysis":
    quiz_multiple_bias = st.radio(
        "Why is multiple bias analysis important?",
        [
            "It accounts for different sources of bias simultaneously",
            "It replaces the need for randomized controlled trials",
            "It guarantees a completely unbiased estimate",
            "It only applies to case-control studies"
        ]
    )
    if quiz_multiple_bias == "It accounts for different sources of bias simultaneously":
        st.success("Correct! Multiple bias analysis helps assess the combined impact of different types of bias in a study.")
    else:
        st.error("Not quite. Multiple bias analysis does not remove bias, but it allows researchers to estimate its impact.")
