# pages/15_target_trial.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from scipy import stats

st.title("Target Trial Emulation in Epidemiology")

# Method selector
method = st.selectbox(
    "Select Method",
    ["Trial Design", "Eligibility Assessment", 
     "Treatment Strategy", "Outcome Assessment",
     "Causal Analysis"]
)

# Add these helper functions before the method selector
def create_timeline_plot(treatment_type, outcome_type, followup_time=12, n_timepoints=3):
    """Create a Gantt chart timeline for the trial design"""
    # Create timeline data
    start = datetime.now()
    df = []
    
    # Enrollment period (3 months)
    df.append(dict(Task="Enrollment", 
                  Start=start, 
                  Finish=start + timedelta(days=90),
                  Resource='Planning'))
    
    # Treatment period
    treatment_start = start + timedelta(days=90)
    treatment_end = treatment_start + timedelta(days=30*followup_time)
    
    df.append(dict(Task="Treatment", 
                  Start=treatment_start,
                  Finish=treatment_end,
                  Resource='Intervention'))
    
    # Add assessment points for time-varying treatment
    if treatment_type == "Time-varying":
        interval = followup_time / n_timepoints
        for i in range(n_timepoints):
            assessment_date = treatment_start + timedelta(days=30*interval*i)
            df.append(dict(Task=f"Assessment {i+1}",
                         Start=assessment_date,
                         Finish=assessment_date + timedelta(days=5),
                         Resource='Milestone'))
    
    # Outcome assessment
    if outcome_type == "Time-to-event":
        df.append(dict(Task="Outcome Monitoring",
                      Start=treatment_start,
                      Finish=treatment_end,
                      Resource='Outcome'))
    else:
        df.append(dict(Task="Outcome Assessment",
                      Start=treatment_end - timedelta(days=30),
                      Finish=treatment_end,
                      Resource='Outcome'))
    
    colors = {'Planning': 'rgb(46, 137, 205)',
              'Intervention': 'rgb(114, 44, 121)',
              'Milestone': 'rgb(198, 47, 105)',
              'Outcome': 'rgb(58, 149, 136)'}
    
    fig = ff.create_gantt(df, 
                         colors=colors,
                         index_col='Resource',
                         title='Trial Timeline',
                         show_colorbar=True,
                         group_tasks=True,
                         showgrid_x=True,
                         showgrid_y=True)
    
    fig.update_layout(height=300)
    return fig

def calculate_sample_size(outcome_type, effect_size=0.15, power=0.8, alpha=0.05):
    """Calculate estimated sample size based on trial parameters"""
    if outcome_type == "Binary":
        # Sample size calculation for binary outcome
        baseline_rate = 0.3
        p1 = baseline_rate
        p2 = baseline_rate + effect_size
        p = (p1 + p2) / 2
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * p * (1-p) * (z_alpha + z_beta)**2 / effect_size**2
        return int(np.ceil(n))
    
    elif outcome_type == "Continuous":
        # Sample size calculation for continuous outcome
        std_dev = 1.0  # Assumed standard deviation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) * std_dev / effect_size)**2
        return int(np.ceil(n))
    
    else:  # Time-to-event
        # Sample size calculation for survival outcome
        hazard_ratio = 0.7  # Expected hazard ratio
        prob_event = 0.3    # Expected event probability
        
        n = -(4 * (z_alpha + z_beta)**2) / \
            (prob_event * np.log(hazard_ratio)**2)
        return int(np.ceil(n))

def plot_eligibility_distribution(min_age, max_age, include_comorbidity):
    """Create a visualization of the expected study population"""
    # Generate synthetic population data
    n_samples = 1000
    age_data = np.random.normal(50, 15, n_samples)
    
    # Apply eligibility criteria
    eligible = (age_data >= min_age) & (age_data <= max_age)
    
    # Create distribution plot
    fig = go.Figure()
    
    # Add overall population distribution
    fig.add_trace(go.Histogram(
        x=age_data,
        name='Total Population',
        opacity=0.5
    ))
    
    # Add eligible population distribution
    fig.add_trace(go.Histogram(
        x=age_data[eligible],
        name='Eligible Population',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Expected Study Population Distribution',
        xaxis_title='Age',
        yaxis_title='Count',
        barmode='overlay',
        height=300
    )
    
    return fig

if method == "Trial Design":
    st.header("Target Trial Design Specification")
    
    # Trial design parameters
    st.subheader("Protocol Components")
    
    # Eligibility criteria
    st.write("**1. Eligibility Criteria**")
    col1, col2 = st.columns(2)
    with col1:
        min_age = st.number_input("Minimum Age", 18, 100, 18)
        max_age = st.number_input("Maximum Age", min_age, 100, 80)
    with col2:
        include_comorbidity = st.multiselect(
            "Include Comorbidities",
            ["Hypertension", "Diabetes", "Heart Disease", "None"],
            ["None"]
        )
    
    # Treatment strategies
    st.write("**2. Treatment Strategies**")
    treatment_type = st.selectbox(
        "Treatment Type",
        ["Binary", "Time-varying", "Dose-dependent"]
    )
    
    if treatment_type == "Binary":
        treatment_options = st.text_input(
            "Treatment Options (comma-separated)",
            "Treatment A, Treatment B"
        ).split(",")
    elif treatment_type == "Time-varying":
        n_timepoints = st.slider("Number of Time Points", 2, 10, 3)
        st.write(f"Will assess treatment at {n_timepoints} time points")
    else:
        dose_levels = st.slider("Number of Dose Levels", 2, 5, 3)
        st.write(f"Will compare {dose_levels} dose levels")
    
    # Outcome
    st.write("**3. Outcome Definition**")
    outcome_type = st.selectbox(
        "Outcome Type",
        ["Binary", "Continuous", "Time-to-event"]
    )
    
    if outcome_type == "Time-to-event":
        followup_time = st.slider("Follow-up Time (months)", 1, 60, 12)
    
    # Generate example protocol
    st.subheader("Trial Protocol Summary")
    
    protocol_text = f"""
    **Target Trial Protocol**
    
    1. **Eligibility Criteria**:
       - Age: {min_age}-{max_age} years
       - Comorbidities: {', '.join(include_comorbidity)}
    
    2. **Treatment Strategy**:
       - Type: {treatment_type}
       {"- Options: " + ', '.join(treatment_options) if treatment_type == "Binary" else ""}
       {"- Time points: " + str(n_timepoints) if treatment_type == "Time-varying" else ""}
       {"- Dose levels: " + str(dose_levels) if treatment_type == "Dose-dependent" else ""}
    
    3. **Outcome**:
       - Type: {outcome_type}
       {"- Follow-up: " + str(followup_time) + " months" if outcome_type == "Time-to-event" else ""}
    """
    
    st.markdown(protocol_text)
if method == "Trial Design":
    st.header("Target Trial Design Specification")
    
    # [Your existing eligibility criteria, treatment strategy, and outcome code stays here]
    
    # After st.markdown(protocol_text), add:
    
    # Display interactive timeline
    st.subheader("Trial Timeline Visualization")
    timeline_fig = create_timeline_plot(
        treatment_type=treatment_type,
        outcome_type=outcome_type,
        followup_time=followup_time if 'followup_time' in locals() else 12,
        n_timepoints=n_timepoints if 'n_timepoints' in locals() else 3
    )
    st.plotly_chart(timeline_fig, use_container_width=True)

    # Display sample size calculation
    st.subheader("Sample Size Estimation")
    col1, col2 = st.columns(2)

    with col1:
        effect_size = st.slider("Expected Effect Size", 0.1, 0.5, 0.15, 0.05)
        power = st.slider("Statistical Power", 0.7, 0.9, 0.8, 0.05)

    with col2:
        alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05, 0.01)
        
    estimated_n = calculate_sample_size(
        outcome_type=outcome_type,
        effect_size=effect_size,
        power=power,
        alpha=alpha
    )

    st.metric("Estimated Required Sample Size", estimated_n, 
              help="Based on specified effect size, power, and significance level")

    # Display population distribution
    st.subheader("Expected Study Population")
    population_fig = plot_eligibility_distribution(
        min_age=min_age,
        max_age=max_age,
        include_comorbidity=include_comorbidity
    )
    st.plotly_chart(population_fig, use_container_width=True)

    # Add key assumptions and notes
    st.info("""
    **Key Assumptions:**
    - Randomization ratio: 1:1
    - Loss to follow-up: 20%
    - Analysis will use intention-to-treat principle
    """)

    # Add risk assessment
    risk_level = len(include_comorbidity) > 1 or outcome_type == "Time-to-event"
    if risk_level:
        st.warning("""
        **Risk Considerations:**
        - Multiple comorbidities may increase complexity
        - Extended follow-up period may lead to higher dropout
        - Consider interim analyses
        """)
        
elif method == "Eligibility Assessment":
    st.header("Eligibility Assessment and Cohort Selection")
    
    # Generate synthetic patient data
    n_patients = st.slider("Sample Size", 100, 1000, 500)
    
    def generate_patient_data(n):
        data = pd.DataFrame({
            'age': np.random.normal(50, 15, n),
            'sex': np.random.binomial(1, 0.5, n),
            'comorbidity_score': np.random.poisson(2, n),
            'prior_treatment': np.random.binomial(1, 0.3, n),
            'contraindication': np.random.binomial(1, 0.1, n)
        })
        return data
    
    data = generate_patient_data(n_patients)
    
    # Eligibility criteria
    st.subheader("Set Eligibility Criteria")
    
    col1, col2 = st.columns(2)
    with col1:
        min_age = st.slider("Minimum Age", 18, 80, 18)
        max_age = st.slider("Maximum Age", min_age, 100, 80)
    with col2:
        max_comorbidity = st.slider("Maximum Comorbidity Score", 0, 10, 5)
    
    exclude_prior = st.checkbox("Exclude Patients with Prior Treatment")
    exclude_contraindication = st.checkbox("Exclude Contraindications")
    
    # Apply eligibility criteria
    eligible = (
        (data['age'] >= min_age) &
        (data['age'] <= max_age) &
        (data['comorbidity_score'] <= max_comorbidity)
    )
    
    if exclude_prior:
        eligible &= ~data['prior_treatment']
    if exclude_contraindication:
        eligible &= ~data['contraindication']
    
    data['eligible'] = eligible
    
    # Display results
    st.subheader("Eligibility Assessment Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Patients", n_patients)
        st.metric("Eligible Patients", sum(eligible))
    with col2:
        st.metric("Exclusion Rate", f"{(1 - sum(eligible)/n_patients):.1%}")
    
    # Visualize characteristics
    fig = go.Figure()
    
    # Age distribution
    fig.add_trace(go.Histogram(
        x=data[eligible]['age'],
        name='Eligible',
        opacity=0.75
    ))
    
    fig.add_trace(go.Histogram(
        x=data[~eligible]['age'],
        name='Excluded',
        opacity=0.75
    ))
    
    fig.update_layout(
        title='Age Distribution by Eligibility',
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Characteristics table
    characteristics = pd.DataFrame({
        'Characteristic': ['Age (mean ± SD)', 'Sex (% female)', 
                         'Comorbidity Score (mean ± SD)'],
        'Eligible': [
            f"{data[eligible]['age'].mean():.1f} ± {data[eligible]['age'].std():.1f}",
            f"{(data[eligible]['sex']).mean():.1%}",
            f"{data[eligible]['comorbidity_score'].mean():.1f} ± {data[eligible]['comorbidity_score'].std():.1f}"
        ],
        'Excluded': [
            f"{data[~eligible]['age'].mean():.1f} ± {data[~eligible]['age'].std():.1f}",
            f"{(data[~eligible]['sex']).mean():.1%}",
            f"{data[~eligible]['comorbidity_score'].mean():.1f} ± {data[~eligible]['comorbidity_score'].std():.1f}"
        ]
    })
    
    st.table(characteristics)

elif method == "Treatment Strategy":
    st.header("Treatment Strategy Definition")
    
    # Parameters
    n_patients = st.slider("Sample Size", 100, 1000, 500)
    n_timepoints = st.slider("Number of Time Points", 2, 10, 4)
    
    # Generate synthetic treatment data
    def generate_treatment_data(n, t):
        data = pd.DataFrame(index=range(n))
        
        # Baseline characteristics
        data['age'] = np.random.normal(50, 15, n)
        data['sex'] = np.random.binomial(1, 0.5, n)
        data['severity'] = np.random.choice(['Mild', 'Moderate', 'Severe'], n)
        
        # Time-varying treatment
        for i in range(t):
            # Treatment influenced by severity and previous treatment
            if i == 0:
                prob_treat = np.where(data['severity'] == 'Severe', 0.8,
                                    np.where(data['severity'] == 'Moderate', 0.5, 0.2))
            else:
                prev_treat = data[f'treatment_{i-1}']
                prob_treat = np.where(prev_treat == 1, 0.8, 0.2)
            
            data[f'treatment_{i}'] = np.random.binomial(1, prob_treat)
        
        return data
    
    data = generate_treatment_data(n_patients, n_timepoints)
    
    # Treatment strategy options
    st.subheader("Define Treatment Strategies")
    
    strategy_type = st.selectbox(
        "Strategy Type",
        ["Static", "Dynamic"]
    )
    
    if strategy_type == "Static":
        st.write("Static Strategy: Same treatment decision throughout follow-up")
        treatment_threshold = st.slider(
            "Treatment Probability Threshold",
            0.0, 1.0, 0.5
        )
    else:
        st.write("Dynamic Strategy: Treatment decision based on time-varying factors")
        st.write("Treatment initiated if:")
        severity_threshold = st.selectbox(
            "Minimum Severity",
            ['Mild', 'Moderate', 'Severe']
        )
    
    # Apply strategies
    def apply_strategy(row, static=True):
        if static:
            return all(row[[f'treatment_{i}' for i in range(n_timepoints)]] > treatment_threshold)
        else:
            return row['severity'] >= severity_threshold
    
    data['strategy_adherent'] = data.apply(
        lambda x: apply_strategy(x, static=(strategy_type=="Static")),
        axis=1
    )
    
    # Visualize treatment patterns
    st.subheader("Treatment Patterns")
    
    # Treatment trajectories
    treatment_cols = [f'treatment_{i}' for i in range(n_timepoints)]
    pattern_counts = data[treatment_cols].value_counts().reset_index(name='count')
    pattern_counts['pattern'] = pattern_counts[treatment_cols].apply(
        lambda x: ''.join(x.astype(str)), axis=1
    )
    
    fig = px.bar(
        pattern_counts,
        x='pattern',
        y='count',
        title='Treatment Pattern Frequencies'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy adherence
    st.metric(
        "Strategy Adherence",
        f"{data['strategy_adherent'].mean():.1%}"
    )

elif method == "Outcome Assessment":
    st.header("Outcome Assessment")
    
    # Parameters
    n_patients = st.slider("Sample Size", 100, 1000, 500)
    followup_time = st.slider("Follow-up Time (months)", 6, 36, 12)
    
    # Generate outcome data
    def generate_outcome_data(n, t):
        data = pd.DataFrame({
            'age': np.random.normal(50, 15, n),
            'treatment': np.random.binomial(1, 0.5, n),
            'risk_score': np.random.normal(0, 1, n)
        })
        
        # Generate time-to-event outcome
        baseline_hazard = 0.1
        treatment_effect = -0.5
        
        lambda_i = baseline_hazard * np.exp(
            treatment_effect * data['treatment'] +
            0.02 * data['age'] +
            0.5 * data['risk_score']
        )
        
        data['time_to_event'] = np.random.exponential(1/lambda_i)
        data['censoring_time'] = np.random.uniform(0, t, n)
        
        data['observed_time'] = np.minimum(data['time_to_event'], 
                                         data['censoring_time'])
        data['event'] = (data['time_to_event'] <= data['censoring_time']).astype(int)
        
        return data
    
    data = generate_outcome_data(n_patients, followup_time)
    
    # Outcome definition
    st.subheader("Outcome Definition")
    
    outcome_type = st.selectbox(
        "Outcome Type",
        ["Time-to-event", "Binary at fixed time"]
    )
    
    if outcome_type == "Binary at fixed time":
        assessment_time = st.slider(
            "Assessment Time (months)",
            1, followup_time, followup_time//2
        )
        data['outcome'] = (data['time_to_event'] <= assessment_time).astype(int)
    
    # Visualize outcomes
    st.subheader("Outcome Analysis")
    
    if outcome_type == "Time-to-event":
        # Kaplan-Meier curves
        from lifelines import KaplanMeierFitter
        
        kmf = KaplanMeierFitter()
        
        fig = go.Figure()
        
        for treatment in [0, 1]:
            mask = data['treatment'] == treatment
            kmf.fit(
                data.loc[mask, 'observed_time'],
                data.loc[mask, 'event'],
                label=f'Treatment={treatment}'
            )
            
            fig.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_.values.flatten(),
                name=f'Treatment {treatment}'
            ))
        
        fig.update_layout(
            title='Survival Curves by Treatment',
            xaxis_title='Time',
            yaxis_title='Survival Probability'
        )
        
    else:
        # Binary outcome analysis
        outcome_by_treatment = pd.crosstab(
            data['treatment'],
            data['outcome']
        )
        
        fig = px.bar(
            outcome_by_treatment,
            title='Outcomes by Treatment Group'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Effect estimates
    st.subheader("Effect Estimates")
    
    if outcome_type == "Time-to-event":
        from lifelines import CoxPHFitter
        cph = CoxPHFitter()
        model_df = data[['observed_time', 'event', 'treatment', 'age', 'risk_score']]
        cph.fit(model_df, duration_col='observed_time', event_col='event')
        
        st.write("Cox Proportional Hazards Model Results:")
        st.write(pd.DataFrame({
            'Hazard Ratio': np.exp(cph.params_),
            'p-value': cph.summary['p']
        }))
        
        st.write("Cox Proportional Hazards Model Results:")
        st.write(pd.DataFrame({
            'Hazard Ratio': np.exp(cph.params_),
            'p-value': cph.summary['p']
        }))
    else:
        model = LogisticRegression()
        X = data[['treatment', 'age', 'risk_score']]
        y = data['outcome']
        
        model.fit(X, y)
        
        st.write("Logistic Regression Results:")
        st.write(pd.DataFrame({
            'Odds Ratio': np.exp(model.coef_[0]),
            'p-value': stats.norm.sf(abs(model.coef_[0]/np.sqrt(1/len(y))))*2
        }).round(3))

elif method == "Causal Analysis":
    st.header("Causal Analysis")
    
    # Generate synthetic data for causal analysis
    n_patients = st.slider("Sample Size", 100, 1000, 500)
    
    def generate_causal_data(n):
        # Baseline confounders
        age = np.random.normal(50, 15, n)
        comorbidity = np.random.poisson(2, n)
        severity = np.random.normal(0, 1, n)
        
        # Treatment assignment affected by confounders
        ps_logit = -1 + 0.02*age + 0.3*comorbidity + 0.5*severity
        ps = 1 / (1 + np.exp(-ps_logit))
        treatment = np.random.binomial(1, ps)
        
        # Outcome affected by treatment and confounders
        outcome_prob = 1 / (1 + np.exp(
            -(0.01*age + 0.2*comorbidity + 0.4*severity - 0.5*treatment)
        ))
        outcome = np.random.binomial(1, outcome_prob)
        
        return pd.DataFrame({
            'age': age,
            'comorbidity': comorbidity,
            'severity': severity,
            'treatment': treatment,
            'outcome': outcome,
            'ps': ps
        })
    
    data = generate_causal_data(n_patients)
    
    # Analysis methods
    analysis_method = st.selectbox(
        "Analysis Method",
        ["Propensity Score Matching", "Inverse Probability Weighting", 
         "Standardization"]
    )
    
    if analysis_method == "Propensity Score Matching":
        # Matching parameters
        caliper = st.slider("Caliper Width (SD)", 0.1, 1.0, 0.2)
        
        # Perform matching
        def match_on_ps(df, caliper_width):
            treated = df[df['treatment'] == 1]
            control = df[df['treatment'] == 0]
            
            matches = []
            used_control = set()
            
            for _, treated_unit in treated.iterrows():
                ps_treated = treated_unit['ps']
                
                # Find eligible matches
                eligible = control[
                    (abs(control['ps'] - ps_treated) < caliper_width)
                ]
                
                if not eligible.empty:
                    # Find closest match
                    distances = abs(eligible['ps'] - ps_treated)
                    best_match_idx = distances.idxmin()
                    
                    if best_match_idx not in used_control:
                        matches.append(pd.concat([
                            treated_unit.to_frame().T,
                            control.loc[[best_match_idx]]
                        ]))
                        used_control.add(best_match_idx)
            
            return pd.concat(matches) if matches else pd.DataFrame()
        
        matched_data = match_on_ps(data, caliper)
        
        # Display matching results
        st.subheader("Matching Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Matched Pairs", len(matched_data)//2)
        with col2:
            st.metric("Match Rate", f"{len(matched_data)/len(data):.1%}")
        
        # Balance plot
        fig = go.Figure()
        
        for var in ['age', 'comorbidity', 'severity']:
            smd_before = (data[data['treatment']==1][var].mean() - 
                         data[data['treatment']==0][var].mean()) / \
                        data[var].std()
            
            smd_after = (matched_data[matched_data['treatment']==1][var].mean() - 
                        matched_data[matched_data['treatment']==0][var].mean()) / \
                        matched_data[var].std()
            
            fig.add_trace(go.Scatter(
                x=[abs(smd_before), abs(smd_after)],
                y=[var, var],
                mode='markers',
                name=var
            ))
        
        fig.add_vline(x=0.1, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Standardized Mean Differences Before/After Matching',
            xaxis_title='Absolute Standardized Mean Difference',
            yaxis_title='Variable',
            xaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Treatment effect estimation
        effect = (matched_data[matched_data['treatment']==1]['outcome'].mean() - 
                 matched_data[matched_data['treatment']==0]['outcome'].mean())
        
        st.metric("Average Treatment Effect", f"{effect:.3f}")
        
    elif analysis_method == "Inverse Probability Weighting":
        # Calculate weights
        data['ipw'] = np.where(
            data['treatment'] == 1,
            1/data['ps'],
            1/(1-data['ps'])
        )
        
        # Trim extreme weights
        weight_threshold = st.slider("Weight Trimming Percentile", 90, 99, 95)
        max_weight = np.percentile(data['ipw'], weight_threshold)
        data['ipw_trimmed'] = np.minimum(data['ipw'], max_weight)
        
        # Display weight distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data['ipw_trimmed'],
            nbinsx=30,
            name='Trimmed Weights'
        ))
        
        fig.update_layout(
            title='Distribution of IPW Weights',
            xaxis_title='Weight',
            yaxis_title='Count'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weighted analysis
        weighted_effect = np.average(
            data['outcome'],
            weights=data['ipw_trimmed'],
            groups=data['treatment']
        )
        weighted_effect = weighted_effect[1] - weighted_effect[0]
        
        st.metric("IPW Treatment Effect", f"{weighted_effect:.3f}")
        
    else:  # Standardization
        # Fit outcome model
        X = data[['age', 'comorbidity', 'severity', 'treatment']]
        y = data['outcome']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Predict outcomes under both treatment scenarios
        data_t1 = data.copy()
        data_t1['treatment'] = 1
        data_t0 = data.copy()
        data_t0['treatment'] = 0
        
        pred_t1 = model.predict_proba(data_t1[X.columns])[:, 1]
        pred_t0 = model.predict_proba(data_t0[X.columns])[:, 1]
        
        # Calculate standardized effect
        effect = np.mean(pred_t1 - pred_t0)
        
        st.metric("Standardized Treatment Effect", f"{effect:.3f}")
        
        # Plot individual treatment effects
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pred_t1 - pred_t0,
            nbinsx=30,
            name='Individual Effects'
        ))
        
        fig.update_layout(
            title='Distribution of Individual Treatment Effects',
            xaxis_title='Effect Size',
            yaxis_title='Count'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add educational content
st.header("Method Details")

if method == "Trial Design":
    st.write("""
    **Trial design components:**
    1. Eligibility criteria
    2. Treatment strategies
    3. Outcome definition
    4. Follow-up period
    5. Causal contrast of interest
    """)

elif method == "Eligibility Assessment":
    st.write("""
    **Key considerations:**
    1. Clear inclusion/exclusion criteria
    2. Baseline covariate assessment
    3. Treatment history evaluation
    4. Contraindication screening
    """)

elif method == "Treatment Strategy":
    st.write("""
    **Strategy types:**
    1. Static strategies
    2. Dynamic strategies
    3. Time-varying treatments
    4. Treatment adherence
    """)

elif method == "Outcome Assessment":
    st.write("""
    **Assessment components:**
    1. Outcome definition
    2. Measurement timing
    3. Competing events
    4. Missing data handling
    """)

elif method == "Causal Analysis":
    st.write("""
    **Analysis methods:**
    1. Propensity score matching
    2. Inverse probability weighting
    3. Standardization
    4. Sensitivity analysis
    """)

# Add references
st.header("Further Reading")
st.write("""
1. Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available
2. Danaei G, et al. Guidelines for Conducting and Reporting Target Trial Emulation Studies
3. Lodi S, et al. Effect Estimates in Randomized Trials and Observational Studies
""")       

st.header("Check your understanding")
if method == "Trial Design":
    quiz_trial = st.radio(
        "What is the main goal of a target trial emulation?",
        [
            "To replace all randomized controlled trials",
            "To analyze observational data while minimizing bias",
            "To predict future clinical trial results",
            "To compare multiple observational datasets"
        ]
    )
    if quiz_trial == "To analyze observational data while minimizing bias":
        st.success("Correct! The goal of target trial emulation is to structure observational data like an RCT to reduce bias.")
    else:
        st.error("Not quite. The main purpose is to emulate an RCT using observational data.")


elif method == "Eligibility Assessment":  
    quiz_eligibility = st.radio(
        "Why is eligibility assessment important in target trials?",
        [
            "To increase the sample size as much as possible",
            "To minimize bias and create a comparable study population",
            "To exclude all patients with comorbidities",
            "To ensure that treatment is assigned randomly"
        ]
    )
    if quiz_eligibility == "To minimize bias and create a comparable study population":
        st.success("Correct! Eligibility criteria help create a comparable study population and reduce confounding bias.")
    else:
        st.error("Not quite. The goal is to create a comparable population while reducing bias.")

elif method == "Treatment Strategy":
    quiz_treatment = st.radio(
        "What is the key difference between a static and a dynamic treatment strategy?",
        [
            "Static strategies change treatment decisions over time, while dynamic strategies remain the same",
            "Static strategies involve random treatment assignment",
            "Dynamic strategies adjust based on patient characteristics over time",
            "Static strategies only include a single dose level"
        ]
    )
    if quiz_treatment == "Dynamic strategies adjust based on patient characteristics over time":
        st.success("Correct! Dynamic strategies allow treatment decisions to change based on patient characteristics.")
    else:
        st.error("Not quite. Dynamic strategies modify treatment based on evolving patient conditions.")

elif method == "Outcome Assessment":
    quiz_outcome = st.radio(
        "Which outcome type is most commonly used in survival analysis?",
        [
            "Binary outcomes",
            "Continuous outcomes",
            "Time-to-event outcomes",
            "Dose-response outcomes"
        ]
    )
    if quiz_outcome == "Time-to-event outcomes":
        st.success("Correct! Time-to-event outcomes are central to survival analysis, measuring the time until an event occurs.")
    else:
        st.error("Not quite. Time-to-event outcomes are typically analyzed using survival models.")

elif method == "Causal Analysis":
    quiz_causal = st.radio(
        "What is the purpose of inverse probability weighting (IPW) in causal analysis?",
        [
            "To remove all confounders from the study",
            "To assign equal weight to all participants",
            "To create a pseudo-population where treatment is independent of confounders",
            "To match treated individuals with similar untreated individuals"
        ]
    )
    if quiz_causal == "To create a pseudo-population where treatment is independent of confounders":
        st.success("Correct! IPW adjusts for confounders by weighting observations to mimic randomization.")
    else:
        st.error("Not quite. IPW aims to balance confounders between treatment groups.")
