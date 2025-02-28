# pages/7_study_designs.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("Epidemiological Study Designs")

# Study design selector
study_design = st.selectbox(
    "Select Study Design",
    ["Cohort Study", "Case-Control Study", "Randomized Controlled Trial", 
     "Cross-sectional Study", "Nested Case-Control Study", "Case-Cohort Study"]
)

if study_design == "Cohort Study":
    st.header("Cohort Study Simulation")
    
    # Parameters for cohort study
    n_subjects = st.slider("Number of Subjects", 100, 1000, 500)
    followup_years = st.slider("Follow-up Years", 1, 10, 5)
    baseline_risk = st.slider("Baseline Risk per Year", 0.01, 0.10, 0.05)
    relative_risk = st.slider("Relative Risk for Exposed", 1.0, 5.0, 2.0)
    
    # Generate cohort data
    np.random.seed(42)
    exposed = np.random.binomial(1, 0.5, n_subjects)
    years_to_event = np.array([
        np.random.exponential(1 / (baseline_risk * (relative_risk if e else 1)))
        for e in exposed
    ])
    censored = years_to_event > followup_years
    years_to_event[censored] = followup_years
    
    cohort_data = pd.DataFrame({
        'Exposed': exposed,
        'Years_to_Event': years_to_event,
        'Censored': censored
    })
    
    # Visualization
    fig = px.scatter(
        cohort_data,
        x='Years_to_Event',
        y=np.random.normal(0, 0.1, n_subjects),  # Jitter
        color='Exposed',
        opacity=0.6,
        title='Time to Event by Exposure Status'
    )
    st.plotly_chart(fig, use_container_width=True)
    

elif study_design == "Case-Control Study":
    st.header("Case-Control Study Simulation")
    
    # Parameters for case-control study
    n_cases = st.slider("Number of Cases", 50, 500, 200)
    control_ratio = st.slider("Control to Case Ratio", 1, 4, 2)
    odds_ratio = st.slider("True Odds Ratio", 1.0, 5.0, 2.0)
    
    # Generate case-control data
    n_controls = n_cases * control_ratio
    
    # Cases
    case_exposure = np.random.binomial(1, 0.4, n_cases)
    case_data = pd.DataFrame({
        'Status': 'Case',
        'Exposed': case_exposure
    })
    
    # Controls
    control_exposure = np.random.binomial(1, 0.4 / odds_ratio, n_controls)
    control_data = pd.DataFrame({
        'Status': 'Control',
        'Exposed': control_exposure
    })
    
    cc_data = pd.concat([case_data, control_data])
    
    # Visualization
    contingency = pd.crosstab(cc_data['Status'], cc_data['Exposed'])
    fig = px.bar(
        contingency,
        barmode='group',
        title='Exposure Distribution in Cases and Controls'
    )
    st.plotly_chart(fig, use_container_width=True)

elif study_design == "Randomized Controlled Trial":
    st.header("Randomized Controlled Trial Simulation")
    
    # Parameters for RCT
    n_participants = st.slider("Number of Participants", 100, 1000, 400)
    effect_size = st.slider("Treatment Effect Size", 0.0, 1.0, 0.3)
    
    # Generate RCT data
    treatment = np.random.binomial(1, 0.5, n_participants)
    outcome = np.random.normal(
        effect_size * treatment,
        1,
        n_participants
    )
    
    rct_data = pd.DataFrame({
        'Treatment': treatment,
        'Outcome': outcome
    })
    
    # Visualization
    fig = px.box(
        rct_data,
        x='Treatment',
        y='Outcome',
        title='Treatment Effect Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)
# Previous content remains the same until the study design selector...

elif study_design == "Cross-sectional Study":
    st.header("Cross-sectional Study Simulation")
    
    # Parameters for cross-sectional study
    n_participants = st.slider("Number of Participants", 100, 2000, 500)
    prevalence = st.slider("Disease Prevalence", 0.05, 0.50, 0.20)
    exposure_prevalence = st.slider("Exposure Prevalence", 0.10, 0.70, 0.30)
    association_strength = st.slider("Association Strength (OR)", 1.0, 5.0, 2.0)

    # Generate cross-sectional data
    def generate_cross_sectional_data(n, disease_prev, exp_prev, odds_ratio):
        # Generate exposure status
        exposure = np.random.binomial(1, exp_prev, n)
        
        # Calculate disease probability based on exposure
        # Using odds ratio to determine disease probability in exposed
        odds_unexposed = disease_prev / (1 - disease_prev)
        odds_exposed = odds_unexposed * odds_ratio
        prob_exposed = odds_exposed / (1 + odds_exposed)
        
        # Generate disease status
        disease = np.zeros(n)
        disease[exposure == 0] = np.random.binomial(1, disease_prev, sum(exposure == 0))
        disease[exposure == 1] = np.random.binomial(1, prob_exposed, sum(exposure == 1))
        
        # Add some covariates
        age = np.random.normal(45, 15, n)
        gender = np.random.binomial(1, 0.5, n)
        
        return pd.DataFrame({
            'Exposure': exposure,
            'Disease': disease,
            'Age': age,
            'Gender': gender
        })

    # Generate data
    cs_data = generate_cross_sectional_data(
        n_participants, 
        prevalence,
        exposure_prevalence,
        association_strength
    )

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Disease prevalence by exposure status
        fig1 = px.bar(
            cs_data.groupby('Exposure')['Disease'].mean().reset_index(),
            x='Exposure',
            y='Disease',
            title='Disease Prevalence by Exposure Status',
            labels={'Disease': 'Disease Prevalence', 'Exposure': 'Exposure Status'}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Age distribution by disease status
        fig2 = px.box(
            cs_data,
            x='Disease',
            y='Age',
            title='Age Distribution by Disease Status'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Calculate and display key statistics
    st.subheader("Key Statistics")
    
    # Create 2x2 table
    contingency = pd.crosstab(cs_data['Exposure'], cs_data['Disease'])
    
    # Calculate prevalence ratio
    prev_exposed = (contingency.loc[1, 1] / (contingency.loc[1, 0] + contingency.loc[1, 1]))
    prev_unexposed = (contingency.loc[0, 1] / (contingency.loc[0, 0] + contingency.loc[0, 1]))
    prevalence_ratio = prev_exposed / prev_unexposed
    
    # Calculate odds ratio
    odds_ratio = (contingency.loc[1, 1] * contingency.loc[0, 0]) / (contingency.loc[1, 0] * contingency.loc[0, 1])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Prevalence", f"{cs_data['Disease'].mean():.3f}")
    with col2:
        st.metric("Prevalence Ratio", f"{prevalence_ratio:.2f}")
    with col3:
        st.metric("Odds Ratio", f"{odds_ratio:.2f}")

    # Display 2x2 table
    st.subheader("2x2 Contingency Table")
    st.dataframe(contingency)

    # Stratified analysis
    st.subheader("Stratified Analysis by Gender")
    gender_strata = cs_data.groupby('Gender').apply(
        lambda x: pd.crosstab(x['Exposure'], x['Disease'])
    )
    st.dataframe(gender_strata)

    # Educational content
    st.header("Cross-sectional Study Characteristics")
    st.write("""
    Key features of cross-sectional studies:
    1. Data collected at a single point in time
    2. Can measure prevalence of disease and exposures
    3. Cannot establish temporality (which came first)
    4. Useful for:
       - Disease surveillance
       - Health service planning
       - Generating hypotheses
       - Studying stable characteristics
    
    Strengths:
    - Quick and relatively inexpensive
    - Can study multiple exposures and outcomes
    - No loss to follow-up
    - Good for prevalence estimation
    
    Limitations:
    - Cannot determine causation
    - Subject to prevalence-incidence bias
    - May miss temporal changes
    - Survival bias for chronic conditions
    """)

    st.subheader("Measures of Association in Cross-sectional Studies")
    st.write("""
    1. Prevalence Ratio (PR):
       - Ratio of prevalence in exposed vs unexposed
       - Similar interpretation to risk ratio
       - Preferred for common outcomes
    
    2. Prevalence Odds Ratio (POR):
       - Ratio of odds of disease in exposed vs unexposed
       - Similar to odds ratio in case-control studies
       - Less intuitive but mathematically convenient
    
    3. When to use each:
       - PR preferred for communication
       - POR useful for adjustment in logistic regression
       - Both valid for cross-sectional data
    """)

    st.subheader("Best Practices")
    st.write("""
    1. Sampling:
       - Ensure representative sample
       - Calculate adequate sample size
       - Consider sampling weights
    
    2. Analysis:
       - Examine confounding
       - Consider effect modification
       - Account for sampling design
    
    3. Interpretation:
       - Be cautious about causal inference
       - Consider selection bias
       - Account for prevalence-incidence bias
    """)
elif study_design == "Nested Case-Control Study":
    st.header("Nested Case-Control Study Simulation")
    
    # Parameters for cohort simulation
    n_cohort = st.slider("Initial Cohort Size", 1000, 10000, 5000)
    followup_years = st.slider("Follow-up Years", 1, 10, 5)
    n_controls = st.slider("Number of Controls per Case", 1, 5, 4)
    exposure_effect = st.slider("Exposure Effect (Hazard Ratio)", 1.0, 5.0, 2.0)
    
    # Generate cohort data
    def generate_nested_cc_data(n_cohort, followup_years, exposure_effect):
        # Generate baseline data
        data = pd.DataFrame({
            'id': range(n_cohort),
            'exposure': np.random.binomial(1, 0.3, n_cohort),
            'age': np.random.normal(50, 10, n_cohort),
            'sex': np.random.binomial(1, 0.5, n_cohort)
        })
        
        # Generate survival times
        baseline_hazard = 0.1
        lambda_i = baseline_hazard * np.exp(np.log(exposure_effect) * data['exposure'])
        data['time_to_event'] = np.random.exponential(1/lambda_i)
        data['time_to_censoring'] = np.random.uniform(0, followup_years, n_cohort)
        
        # Determine observed time and case status
        data['observed_time'] = np.minimum(data['time_to_event'], 
                                         np.minimum(data['time_to_censoring'], followup_years))
        data['is_case'] = (data['time_to_event'] <= data['time_to_censoring']) & \
                         (data['time_to_event'] <= followup_years)
        
        return data
    
    cohort_data = generate_nested_cc_data(n_cohort, followup_years, exposure_effect)
    
    # Select cases and controls
    cases = cohort_data[cohort_data['is_case']].copy()
    potential_controls = cohort_data[~cohort_data['is_case']].copy()
    
    # Matching controls for each case
    nested_cc_data = []
    for _, case in cases.iterrows():
        # Find eligible controls (those still at risk at case's event time)
        eligible_controls = potential_controls[
            potential_controls['observed_time'] >= case['observed_time']
        ]
        
        if len(eligible_controls) >= n_controls:
            matched_controls = eligible_controls.sample(n_controls)
            nested_cc_data.append(pd.concat([
                case.to_frame().T,
                matched_controls
            ]))
    
    nested_cc_data = pd.concat(nested_cc_data)
    
    # Display results
    st.subheader("Study Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cases", len(cases))
        st.metric("Total Controls", len(cases) * n_controls)
    
    with col2:
        # Calculate odds ratio
        table = pd.crosstab(nested_cc_data['is_case'], nested_cc_data['exposure'])
        odds_ratio = (table[1][1] * table[0][0]) / (table[1][0] * table[0][1])
        st.metric("Estimated Odds Ratio", f"{odds_ratio:.2f}")
        st.metric("True Hazard Ratio", f"{exposure_effect:.2f}")
    
    # Visualization of exposure distribution
    fig = go.Figure()
    
    for status in [True, False]:
        subset = nested_cc_data[nested_cc_data['is_case'] == status]
        fig.add_trace(go.Histogram(
            x=subset['exposure'],
            name='Cases' if status else 'Controls',
            opacity=0.75
        ))
    
    fig.update_layout(
        title='Exposure Distribution in Cases and Controls',
        xaxis_title='Exposure Status',
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif study_design == "Case-Cohort Study":
    st.header("Case-Cohort Study Simulation")
    
    # Parameters
    n_cohort = st.slider("Initial Cohort Size", 1000, 10000, 5000, key='cc_cohort_size')
    subcohort_fraction = st.slider("Subcohort Sampling Fraction", 0.1, 0.5, 0.2)
    followup_years = st.slider("Follow-up Years", 1, 10, 5, key='cc_followup')
    exposure_effect = st.slider("Exposure Effect (Hazard Ratio)", 1.0, 5.0, 2.0, key='cc_effect')
    
    # Generate cohort data
    def generate_case_cohort_data(n_cohort, followup_years, exposure_effect):
        # Generate baseline data
        data = pd.DataFrame({
            'id': range(n_cohort),
            'exposure': np.random.binomial(1, 0.3, n_cohort),
            'age': np.random.normal(50, 10, n_cohort),
            'sex': np.random.binomial(1, 0.5, n_cohort)
        })
        
        # Generate survival times
        baseline_hazard = 0.1
        lambda_i = baseline_hazard * np.exp(np.log(exposure_effect) * data['exposure'])
        data['time_to_event'] = np.random.exponential(1/lambda_i)
        data['time_to_censoring'] = np.random.uniform(0, followup_years, n_cohort)
        
        # Determine observed time and case status
        data['observed_time'] = np.minimum(data['time_to_event'], 
                                         np.minimum(data['time_to_censoring'], followup_years))
        data['is_case'] = (data['time_to_event'] <= data['time_to_censoring']) & \
                         (data['time_to_event'] <= followup_years)
        
        return data
    
    cohort_data = generate_case_cohort_data(n_cohort, followup_years, exposure_effect)
    
    # Select subcohort
    n_subcohort = int(n_cohort * subcohort_fraction)
    subcohort = cohort_data.sample(n_subcohort)
    cases = cohort_data[cohort_data['is_case']].copy()
    
    # Combine subcohort and cases
    case_cohort_data = pd.concat([
        subcohort,
        cases[~cases['id'].isin(subcohort['id'])]
    ]).drop_duplicates()
    
    # Display results
    st.subheader("Study Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cases", len(cases))
        st.metric("Subcohort Size", n_subcohort)
    
    with col2:
        st.metric("Cases in Subcohort", len(subcohort[subcohort['is_case']]))
        st.metric("Total Sample Size", len(case_cohort_data))
    
    # Calculate sampling weights
    case_cohort_data['weight'] = 1.0
    case_cohort_data.loc[
        (~case_cohort_data['is_case']) & 
        (case_cohort_data['id'].isin(subcohort['id'])),
        'weight'
    ] = 1/subcohort_fraction
    
    # Visualization of subcohort selection
    fig = px.scatter(
        case_cohort_data,
        x='age',
        y='observed_time',
        color='is_case',
        opacity=0.6,
        title='Case-Cohort Study Design Visualization'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time-to-event curves
    fig = go.Figure()
    
    for exp in [0, 1]:
        subset = case_cohort_data[case_cohort_data['exposure'] == exp]
        
        fig.add_trace(go.Scatter(
            x=sorted(subset['observed_time']),
            y=1 - np.arange(len(subset))/len(subset),
            name=f'Exposure = {exp}',
            mode='lines'
        ))
    
    fig.update_layout(
        title='Survival Curves by Exposure Status',
        xaxis_title='Time',
        yaxis_title='Survival Probability'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add educational content for new study designs
if study_design == "Nested Case-Control Study":
    st.write("""
    **Key features of nested case-control studies:**
    
    1. **Advantages**
    - More efficient than full cohort analysis
    - Reduces data collection costs
    - Maintains temporal sequence
    - Good for rare outcomes
    
    2. **Design Elements**
    - Starts with a defined cohort
    - Cases identified during follow-up
    - Controls selected from risk set
    - Matching on time at risk
    
    3. **Analysis Considerations**
    - Conditional logistic regression
    - Time-dependent variables possible
    - Risk set sampling
    - Matching factors
    """)

elif study_design == "Case-Cohort Study":
    st.write("""
    **Key features of case-cohort studies:**
    
    1. **Advantages**
    - Efficient for multiple outcomes
    - Subcohort represents full cohort
    - No matching required
    - Flexible analysis options
    
    2. **Design Elements**
    - Random subcohort selection
    - All cases included
    - Time-to-event data
    - Representative subcohort
    
    3. **Analysis Considerations**
    - Weighted analysis required
    - Modified survival methods
    - Robust variance estimation
    - Time-varying covariates possible
    """)

# Add to references section
st.write("""
Additional References for Nested Designs:
1. Langholz B, et al. Nested Case-Control Studies: Matching on Time and Risk Set Sampling
2. Prentice RL. A Case-Cohort Design for Epidemiologic Cohort Studies and Disease Prevention Trials
""")

# Educational content for each design
st.header("Study Design Characteristics")

if study_design == "Cohort Study":
    st.write("""
    **Key features of cohort studies:**
    1. Follow participants over time
    2. Compare exposed and unexposed groups
    3. Can measure multiple outcomes
    4. Can calculate incidence and relative risks
    5. May have issues with loss to follow-up
    """)
elif study_design == "Case-Control Study":
    st.write("""
    **Key features of case-control studies:**
    1. Start with disease status (cases and controls)
    2. Look backwards at exposures
    3. Efficient for rare diseases
    4. Can only calculate odds ratios
    5. May have recall bias
    """)
elif study_design == "Randomized Controlled Trial":
    st.write("""
    **Key features of RCTs:**
    1. Random allocation to intervention
    2. Can establish causality
    3. Balances confounders
    4. May have limited generalizability
    5. May have ethical constraints
    """)
    
elif study_design == "Nested Case-Control Study":
    st.write("""
    **Key features of Nested Case-Control:**
    1. **Efficiency and Resources**
    - More cost-efficient than analyzing entire cohort
    - Reduces data collection costs while maintaining validity
    - Particularly useful for expensive or labor-intensive exposure measurements
    - Perfect for biomarker studies or when specimen analysis is costly
    
    2. **Temporal Aspects**
    - Preserves temporal relationship between exposure and outcome
    - Allows study of time-varying exposures
    - Can assess multiple time windows of exposure
    - Enables evaluation of exposure changes over time
    
    3. **Statistical Considerations**
    - Maintains the same validity as full cohort analysis
    - Allows matching on time-at-risk
    - Can control for confounding through matching
    - Power depends on number of controls per case
    
    4. **Best Use Scenarios**
    - Rare diseases in large cohorts
    - Studies requiring expensive biomarker analysis
    - When complete exposure assessment is impractical
    - When efficiency is needed without compromising validity
    """)
    
elif study_design == "Case-Cohort Study":
    st.write("""
    Key features of Case-Cohort:
    
    1. **Design Features**
    - Random sample of the entire cohort (subcohort) plus all cases
    - No matching required
    - Can study multiple disease outcomes
    - Subcohort serves as a comparison group for all outcomes

    2. **Analytical Advantages**
    - Can estimate absolute risks and rates
    - Allows for time-varying covariates
    - Flexible analysis options
    - Can handle competing risks
    
    3. **Practical Benefits**
    - Exposure data needed only for subcohort and cases
    - Subcohort can be used for multiple outcomes
    - Can add new outcomes during study
    - More efficient than full cohort analysis
    
    4. **Best Use Scenarios**
    - Studies with multiple outcomes of interest
    - When exposure measurement is expensive
    - When population rates are needed
    - When matching is impractical or undesirable
    """)
    
    

# Educational content for each design
st.header("Check your understanding of study designs")

if study_design == "Cohort Study":
    quiz_cohort = st.radio(
        "Which measure can be directly computed from a cohort study?",
        ("Odds Ratio", "Risk Ratio", "Neither can be computed")
    )
    if quiz_cohort == "Risk Ratio":
        st.success("Correct! Cohort studies allow you to calculate incidence and thus risk ratios.")
    else:
        st.error("Not quite. While you can compute an odds ratio, the primary advantage of a cohort design is that you can directly compute a Risk Ratio.")


elif study_design == "Case-Control Study":
    quiz_case_control = st.radio(
        "Which measure of association is primarily used in a case-control study?",
        ("Risk Ratio (RR)", "Odds Ratio (OR)", "Risk Difference (RD)")
    )
    if quiz_case_control == "Odds Ratio (OR)":
        st.success("Correct! Case-control designs typically estimate the odds ratio.")
    else:
        st.error("In a case-control study, incidence cannot be directly computed, so the odds ratio is the primary measure.")
   

elif study_design == "Randomized Controlled Trial":
    quiz_rct = st.radio(
        "Why are RCTs considered the gold standard in epidemiological research?",
        (
            "Because they are cheapest to conduct.",
            "They randomly assign exposure, minimizing confounding.",
            "They never lose participants to follow-up."
        )
    )
    if quiz_rct == "They randomly assign exposure, minimizing confounding.":
        st.success("Correct! Randomization helps ensure confounders are evenly distributed.")
    else:
        st.error("Not quite. The key advantage is the random assignment of exposure.")

elif study_design == "Cross-sectional Study":
    quiz_cross = st.radio(
        "What is a major limitation of cross-sectional studies for determining causality?",
        (
            "They rely on randomization.",
            "They only capture data at one point in time, making temporality unclear.",
            "They require large budgets and are always unethical."
        )
    )
    if quiz_cross == "They only capture data at one point in time, making temporality unclear.":
        st.success("Exactly. In cross-sectional studies, exposure and outcome are measured simultaneously, so we can't confirm which came first.")
    else:
        st.error("That's not the main reason. The biggest limitation is that we can't determine temporality with cross-sectional data.")

    
elif study_design == "Nested Case-Control Study":
    quiz_nested_cc = st.radio(
        "Which of the following is a key advantage of a nested case-control study?",
        (
            "You can easily compute incidence in the sub-sample.",
            "It is more cost-efficient than assessing the entire cohort.",
            "All participants are followed for the full duration, guaranteeing no loss to follow-up."
        )
    )
    if quiz_nested_cc == "It is more cost-efficient than assessing the entire cohort.":
        st.success("Correct! Nested designs use fewer resources while maintaining validity.")
    else:
        st.error("Not quite. The primary advantage is cost and resource efficiency while maintaining many cohort advantages.")

    
elif study_design == "Case-Cohort Study":
    quiz_case_cohort = st.radio(
        "What is a main advantage of a case-cohort study design?",
        (
            "It only requires cases, no controls.",
            "You can investigate multiple outcomes using the same subcohort.",
            "It does not require any follow-up period."
        )
    )
    if quiz_case_cohort == "You can investigate multiple outcomes using the same subcohort.":
        st.success("Correct! The subcohort can be used to study various diseases that arise.")
    else:
        st.error("Not quite. The hallmark advantage is the ability to examine multiple outcomes.")

    