# pages/11_disease_frequency.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from lifelines import KaplanMeierFitter

st.title("Disease Frequency and Measures")

# Method selector
method = st.selectbox(
    "Select Analysis Method",
    ["Disease Clustering", "Age-Period-Cohort Analysis", 
     "Standardization", "Life Tables", "Population Attributable Risk",
     "Number Needed to Treat"]
)

if method == "Disease Clustering":
    st.header("Disease Clustering and Outbreak Detection")
    
    # Temporal clustering analysis
    st.subheader("Temporal Clustering Analysis")
    
    # Parameters for simulation
    n_days = st.slider("Number of Days", 30, 365, 90)
    baseline_rate = st.slider("Baseline Daily Cases", 1, 20, 5)
    outbreak_size = st.slider("Outbreak Size", 1, 50, 15)
    outbreak_duration = st.slider("Outbreak Duration (days)", 1, 30, 7)
    
    # Generate time series data
    def generate_outbreak_data(n_days, baseline, outbreak_size, outbreak_duration):
        # Baseline cases
        cases = np.random.poisson(baseline, n_days)
        
        # Add outbreak
        outbreak_start = n_days // 3
        outbreak_cases = np.random.poisson(outbreak_size, outbreak_duration)
        cases[outbreak_start:outbreak_start + outbreak_duration] += outbreak_cases
        
        dates = pd.date_range(start='2024-01-01', periods=n_days)
        return pd.DataFrame({'Date': dates, 'Cases': cases})
    
    data = generate_outbreak_data(n_days, baseline_rate, outbreak_size, outbreak_duration)
    
    # Calculate moving average
    data['MA7'] = data['Cases'].rolling(window=7).mean()
    
    # Plot time series
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Cases'],
        mode='markers+lines',
        name='Daily Cases'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['MA7'],
        mode='lines',
        name='7-day Moving Average',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Disease Cases Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Cases'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detect outliers using Z-score
    z_scores = np.abs(stats.zscore(data['Cases']))
    outliers = z_scores > 2
    
    st.subheader("Outbreak Detection")
    st.write(f"Number of potential outbreak days detected: {sum(outliers)}")
    
    # Spatial clustering example
    st.subheader("Spatial Clustering")
    
    # Generate spatial data
    n_locations = 20
    x_coords = np.random.uniform(0, 10, n_locations)
    y_coords = np.random.uniform(0, 10, n_locations)
    cases = np.random.poisson(5, n_locations)
    
    # Add cluster
    cluster_x = 5
    cluster_y = 5
    cluster_cases = np.random.poisson(15, 5)
    x_coords = np.append(x_coords, np.random.normal(cluster_x, 0.5, 5))
    y_coords = np.append(y_coords, np.random.normal(cluster_y, 0.5, 5))
    cases = np.append(cases, cluster_cases)
    
    spatial_data = pd.DataFrame({
        'X': x_coords,
        'Y': y_coords,
        'Cases': cases
    })
    
    # Plot spatial distribution
    fig = px.scatter(
        spatial_data,
        x='X',
        y='Y',
        size='Cases',
        title='Spatial Distribution of Cases'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Age-Period-Cohort Analysis":
    st.header("Age-Period-Cohort Analysis")
    
    # Generate synthetic APC data
    n_years = 5
    n_age_groups = 6
    
    # Create age groups and periods
    age_groups = [f"{i*10}-{(i+1)*10-1}" for i in range(n_age_groups)]
    periods = list(range(2020, 2020 + n_years))
    
    # Generate data
    def generate_apc_data():
        # Age effects
        age_effects = np.array([1, 2, 3, 4, 3, 2])
        
        # Period effects
        period_effects = np.array([1, 1.2, 1.1, 0.9, 1.3])
        
        # Create rates matrix
        rates = np.zeros((n_age_groups, n_years))
        for i in range(n_age_groups):
            for j in range(n_years):
                cohort_idx = n_age_groups - i + j
                cohort_effect = 1 + 0.1 * cohort_idx
                rates[i, j] = age_effects[i] * period_effects[j] * cohort_effect
        
        return rates
    
    rates = generate_apc_data()
    
    # Create long format data
    data_long = []
    for i, age in enumerate(age_groups):
        for j, period in enumerate(periods):
            data_long.append({
                'Age Group': age,
                'Period': period,
                'Rate': rates[i, j]
            })
    
    df_apc = pd.DataFrame(data_long)
    
    # Visualization
    # Age-specific rates by period
    fig = px.line(
        df_apc,
        x='Period',
        y='Rate',
        color='Age Group',
        title='Age-specific Rates by Period'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Period-specific rates by age
    fig = px.line(
        df_apc,
        x='Age Group',
        y='Rate',
        color='Period',
        title='Period-specific Rates by Age'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Standardization":
    st.header("Standardization Methods")
    
    # Interactive population input
    st.subheader("Population Structure and Rates")
    
    # Number of age groups selector
    n_age_groups = st.slider("Number of Age Groups", 3, 8, 4)
    
    # Create expandable sections for each population
    with st.expander("Population A Settings", expanded=True):
        pop_a_data = []
        for i in range(n_age_groups):
            col1, col2, col3 = st.columns(3)
            with col1:
                age_group = f"{i*10}-{(i+1)*10-1}"
                st.write(f"Age Group: {age_group}")
            with col2:
                pop_size = st.number_input(
                    f"Population Size A {i}",
                    min_value=100,
                    max_value=10000,
                    value=1000 + i*500,
                    key=f"pop_a_size_{i}"
                )
            with col3:
                rate = st.number_input(
                    f"Rate per 1000 A {i}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(2 + i*3),
                    key=f"rate_a_{i}"
                )
            pop_a_data.append({
                'Age_Group': age_group,
                'Pop_Size': pop_size,
                'Rate': rate
            })
    
    with st.expander("Population B Settings", expanded=True):
        pop_b_data = []
        for i in range(n_age_groups):
            col1, col2, col3 = st.columns(3)
            with col1:
                age_group = f"{i*10}-{(i+1)*10-1}"
                st.write(f"Age Group: {age_group}")
            with col2:
                pop_size = st.number_input(
                    f"Population Size B {i}",
                    min_value=100,
                    max_value=10000,
                    value=1500 + i*400,
                    key=f"pop_b_size_{i}"
                )
            with col3:
                rate = st.number_input(
                    f"Rate per 1000 B {i}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(3 + i*2),
                    key=f"rate_b_{i}"
                )
            pop_b_data.append({
                'Age_Group': age_group,
                'Pop_Size': pop_size,
                'Rate': rate
            })
    
    # Standard population options
    standard_pop_option = st.radio(
        "Choose Standard Population",
        ["World Standard", "European Standard", "Custom"]
    )
    
    if standard_pop_option == "Custom":
        st.subheader("Custom Standard Population")
        standard_pop_data = []
        for i in range(n_age_groups):
            col1, col2 = st.columns(2)
            with col1:
                age_group = f"{i*10}-{(i+1)*10-1}"
                st.write(f"Age Group: {age_group}")
            with col2:
                pop_size = st.number_input(
                    f"Standard Population Size {i}",
                    min_value=100,
                    max_value=10000,
                    value=2000,
                    key=f"std_pop_{i}"
                )
            standard_pop_data.append({
                'Age_Group': age_group,
                'Pop_Size': pop_size
            })
    else:
        # Predefined standard populations
        if standard_pop_option == "World Standard":
            weights = [2500, 3500, 2500, 1500] * (n_age_groups // 4 + 1)
        else:  # European Standard
            weights = [2000, 4000, 3000, 1000] * (n_age_groups // 4 + 1)
        
        standard_pop_data = [
            {'Age_Group': f"{i*10}-{(i+1)*10-1}", 'Pop_Size': weights[i]}
            for i in range(n_age_groups)
        ]
    
    # Create DataFrames
    pop_a_df = pd.DataFrame(pop_a_data)
    pop_b_df = pd.DataFrame(pop_b_data)
    standard_pop_df = pd.DataFrame(standard_pop_data)
    
    # Combine data
    combined_data = pd.DataFrame({
        'Age_Group': pop_a_df['Age_Group'],
        'Pop_A_Size': pop_a_df['Pop_Size'],
        'Pop_B_Size': pop_b_df['Pop_Size'],
        'Standard_Pop': standard_pop_df['Pop_Size'],
        'Rate_A': pop_a_df['Rate'],
        'Rate_B': pop_b_df['Rate']
    })
    
    # Calculate crude rates
    crude_rate_a = (combined_data['Pop_A_Size'] * combined_data['Rate_A']).sum() / combined_data['Pop_A_Size'].sum()
    crude_rate_b = (combined_data['Pop_B_Size'] * combined_data['Rate_B']).sum() / combined_data['Pop_B_Size'].sum()
    
    # Direct standardization
    direct_rate_a = (combined_data['Standard_Pop'] * combined_data['Rate_A']).sum() / combined_data['Standard_Pop'].sum()
    direct_rate_b = (combined_data['Standard_Pop'] * combined_data['Rate_B']).sum() / combined_data['Standard_Pop'].sum()
    
    # Display results
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Crude Rate - Population A", f"{crude_rate_a:.1f} per 1000")
        st.metric("Standardized Rate - Population A", f"{direct_rate_a:.1f} per 1000")
    with col2:
        st.metric("Crude Rate - Population B", f"{crude_rate_b:.1f} per 1000")
        st.metric("Standardized Rate - Population B", f"{direct_rate_b:.1f} per 1000")
    
    # Visualization of population structures
    st.subheader("Population Age Structures")
    
    # Prepare data for population pyramid
    pyramid_data = pd.DataFrame({
        'Age_Group': combined_data['Age_Group'],
        'Population A': combined_data['Pop_A_Size'] / combined_data['Pop_A_Size'].sum() * 100,
        'Population B': combined_data['Pop_B_Size'] / combined_data['Pop_B_Size'].sum() * 100,
        'Standard': combined_data['Standard_Pop'] / combined_data['Standard_Pop'].sum() * 100
    })
    
    fig = go.Figure()
    
    # Add population bars
    fig.add_trace(go.Bar(
        y=pyramid_data['Age_Group'],
        x=pyramid_data['Population A'],
        name='Population A',
        orientation='h'
    ))
    
    fig.add_trace(go.Bar(
        y=pyramid_data['Age_Group'],
        x=pyramid_data['Population B'],
        name='Population B',
        orientation='h'
    ))
    
    fig.add_trace(go.Bar(
        y=pyramid_data['Age_Group'],
        x=pyramid_data['Standard'],
        name='Standard Population',
        orientation='h'
    ))
    
    fig.update_layout(
        title='Population Age Structures (% of total)',
        xaxis_title='Percentage',
        yaxis_title='Age Group',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualization of age-specific rates
    st.subheader("Age-specific Rates")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=combined_data['Age_Group'],
        y=combined_data['Rate_A'],
        mode='lines+markers',
        name='Population A'
    ))
    
    fig.add_trace(go.Scatter(
        x=combined_data['Age_Group'],
        y=combined_data['Rate_B'],
        mode='lines+markers',
        name='Population B'
    ))
    
    fig.update_layout(
        title='Age-specific Rates',
        xaxis_title='Age Group',
        yaxis_title='Rate per 1000'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Effect of standardization visualization
    st.subheader("Effect of Standardization")
    
    comparison_data = pd.DataFrame({
        'Population': ['A', 'A', 'B', 'B'],
        'Rate_Type': ['Crude', 'Standardized', 'Crude', 'Standardized'],
        'Rate': [crude_rate_a, direct_rate_a, crude_rate_b, direct_rate_b]
    })
    
    fig = px.bar(
        comparison_data,
        x='Population',
        y='Rate',
        color='Rate_Type',
        barmode='group',
        title='Crude vs Standardized Rates'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
elif method == "Life Tables":
    st.header("Life Tables and Survival Analysis")
    
    # Generate survival data
    n_subjects = st.slider("Number of Subjects", 100, 1000, 500)
    max_time = st.slider("Follow-up Time (years)", 1, 20, 10)
    
    # Generate data
    def generate_survival_data(n, max_time):
        # Generate survival times
        times = np.random.exponential(max_time/2, n)
        times = np.minimum(times, max_time)
        
        # Generate censoring
        censoring = np.random.uniform(0, max_time, n)
        observed = times <= censoring
        final_times = np.minimum(times, censoring)
        
        # Add groups
        groups = np.random.binomial(1, 0.5, n)
        
        return pd.DataFrame({
            'time': final_times,
            'event': observed,
            'group': groups
        })
    
    survival_data = generate_survival_data(n_subjects, max_time)
    
    # Fit Kaplan-Meier
    kmf = KaplanMeierFitter()
    
    # Plot survival curves by group
    fig = go.Figure()
    
    for group in [0, 1]:
        mask = survival_data['group'] == group
        kmf.fit(
            survival_data.loc[mask, 'time'],
            survival_data.loc[mask, 'event'],
            label=f'Group {group}'
        )
        
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.survival_function_.values.flatten(),
            name=f'Group {group}'
        ))
    
    fig.update_layout(
        title='Kaplan-Meier Survival Curves',
        xaxis_title='Time',
        yaxis_title='Survival Probability'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Life table calculations
    st.subheader("Life Table")
    
    # Create life table
    intervals = np.linspace(0, max_time, 6)
    life_table = pd.DataFrame()
    
    for i in range(len(intervals)-1):
        start, end = intervals[i], intervals[i+1]
        mask = (survival_data['time'] >= start) & (survival_data['time'] < end)
        
        n_start = sum(survival_data['time'] >= start)
        events = sum(mask & survival_data['event'])
        censored = sum(mask & ~survival_data['event'])
        
        life_table.loc[i, 'Interval'] = f"{start:.1f}-{end:.1f}"
        life_table.loc[i, 'At Risk'] = n_start
        life_table.loc[i, 'Events'] = events
        life_table.loc[i, 'Censored'] = censored
        life_table.loc[i, 'Survival'] = 1 - events/n_start if n_start > 0 else 0
    
    st.dataframe(life_table)

elif method == "Population Attributable Risk":
    st.header("Population Attributable Risk (PAR)")
    
    # Input parameters
    st.subheader("Input Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        exposure_prev = st.slider("Exposure Prevalence", 0.0, 1.0, 0.3)
        relative_risk = st.slider("Relative Risk", 1.0, 10.0, 2.5)
    
    # Calculate PAR
    par = (exposure_prev * (relative_risk - 1)) / (1 + exposure_prev * (relative_risk - 1))
    par_percent = par * 100
    
    # Display results
    st.metric("Population Attributable Risk", f"{par_percent:.1f}%")
    
    # Visualization of PAR
    rr_range = np.linspace(1, 10, 100)
    prev_range = np.linspace(0, 1, 100)
    RR, PREV = np.meshgrid(rr_range, prev_range)
    
    PAR = (PREV * (RR - 1)) / (1 + PREV * (RR - 1))
    
    fig = go.Figure(data=[
        go.Contour(
            z=PAR * 100,
            x=rr_range,
            y=prev_range,
            colorscale='Viridis',
            colorbar=dict(title='PAR %')
        )
    ])
    
    fig.update_layout(
        title='Population Attributable Risk by Exposure Prevalence and Relative Risk',
        xaxis_title='Relative Risk',
        yaxis_title='Exposure Prevalence',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # PAR calculator for multiple risk factors
    st.subheader("Multiple Risk Factors PAR Calculator")
    
    n_factors = st.number_input("Number of Risk Factors", min_value=1, max_value=5, value=2)
    
    total_par = 0
    data_factors = []
    
    for i in range(n_factors):
        st.write(f"Risk Factor {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            prev = st.slider(f"Prevalence {i+1}", 0.0, 1.0, 0.2, key=f"prev_{i}")
        with col2:
            rr = st.slider(f"Relative Risk {i+1}", 1.0, 10.0, 1.5, key=f"rr_{i}")
        
        par_i = (prev * (rr - 1)) / (1 + prev * (rr - 1))
        data_factors.append({
            'Risk Factor': f"Factor {i+1}",
            'PAR': par_i * 100
        })
        total_par += par_i
    
    # Display individual and combined PAR
    df_factors = pd.DataFrame(data_factors)
    
    fig = px.bar(
        df_factors,
        x='Risk Factor',
        y='PAR',
        title='Population Attributable Risk by Factor'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Combined PAR", f"{min(total_par * 100, 100):.1f}%")

elif method == "Number Needed to Treat":
    st.header("Number Needed to Treat/Harm")
    
    # Input parameters
    st.subheader("Input Data")
    
    col1, col2 = st.columns(2)
    with col1:
        control_risk = st.slider("Control Group Risk", 0.0, 1.0, 0.3, help="Event rate in control group")
        treatment_risk = st.slider("Treatment Group Risk", 0.0, 1.0, 0.2, help="Event rate in treatment group")
    
    # Calculate NNT
    risk_difference = control_risk - treatment_risk
    if risk_difference == 0:
        nnt = float('inf')
    else:
        nnt = abs(1 / risk_difference)
    
    # Determine if NNT or NNH
    if risk_difference > 0:
        metric_name = "Number Needed to Treat (NNT)"
    else:
        metric_name = "Number Needed to Harm (NNH)"
    
    # Display results
    st.metric(metric_name, f"{nnt:.1f}")
    st.metric("Absolute Risk Reduction", f"{abs(risk_difference):.3f}")
    st.metric("Relative Risk Reduction", f"{(1 - treatment_risk/control_risk):.1%}")
    
    # Visualization of treatment effect
    fig = go.Figure()
    
    # Add bars for control and treatment risks
    fig.add_trace(go.Bar(
        x=['Control Group', 'Treatment Group'],
        y=[control_risk, treatment_risk],
        text=[f"{control_risk:.1%}", f"{treatment_risk:.1%}"],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Event Rates in Control and Treatment Groups',
        yaxis_title='Event Rate',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # NNT by baseline risk visualization
    st.subheader("NNT by Baseline Risk")
    
    baseline_risks = np.linspace(0.1, 0.9, 100)
    relative_risk = treatment_risk / control_risk
    nnts = 1 / (baseline_risks * (1 - relative_risk))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=baseline_risks,
        y=nnts,
        mode='lines',
        name='NNT'
    ))
    
    fig.update_layout(
        title='Number Needed to Treat by Baseline Risk',
        xaxis_title='Baseline Risk',
        yaxis_title='NNT'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Educational content
st.header("Method Description")

if method == "Disease Clustering":
    st.write("""
    Disease clustering analysis helps identify unusual aggregations of disease in:
    
    1. Time (temporal clustering)
    - Outbreak detection
    - Seasonal patterns
    - Trend analysis
    
    2. Space (spatial clustering)
    - Geographic hotspots
    - Environmental risk factors
    - Resource allocation
    
    3. Space-time interaction
    - Moving clusters
    - Emerging outbreaks
    - Transmission patterns
    """)

elif method == "Age-Period-Cohort Analysis":
    st.write("""
    Age-Period-Cohort (APC) analysis separates three time-related effects:
    
    1. Age Effects
    - Changes in disease risk with aging
    - Biological patterns
    - Age-related exposures
    
    2. Period Effects
    - Temporal changes affecting all ages
    - Environmental changes
    - Healthcare improvements
    
    3. Cohort Effects
    - Generation-specific risks
    - Early life exposures
    - Societal changes
    """)

elif method == "Standardization":
    st.write("""
    Standardization methods adjust for population differences:
    
    1. Direct Standardization
    - Applies study population rates to standard population
    - Comparable across populations
    - Requires age-specific rates
    
    2. Indirect Standardization
    - Applies standard rates to study population
    - Useful when age-specific rates are unstable
    - Produces Standardized Mortality/Morbidity Ratio (SMR)
    """)

elif method == "Life Tables":
    st.write("""
    Life tables and survival analysis examine time-to-event data:
    
    1. Life Tables
    - Mortality/survival by age
    - Life expectancy
    - Population dynamics
    
    2. Survival Analysis
    - Time-to-event analysis
    - Censoring handling
    - Comparison between groups
    """)

elif method == "Population Attributable Risk":
    st.write("""
    Population Attributable Risk (PAR) measures population impact:
    
    1. Interpretation
    - Proportion of disease attributable to exposure
    - Population prevention potential
    - Public health impact
    
    2. Applications
    - Prevention planning
    - Resource allocation
    - Policy making
    """)

elif method == "Number Needed to Treat":
    st.write("""
    Number Needed to Treat (NNT) and Number Needed to Harm (NNH):
    
    1. Interpretation
    - Patients needed to treat to prevent one event
    - Patients exposed for one additional adverse event
    - Clinical significance measure
    
    2. Applications
    - Treatment decisions
    - Risk communication
    - Cost-effectiveness analysis
    """)

# Add references
st.header("Further Reading")
st.write("""
1. Rothman KJ, et al. Modern Epidemiology.
2. Clayton D, Hills M. Statistical Models in Epidemiology.
3. Breslow NE, Day NE. Statistical Methods in Cancer Research.
4. Yang Y, Land KC. Age-Period-Cohort Analysis: New Models, Methods, and Empirical Applications.
""")

st.header("Check your understanding")

if method == "Disease Clustering":
    quiz_clustering = st.radio(
        "Which technique(s) can help identify unusual aggregations of disease in space or time?",
        [
            "Temporal analysis (time series)",
            "Spatial analysis (maps, location data)",
            "Spatio-temporal analysis (combination)",
            "All of the above"
        ]
    )
    if quiz_clustering == "All of the above":
        st.success("Correct! All these methods can help detect clusters or outbreaks.")
    else:
        st.error("Not quite. In practice, we often combine multiple approaches to detect clustering.")
    
elif method == "Age-Period-Cohort Analysis":
    quiz_apc = st.radio(
        "Which effect refers to differences in disease risk among people born around the same time?",
        ["Age effect", "Period effect", "Cohort effect"]
    )
    if quiz_apc == "Cohort effect":
        st.success("Correct! Cohort effects focus on the unique exposures shared by a birth cohort.")
    else:
        st.error("Close. Age and period effects are different from a birth cohort effect.")

elif method == "Standardization":
    quiz_std = st.radio(
        "Why do we use direct standardization?",
        [
            "To remove population structure differences in comparisons",
            "To estimate the prevalence of a disease",
            "To randomly allocate participants into groups"
        ]
    )
    if quiz_std == "To remove population structure differences in comparisons":
        st.success("Correct! Standardization adjusts for demographic/age structure so rates are comparable.")
    else:
        st.error("Not quite. Standardization specifically tackles differences in population structures.")

elif method == "Life Tables":
    quiz_life = st.radio(
        "Which measure does a life table primarily provide?",
        [
            "Age-specific incidence rates",
            "Life expectancy or survival probabilities",
            "Risk ratios for different exposures"
        ]
    )
    if quiz_life == "Life expectancy or survival probabilities":
        st.success("Correct! Life tables show survival over intervals, giving life expectancy estimates.")
    else:
        st.error("Not quite. Life tables are about survival probabilities, not direct incidence or exposures.")

elif method == "Population Attributable Risk":
    quiz_par = st.radio(
        "PAR is best interpreted as:",
        [
            "The fraction of disease in the entire population that can be attributed to an exposure",
            "The risk among the exposed group only",
            "The difference in incidence between two random groups"
        ]
    )
    if quiz_par == "The fraction of disease in the entire population that can be attributed to an exposure":
        st.success("Correct! PAR is about the proportion of disease in the total population due to that exposure.")
    else:
        st.error("Not quite. PAR is specifically about the fraction of disease in the *population* caused by exposure.")

elif method == "Number Needed to Treat":
    quiz_nnt = st.radio(
        "If the absolute risk reduction (ARR) is 0.10 (10%), what is the NNT?",
        ["NNT = 1/ARR = 10", "NNT = ARR × 100 = 10", "NNT = 10 × ARR = 1"]
    )
    if quiz_nnt == "NNT = 1/ARR = 10":
        st.success("Correct! If ARR = 0.10, then NNT = 1 / 0.10 = 10.")
    else:
        st.error("That's not correct. NNT is the inverse of the absolute risk reduction (1 / 0.10 = 10).")
