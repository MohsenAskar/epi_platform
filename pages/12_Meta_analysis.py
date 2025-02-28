# pages/12_meta_analysis.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.title("Meta-Analysis and Systematic Reviews")

# Method selector
method = st.selectbox(
    "Select Analysis Method",
    ["Forest Plot Builder", "Heterogeneity Analysis", 
     "Publication Bias Assessment", "Quality Assessment",
     "Subgroup Analysis"]
)

if method == "Forest Plot Builder":
    st.header("Interactive Forest Plot Builder")
    
    # Number of studies
    n_studies = st.slider("Number of Studies", 2, 10, 5)
    
    # Effect measure selection
    effect_measure = st.selectbox(
        "Effect Measure",
        ["Risk Ratio", "Odds Ratio", "Mean Difference"]
    )
    
    # Create data input interface
    study_data = []
    
    for i in range(n_studies):
        st.subheader(f"Study {i+1}")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            study_name = st.text_input(f"Study Name {i+1}", f"Study {i+1}")
        with col2:
            effect = st.number_input(f"Effect Size {i+1}", 0.01, 10.0, 1.0)
        with col3:
            lower_ci = st.number_input(f"Lower CI {i+1}", 0.01, effect, effect*0.8)
        with col4:
            upper_ci = st.number_input(f"Upper CI {i+1}", effect, 10.0, effect*1.2)
        
        study_data.append({
            'Study': study_name,
            'Effect': effect,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci,
            'Weight': np.random.uniform(5, 20)  # Random weights for demonstration
        })
    
    df = pd.DataFrame(study_data)
    
    # Calculate random effects pooled estimate
    weights = 1 / ((df['Upper_CI'] - df['Lower_CI'])/3.92)**2
    pooled_effect = np.sum(df['Effect'] * weights) / np.sum(weights)
    
    # Forest plot
    fig = go.Figure()
    
    # Add individual study effects
    for idx, row in df.iterrows():
        # Effect estimate
        fig.add_trace(go.Scatter(
            x=[row['Effect']],
            y=[row['Study']],
            mode='markers',
            name=row['Study'],
            marker=dict(size=10*np.sqrt(row['Weight'])),
            showlegend=False
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=[row['Lower_CI'], row['Upper_CI']],
            y=[row['Study'], row['Study']],
            mode='lines',
            line=dict(width=1),
            showlegend=False
        ))
    
    # Add pooled effect
    fig.add_vline(x=pooled_effect, line_dash="dash", line_color="red")
    
    # Add null effect line
    fig.add_vline(x=1 if effect_measure in ["Risk Ratio", "Odds Ratio"] else 0,
                 line_dash="dot")
    
    fig.update_layout(
        title='Forest Plot',
        xaxis_title=effect_measure,
        yaxis_title='Study',
        showlegend=False
    )
    
    # Log scale for ratio measures
    if effect_measure in ["Risk Ratio", "Odds Ratio"]:
        fig.update_xaxes(type="log")
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Heterogeneity Analysis":
    st.header("Heterogeneity Analysis")
    
    # Parameters for simulation
    n_studies = st.slider("Number of Studies", 3, 15, 8)
    true_effect = st.slider("True Effect", 0.0, 2.0, 1.0)
    heterogeneity = st.slider("Heterogeneity (τ²)", 0.0, 1.0, 0.2)
    
    # Generate random effects meta-analysis data
    def generate_meta_data(n, true_effect, tau2):
        studies = []
        for i in range(n):
            # Random sample size
            n_i = np.random.randint(50, 500)
            
            # Random effect for study
            theta_i = np.random.normal(true_effect, np.sqrt(tau2))
            
            # Within-study variance
            var_i = 0.5/n_i
            
            # Observed effect
            y_i = np.random.normal(theta_i, np.sqrt(var_i))
            
            studies.append({
                'Study': f'Study {i+1}',
                'Effect': y_i,
                'Variance': var_i,
                'Sample_Size': n_i
            })
        
        return pd.DataFrame(studies)
    
    meta_data = generate_meta_data(n_studies, true_effect, heterogeneity)
    
    # Calculate heterogeneity statistics
    weights = 1/meta_data['Variance']
    weighted_mean = np.sum(meta_data['Effect'] * weights) / np.sum(weights)
    Q = np.sum(weights * (meta_data['Effect'] - weighted_mean)**2)
    df = n_studies - 1
    I2 = max(0, (Q - df)/Q * 100)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Q statistic", f"{Q:.2f}")
    with col2:
        st.metric("I² statistic", f"{I2:.1f}%")
    with col3:
        st.metric("τ² (input)", f"{heterogeneity:.3f}")
    
    # Visualization of study effects
    fig = go.Figure()
    
    # Add individual study effects
    fig.add_trace(go.Scatter(
        x=meta_data['Effect'],
        y=meta_data['Study'],
        mode='markers',
        marker=dict(
            size=10*np.sqrt(1/meta_data['Variance']),
            color=meta_data['Sample_Size'],
            colorscale='Viridis',
            showscale=True
        ),
        name='Study Effects'
    ))
    
    # Add overall effect
    fig.add_vline(x=weighted_mean, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f'Study Effects (I² = {I2:.1f}%)',
        xaxis_title='Effect Size',
        yaxis_title='Study'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Publication Bias Assessment":
    st.header("Publication Bias Assessment")
    
    # Parameters for funnel plot simulation
    n_studies = st.slider("Number of Studies", 5, 30, 15)
    publication_bias = st.slider("Publication Bias Strength", 0.0, 1.0, 0.3)
    
    # Generate funnel plot data
    def generate_funnel_data(n, bias_strength):
        studies = []
        for i in range(n):
            precision = np.random.uniform(0.5, 2.0)
            effect = np.random.normal(0.5, 1/precision)
            
            # Apply publication bias
            if effect < 0 and np.random.random() < bias_strength:
                continue
                
            studies.append({
                'Study': f'Study {i+1}',
                'Effect': effect,
                'Precision': precision
            })
        
        return pd.DataFrame(studies)
    
    funnel_data = generate_funnel_data(n_studies, publication_bias)
    
    # Calculate pooled effect
    pooled_effect = np.average(funnel_data['Effect'], 
                             weights=funnel_data['Precision'])
    
    # Create funnel plot
    fig = go.Figure()
    
    # Add study points
    fig.add_trace(go.Scatter(
        x=funnel_data['Effect'],
        y=funnel_data['Precision'],
        mode='markers',
        marker=dict(size=10),
        name='Studies'
    ))
    
    # Add vertical line for pooled effect
    fig.add_vline(x=pooled_effect, line_dash="dash", line_color="red")
    
    # Update layout
    fig.update_layout(
        title='Funnel Plot',
        xaxis_title='Effect Size',
        yaxis_title='Precision (1/SE)',
        yaxis_autorange="reversed"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Egger's test
    X = np.column_stack([np.ones(len(funnel_data)), 1/funnel_data['Precision']])
    y = funnel_data['Effect'] * funnel_data['Precision']
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    st.write(f"Egger's test intercept: {beta[0]:.3f}")

elif method == "Quality Assessment":
    st.header("Study Quality Assessment")
    
    # Quality domains
    domains = [
        "Random sequence generation",
        "Allocation concealment",
        "Blinding of participants",
        "Blinding of outcome assessment",
        "Incomplete outcome data",
        "Selective reporting"
    ]
    
    # Number of studies
    n_studies = st.slider("Number of Studies", 2, 10, 5)
    
    # Create quality assessment interface
    quality_data = []
    
    for i in range(n_studies):
        st.subheader(f"Study {i+1}")
        study_quality = {}
        
        for domain in domains:
            status = st.selectbox(
                f"{domain} - Study {i+1}",
                ["Low Risk", "High Risk", "Unclear"],
                key=f"{domain}_{i}"
            )
            study_quality[domain] = status
        
        quality_data.append({
            'Study': f'Study {i+1}',
            **study_quality
        })
    
    df_quality = pd.DataFrame(quality_data)
    
    # Create risk of bias visualization
    fig = go.Figure()
    
    # Calculate percentages for each domain
    for domain in domains:
        low_risk = (df_quality[domain] == "Low Risk").mean() * 100
        high_risk = (df_quality[domain] == "High Risk").mean() * 100
        unclear = (df_quality[domain] == "Unclear").mean() * 100
        
        fig.add_trace(go.Bar(
            name="Low Risk",
            y=[domain],
            x=[low_risk],
            orientation='h',
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            name="High Risk",
            y=[domain],
            x=[high_risk],
            orientation='h',
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            name="Unclear",
            y=[domain],
            x=[unclear],
            orientation='h',
            marker_color='gray'
        ))
    
    fig.update_layout(
        barmode='stack',
        title='Risk of Bias Assessment',
        xaxis_title='Percentage of Studies',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Subgroup Analysis":
    st.header("Subgroup Analysis")
    
    # Parameters
    n_studies = st.slider("Number of Studies", 4, 20, 10)
    n_subgroups = st.slider("Number of Subgroups", 2, 4, 2)
    between_subgroup_variance = st.slider("Between-subgroup Variance", 0.0, 1.0, 0.3)
    
    # Generate subgroup data
    def generate_subgroup_data(n_studies, n_subgroups, between_var):
        studies = []
        subgroup_effects = np.random.normal(0.5, np.sqrt(between_var), n_subgroups)
        
        for i in range(n_studies):
            subgroup = np.random.randint(0, n_subgroups)
            true_effect = subgroup_effects[subgroup]
            
            precision = np.random.uniform(0.5, 2.0)
            effect = np.random.normal(true_effect, 1/precision)
            
            studies.append({
                'Study': f'Study {i+1}',
                'Subgroup': f'Subgroup {subgroup+1}',
                'Effect': effect,
                'Precision': precision
            })
        
        return pd.DataFrame(studies)
    
    subgroup_data = generate_subgroup_data(n_studies, n_subgroups, between_subgroup_variance)
    
    # Calculate subgroup effects
    subgroup_effects = []
    for subgroup in range(n_subgroups):
        mask = subgroup_data['Subgroup'] == f'Subgroup {subgroup+1}'
        subset = subgroup_data[mask]
        
        weights = subset['Precision']
        effect = np.average(subset['Effect'], weights=weights)
        
        subgroup_effects.append({
            'Subgroup': f'Subgroup {subgroup+1}',
            'Effect': effect
        })
    
    df_subgroups = pd.DataFrame(subgroup_effects)
    
    # Create forest plot by subgroup
    fig = go.Figure()
    
    # Add individual study effects
    for idx, row in subgroup_data.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Effect']],
            y=[f"{row['Study']} ({row['Subgroup']})"],
            mode='markers',
            marker=dict(size=10*row['Precision']),
            name=row['Subgroup'],
            showlegend=False
        ))
    
    # Add subgroup effects
    for idx, row in df_subgroups.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Effect']],
            y=[f"Overall ({row['Subgroup']})"],
            mode='markers',
            marker=dict(
                size=15,
                symbol='diamond'
            ),
            name=row['Subgroup']
        ))
    
    fig.update_layout(
        title='Subgroup Analysis Forest Plot',
        xaxis_title='Effect Size',
        yaxis_title='Study',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add educational content
st.header("Method Description")

# Continuing from previous code...

if method == "Forest Plot Builder":
    st.write("""
    Forest plots visualize meta-analysis results:
    
    1. Elements:
    - Individual study effects shown as squares or circles
    - Size of shapes represents study weight
    - Horizontal lines show confidence intervals
    - Diamond shows pooled effect estimate
    - Vertical line shows null effect
    
    2. Interpretation:
    - Position shows effect size
    - Width of confidence intervals shows precision
    - Overlap indicates consistency
    - Overall diamond shows final conclusion
    
    3. Best Practices:
    - Order studies by year or weight
    - Include study details
    - Show heterogeneity statistics
    - Include subgroup analyses when appropriate
    """)

elif method == "Heterogeneity Analysis":
    st.write("""
    Heterogeneity assessment examines consistency between studies:
    
    1. Statistics:
    - Q statistic: measures observed variation
    - I² statistic: percentage of variation due to heterogeneity
    - τ² (tau-squared): between-study variance
    
    2. Interpretation:
    - I² < 25%: Low heterogeneity
    - I² 25-75%: Moderate heterogeneity
    - I² > 75%: High heterogeneity
    
    3. Implications:
    - Affects choice of meta-analysis model
    - Guides subgroup analyses
    - Influences strength of conclusions
    - Directs future research
    """)

elif method == "Publication Bias Assessment":
    st.write("""
    Publication bias assessment evaluates reporting bias:
    
    1. Funnel Plot:
    - X-axis shows effect size
    - Y-axis shows precision or sample size
    - Asymmetry suggests possible bias
    - Gaps indicate missing studies
    
    2. Statistical Tests:
    - Egger's test
    - Begg's test
    - Trim and fill method
    - p-curve analysis
    
    3. Interpretation:
    - Symmetrical funnel suggests less bias
    - Small study effects
    - Impact on overall conclusions
    - Need for sensitivity analyses
    """)

elif method == "Quality Assessment":
    st.write("""
    Quality assessment evaluates study validity:
    
    1. Risk of Bias Domains:
    - Selection bias
    - Performance bias
    - Detection bias
    - Attrition bias
    - Reporting bias
    
    2. Assessment Tools:
    - Cochrane Risk of Bias Tool
    - Newcastle-Ottawa Scale
    - GRADE approach
    - ROBINS-I tool
    
    3. Impact Analysis:
    - Sensitivity analyses
    - Subgroup analyses by quality
    - Weighting by quality
    - Reporting standards
    """)

elif method == "Subgroup Analysis":
    st.write("""
    Subgroup analysis explores effect modifiers:
    
    1. Purpose:
    - Investigate heterogeneity
    - Identify effect modifiers
    - Guide clinical application
    - Generate hypotheses
    
    2. Considerations:
    - Pre-specified analyses
    - Biological plausibility
    - Number of subgroups
    - Multiple testing issues
    
    3. Interpretation:
    - Between-group differences
    - Within-group heterogeneity
    - Interaction tests
    - Clinical relevance
    """)

# Add best practices section
st.header("Best Practices")

if method == "Forest Plot Builder":
    st.write("""
    1. Data Presentation:
    - Include all essential study information
    - Show numerical results
    - Use consistent scales
    - Clear labeling
    
    2. Visual Elements:
    - Appropriate symbol sizes
    - Clear confidence intervals
    - Distinct subgroup formatting
    - Legible text size
    
    3. Documentation:
    - Effect measure used
    - Model specifications
    - Software details
    - Data sources
    """)

elif method == "Heterogeneity Analysis":
    st.write("""
    1. Analysis Steps:
    - Visual inspection
    - Statistical testing
    - Sensitivity analyses
    - Subgroup exploration
    
    2. Reporting:
    - All heterogeneity metrics
    - Confidence intervals
    - Model justification
    - Exploration results
    
    3. Decision Making:
    - Model selection
    - Subgroup analyses
    - Meta-regression
    - Conclusion strength
    """)

elif method == "Publication Bias Assessment":
    st.write("""
    1. Comprehensive Search:
    - Multiple databases
    - Grey literature
    - Conference abstracts
    - Expert consultation
    
    2. Analysis Approach:
    - Multiple methods
    - Sensitivity analyses
    - Trim and fill
    - Selection models
    
    3. Reporting:
    - Search strategy
    - Excluded studies
    - Bias assessment
    - Impact analysis
    """)

elif method == "Quality Assessment":
    st.write("""
    1. Assessment Process:
    - Independent reviewers
    - Standardized tools
    - Documentation
    - Resolution process
    
    2. Integration:
    - Quality-based analyses
    - Sensitivity testing
    - Reporting impact
    - Recommendations
    
    3. Transparency:
    - Assessment criteria
    - Reviewer decisions
    - Quality summaries
    - Limitations
    """)

elif method == "Subgroup Analysis":
    st.write("""
    1. Planning:
    - Pre-specification
    - Power considerations
    - Number of analyses
    - Clinical relevance
    
    2. Analysis:
    - Interaction testing
    - Within-group assessment
    - Between-group comparison
    - Sensitivity analyses
    
    3. Reporting:
    - All planned analyses
    - Statistical methods
    - Clinical implications
    - Limitations
    """)

# Add references section
st.header("Further Reading")
st.write("""
1. Higgins JPT, et al. Cochrane Handbook for Systematic Reviews of Interventions
2. Borenstein M, et al. Introduction to Meta-Analysis
3. Cooper H, et al. The Handbook of Research Synthesis and Meta-Analysis
4. Egger M, et al. Systematic Reviews in Health Care: Meta-Analysis in Context
""")

# Add practical examples
st.header("Practical Examples")
st.write("""
Visit the Cochrane Library (www.cochranelibrary.com) for examples of high-quality systematic reviews and meta-analyses in healthcare.

Key examples to study:
1. Intervention reviews
2. Diagnostic test accuracy reviews
3. Network meta-analyses
4. Individual participant data meta-analyses
""")

st.header("Check your understanding")
if method == "Forest Plot Builder":
    quiz_forest = st.radio(
        "What does the diamond in a forest plot represent?",
        [
            "The pooled effect estimate from all studies",
            "The largest individual study effect",
            "A marker for publication bias",
            "The study with the highest weight"
        ]
    )
    if quiz_forest == "The pooled effect estimate from all studies":
        st.success("Correct! The diamond represents the overall summary effect size with its confidence interval.")
    else:
        st.error("Not quite. The diamond represents the overall effect estimate from the meta-analysis.")
        
elif method == "Heterogeneity Analysis":
    quiz_heterogeneity = st.radio(
        "What does an I² statistic of 80% indicate?",
        [
            "Low heterogeneity between studies",
            "Moderate heterogeneity between studies",
            "High heterogeneity between studies",
            "No heterogeneity detected"
        ]
    )
    if quiz_heterogeneity == "High heterogeneity between studies":
        st.success("Correct! An I² of 80% suggests substantial variability between study results.")
    else:
        st.error("Not quite. Higher I² values indicate greater heterogeneity.")

elif method == "Publication Bias Assessment":
    quiz_pub_bias = st.radio(
        "What is a key characteristic of publication bias in a funnel plot?",
        [
            "A symmetrical distribution of studies",
            "A skewed or asymmetric funnel shape",
            "Larger studies clustered at the bottom",
            "A wider funnel with increasing precision"
        ]
    )
    if quiz_pub_bias == "A skewed or asymmetric funnel shape":
        st.success("Correct! An asymmetrical funnel suggests missing studies, likely due to publication bias.")
    else:
        st.error("Not quite. Publication bias often leads to an asymmetrical funnel plot.")

elif method == "Quality Assessment":
    quiz_quality = st.radio(
        "Which domain is commonly assessed in a risk of bias evaluation?",
        [
            "Effect size estimation",
            "Blinding of participants and assessors",
            "Funnel plot asymmetry",
            "Number of included studies"
        ]
    )
    if quiz_quality == "Blinding of participants and assessors":
        st.success("Correct! Blinding is crucial to minimize performance and detection bias in studies.")
    else:
        st.error("Not quite. Risk of bias assessments focus on study design, including blinding and allocation concealment.")

elif method == "Subgroup Analysis":
    quiz_subgroup = st.radio(
        "What is the primary reason for conducting a subgroup analysis?",
        [
            "To confirm that all studies report the same results",
            "To explore potential sources of heterogeneity",
            "To remove studies that do not fit the hypothesis",
            "To make the meta-analysis results appear stronger"
        ]
    )
    if quiz_subgroup == "To explore potential sources of heterogeneity":
        st.success("Correct! Subgroup analysis helps identify factors that may explain differences between study results.")
    else:
        st.error("Not quite. Subgroup analysis is used to investigate sources of variability in study outcomes.")
