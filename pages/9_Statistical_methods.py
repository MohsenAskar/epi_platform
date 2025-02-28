# pages/9_statistical_methods.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter

st.title("Statistical Methods in Epidemiology")

# Method selector
method = st.selectbox(
    "Select Statistical Method",
    ["Logistic Regression", "Cox Proportional Hazards", "Poisson Regression", 
     "Propensity Score Analysis"]
)


if method == "Logistic Regression":
    st.header("Logistic Regression Analysis")
    
    # Parameters
    n_samples = st.slider("Sample Size", 100, 1000, 500)
    effect_size = st.slider("Effect Size (Log Odds)", -2.0, 2.0, 1.0)
    confounder_strength = st.slider("Confounder Strength", 0.0, 2.0, 1.0)
    
    # Generate data
    def generate_logistic_data(n, effect, conf_strength):
        # Generate exposure and confounder
        confounder = np.random.normal(0, 1, n)
        exposure = np.random.binomial(1, 0.5, n)
        
        # Generate outcome
        logit = -2 + effect * exposure + conf_strength * confounder
        prob = 1 / (1 + np.exp(-logit))
        outcome = np.random.binomial(1, prob)
        
        return pd.DataFrame({
            'exposure': exposure,
            'confounder': confounder,
            'outcome': outcome
        })
    
    data = generate_logistic_data(n_samples, effect_size, confounder_strength)
    
    try:
        # Fit models
        # Unadjusted
        X_unadj = data['exposure'].values.reshape(-1, 1)
        X_unadj = sm.add_constant(X_unadj)  # Add constant term
        y = data['outcome'].values
        model_unadj = sm.Logit(y, X_unadj).fit(disp=0)
        
        # Adjusted
        X_adj = data[['exposure', 'confounder']].values
        X_adj = sm.add_constant(X_adj)  # Add constant term
        model_adj = sm.Logit(y, X_adj).fit(disp=0)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Unadjusted Analysis")
            # Use direct array indexing for numpy arrays
            unadj_or = np.exp(model_unadj.params[1]).round(2)
            unadj_ci = np.exp(model_unadj.conf_int()[1]).round(2)
            st.write("Odds Ratio:", unadj_or)
            st.write("95% CI:", [unadj_ci[0], unadj_ci[1]])
        
        with col2:
            st.subheader("Adjusted Analysis")
            # Use direct array indexing for numpy arrays
            adj_or = np.exp(model_adj.params[1]).round(2)
            adj_ci = np.exp(model_adj.conf_int()[1]).round(2)
            st.write("Odds Ratio:", adj_or)
            st.write("95% CI:", [adj_ci[0], adj_ci[1]])
        
        # Visualization
        fig = go.Figure()
        
        # Predicted probabilities
        confounder_range = np.linspace(data['confounder'].min(), data['confounder'].max(), 100)
        for exposure_val in [0, 1]:
            X_pred = np.column_stack([
                np.ones(100),
                np.repeat(exposure_val, 100),
                confounder_range
            ])
            y_pred = model_adj.predict(X_pred)
            
            fig.add_trace(go.Scatter(
                x=confounder_range,
                y=y_pred,
                name=f'Exposure = {exposure_val}',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Predicted Probabilities by Confounder Value',
            xaxis_title='Confounder',
            yaxis_title='Predicted Probability'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add model summary statistics
        st.subheader("Model Summary")
        st.write("Unadjusted Model:")
        st.write(f"- Pseudo R-squared: {model_unadj.prsquared:.3f}")
        st.write(f"- Log-likelihood: {model_unadj.llf:.3f}")
        st.write(f"- AIC: {model_unadj.aic:.3f}")
        
        st.write("Adjusted Model:")
        st.write(f"- Pseudo R-squared: {model_adj.prsquared:.3f}")
        st.write(f"- Log-likelihood: {model_adj.llf:.3f}")
        st.write(f"- AIC: {model_adj.aic:.3f}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug information:")
        st.write("Data shape:", data.shape)
        st.write("X_unadj shape:", X_unadj.shape)
        st.write("X_adj shape:", X_adj.shape)
        st.write("y shape:", y.shape)
        
# Cox Proportional Hazards section with fixed model fitting

elif method == "Cox Proportional Hazards":
    st.header("Cox Proportional Hazards Analysis")
    
    # Parameters
    n_samples = st.slider("Sample Size", 100, 1000, 500)
    hazard_ratio = st.slider("Hazard Ratio", 1.0, 5.0, 2.0)
    censoring_rate = st.slider("Censoring Rate", 0.0, 0.8, 0.3)
    
    # Generate survival data
    def generate_survival_data(n, hr, censor_rate):
        exposure = np.random.binomial(1, 0.5, n)
        
        # Generate survival times
        baseline = np.random.exponential(1, n)
        time = baseline * np.exp(-np.log(hr) * exposure)
        
        # Generate censoring
        c_time = np.random.exponential(1/censor_rate, n) if censor_rate > 0 else np.inf
        observed_time = np.minimum(time, c_time)
        event = (time <= c_time).astype(int)
        
        # Add some covariates for more realistic data
        age = np.random.normal(60, 10, n)
        sex = np.random.binomial(1, 0.5, n)
        
        return pd.DataFrame({
            'duration': observed_time,
            'event': event,
            'exposure': exposure,
            'age': age,
            'sex': sex
        })
    
    try:
        data = generate_survival_data(n_samples, hazard_ratio, censoring_rate)
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(data, 'duration', 'event')
        
        # Display results
        st.subheader("Cox Model Results")
        
        # Create a formatted results table using model attributes directly
        results_df = pd.DataFrame({
            'Hazard Ratio': np.exp(cph.params_),
            'Lower 95% CI': np.exp(cph.confidence_intervals_['95% lower-bound']),
            'Upper 95% CI': np.exp(cph.confidence_intervals_['95% upper-bound'])
        }).round(3)
        
        st.write("Model Coefficients:")
        st.dataframe(results_df)
        
        # Model diagnostics
        st.subheader("Model Diagnostics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Model Statistics:")
            st.write(f"Concordance Index: {cph.concordance_index_:.3f}")
            st.write(f"Partial AIC: {cph.AIC_partial_:.3f}")
            st.write(f"Log Likelihood: {cph.log_likelihood_:.3f}")
        
        with col2:
            st.write("Number of Observations:")
            st.write(f"Total: {len(data)}")
            st.write(f"Events: {sum(data['event'])}")
            st.write(f"Censored: {len(data) - sum(data['event'])} ({(1 - sum(data['event'])/len(data))*100:.1f}%)")
        
        # Visualization
        st.subheader("Survival Curves by Exposure Status")
        
        # Create groups for plotting
        groups = data['exposure'].unique()
        fig = go.Figure()
        
        # Get median values for other covariates
        median_data = data.copy()
        for col in ['age', 'sex']:
            median_data[col] = data[col].median()
        
        for group in sorted(groups):
            # Create prediction data
            pred_data = median_data.iloc[0:1].copy()
            pred_data['exposure'] = group
            
            # Predict survival curve
            sf = cph.predict_survival_function(pred_data)
            
            fig.add_trace(go.Scatter(
                x=sf.index,
                y=sf.values.flatten(),
                name=f'Exposure = {group}',
                mode='lines'
            ))
        
        fig.update_layout(
            title='Predicted Survival Curves (adjusted for median covariates)',
            xaxis_title='Time',
            yaxis_title='Survival Probability',
            yaxis_range=[0, 1],
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add baseline hazard plot
        st.subheader("Baseline Cumulative Hazard")
        
        baseline_hazard = cph.baseline_cumulative_hazard_
        fig_hazard = go.Figure()
        
        fig_hazard.add_trace(go.Scatter(
            x=baseline_hazard.index,
            y=baseline_hazard.values.flatten(),
            mode='lines',
            name='Baseline Cumulative Hazard'
        ))
        
        fig_hazard.update_layout(
            title='Baseline Cumulative Hazard Function',
            xaxis_title='Time',
            yaxis_title='Cumulative Hazard',
            showlegend=True
        )
        
        st.plotly_chart(fig_hazard, use_container_width=True)
        
        # Add proportional hazards assumption test
        st.subheader("Proportional Hazards Assumption Test")
        try:
            ph_test = cph.check_assumptions(data, show_plots=False)
            if ph_test is not None:
                st.write("Schoenfeld Residuals Test p-values:")
                st.dataframe(ph_test.round(3))
            else:
                st.write("Proportional hazards test results not available.")
        except Exception as e:
            st.write("Could not perform proportional hazards assumption test.")
            st.write(f"Error: {str(e)}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug information:")
        st.write("Data shape:", data.shape)
        st.write("Data columns:", data.columns.tolist())
        st.write("Number of events:", data['event'].sum())
        st.write("Duration range:", [data['duration'].min(), data['duration'].max()])
elif method == "Poisson Regression":
    st.header("Poisson Regression Analysis")
    
    # Parameters
    n_groups = st.slider("Number of Groups", 10, 50, 20)
    rate_ratio = st.slider("Rate Ratio", 1.0, 5.0, 2.0)
    baseline_rate = st.slider("Baseline Rate (per 1000)", 1, 50, 10)
    
    # Generate count data
    def generate_poisson_data(n_groups, rr, baseline):
        exposure = np.random.binomial(1, 0.5, n_groups)
        population = np.random.uniform(1000, 5000, n_groups)
        
        # Generate counts
        rate = baseline/1000 * np.exp(np.log(rr) * exposure)
        counts = np.random.poisson(rate * population)
        
        return pd.DataFrame({
            'counts': counts,
            'population': population,
            'exposure': exposure,
            'rate': counts/population * 1000
        })
    
    data = generate_poisson_data(n_groups, rate_ratio, baseline_rate)
    
    # Fit Poisson model
    model = sm.GLM(
        data['counts'],
        sm.add_constant(data['exposure']),
        offset=np.log(data['population']),
        family=sm.families.Poisson()
    ).fit()
    
    # Display results
    st.subheader("Poisson Model Results")
    st.write("Rate Ratio:", np.exp(model.params['exposure']).round(2))
    st.write("95% CI:", np.exp(model.conf_int().loc['exposure']).round(2).tolist())
    
    # Visualization
    fig = px.scatter(
        data,
        x='population',
        y='rate',
        color='exposure',
        title='Rates by Population Size and Exposure Status'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Propensity Score Analysis":
    st.header("Propensity Score Analysis")
    
    # Parameters
    n_samples = st.slider("Sample Size", 100, 1000, 500)
    true_effect = st.slider("True Effect", -2.0, 2.0, 1.0)
    confounding_strength = st.slider("Confounding Strength", 0.0, 2.0, 1.0)
    
    # Generate data with confounding
    def generate_ps_data(n, effect, conf_strength):
        # Generate confounders
        age = np.random.normal(50, 10, n)
        income = np.random.normal(50000, 20000, n)
        
        # Generate exposure (treatment) based on confounders
        logit_treat = -1 + 0.03 * age + income/50000
        p_treat = 1 / (1 + np.exp(-logit_treat))
        treatment = np.random.binomial(1, p_treat)
        
        # Generate outcome
        outcome = (2 + effect * treatment + 
                  conf_strength * (age/10) + 
                  conf_strength * (income/50000) +
                  np.random.normal(0, 1, n))
        
        return pd.DataFrame({
            'age': age,
            'income': income,
            'treatment': treatment,
            'outcome': outcome
        })
    
    data = generate_ps_data(n_samples, true_effect, confounding_strength)
    
    # Fit propensity score model
    ps_model = LogisticRegression()
    ps_model.fit(data[['age', 'income']], data['treatment'])
    ps_scores = ps_model.predict_proba(data[['age', 'income']])
    data['propensity_score'] = ps_scores[:, 1]  # Get probability of treatment
    
    # Create PS matched groups
    def match_on_ps(df, caliper=0.2):
        # Ensure we have the required columns
        required_cols = ['treatment', 'propensity_score', 'outcome']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")
            
        # Split into treated and control groups
        treated = df[df['treatment'] == 1].copy()
        control = df[df['treatment'] == 0].copy()
        
        if treated.empty or control.empty:
            st.error("No treated or control units found in the data")
            return df
            
        matched_pairs = []
        used_control = set()
        
        for idx, treated_unit in treated.iterrows():
            ps_treated = treated_unit['propensity_score']
            
            # Find closest control unit within caliper
            distances = abs(control['propensity_score'] - ps_treated)
            valid_matches = distances[distances < caliper]
            
            if not valid_matches.empty:
                best_match_idx = valid_matches.idxmin()
                if best_match_idx not in used_control:
                    matched_pairs.append((treated_unit, control.loc[best_match_idx]))
                    used_control.add(best_match_idx)
        
        if not matched_pairs:
            st.warning("No matches found within caliper distance")
            return df
            
        return pd.concat([pd.DataFrame([t, c]) for t, c in matched_pairs])
    
    matched_data = match_on_ps(data.copy())
    
    # Calculate effects
    naive_effect = data.groupby('treatment')['outcome'].mean().diff().iloc[-1]
    matched_effect = matched_data.groupby('treatment')['outcome'].mean().diff().iloc[-1]
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Naive Analysis")
        st.write("Treatment Effect:", round(naive_effect, 2))
    
    with col2:
        st.subheader("PS Matched Analysis")
        st.write("Treatment Effect:", round(matched_effect, 2))
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data[data['treatment']==1]['propensity_score'],
        name='Treated',
        opacity=0.75
    ))
    
    fig.add_trace(go.Histogram(
        x=data[data['treatment']==0]['propensity_score'],
        name='Control',
        opacity=0.75
    ))
    
    fig.update_layout(
        title='Propensity Score Distribution',
        xaxis_title='Propensity Score',
        yaxis_title='Count',
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Educational content
st.header("Method Description")

if method == "Logistic Regression":
    st.write("""
    Logistic regression is used for binary outcomes in epidemiology. Key features:
    
    1. Models probability of outcome on log-odds scale
    2. Coefficients interpreted as log odds ratios
    3. Useful for:
       - Case-control studies
       - Cross-sectional studies
       - Cohort studies with binary outcomes
    
    Assumptions:
    - Linearity in the logit
    - Independence of observations
    - No perfect separation
    """)

elif method == "Cox Proportional Hazards":
    st.write("""
    Cox proportional hazards models analyze time-to-event data. Key features:
    
    1. Models hazard rate over time
    2. Handles censored observations
    3. Coefficients interpreted as log hazard ratios
    
    Assumptions:
    - Proportional hazards
    - Independent censoring
    - Non-informative censoring
    """)

elif method == "Poisson Regression":
    st.write("""
    Poisson regression models count data or rates. Key features:
    
    1. Models count data with exposure time
    2. Coefficients interpreted as log rate ratios
    3. Useful for:
       - Incidence rates
       - Rare events
       - Person-time data
    
    Assumptions:
    - Mean equals variance
    - Independence of events
    - Constant rate over time period
    """)

elif method == "Propensity Score Analysis":
    st.write("""
    Propensity score analysis helps control for confounding. Key features:
    
    1. Balances covariates between groups
    2. Reduces dimensionality of adjustment
    3. Methods include:
       - Matching
       - Stratification
       - Inverse probability weighting
    
    Assumptions:
    - No unmeasured confounding
    - Positivity
    - Correct PS model specification
    """)

st.header("When to Use This Method")
if method == "Logistic Regression":
    st.write("""
    Use logistic regression when:
    1. Outcome is binary (yes/no)
    2. Need to adjust for multiple covariates
    3. Interest is in odds ratios
    4. Multiple predictors need to be evaluated
    """)

elif method == "Cox Proportional Hazards":
    st.write("""
    Use Cox models when:
    1. Studying time to event
    2. Have censored observations
    3. Interest is in survival probabilities
    4. Need to compare survival between groups
    """)

# Continuing from the previous code...

elif method == "Poisson Regression":
    st.write("""
    Use Poisson regression when:
    1. Analyzing count data
    2. Studying rare events
    3. Calculating incidence rates
    4. Working with person-time data
    5. Evaluating disease clustering
    6. Analyzing temporal trends in disease occurrence
    
    Common applications:
    - Disease surveillance
    - Cancer registries
    - Mortality studies
    - Hospital admission rates
    - Event clustering in time or space
    """)

elif method == "Propensity Score Analysis":
    st.write("""
    Use propensity score analysis when:
    1. Conducting observational studies
    2. Having many potential confounders
    3. Treatment groups are imbalanced
    4. Need to mimic randomization
    5. Sample size allows for matching/stratification
    
    Common applications:
    - Pharmacoepidemiologic studies
    - Health services research
    - Policy evaluation
    - Treatment effectiveness studies
    """)

# Add practical considerations section
st.header("Practical Considerations")

if method == "Logistic Regression":
    st.write("""
    Key considerations for logistic regression:
    
    1. Sample Size Requirements
    - Rule of thumb: 10-20 events per predictor variable
    - More events needed for rare outcomes
    - Consider sparse data bias
    
    2. Model Building
    - Check for multicollinearity
    - Assess linearity assumptions
    - Consider interaction terms
    - Evaluate model fit (e.g., Hosmer-Lemeshow test)
    
    3. Common Pitfalls
    - Overfitting with too many predictors
    - Incorrect handling of continuous variables
    - Missing data issues
    - Selection bias in case-control studies
    
    4. Reporting Guidelines
    - Report odds ratios with confidence intervals
    - Describe model building strategy
    - Present goodness-of-fit statistics
    - Document variable selection process
    """)

elif method == "Cox Proportional Hazards":
    st.write("""
    Key considerations for Cox models:
    
    1. Testing Assumptions
    - Check proportional hazards using:
      * Schoenfeld residuals
      * Log-log plots
      * Time-dependent covariates
    
    2. Handling Violations
    - Stratification
    - Time-dependent coefficients
    - Alternative parametric models
    
    3. Dealing with Tied Events
    - Breslow approximation
    - Exact method for small datasets
    - Impact on estimation
    
    4. Special Situations
    - Competing risks
    - Recurrent events
    - Time-varying covariates
    - Left truncation
    """)

elif method == "Poisson Regression":
    st.write("""
    Key considerations for Poisson regression:
    
    1. Overdispersion
    - Check variance-mean relationship
    - Consider negative binomial alternative
    - Use quasi-Poisson methods
    - Calculate dispersion parameter
    
    2. Zero-Inflation
    - Assess zero-inflation
    - Consider zero-inflated models
    - Use hurdle models when appropriate
    
    3. Exposure Time
    - Proper use of offset terms
    - Handling varying follow-up times
    - Accounting for population size
    
    4. Model Diagnostics
    - Residual analysis
    - Goodness-of-fit tests
    - Influence diagnostics
    """)

elif method == "Propensity Score Analysis":
    st.write("""
    Key considerations for propensity score analysis:
    
    1. Variable Selection
    - Include all potential confounders
    - Avoid colliders
    - Consider instrumental variables
    - Balance pre-treatment variables
    
    2. Matching Considerations
    - Choose appropriate caliper width
    - Decide on matching ratio
    - With/without replacement
    - Assess match quality
    
    3. Alternative Methods
    - Stratification
    - Inverse probability weighting
    - Covariate adjustment
    - Doubly robust estimation
    
    4. Balance Assessment
    - Standardized differences
    - Love plot
    - Common support evaluation
    - Covariate balance plots
    """)

# Add example code section
st.header("Example Code Implementation")

if method == "Logistic Regression":
    st.code("""
    # Example R code
    model <- glm(outcome ~ exposure + confounder1 + confounder2, 
                family = binomial(link = "logit"), 
                data = your_data)
    
    # Example Python code
    from statsmodels.api import Logit
    model = Logit(y, X).fit()
    """, language='python')
    # Fit models
    # Unadjusted
    X_unadj = sm.add_constant(data['exposure'].values.reshape(-1, 1))
    model_unadj = sm.Logit(data['outcome'], X_unadj).fit(disp=0)
    
    # Adjusted
    X_adj = np.column_stack((
        np.ones(len(data)),
        data['exposure'].values,
        data['confounder'].values
    ))
    model_adj = sm.Logit(data['outcome'], X_adj).fit(disp=0)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Unadjusted Analysis")
        st.write("Odds Ratio:", np.exp(model_unadj.params[1]).round(2))  # Index 1 for exposure
        conf_int_unadj = model_unadj.conf_int().iloc[1].values  # Index 1 for exposure
        st.write("95% CI:", np.exp(conf_int_unadj).round(2).tolist())
    
    with col2:
        st.subheader("Adjusted Analysis")
        st.write("Odds Ratio:", np.exp(model_adj.params[1]).round(2))  # Index 1 for exposure
        conf_int_adj = model_adj.conf_int().iloc[1].values  # Index 1 for exposure
        st.write("95% CI:", np.exp(conf_int_adj).round(2).tolist())
    

elif method == "Cox Proportional Hazards":
    st.code("""
    # Example R code
    library(survival)
    model <- coxph(Surv(time, event) ~ exposure + strata(group), 
                   data = your_data)
    
    # Example Python code
    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    cph.fit(df, 'time', 'event')
    """, language='python')
    # Generate survival data
    def generate_survival_data(n, hr, censor_rate):
        exposure = np.random.binomial(1, 0.5, n)
        
        # Generate survival times
        baseline = np.random.exponential(1, n)
        time = baseline * np.exp(-np.log(hr) * exposure)
        
        # Generate censoring
        c_time = np.random.exponential(1/censor_rate, n)
        observed_time = np.minimum(time, c_time)
        event = (time <= c_time).astype(int)
        
        return pd.DataFrame({
            'duration': observed_time,  # Changed 'time' to 'duration'
            'event': event,
            'exposure': exposure
        })
    
    data = generate_survival_data(n_samples, hazard_ratio, censoring_rate)
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(data, duration_col='duration', event_col='event')  # Removed covariates parameter
    
    # Display results
    st.subheader("Cox Model Results")
    st.write("Hazard Ratio:", np.exp(cph.params_['exposure']).round(2))
    
    # Calculate survival curves for plotting
    survival_curves = cph.predict_survival_function(data)
    
    # Visualization
    fig = go.Figure()
    
    for exposure_val in [0, 1]:
        mask = data['exposure'] == exposure_val
        if mask.any():
            mean_survival = survival_curves.loc[:, mask].mean(axis=1)
            fig.add_trace(go.Scatter(
                x=survival_curves.index,
                y=mean_survival,
                name=f'Exposure = {exposure_val}'
            ))

elif method == "Poisson Regression":
    st.code("""
    # Example R code
    model <- glm(counts ~ exposure + offset(log(time)), 
                family = poisson(link = "log"), 
                data = your_data)
    
    # Example Python code
    import statsmodels.api as sm
    model = sm.GLM(y, X, family=sm.families.Poisson(), 
                   offset=np.log(exposure_time)).fit()
    """, language='python')

elif method == "Propensity Score Analysis":
    st.code("""
    # Example R code
    library(MatchIt)
    ps_model <- matchit(treatment ~ x1 + x2 + x3, 
                       method = "nearest", 
                       data = your_data)
    
    # Example Python code
    from sklearn.linear_model import LogisticRegression
    ps_model = LogisticRegression()
    ps_scores = ps_model.fit_predict_proba(X, treatment)
    """, language='python')

    # Generate data with confounding
    def generate_ps_data(n, effect, conf_strength):
        # Generate confounders
        age = np.random.normal(50, 10, n)
        income = np.random.normal(50000, 20000, n)
        
        # Generate exposure (treatment) based on confounders
        logit_treat = -1 + 0.03 * age + income/50000
        p_treat = 1 / (1 + np.exp(-logit_treat))
        treatment = np.random.binomial(1, p_treat)  # Changed 'treatment' to 'exposure'
        
        # Generate outcome
        outcome = (2 + effect * treatment  + 
                  conf_strength * (age/10) + 
                  conf_strength * (income/50000) +
                  np.random.normal(0, 1, n))
        
        return pd.DataFrame({
            'age': age,
            'income': income,
            'treatment': treatment,  # Changed 'treatment' to 'exposure'
            'outcome': outcome
        })
    
    data = generate_ps_data(n_samples, true_effect, confounding_strength)
    
    # Fit propensity score model
    ps_model = LogisticRegression()
    ps_model.fit(data[['age', 'income']], data['treatment'])  # Changed 'treatment' to 'exposure'
    data['propensity_score'] = ps_model.predict_proba(data[['age', 'income']])[:, 1]
      
    # Calculate effects
    naive_effect = data.groupby('treatment')['outcome'].mean().diff().iloc[-1]  # Changed 'treatment' to 'exposure'
    
    # Only calculate matched effect if we have matched data
    if not matched_data.empty:
        matched_effect = matched_data.groupby('treatment')['outcome'].mean().diff().iloc[-1]  # Changed 'treatment' to 'exposure'
    else:
        matched_effect = None

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Naive Analysis")
        st.write("Treatment Effect:", round(naive_effect, 2))
    
    with col2:
        st.subheader("PS Matched Analysis")
        if matched_effect is not None:
            st.write("Treatment Effect:", round(matched_effect, 2))
        else:
            st.write("No matched pairs found with current settings")

# Add references section
st.header("Further Reading")
st.write("""
1. Rothman KJ, Greenland S, Lash TL. Modern Epidemiology, 3rd Edition.
2. Hosmer DW, Lemeshow S. Applied Logistic Regression.
3. Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox Model.
4. Rosenbaum PR. Observational Studies, 2nd Edition.
""")

st.header("Check your understanding")

if method == "Logistic Regression":
    quiz_logistic = st.radio(
        "Which value is primarily interpreted as the strength of association in logistic regression?",
        ["Coefficient (Beta)", "Odds Ratio (exp(Beta))", "Mean difference in outcome"]
    )
    if quiz_logistic == "Odds Ratio (exp(Beta))":
        st.success("Correct! In logistic regression, we typically interpret the exponentiated coefficients.")
    else:
        st.error("Not quite. The logistic regression coefficient itself is on the log-odds scale; we usually use exp(Beta).")

elif method == "Cox Proportional Hazards":
    quiz_cox = st.radio(
        "What assumption must hold true for valid interpretation of HRs in a Cox model?",
        [
            "Time-invariant hazards for each individual",
            "Proportional hazards (the ratio of hazards is constant over time)",
            "No randomization needed"
        ]
    )
    if quiz_cox == "Proportional hazards (the ratio of hazards is constant over time)":
        st.success("Correct! The core assumption is that the hazard ratio is constant over time.")
    else:
        st.error("That's not correct. The key assumption is that hazards remain proportional.")

elif method == "Poisson Regression":
    quiz_poisson = st.radio(
        "Which type of data is Poisson regression best suited for?",
        ["Binary outcome data", "Time-to-event data", "Count or rate data"]
    )
    if quiz_poisson == "Count or rate data":
        st.success("Correct! Poisson regression is designed for modeling counts (and optionally offset by exposure time).")
    else:
        st.error("Not quite. Poisson regression isn't used for binary or time-to-event data by default.")

elif method == "Propensity Score Analysis":
    quiz_ps = st.radio(
        "What is the primary goal of using propensity scores in an observational study?",
        [
            "To make the treatment effect larger",
            "To balance observed covariates between treated and control groups",
            "To remove the need for randomization"
        ]
    )
    if quiz_ps == "To balance observed covariates between treated and control groups":
        st.success("Correct! Propensity scores aim to mimic randomization by creating balance in measured covariates.")
    else:
        st.error("Not quite. The main purpose is to achieve balance in observed confounders.")