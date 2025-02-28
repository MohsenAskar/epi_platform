# pages/13_machine_learning.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.title("Machine Learning in Epidemiology")

# Method selector
method = st.selectbox(
    "Select Analysis Method",
    ["Prediction Models", "Risk Stratification", 
     "Feature Selection", "Model Comparison",
     "Outbreak Prediction"]
)

if method == "Prediction Models":
    st.header("Disease Prediction Models")
    
    # Parameters for data generation
    n_samples = st.slider("Sample Size", 100, 1000, 500)
    n_risk_factors = st.slider("Number of Risk Factors", 2, 10, 5)
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.2)
    
    # Generate synthetic data
    def generate_prediction_data(n_samples, n_features, noise):
        X = np.random.normal(0, 1, (n_samples, n_features))
        # Create true relationship
        true_betas = np.random.uniform(-1, 1, n_features)
        logits = X @ true_betas + np.random.normal(0, noise, n_samples)
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)
        
        return X, y, true_betas
    
    X, y, true_coefficients = generate_prediction_data(n_samples, n_risk_factors, noise_level)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Logistic Regression", "Random Forest", "SVM"]
    )
    
    # Train selected model
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC(probability=True)
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.2f})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash='dash'),
        name='Random'
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if model_type == "Logistic Regression":
        importance = pd.DataFrame({
            'Feature': [f'Risk Factor {i+1}' for i in range(n_risk_factors)],
            'Importance': abs(model.coef_[0])
        })
    elif model_type == "Random Forest":
        importance = pd.DataFrame({
            'Feature': [f'Risk Factor {i+1}' for i in range(n_risk_factors)],
            'Importance': model.feature_importances_
        })
    else:
        importance = pd.DataFrame({
            'Feature': [f'Risk Factor {i+1}' for i in range(n_risk_factors)],
            'Importance': abs(true_coefficients)  # Use true coefficients for SVM
        })
    
    fig = px.bar(
        importance.sort_values('Importance', ascending=False),
        x='Feature',
        y='Importance',
        title='Feature Importance'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Risk Stratification":
    st.header("Risk Stratification Analysis")
    
    # Parameters
    n_patients = st.slider("Number of Patients", 100, 1000, 500)
    n_features = 3  # Fixed number for simplicity
    
    # Generate patient data
    def generate_patient_data(n_patients):
        data = pd.DataFrame({
            'Age': np.random.normal(60, 10, n_patients),
            'BP': np.random.normal(130, 20, n_patients),
            'BMI': np.random.normal(25, 5, n_patients)
        })
        
        # Generate risk scores
        risk_score = (
            0.02 * data['Age'] +
            0.01 * data['BP'] +
            0.03 * data['BMI'] +
            np.random.normal(0, 0.5, n_patients)
        )
        
        data['Risk_Score'] = risk_score
        data['Risk_Category'] = pd.qcut(risk_score, q=3, labels=['Low', 'Medium', 'High'])
        
        return data
    
    patient_data = generate_patient_data(n_patients)
    
    # Risk distribution
    fig = px.histogram(
        patient_data,
        x='Risk_Score',
        color='Risk_Category',
        title='Distribution of Risk Scores'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    fig = px.scatter_matrix(
        patient_data,
        dimensions=['Age', 'BP', 'BMI'],
        color='Risk_Category',
        title='Feature Relationships'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive risk calculator
    st.subheader("Risk Calculator")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 30, 90, 60)
    with col2:
        bp = st.number_input("Blood Pressure", 90, 200, 130)
    with col3:
        bmi = st.number_input("BMI", 15, 45, 25)
    
    # Calculate risk
    risk = 0.02 * age + 0.01 * bp + 0.03 * bmi
    
    # Determine risk category
    risk_thresholds = patient_data['Risk_Score'].quantile([0.33, 0.67])
    if risk < risk_thresholds.iloc[0]:
        category = "Low"
        color = "green"
    elif risk < risk_thresholds.iloc[1]:
        category = "Medium"
        color = "orange"
    else:
        category = "High"
        color = "red"
    
    st.metric("Calculated Risk Score", f"{risk:.2f}")
    st.markdown(f"Risk Category: <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)

elif method == "Feature Selection":
    st.header("Feature Selection Methods")
    
    # Parameters
    n_samples = st.slider("Sample Size", 100, 1000, 500)
    n_features = st.slider("Number of Features", 5, 20, 10)
    n_informative = st.slider("Number of Informative Features", 2, n_features, 5)
    
    # Generate data with known important features
    def generate_feature_data(n_samples, n_features, n_informative):
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Create outcome using only informative features
        informative_coef = np.zeros(n_features)
        informative_coef[:n_informative] = np.random.uniform(0.5, 1.5, n_informative)
        
        y = X @ informative_coef + np.random.normal(0, 0.1, n_samples)
        y = (y > np.median(y)).astype(int)  # Convert to binary outcome
        
        return X, y, informative_coef
    
    X, y, true_importance = generate_feature_data(n_samples, n_features, n_informative)
    
    # Feature selection method
    selection_method = st.selectbox(
        "Select Feature Selection Method",
        ["Random Forest Importance", "Univariate Selection", "L1 Regularization"]
    )
    
    # Apply selection method
    if selection_method == "Random Forest Importance":
        rf = RandomForestClassifier()
        rf.fit(X, y)
        importance = rf.feature_importances_
    elif selection_method == "Univariate Selection":
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X, y)
        importance = selector.scores_
    else:  # L1 Regularization
        l1_model = LogisticRegression(penalty='l1', solver='liblinear')
        l1_model.fit(X, y)
        importance = abs(l1_model.coef_[0])
    
    # Visualize feature importance
    importance_df = pd.DataFrame({
        'Feature': [f'Feature {i+1}' for i in range(n_features)],
        'Importance': importance,
        'True_Importance': abs(true_importance)
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance_df['Feature'],
        y=importance_df['Importance'],
        name='Estimated Importance'
    ))
    
    fig.add_trace(go.Bar(
        x=importance_df['Feature'],
        y=importance_df['True_Importance'],
        name='True Importance'
    ))
    
    fig.update_layout(
        title='Feature Importance Comparison',
        xaxis_title='Features',
        yaxis_title='Importance',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Model Comparison":
    st.header("Model Comparison Analysis")
    
    # Parameters
    n_samples = st.slider("Sample Size", 100, 1000, 500)
    complexity = st.slider("Data Complexity", 0.1, 2.0, 1.0)
    
    # Generate complex data
    def generate_complex_data(n_samples, complexity):
        X = np.random.normal(0, 1, (n_samples, 2))
        
        # Generate non-linear decision boundary
        radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        angle = np.arctan2(X[:, 1], X[:, 0])
        
        y = (np.sin(complexity * radius) + np.cos(complexity * angle) > 0).astype(int)
        
        return X, y
    
    X, y = generate_complex_data(n_samples, complexity)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train different models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        results[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
    
    # Plot ROC curves
    fig = go.Figure()
    
    for name, res in results.items():
        fig.add_trace(go.Scatter(
            x=res['fpr'],
            y=res['tpr'],
            name=f'{name} (AUC = {res["auc"]:.2f})'
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        line=dict(dash='dash'),
        name='Random'
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif method == "Outbreak Prediction":
    st.header("Outbreak Prediction Analysis")
    
    # Parameters
    n_days = st.slider("Number of Days", 30, 365, 180)
    seasonality = st.slider("Seasonality Strength", 0.0, 1.0, 0.5)
    trend = st.slider("Trend Strength", -0.5, 0.5, 0.1)
    
    # Generate time series data
    def generate_outbreak_data(n_days, seasonality, trend):
        time = np.arange(n_days)
        
        # Base signal
        seasonal = seasonality * np.sin(2 * np.pi * time / 365)
        trend_component = trend * time / n_days
        
        # Add outbreaks
        n_outbreaks = np.random.randint(2, 5)
        outbreak_times = np.random.choice(n_days, n_outbreaks, replace=False)
        outbreak_sizes = np.random.uniform(1, 3, n_outbreaks)
        
        outbreaks = np.zeros(n_days)
        for t, s in zip(outbreak_times, outbreak_sizes):
            outbreak_duration = np.random.randint(7, 21)
            outbreak = s * np.exp(-0.5 * ((time - t) / (outbreak_duration/3))**2)
            outbreaks += outbreak
        
        signal = seasonal + trend_component + outbreaks
        cases = np.random.poisson(np.exp(signal))
        
        return pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=n_days),
            'Cases': cases,
            'Trend': trend_component,
            'Seasonal': seasonal,
            'Outbreaks': outbreaks
        })
    data = generate_outbreak_data(n_days, seasonality, trend)
    
    # Plot components
    fig = go.Figure()
    
    # Original cases
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Cases'],
        name='Observed Cases',
        mode='lines+markers'
    ))
    
    # Trend
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=np.exp(data['Trend']),
        name='Trend Component',
        line=dict(dash='dash')
    ))
    
    # Seasonal pattern
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=np.exp(data['Seasonal']),
        name='Seasonal Component',
        line=dict(dash='dot')
    ))
    
    fig.update_layout(
        title='Disease Cases and Components',
        xaxis_title='Date',
        yaxis_title='Number of Cases'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Outbreak detection
    st.subheader("Outbreak Detection")
    
    # Calculate moving average and standard deviation
    window_size = st.slider("Detection Window (days)", 7, 30, 14)
    data['MA'] = data['Cases'].rolling(window=window_size).mean()
    data['SD'] = data['Cases'].rolling(window=window_size).std()
    
    # Define threshold for outbreak detection
    threshold_multiplier = st.slider("Detection Threshold (SD)", 1.0, 4.0, 2.0)
    data['Upper_Bound'] = data['MA'] + threshold_multiplier * data['SD']
    data['Is_Outbreak'] = data['Cases'] > data['Upper_Bound']
    
    # Plot with detection bounds
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Cases'],
        name='Observed Cases',
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['MA'],
        name='Moving Average',
        line=dict(color='yellow')
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Upper_Bound'],
        name=f'Threshold ({threshold_multiplier}σ)',
        line=dict(color='red', dash='dash')
    ))
    
    # Highlight outbreak points
    outbreak_points = data[data['Is_Outbreak']]
    fig.add_trace(go.Scatter(
        x=outbreak_points['Date'],
        y=outbreak_points['Cases'],
        mode='markers',
        name='Detected Outbreaks',
        marker=dict(size=12, color='red')
    ))
    
    fig.update_layout(
        title='Outbreak Detection Results',
        xaxis_title='Date',
        yaxis_title='Number of Cases'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface
    st.subheader("Future Prediction")
    
    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    
    # Simple prediction model using trend and seasonality
    future_dates = pd.date_range(
        start=data['Date'].iloc[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )
    
    # Generate predictions
    time_extended = np.arange(n_days + forecast_days)
    seasonal_extended = seasonality * np.sin(2 * np.pi * time_extended / 365)
    trend_extended = trend * time_extended / len(time_extended)
    
    future_signal = seasonal_extended[-forecast_days:] + trend_extended[-forecast_days:]
    predictions = np.exp(future_signal)
    
    # Create prediction intervals
    prediction_std = np.std(np.log(data['Cases']))
    lower_bound = np.exp(np.log(predictions) - 1.96 * prediction_std)
    upper_bound = np.exp(np.log(predictions) + 1.96 * prediction_std)
    
    # Plot predictions
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Cases'],
        name='Historical Cases',
        mode='lines+markers'
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name='Predicted Cases',
        mode='lines',
        line=dict(dash='dash')
    ))
    
    # Prediction interval
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        name='95% Prediction Interval',
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        name='95% Prediction Interval',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=True
    ))
    
    fig.update_layout(
        title='Disease Cases Forecast',
        xaxis_title='Date',
        yaxis_title='Number of Cases'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add educational content for each method
st.header("Technical Details")

if method == "Prediction Models":
    st.write("""
    Disease prediction models in epidemiology:
    
    1. Model Types:
    - Logistic Regression: Linear relationships
    - Random Forest: Complex patterns, interactions
    - SVM: Non-linear boundaries, robust to outliers
    
    2. Model Evaluation:
    - ROC curves and AUC
    - Sensitivity and specificity
    - Calibration assessment
    - Cross-validation
    
    3. Feature Importance:
    - Coefficient magnitudes
    - Variable rankings
    - Contribution to predictions
    """)

elif method == "Risk Stratification":
    st.write("""
    Risk stratification methods:
    
    1. Purpose:
    - Patient classification
    - Resource allocation
    - Intervention targeting
    - Prevention planning
    
    2. Approaches:
    - Score-based systems
    - Machine learning classifiers
    - Clinical prediction rules
    - Risk thresholds
    
    3. Validation:
    - Internal validation
    - External validation
    - Temporal validation
    - Geographic validation
    """)

elif method == "Feature Selection":
    st.write("""
    Feature selection in epidemiology:
    
    1. Methods:
    - Filter methods (univariate)
    - Wrapper methods (recursive)
    - Embedded methods (LASSO)
    - Domain knowledge
    
    2. Considerations:
    - Collinearity
    - Missing data
    - Interaction effects
    - Causal relationships
    
    3. Validation:
    - Stability selection
    - Cross-validation
    - External validation
    - Clinical relevance
    """)

elif method == "Model Comparison":
    st.write("""
    Model comparison approaches:
    
    1. Performance Metrics:
    - Discrimination (AUC)
    - Calibration
    - Net reclassification
    - Decision curves
    
    2. Validation Strategies:
    - Cross-validation
    - Bootstrap
    - External validation
    - Temporal validation
    
    3. Trade-offs:
    - Complexity vs. interpretability
    - Accuracy vs. generalizability
    - Computational cost
    - Implementation feasibility
    """)

elif method == "Outbreak Prediction":
    st.write("""
    Outbreak prediction methods:
    
    1. Components:
    - Trend analysis
    - Seasonality patterns
    - Outbreak detection
    - Prediction intervals
    
    2. Detection Methods:
    - Moving averages
    - Statistical process control
    - Change point detection
    - Anomaly detection
    
    3. Forecasting:
    - Time series models
    - Machine learning approaches
    - Ensemble methods
    - Uncertainty quantification
    """)

# Add references
st.header("Further Reading")
st.write("""
1. Hastie T, et al. The Elements of Statistical Learning
2. Kuhn M, Johnson K. Applied Predictive Modeling
3. Brookmeyer R, Stroup DF. Monitoring the Health of Populations
4. Wiemken TL, et al. Machine Learning in Epidemiology and Health Outcomes Research
""")

st.header("Check your understanding")
if method == "Prediction Models":
    quiz_pred = st.radio(
        "Which of the following is a common metric used to evaluate classification models?",
        [
            "R-squared (R²)",
            "Mean Absolute Error (MAE)",
            "Area Under the ROC Curve (AUC)",
            "Structural Similarity Index (SSIM)"
        ]
    )
    if quiz_pred == "Area Under the ROC Curve (AUC)":
        st.success("Correct! AUC measures a model's ability to distinguish between classes.")
    else:
        st.error("Not quite. AUC is widely used to evaluate classification models.")
        
elif method == "Risk Stratification":
    quiz_risk = st.radio(
        "What is the main goal of risk stratification in healthcare?",
        [
            "To predict stock market trends",
            "To classify patients into different risk categories",
            "To calculate the average height of a population",
            "To optimize hospital building design"
        ]
    )
    if quiz_risk == "To classify patients into different risk categories":
        st.success("Correct! Risk stratification helps categorize patients based on their likelihood of developing a condition.")
    else:
        st.error("Not quite. Risk stratification is used to classify patient risk levels.")
        
elif method == "Feature Selection":
    quiz_feature = st.radio(
        "Which feature selection method uses penalties to shrink less important coefficients to zero?",
        [
            "Recursive Feature Elimination (RFE)",
            "L1 Regularization (LASSO)",
            "Principal Component Analysis (PCA)",
            "Random Forest Feature Importance"
        ]
    )
    if quiz_feature == "L1 Regularization (LASSO)":
        st.success("Correct! LASSO regression applies an L1 penalty, forcing some coefficients to be zero.")
    else:
        st.error("Not quite. LASSO regression is known for its ability to perform feature selection.")

elif method == "Model Comparison":
    quiz_compare = st.radio(
        "Which of the following models is best suited for capturing non-linear relationships in data?",
        [
            "Logistic Regression",
            "Linear Regression",
            "Support Vector Machine (SVM) with RBF Kernel",
            "Ordinary Least Squares (OLS) Regression"
        ]
    )
    if quiz_compare == "Support Vector Machine (SVM) with RBF Kernel":
        st.success("Correct! SVM with a non-linear kernel can capture complex decision boundaries.")
    else:
        st.error("Not quite. Non-linear kernels allow SVMs to capture non-linear patterns.")

elif method == "Outbreak Prediction":
    quiz_outbreak = st.radio(
        "Which time series method is commonly used for predicting outbreaks?",
        [
            "K-Means Clustering",
            "Linear Regression",
            "Autoregressive Integrated Moving Average (ARIMA)",
            "Decision Tree Classifier"
        ]
    )
    if quiz_outbreak == "Autoregressive Integrated Moving Average (ARIMA)":
        st.success("Correct! ARIMA models are widely used in time series forecasting, including disease outbreak predictions.")
    else:
        st.error("Not quite. ARIMA models are designed for analyzing and forecasting time series data.")
