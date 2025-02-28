import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def generate_synthetic_data(n_patients=100):
    """Generate synthetic patient adherence data"""
    np.random.seed(42)
    
    # Generate patient data
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(65, 15, n_patients).astype(int),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'n_medications': np.random.randint(1, 6, n_patients),
        'education_years': np.random.normal(14, 3, n_patients).astype(int),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_patients),
        'living_alone': np.random.choice([0, 1], n_patients),
        'chronic_conditions': np.random.randint(0, 5, n_patients)
    }
    
    # Calculate adherence score (synthetic relationship)
    adherence_base = (
        -0.3 * data['n_medications'] +
        0.2 * (data['education_years'] - 10) +
        -5 * data['living_alone'] +
        -0.1 * (data['age'] - 60) +
        -2 * data['chronic_conditions']
    )
    
    # Add some random noise
    adherence_base += np.random.normal(0, 5, n_patients)
    
    # Convert to 0-100 scale
    data['adherence_score'] = (adherence_base - adherence_base.min()) / (adherence_base.max() - adherence_base.min()) * 100
    data['adherence_score'] = np.clip(data['adherence_score'], 0, 100)
    
    # Generate daily adherence data for last 30 days
    daily_data = []
    for pid in range(1, n_patients + 1):
        base_prob = data['adherence_score'][pid-1] / 100
        for day in range(30):
            took_meds = np.random.random() < base_prob
            daily_data.append({
                'patient_id': pid,
                'day': day + 1,
                'took_medication': took_meds
            })
    
    return pd.DataFrame(data), pd.DataFrame(daily_data)

def calculate_adherence_metrics(patient_data, daily_data):
    """Calculate various adherence metrics"""
    # Overall adherence rate
    overall_adherence = daily_data['took_medication'].mean() * 100
    
    # Adherence by day of week
    daily_data['day_of_week'] = daily_data['day'] % 7
    dow_adherence = daily_data.groupby('day_of_week')['took_medication'].mean() * 100
    
    # Patient segments
    patient_data['adherence_category'] = pd.qcut(patient_data['adherence_score'], 
                                               q=3, 
                                               labels=['Low', 'Medium', 'High'])
    
    return overall_adherence, dow_adherence, patient_data

def train_adherence_model(patient_data):
    """Train a simple ML model to predict adherence risk"""
    # Prepare features
    features = ['age', 'n_medications', 'education_years', 'living_alone', 'chronic_conditions']
    X = patient_data[features]
    
    # Create binary target (low adherence = True)
    y = patient_data['adherence_score'] < patient_data['adherence_score'].median()
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, importance

def main():
    st.title("Medication Adherence Analysis")
    
    # Generate or load data
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data, st.session_state.daily_data = generate_synthetic_data()
    
    # Navigation
    page = st.radio("Select Analysis", [
        "Adherence Overview",
        "Patient Analysis",
        "Risk Prediction",
        "Intervention Planning"
    ], horizontal=True)
    
    if page == "Adherence Overview":
        st.markdown("""
        ### Medication Adherence Overview
        
        Analyze overall adherence patterns and trends.
        """)
        
        # Calculate metrics
        overall_adherence, dow_adherence, patient_data = calculate_adherence_metrics(
            st.session_state.patient_data, st.session_state.daily_data)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Adherence", f"{overall_adherence:.1f}%")
        with col2:
            st.metric("Patients at Risk", 
                     f"{(patient_data['adherence_score'] < 60).sum()}")
        with col3:
            st.metric("Perfect Adherence", 
                     f"{(patient_data['adherence_score'] > 90).sum()}")
        
        # Adherence heatmap
        st.subheader("Daily Adherence Patterns")
        
        # Reshape data for heatmap
        heatmap_data = st.session_state.daily_data.pivot_table(
            index='patient_id',
            columns='day',
            values='took_medication',
            aggfunc='first'
        ).iloc[:20]  # Show first 20 patients
        
        fig = px.imshow(heatmap_data,
                       labels=dict(x="Day", y="Patient ID", color="Took Medication"),
                       color_continuous_scale="RdYlGn")
        
        st.plotly_chart(fig)
        
        # Day of week patterns
        st.subheader("Adherence by Day of Week")
        fig = px.bar(dow_adherence, 
                    labels={'value': 'Adherence Rate (%)', 'day_of_week': 'Day of Week'})
        st.plotly_chart(fig)
        
    elif page == "Patient Analysis":
        st.markdown("""
        ### Individual Patient Analysis
        
        Examine adherence patterns for specific patients.
        """)
        
        # Patient selector
        patient_id = st.selectbox(
            "Select Patient",
            st.session_state.patient_data['patient_id'].tolist()
        )
        
        # Get patient data
        patient = st.session_state.patient_data[
            st.session_state.patient_data['patient_id'] == patient_id].iloc[0]
        patient_daily = st.session_state.daily_data[
            st.session_state.daily_data['patient_id'] == patient_id]
        
        # Display patient info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age", patient['age'])
            st.metric("Medications", patient['n_medications'])
        with col2:
            st.metric("Adherence Score", f"{patient['adherence_score']:.1f}")
            st.metric("Chronic Conditions", patient['chronic_conditions'])
        with col3:
            st.metric("Living Alone", "Yes" if patient['living_alone'] else "No")
            st.metric("Education (years)", patient['education_years'])
        
        # Show adherence timeline
        fig = px.line(patient_daily, x='day', y='took_medication',
                     title="30-Day Adherence Pattern")
        st.plotly_chart(fig)
        
    elif page == "Risk Prediction":
        st.markdown("""
        ### Adherence Risk Prediction
        
        Predict adherence risk based on patient characteristics.
        """)
        
        # Train model
        model, scaler, importance = train_adherence_model(st.session_state.patient_data)
        
        # Feature importance plot
        st.subheader("Risk Factors Importance")
        fig = px.bar(importance, x='feature', y='importance',
                    title="Feature Importance in Predicting Non-adherence")
        st.plotly_chart(fig)
        
        # Risk calculator
        st.subheader("Risk Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 65)
            n_medications = st.number_input("Number of Medications", 1, 10, 3)
            education = st.number_input("Years of Education", 0, 20, 12)
        with col2:
            living_alone = st.checkbox("Living Alone")
            conditions = st.number_input("Number of Chronic Conditions", 0, 10, 2)
        
        # Make prediction
        input_data = np.array([[age, n_medications, education, living_alone, conditions]])
        input_scaled = scaler.transform(input_data)
        risk_prob = model.predict_proba(input_scaled)[0][1]
        
        st.metric("Risk of Non-adherence", f"{risk_prob*100:.1f}%")
        
        risk_level = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.3 else "Low"
        risk_color = "red" if risk_prob > 0.7 else "yellow" if risk_prob > 0.3 else "green"
        
        st.markdown(f"Risk Level: **:{risk_color}[{risk_level}]**")
        
    elif page == "Intervention Planning":
        st.markdown("""
        ### Intervention Planning
        
        Plan and track interventions for at-risk patients.
        """)
        
        # Calculate risk scores
        risk_threshold = st.slider("Risk Threshold (Adherence Score)", 0, 100, 60)
        at_risk = st.session_state.patient_data[
            st.session_state.patient_data['adherence_score'] < risk_threshold]
        
        st.subheader(f"Patients Requiring Intervention (n={len(at_risk)})")
        
        # Display risk factors and recommended interventions
        for _, patient in at_risk.iterrows():
            with st.expander(f"Patient {patient['patient_id']} "
                           f"(Score: {patient['adherence_score']:.1f})"):
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Risk Factors:**")
                    if patient['n_medications'] >= 4:
                        st.write("- High medication burden")
                    if patient['living_alone']:
                        st.write("- Lives alone")
                    if patient['chronic_conditions'] >= 3:
                        st.write("- Multiple chronic conditions")
                    if patient['age'] > 75:
                        st.write("- Advanced age")
                
                with col2:
                    st.markdown("**Recommended Interventions:**")
                    if patient['n_medications'] >= 4:
                        st.write("- Medication review and simplification")
                    if patient['living_alone']:
                        st.write("- Caregiver support program")
                    if patient['adherence_score'] < 40:
                        st.write("- Daily reminder system")
                        st.write("- Weekly pharmacist check-in")

if __name__ == "__main__":
    main()