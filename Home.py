
# Home.py
import streamlit as st
import base64
import os
st.set_page_config(
    page_title="Interactive Epidemiology Platform",
    page_icon="🔬",
    layout="wide"
)


# Convert the image to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load your image from a local path
image_path = (r"C:\Users\mas082\OneDrive - UiT Office 365\Desktop\Introduce_Your_Self\cartoon.JPG")
# Get the base64 string of the image
image_base64 = image_to_base64(image_path)

# Display your image and name in the top right corner
st.markdown(
    f"""
    <style>
    .header {{
        position: absolute;  /* Fix the position */
        top: -60px;  /* Adjust as needed */
        right: -40px;  /* Align to the right */
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 10px;
        flex-direction: column; /* Stack items vertically */
        text-align: center; /* Ensures text is centrally aligned */
    }}
    .header img {{
        border-radius: 50%;
        width: 50px;
        height: 50px;
        margin-bottom: 5px; /* Space between image and text */
    }}
    .header-text {{
        font-size: 12px;
        font-weight: normal; /* Regular weight for text */
        text-align: center;
    }}
    </style>
    <div class="header">
        <img src="data:image/jpeg;base64,{image_base64}" alt="Mohsen Askar">
        <div class="header-text">Developed by: Mohsen Askar</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Interactive Epidemiology Concepts")
      
st.write("""
Welcome to the Interactive Epidemiology Platform! 

This educational tool helps you understand key 
epidemiological concepts through interactive visualizations and examples.

### Available epidemiological modules:
- **Correlation**: Understand the relationship between two variables
- **Stratification**: Observe how data is split into groups
- **Confounding**: See how a third variable affects the relationship between two others
- **Effect Modification**: Observe how the effect of one variable changes based on another
- **Selection Bias**: Understand how bias can affect study results
- **Measures of Association**: Learn about key measures like Risk Ratio, Odds Ratio, and Hazard Ratio
- **Epidemiological Study Designs**: Explore different study designs like Cohort, Case-Control, and Cross-Sectional
- **Causal Inference and DAGs**: Understand the concept of causality and Directed Acyclic Graphs
- **Statistical Methods in Epidemiology**: Learn about statistical tests like logistic regression, Cox proportional hazards, and more
- **Screening and Diagnostic Tests**: Understand the concepts of sensitivity, specificity, positive predictive value, and negative predictive value
- **Disease Frequency and Measures**: Learn about measures of disease frequency
- **Meta-Analysis and Systematic Reviews**: Understand the process of combining results from multiple studies
- **Machine Learning in Epidemiology**: Explore the basic concepts of machine learning in epidemiology
- **Network Analysis in Epidemiology**: Learn about network analysis and its applications in epidemiology
- **Target Trial Emulation in Epidemiology**: Understand the concept of target trial emulation
- **Quantitative Bias Analysis**: Learn about the concept of quantitative bias analysis
- **Clinical Epidemiology**: Learn about the application of epidemiology in clinical settings
- **Environmental and Occupational Epidemiology**: Learn about the application of epidemiology in environmental and occupational settings
- **Time-to-Event (Survival) Analysis**: Understand the concept of time-to-event analysis
- **Longitudinal Data Analysis**: Learn about the analysis of longitudinal data
- **Time Series Analysis**: Learn about the analysis of time series data
- **Bayesian Methods in Epidemiology**:  Learn about the application of Bayesian methods in epidemiology
- **Data Management & Wrangling for Epidemiology**:  Learn about data management and wrangling techniques for epidemiological data

Select a concept from the sidebar to begin exploring.
""")

# Display featured visualizations or key statistics on the home page
st.header("Quick Start Guide")
st.write("""
1. Use the sidebar to navigate between different concepts
2. Each page contains interactive elements - adjust sliders and inputs to see how they affect the results
3. Read the explanations provided with each visualization
""")



