# pages/3_confounding.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

############################
# Data Generation Function #
############################
def generate_confounding_data(n_samples=1000, confounder_strength=0.5, true_effect=0.3):
    """Generate data with confounding."""
    # Generate confounder (e.g., age)
    confounder = np.random.normal(0, 1, n_samples)
    
    # Generate exposure influenced by confounder
    exposure = confounder * confounder_strength + np.random.normal(0, 1, n_samples)
    
    # Generate outcome influenced by both exposure and confounder
    outcome = (
        exposure * true_effect + 
        confounder * confounder_strength + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    return pd.DataFrame({
        'Confounder': confounder,
        'Exposure': exposure,
        'Outcome': outcome
    })

################
# Page Layout  #
################
st.set_page_config(layout="wide")

# Title Section
st.title("Understanding Confounding in Epidemiology")

######################################
# Create columns for controls & plot #
######################################
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Confounding Controls")

    # Sliders
    confounder_strength = st.slider(
        "Confounder Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    true_effect = st.slider(
        "True Exposure Effect",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )

    st.markdown("""
    **Tip**: The confounder strength dictates how strongly the third variable
    (e.g., age) influences both the exposure and the outcome.

    The **True Exposure Effect** is the actual causal influence that exposure has
    on the outcome (i.e., how much the outcome changes given the exposure).
    """)

with col2:
    # Generate data
    data = generate_confounding_data(
        n_samples=1000,
        confounder_strength=confounder_strength,
        true_effect=true_effect
    )

    # Create visualizations using subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Exposure vs Outcome (Crude)',
            'Confounder vs Exposure',
            'Confounder vs Outcome',
            'Stratified Analysis'
        )
    )

    # 1) Crude association: Exposure vs Outcome
    fig.add_trace(
        go.Scatter(
            x=data['Exposure'],
            y=data['Outcome'],
            mode='markers',
            name='Crude',
            marker=dict(size=5)
        ),
        row=1, col=1
    )

    # 2) Confounder vs Exposure
    fig.add_trace(
        go.Scatter(
            x=data['Confounder'],
            y=data['Exposure'],
            mode='markers',
            name='Conf-Exp',
            marker=dict(size=5)
        ),
        row=1, col=2
    )

    # 3) Confounder vs Outcome
    fig.add_trace(
        go.Scatter(
            x=data['Confounder'],
            y=data['Outcome'],
            mode='markers',
            name='Conf-Out',
            marker=dict(size=5)
        ),
        row=2, col=1
    )

    # 4) Stratified analysis by confounder levels
    strata = pd.qcut(data['Confounder'], q=3, labels=['Low', 'Medium', 'High'])
    colors = ['blue', 'green', 'red']

    for stratum, color in zip(strata.unique(), colors):
        mask = strata == stratum
        fig.add_trace(
            go.Scatter(
                x=data.loc[mask, 'Exposure'],
                y=data.loc[mask, 'Outcome'],
                mode='markers',
                name=f'Stratum {stratum}',
                marker=dict(size=5, color=color)
            ),
            row=2, col=2
        )

    fig.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

########################
# Educational Content  #
########################
st.subheader("What is Confounding?")
st.markdown("""
**Confounding** occurs when a third variable (the confounder) influences both the exposure and outcome,
leading to a spurious (or misleading) association between them. Controlling for confounding is crucial
in epidemiological research.

Key characteristics of a confounder:
1. Associated with the exposure
2. An independent risk factor for the outcome
3. Not on the direct causal pathway between exposure and outcome
""")

st.subheader("Interactive Elements Explanation")
st.write("""
In this visualization:
- Top Left: Shows the crude (unadjusted) relationship between exposure and outcome
- Top Right: Shows how the confounder relates to the exposure
- Bottom Left: Shows how the confounder relates to the outcome
- Bottom Right: Shows the relationship stratified by confounder levels

Try adjusting the sliders to see how:
1. Confounder strength affects the relationships
2. The true exposure effect differs from the crude association
""")

###########################
# Interactive Quiz/Checks #
###########################
st.subheader("Test Your Understanding")
quiz_question = st.radio(
    "Which of the following best describes why confounding can distort the observed relationship?",
    [
        "Because it introduces a new causal pathway for the exposure.",
        "Because the confounder is related to both the exposure and the outcome, making it seem like there is a relationship even if there isn't.",
        "Because it randomly changes the outcome without affecting the exposure."
    ]
)

if quiz_question == "Because the confounder is related to both the exposure and the outcome, making it seem like there is a relationship even if there isn't.":
    st.success("Correct! A confounder is a variable associated with both exposure and outcome, distorting the observed effect.")
else:
    st.error("Not quite. Remember, the key point is that the confounder is associated with both the exposure and the outcome.")

#######################
# Additional Remarks  #
#######################
st.write("""
Try adjusting the **Confounder Strength** and **True Exposure Effect** sliders to see how they alter
these relationships. Notice how a strong confounder can make it appear that the exposure has a bigger
effect than it really does (or vice versa).
""")

st.markdown("""
---
**Further Reading**:
- [Confounding in Epidemiology (CDC)](https://www.cdc.gov/csels/dsepd/ss1978/lesson4/section4.html)
- [Controlling for Confounding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4447039/)
""")


