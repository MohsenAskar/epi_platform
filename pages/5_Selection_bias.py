# pages/5_selection_bias.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

###############################
# Data Generation for Bias    #
###############################
def generate_selection_bias_data(n_samples=1000, selection_strength=0.5, true_effect=0.3):
    """Generate data demonstrating selection bias."""
    # Generate true population
    exposure = np.random.binomial(1, 0.5, n_samples)  # 0/1 exposure
    # Outcome depends on exposure with some baseline probability plus true_effect
    outcome = np.random.binomial(1, 0.3 + true_effect * exposure, n_samples)

    # Generate selection probability:
    # Individuals with outcome=1 have a higher probability of being selected
    # Weighted by selection_strength
    selection_prob = np.clip(outcome * selection_strength + 0.5, 0, 1)
    selected = np.random.binomial(1, selection_prob, n_samples)

    return pd.DataFrame({
        'Exposure': exposure,
        'Outcome': outcome,
        'Selected': selected,
        'Selection_Probability': selection_prob
    })

#################
# Layout Config #
#################
st.set_page_config(layout="wide")

###################
# Title and Intro #
###################
st.title("Understanding Selection Bias")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("Selection Bias Controls")

    # Sidebar controls
    selection_strength = st.slider(
        "Selection Bias Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    st.markdown("""
    **Selection Bias Strength**: Increasing this value means that individuals with outcome=1
    are more likely to be selected into the study. As a result, the sample becomes less
    representative of the true population, distorting the observed association.
    """)

with col2:
    # Generate data
    data = generate_selection_bias_data(n_samples=1000, selection_strength=selection_strength)

    # Calculate total population vs selected sample size
    total_pop = len(data)
    selected_pop = len(data[data['Selected'] == 1])

    # Probability of outcome in unexposed/exposed in the true population
    true_counts = pd.crosstab(data['Exposure'], data['Outcome'])
    true_unexposed_outcome = true_counts.iloc[0, 1] / total_pop
    true_exposed_outcome = true_counts.iloc[1, 1] / total_pop

    # Probability of outcome in unexposed/exposed in the selected sample
    selected_data = data[data['Selected'] == 1]
    if not selected_data.empty:
        sel_counts = pd.crosstab(selected_data['Exposure'], selected_data['Outcome'])
        sel_unexposed_outcome = (sel_counts.iloc[0, 1] / selected_pop) if 0 in sel_counts.index else 0
        sel_exposed_outcome = (sel_counts.iloc[1, 1] / selected_pop) if 1 in sel_counts.index else 0
    else:
        # Edge case if no one is selected
        sel_unexposed_outcome = 0
        sel_exposed_outcome = 0

    # Create bar chart to visualize difference
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Unexposed', 'Exposed'],
        y=[true_unexposed_outcome, true_exposed_outcome],
        name='True Population',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=['Unexposed', 'Exposed'],
        y=[sel_unexposed_outcome, sel_exposed_outcome],
        name='Selected Sample',
        marker_color='red'
    ))

    fig.update_layout(
        title='Impact of Selection Bias on Exposure-Outcome Association',
        xaxis_title='Exposure Status',
        yaxis_title='Probability of Outcome',
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)

##########################
# Educational Content   #
##########################
st.header("What is Selection Bias?")
st.markdown("""
**Selection bias** occurs when the individuals who are included in the study differ systematically
from those who are not included, *in a way related to both exposure and outcome*. This leads
to a distorted or biased estimate of the true exposure-outcome relationship.

**Common Types**:
1. Self-selection bias
2. Loss to follow-up
3. Healthy worker effect
4. Berkson's bias
""")

st.subheader("Interactive Visualization Explanation")
st.markdown("""
- **Blue bars**: Outcome probability among the true population (unexposed vs. exposed)
- **Red bars**: Outcome probability among *only* those selected into the study
- Adjust the *Selection Bias Strength* to see how preferential inclusion of outcome-positive
  individuals warps the observed association.
""")

# Example Quiz Section
st.subheader("Test Your Understanding")
quiz_question = st.radio(
    "Which of the following statements correctly describes selection bias?",
    [
        "It is when the exposure incorrectly influences measurement of the outcome.",
        "It occurs when those in the study differ systematically from those not in the study, in a way that is related to both exposure and outcome.",
        "It is a random error that happens regardless of design or measurement quality."
    ]
)

if quiz_question == "It occurs when those in the study differ systematically from those not in the study, in a way that is related to both exposure and outcome.":
    st.success("Correct! That's a succinct definition of selection bias.")
else:
    st.error("Not quite. Remember, selection bias involves non-random participation/retention that distorts the exposure-outcome relationship.")
