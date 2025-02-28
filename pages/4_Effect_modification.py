# pages/4_effect_modification.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

#############################
# Data Generation Function  #
#############################
def generate_effect_modification_data(n_samples=1000, effect_modifier_strength=0.5):
    """Generate data with effect modification."""
    # Generate effect modifier (e.g., age groups)
    effect_modifier = np.random.choice(['Young', 'Middle', 'Old'], n_samples)

    # Generate exposure
    exposure = np.random.normal(0, 1, n_samples)

    # Generate outcome with different effects by group
    outcome = np.zeros(n_samples)

    for group in ['Young', 'Middle', 'Old']:
        mask = (effect_modifier == group)
        if group == 'Young':
            effect = 0.2
        elif group == 'Middle':
            effect = 0.5
        else:
            effect = 0.8

        outcome[mask] = (
            exposure[mask] * effect * effect_modifier_strength +
            np.random.normal(0, 0.5, sum(mask))
        )

    return pd.DataFrame({
        'Effect_Modifier': effect_modifier,
        'Exposure': exposure,
        'Outcome': outcome
    })

################
# Layout Setup #
################
st.set_page_config(layout="wide")

# Title
st.title("Effect Modification (Effect Measure Modification)")

#################################
# Controls & Data Visualization #
#################################
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Effect Modification Controls")

    effect_strength = st.slider(
        "Effect Modification Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    st.markdown("""
    **Effect Modification Strength**: This controls how strongly the effect of the exposure
    on the outcome differs across the three groups (Young, Middle, Old).
    """)

with col2:
    # Generate data
    data = generate_effect_modification_data(
        n_samples=1000,
        effect_modifier_strength=effect_strength
    )

    # Create scatter plot faceted by the effect modifier
    fig = px.scatter(
        data,
        x='Exposure',
        y='Outcome',
        color='Effect_Modifier',
        facet_col='Effect_Modifier',
        title='Effect Modification by Group'
    )

    st.plotly_chart(fig, use_container_width=True)

#########################
# Educational Content   #
#########################
st.subheader("Understanding Effect Modification")
st.markdown("""
**Effect modification** occurs when the effect of an exposure on an outcome *varies* across levels
of a third variable. Unlike confounding, *effect modification is not a bias* but rather a real
phenomenon that needs to be described and understood.

**Key points**:
1. The relationship between exposure and outcome differs across strata.
2. There's no need to "control for" effect modification.
3. It's important for targeting interventions and understanding heterogeneity.
""")

###########################
# Interactive Quiz/Checks #
###########################
st.subheader("Test Your Understanding")
quiz_answer = st.radio(
    "Which statement best describes effect modification?",
    [
        "An artifact that biases the exposure-outcome relationship.",
        "A phenomenon where the effect of exposure is stronger/weaker in different subgroups.",
        "An error that should be statistically corrected for."
    ]
)

if quiz_answer == "A phenomenon where the effect of exposure is stronger/weaker in different subgroups.":
    st.success("Correct! Effect modification means the exposure-outcome effect differs across strata.")
else:
    st.error("Remember, effect modification is a real difference in effect, not a bias or an artifact.")

st.markdown("""
---
**Further Modules/Reading**:
- Selection Bias Module
- Measures of Association Module
- Study Designs Module
- Causal Inference Module
""")
