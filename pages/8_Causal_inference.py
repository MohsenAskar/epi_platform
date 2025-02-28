# pages/8_causal_inference.py
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

st.title("Causal Inference and DAGs")

# DAG creation tool
st.header("Create and Analyze a DAG")

# Example DAG scenarios
dag_scenario = st.selectbox(
    "Select a Causal Scenario",
    ["Confounding", "Mediation", "Collider", "M-Bias"]
)

def create_dag(scenario):
    G = nx.DiGraph()
    
    if scenario == "Confounding":
        G.add_edges_from([
            ('Confounder', 'Exposure'),
            ('Confounder', 'Outcome'),
            ('Exposure', 'Outcome')
        ])
    elif scenario == "Mediation":
        G.add_edges_from([
            ('Exposure', 'Mediator'),
            ('Mediator', 'Outcome'),
            ('Exposure', 'Outcome')
        ])
    elif scenario == "Collider":
        G.add_edges_from([
            ('Exposure', 'Collider'),
            ('Outcome', 'Collider')
        ])
    elif scenario == "M-Bias":
        G.add_edges_from([
            ('U1', 'Exposure'),
            ('U1', 'M'),
            ('U2', 'Outcome'),
            ('U2', 'M'),
            ('Exposure', 'Outcome')
        ])
    
    return G

# Create and display DAG
G = create_dag(dag_scenario)
fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=10, font_weight='bold',
        arrows=True, edge_color='gray')
st.pyplot(fig)

# Educational content
st.header("Understanding Causal Inference")
st.write("""
Causal inference in epidemiology involves understanding and measuring causal relationships 
between exposures and outcomes. Directed Acyclic Graphs (DAGs) are tools that help us:

1. Identify confounding
2. Understand selection bias
3. Determine appropriate adjustment sets
4. Visualize causal assumptions
""")

st.subheader(f"Current Scenario: {dag_scenario}")
if dag_scenario == "Confounding":
    st.write("""
    In this DAG:
    - The confounder affects both exposure and outcome
    - Creates a backdoor path
    - Must adjust for confounder to get unbiased estimate
    """)
elif dag_scenario == "Mediation":
    st.write("""
    In this DAG:
    - Mediator lies on causal pathway
    - Don't adjust for mediator if interested in total effect
    - Can adjust for mediator to estimate direct effect
    """)
elif dag_scenario == "Collider":
    st.write("""
    In this DAG:
    - Collider is affected by both exposure and outcome
    - Adjusting for collider can create bias
    - No confounding present in unadjusted analysis
    """)
elif dag_scenario == "M-Bias":
    st.write("""
    In this DAG:
    - M creates apparent confounding
    - Adjusting for M can create bias
    - Example of how DAGs help avoid inappropriate adjustment
    """)

# Causal inference methods
st.header("Common Causal Inference Methods")
st.write("""
1. Regression Adjustment
   - Traditional approach
   - Assumes correct model specification
   - May not capture non-linear relationships

2. Propensity Score Methods
   - Balance covariates between groups
   - Useful for high-dimensional confounding
   - Various implementation options (matching, weighting, stratification)

3. Instrumental Variables
   - Useful when unmeasured confounding exists
   - Requires valid instrument
   - Can be difficult to find good instruments

4. Difference-in-Differences
   - Uses time trends to control for confounding
   - Requires parallel trends assumption
   - Useful for policy evaluation
""")

st.header("Assumptions for Causal Inference")
st.write("""
1. Exchangeability (No Unmeasured Confounding)
   - All important confounders are measured and controlled
   - Often untestable assumption

2. Positivity
   - All exposure levels possible in all confounder strata
   - Can check empirically

3. Consistency
   - Well-defined intervention
   - Same exposure leads to same outcome

4. No Interference
   - One unit's exposure doesn't affect another's outcome
   - May be violated in infectious disease studies
""")

#######################
# Quiz for Scenarios  #
#######################
st.subheader("Check Your Understanding: DAG Scenarios")
quiz_causal = st.radio(
    "Which scenario describes a variable that *influences both the exposure and the outcome*?",
    ("Mediation", "Confounding", "Collider", "M-Bias")
)

if quiz_causal == "Confounding":
    st.success("Correct! A confounder affects both exposure and outcome, creating a backdoor path.")
else:
    st.error("Not quite. Confounders are variables associated with both exposure and outcome.")