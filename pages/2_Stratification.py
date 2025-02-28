import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.data_generators import generate_stratified_data

# Make the page layout wider
st.set_page_config(layout="wide")

# Title Section
st.title("📊 Stratification Analysis in Drug Response")

# Educational Content
st.subheader("🔬 What is Stratification?")
st.markdown("""
**Stratification** in clinical research helps us **compare drug effects** across different patient groups.  
It is useful for:
1. **Understanding subgroup differences** in drug response
2. **Detecting effect modification (interaction)**—whether a drug works differently in one group vs. another
3. **Reducing bias** by controlling for confounding factors

🧪 **Example:** A new blood pressure medication might work **better in younger patients** and have a **weaker effect in older patients**.  
To analyze this, we stratify the data by **age group** and observe the differences.
""")

# Create layout: controls in the first column, graphs in the second
col1, col2 = st.columns([2, 3])

with col1:
    st.header("🔧 Stratification Controls")

    # Slider for effect size
    effect_size = st.slider(
        "Effect Size (strength of relationship between variables)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    # Interpretation based on effect size
    if effect_size == 0.0:
        effect_emoji = "• No effect: The relationship is flat across strata."
    elif effect_size < 1.0:
        effect_emoji = "➡️ Mild effect: Some stratification differences, but subtle."
    elif effect_size < 1.5:
        effect_emoji = "🔺 Moderate effect: Noticeable difference in strata relationships."
    else:
        effect_emoji = "🔻 Strong effect: The effect size varies significantly across strata."

    st.markdown(f"**Current Effect Size Interpretation:** {effect_emoji}")

    # Explanation of Variables
    st.subheader("📌 Understanding the Variables")
    st.markdown("""
    - **Variable 1 (Drug Dosage in mg):** The dose of the drug given to patients.
    - **Variable 2 (Blood Pressure Reduction in mmHg):** How much the drug reduced the patient's blood pressure.
    - **Stratum (Age Group):** Patients are categorized into different **age groups** to see if the drug works differently in younger vs. older individuals.
    """)

with col2:
    # Generate stratified data
    df = generate_stratified_data(effect_size)

    # Rename Variables for Better Understanding
    df = df.rename(columns={
        "Variable 1": "Drug Dosage (mg)",
        "Variable 2": "Blood Pressure Reduction (mmHg)",
        "Stratum": "Age Group"
    })

    # Figure 1: Overall population
    fig1 = go.Figure()

    # Calculate overall effect (slope)
    z_overall = np.polyfit(df["Drug Dosage (mg)"], df["Blood Pressure Reduction (mmHg)"], 1)
    overall_slope = z_overall[0]
    p_overall = np.poly1d(z_overall)

    # Add overall population scatter plot
    fig1.add_trace(
        go.Scatter(
            x=df["Drug Dosage (mg)"],
            y=df["Blood Pressure Reduction (mmHg)"],
            mode='markers',
            name='All Patients',
            marker=dict(color='blue', opacity=0.6)
        )
    )

    # Add trend line with effect size
    fig1.add_trace(
        go.Scatter(
            x=df["Drug Dosage (mg)"].sort_values(),
            y=p_overall(df["Drug Dosage (mg)"].sort_values()),
            mode='lines',
            name=f'Overall Trend (Effect: {overall_slope:.2f} mmHg/mg)',
            line=dict(color='red', dash='dash')
        )
    )

    # Update layout for overall population
    fig1.update_layout(
        height=400,
        title=f"Overall Population (All Age Groups Combined)<br>Average Effect: {overall_slope:.2f} mmHg/mg",
        xaxis_title="Drug Dosage (mg)",
        yaxis_title="Blood Pressure Reduction (mmHg)",
        showlegend=True
    )

    # Display the first figure
    st.plotly_chart(fig1, use_container_width=True)

# Move Stratified Analysis to Full Width Below
st.markdown("---")  # Horizontal separator

# Stratified Analysis Container
with st.container():    
    # Figure 2: Stratified view using plotly express
    fig2 = px.scatter(
        df,
        x="Drug Dosage (mg)",
        y="Blood Pressure Reduction (mmHg)",
        color="Age Group",
        facet_col="Age Group",
        title="Stratified Analysis: Drug Effect Across Age Groups",
        height=400
    )

    # Add trend lines to each facet
    age_groups = sorted(df["Age Group"].unique())
    age_group_to_position = {group: idx + 1 for idx, group in enumerate(age_groups)}
    
    # Store slopes for comparison
    slopes = {}

    for age_group in age_groups:
        group_data = df[df["Age Group"] == age_group]
        z = np.polyfit(group_data["Drug Dosage (mg)"], group_data["Blood Pressure Reduction (mmHg)"], 1)
        slope = z[0]
        slopes[age_group] = slope
        p = np.poly1d(z)
        
        fig2.add_trace(
            go.Scatter(
                x=group_data["Drug Dosage (mg)"].sort_values(),
                y=p(group_data["Drug Dosage (mg)"].sort_values()),
                mode='lines',
                name=f'Trend {age_group} (Effect: {slope:.2f} mmHg/mg)',
                line=dict(dash='dash'),
                showlegend=True
            ),
            row=1,
            col=age_group_to_position[age_group]
        )

    # Update annotations to include effect sizes
    fig2.for_each_annotation(
        lambda a: a.update(
            text=f"{a.text}<br>Effect: {slopes[a.text.split('=')[-1].strip()]:.2f} mmHg/mg"
        )
    )

    # Update layout for stratified view
    fig2.update_layout(
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )

    # Display the second figure with full width
    st.plotly_chart(fig2, use_container_width=True)

    # Add effect size comparison text
    st.subheader("📊 Effect Size Comparison")
    st.markdown(f"""
    **Overall Population Effect:** {overall_slope:.2f} mmHg/mg
    
    **Stratified Effects:**
    """)
    
    for age_group, slope in slopes.items():
        difference = slope - overall_slope
        if abs(difference) < 0.01:
            comparison = "similar to"
        elif difference > 0:
            comparison = "stronger than"
        else:
            comparison = "weaker than"
            
        st.markdown(f"- **{age_group}:** {slope:.2f} mmHg/mg ({comparison} overall effect)")

    # Add interpretation based on effect size variation
    max_diff = max(slopes.values()) - min(slopes.values())
    if max_diff < 0.1:
        st.info("🔍 **Interpretation:** The drug effect is relatively consistent across age groups.")
    elif max_diff < 0.5:
        st.warning("🔍 **Interpretation:** There are moderate differences in drug effect across age groups.")
    else:
        st.error("🔍 **Interpretation:** There are substantial differences in drug effect across age groups - strong evidence of effect modification.")

    # Continue with existing "What Do You See in the Graphs?" section...
    # Explain what the student sees in the interactive graphs
    st.subheader("🔍 What Do You See in the Graphs?")
    if effect_size == 0.0:
        st.info("""
        **Observation:**
        - In the **overall population**, you see no relationship between drug dosage and blood pressure reduction.
        - When **stratified by age**, you can confirm that none of the age groups show any effect.
        - This suggests the drug might not be effective at all.
        """)
    elif effect_size < 1.0:
        st.info("""
        **Observation:**
        - The **overall population** shows a weak positive relationship.
        - When **stratified**, you can see that all age groups respond similarly with a mild effect.
        - This suggests the drug has a consistent but weak effect across age groups.
        """)
    elif effect_size < 1.5:
        st.info("""
        **Observation:**
        - The **overall population** shows a moderate positive relationship.
        - However, when **stratified**, you can see that younger patients respond better than older patients.
        - This effect modification would be missed if we only looked at the overall population!
        """)
    else:
        st.warning("""
        **Observation:**
        - The **overall population** might show a misleading average effect.
        - The **stratified view** reveals dramatic differences between age groups.
        - This is a clear example of effect modification that would be hidden in the overall analysis.
        """)

# Interactive Quiz
st.subheader("🧐 Test Your Understanding")
quiz_answer = st.radio(
    "Looking at both the overall population and stratified views, what's the main advantage of stratification?",
    (
        "It always shows stronger relationships than the overall population.",
        "It reveals patterns that might be hidden in the overall population.",
        "It makes the data easier to collect.",
        "It always shows weaker relationships than the overall population."
    )
)

if quiz_answer == "It reveals patterns that might be hidden in the overall population.":
    st.success("✅ Correct! Stratification can uncover important differences between groups that might be masked when looking at the overall population.")
else:
    st.error("❌ Not quite! Think about how the overall trend might hide different patterns in different subgroups.")

st.write("🔄 Adjust the **effect size slider** to see how stratification reveals patterns in the data!")

# Further Reading
st.markdown("""
---
📖 **Further Reading**:
- 🏥 [Stratification in Epidemiology (CDC)](https://www.cdc.gov/csels/dsepd/ss1978/lesson4/section3.html)
- 📈 [Confounding and Interaction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4354892/)
""")