import streamlit as st
import plotly.express as px
from utils.data_generators import generate_correlated_data

st.set_page_config(layout="wide")
# Title Section
st.title("Understanding Correlation")

# Layout example: create columns
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Correlation Controls")

    # Sidebar-like controls but in a column
    correlation = st.slider(
        "Select correlation strength",
        min_value=-1.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    n_samples = st.number_input(
        "Number of samples",
        min_value=10,
        max_value=2000,
        value=100,
        step=10
    )

with col2:
    # Generate data
    data = generate_correlated_data(correlation, n_samples)

    # Determine an emoji based on correlation sign
    if correlation > 0:
        correlation_emoji = "🔺"  # positive
    elif correlation < 0:
        correlation_emoji = "🔻"  # negative
    else:
        correlation_emoji = "•"   # zero

    # Dynamic title
    plot_title = f"Scatter Plot ({correlation_emoji} Correlation = {correlation:.2f})"

    # Create the scatter plot
    fig = px.scatter(
        data,
        x='Variable 1',
        y='Variable 2',
        color='Variable 1',  # coloring by 'Variable 1'
        title=plot_title
    )

    st.plotly_chart(fig, use_container_width=True)

# Educational content
st.subheader("What is Correlation?")
st.markdown("""
**Correlation** measures the *strength* and *direction* of the linear relationship between two variables:

- **+1**: Perfect positive relationship (as one variable increases, the other always increases linearly).
- **-1**: Perfect negative relationship (as one variable increases, the other always decreases linearly).
- **0**: No linear relationship (though there could be a non-linear relationship).

In a real-world example, you might see a positive correlation between the hours spent studying and exam scores.
However, it’s crucial to remember that **correlation ≠ causation**.
""")

# Interactive Quiz
st.subheader("Test Your Understanding")
quiz_answer = st.radio(
    "If correlation = 0, which statement is correct?",
    (
        "There is absolutely no relationship of any kind.",
        "There is no linear relationship, but there could still be a non-linear relationship.",
        "Perfect negative linear relationship."
    )
)

if quiz_answer == "There is no linear relationship, but there could still be a non-linear relationship.":
    st.success("Correct! Correlation only measures linear dependence.")
else:
    st.error("Not quite. Remember that a correlation of 0 indicates no linear relationship.")

# Interactive examples
st.header("Try it yourself!")
st.write("Adjust the correlation slider in the sidebar to see how it affects the relationship between variables.")


# Educational content
st.header("What is Correlation?")
st.write("""
Correlation measures the strength and direction of the relationship between two variables.
- A correlation of 1 indicates a perfect positive relationship
- A correlation of -1 indicates a perfect negative relationship
- A correlation of 0 indicates no linear relationship
""")

# Additional references or concluding statement
st.markdown("""---\n**Further Reading**:\n- [Pearson vs. Spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)\n- [Correlation vs. Causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)\n""")

