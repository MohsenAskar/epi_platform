# 21_bayesian_methods.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import arviz as az
import pymc as pm


########################
# Page Title
########################
st.title("Bayesian Methods in Epidemiology")

########################
# Main Selectbox
########################
analysis_type = st.selectbox(
    "Select a Bayesian Method",
    [
        "Bayesian Basics (Beta-Binomial)",
        "Bayesian Hierarchical Model",
        "Bayesian Updating for Mean/Variance"
    ]
)

###############################################################################
# 1. Bayesian Basics (Beta-Binomial)
###############################################################################
if analysis_type == "Bayesian Basics (Beta-Binomial)":
    st.header("Bayesian Basics: Beta-Binomial")

    st.write("""
    **Scenario**: Suppose we're estimating a disease prevalence (or 
    probability of success) \\(p\\). We collect data: 
    \\(X\\) successes out of \\(n\\) trials (e.g., positive tests). 
    In a Bayesian framework, we combine a prior distribution for \\(p\\) 
    with the likelihood to form a posterior distribution.
    """)

    st.subheader("Simulation Setup & Prior")
    n = st.slider("Number of Trials (n)", 1, 200, 50)
    true_p = st.slider("True Proportion (p)", 0.0, 1.0, 0.3, 0.01)

    # Beta prior parameters
    alpha_prior = st.number_input("Prior α (alpha)", 0.1, 10.0, 1.0, 0.1)
    beta_prior = st.number_input("Prior β (beta)", 0.1, 10.0, 1.0, 0.1)

    # Simulate data
    rng = np.random.default_rng(42)
    X = rng.binomial(n, true_p)

    st.write(f"**Observed Data**: X = {X} successes out of n = {n}")

    # Build a PyMC model for Beta-Binomial
    with pm.Model() as beta_binom_model:
        p = pm.Beta("p", alpha=alpha_prior, beta=beta_prior)
        y = pm.Binomial("y", n=n, p=p, observed=X)

        # Draw samples from posterior
        trace = pm.sample(
            draws=2000, tune=1000, chains=2, 
            progressbar=False, random_seed=42,
            cores=1  # if you want to limit to 1 core for Streamlit
        )

    # Summarize posterior
    posterior_summary = az.summary(trace, var_names=["p"], round_to=3)
    st.write("### Posterior Summary")
    st.write(posterior_summary)

    # Posterior distribution plot
    p_samples = trace.posterior["p"].values.flatten()
    fig = px.histogram(
        p_samples,
        nbins=50,
        title="Posterior Distribution of p",
        labels={"value": "p", "count": "Frequency"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    **Interpretation**:
    - The posterior distribution combines the prior 
      \\(\\text{{Beta}}({alpha_prior:.1f}, {beta_prior:.1f})\\) 
      and the likelihood from the observed binomial data (\\(X={X}\\) out of \\(n={n}\\)).
    - The resulting distribution is effectively 
      \\(\\text{{Beta}}(\\alpha + X, \\beta + n - X)\\), 
      though PyMC obtains it via sampling.
    """)

###############################################################################
# 2. Bayesian Hierarchical Model
###############################################################################
elif analysis_type == "Bayesian Hierarchical Model":
    st.header("Bayesian Hierarchical Model")

    st.write("""
    **Scenario**: We have multiple groups or clusters (e.g., clinics, areas), 
    each with its own rate of disease, but we believe these rates come from a 
    common distribution. A hierarchical (multilevel) model partially pools 
    estimates, shrinking them towards a global mean.
    """)

    st.subheader("Simulation Setup")
    n_groups = st.slider("Number of Groups", 2, 30, 5)
    n_per_group = st.slider("Observations per Group", 5, 100, 20)
    true_mu = st.slider("Overall Mean (log-odds scale)", -3.0, 1.0, -1.5, 0.1)
    true_tau = st.slider("Between-Group SD (log-odds scale)", 0.01, 2.0, 0.7, 0.1)

    rng = np.random.default_rng(123)
    # For each group, draw a random log-odds from Normal(mu, tau)
    group_logodds = rng.normal(loc=true_mu, scale=true_tau, size=n_groups)

    data_list = []
    for i in range(n_groups):
        group_id = i + 1
        logodds_i = group_logodds[i]
        p_i = 1 / (1 + np.exp(-logodds_i))
        # Simulate binary outcomes
        y_i = rng.binomial(n=1, p=p_i, size=n_per_group)
        for val in y_i:
            data_list.append({"group": group_id, "y": val})

    df = pd.DataFrame(data_list)

    st.write("### First 10 Rows of Simulated Data")
    st.dataframe(df.head(10))

    with pm.Model() as hierarchical_model:
        # Hyperpriors for global mean and sd
        mu = pm.Normal("mu", mu=0, sigma=5)  
        tau = pm.HalfNormal("tau", sigma=5)

        # Random intercept for each group
        # centered parameterization: 
        #   each group has intercept = mu + alpha_raw[i]*tau
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_groups)
        alpha = pm.Deterministic("alpha", mu + alpha_raw * tau)

        # Convert group-specific intercept to probability
        # For each observation, we pick intercept from the group
        # -> logistic link
        # We'll need to map each row's group index
        group_idx = df["group"].values - 1  # zero-based

        # Probability for each row
        p = pm.Deterministic("p", pm.math.sigmoid(alpha[group_idx]))

        # Likelihood
        y_obs = pm.Bernoulli("y_obs", p=p, observed=df["y"].values)

        # Sample
        trace_hm = pm.sample(
            draws=2000, tune=1000, chains=2, 
            progressbar=False, random_seed=42, 
            cores=1
        )

    st.write("### Posterior Summary (Hyperparameters)")
    st.write(az.summary(trace_hm, var_names=["mu", "tau"], round_to=3))

    # Posterior means for each group
    alpha_post = trace_hm.posterior["alpha"].mean(dim=("chain","draw")).values
    # Convert log-odds to probability
    alpha_post_prob = 1 / (1 + np.exp(-alpha_post))

    df_groups = pd.DataFrame({
        "group_id": np.arange(1, n_groups+1),
        "true_p": 1 / (1 + np.exp(-group_logodds)),
        "posterior_mean_p": alpha_post_prob
    })

    st.write("### Group-Level Results")
    st.dataframe(df_groups)

    st.markdown("""
    Note how partial pooling "shrinks" each group’s estimate toward the overall mean, 
    especially when the sample size per group is small.
    """)

    fig = px.scatter(
        df_groups,
        x="true_p",
        y="posterior_mean_p",
        text="group_id",
        labels={"true_p": "True Probability", "posterior_mean_p": "Posterior Mean Probability"},
        title="True Probability vs. Posterior Mean Probability (by Group)"
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", color="red"),
        x0=0, x1=1, y0=0, y1=1
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation**: A **Bayesian hierarchical model** can handle variation 
    across groups while borrowing strength (partial pooling) from the overall sample. 
    The hyperparameters \\(\\mu\\) and \\(\\tau\\) describe the distribution of 
    group-level log-odds around a global mean.
    """)

###############################################################################
# 3. Bayesian Updating for Mean/Variance
###############################################################################
elif analysis_type == "Bayesian Updating for Mean/Variance":
    st.header("Bayesian Updating for Mean/Variance")

    st.write("""
    **Scenario**: We have continuous data and assume they're normally distributed. 
    We place a prior on the mean \\(\\mu\\) and variance \\(\\sigma^2\\). 
    We want to see how observing more data updates our beliefs about \\(\\mu\\) 
    and \\(\\sigma^2\\).
    """)

    st.subheader("Simulation Setup & Incremental Data")

    n_total = st.slider("Total Data Points to Simulate", 10, 500, 50)
    true_mu = st.slider("True Mean", -10.0, 10.0, 0.0, 1.0)
    true_sigma = st.slider("True Std Dev", 0.1, 5.0, 1.0, 0.1)

    rng = np.random.default_rng(999)
    data = rng.normal(true_mu, true_sigma, n_total)

    st.write("### Priors for Mean & Std")
    prior_mu_mean = st.number_input("Prior Mean of μ", -10.0, 10.0, 0.0, 1.0)
    prior_mu_sd   = st.number_input("Prior Std of μ", 0.1, 10.0, 1.0, 0.1)
    prior_sigma_upper = st.number_input("Upper Bound for HalfNormal σ Prior", 1.0, 20.0, 10.0, 1.0)

    st.write("We'll do incremental updates: first half of the data, then the second half, to see the posterior shift.")

    # Split data into two parts
    data_part1 = data[: n_total // 2]
    data_part2 = data[n_total // 2 :]

    # First update (Part 1)
    with pm.Model() as model_part1:
        mu_1 = pm.Normal("mu_1", mu=prior_mu_mean, sigma=prior_mu_sd)
        sigma_1 = pm.HalfNormal("sigma_1", sigma=prior_sigma_upper)

        obs_1 = pm.Normal("obs_1", mu=mu_1, sigma=sigma_1, observed=data_part1)

        trace_part1 = pm.sample(
            draws=2000, tune=1000, chains=2,
            progressbar=False, random_seed=42,
            cores=1
        )

    # Posterior from part1 as new prior for part2
    post_mu_part1 = trace_part1.posterior["mu_1"].mean().item()
    post_sigma_part1 = trace_part1.posterior["sigma_1"].mean().item()

    st.write("### Posterior from First Half of Data")
    st.write(az.summary(trace_part1, var_names=["mu_1","sigma_1"], round_to=3))

    st.write(f"**Posterior Mean for μ** after Part 1: {post_mu_part1:.3f}")
    st.write(f"**Posterior Mean for σ** after Part 1: {post_sigma_part1:.3f}")

    # Second update (Part 2) - use posterior means from part1 as priors 
    # (Naive approach for demonstration: we might choose a normal prior for mu_2 with mean=post_mu_part1, etc.)
    with pm.Model() as model_part2:
        mu_2 = pm.Normal("mu_2", mu=post_mu_part1, sigma=prior_mu_sd/2)  # narrower prior if desired
        sigma_2 = pm.HalfNormal("sigma_2", sigma=max(1e-3, post_sigma_part1*2))  # scale it or keep as halfnormal

        obs_2 = pm.Normal("obs_2", mu=mu_2, sigma=sigma_2, observed=data_part2)

        trace_part2 = pm.sample(
            draws=2000, tune=1000, chains=2,
            progressbar=False, random_seed=84,
            cores=1
        )

    st.write("### Posterior after Second Half of Data")
    st.write(az.summary(trace_part2, var_names=["mu_2","sigma_2"], round_to=3))

    # Plot final posterior distributions
    mu_2_samples = trace_part2.posterior["mu_2"].values.flatten()
    sigma_2_samples = trace_part2.posterior["sigma_2"].values.flatten()

    col1, col2 = st.columns(2)
    with col1:
        fig_mu = px.histogram(
            mu_2_samples, nbins=50,
            title="Posterior Distribution of μ (after Part 2)",
            labels={"value": "μ", "count": "Frequency"}
        )
        st.plotly_chart(fig_mu, use_container_width=True)
    with col2:
        fig_sigma = px.histogram(
            sigma_2_samples, nbins=50,
            title="Posterior Distribution of σ (after Part 2)",
            labels={"value": "σ", "count": "Frequency"}
        )
        st.plotly_chart(fig_sigma, use_container_width=True)

    st.markdown(f"""
    **Interpretation**:
    - We first updated our beliefs about \\(\\mu\\) and \\(\\sigma\\) using 
      the first half of the data. 
    - Then, we used that posterior (roughly) as a new prior for the second half. 
    - Over more observations, the posterior should concentrate around the true values 
      (\\(\\mu={true_mu}\\), \\(\\sigma={true_sigma}\\)) 
      given enough data and well-specified priors.
    """)

########################
# Method Description
########################
st.header("Method Descriptions")
if analysis_type == "Bayesian Basics (Beta-Binomial)":
    st.write("""
    The **Beta-Binomial** model is a classic conjugate pair for 
    binomial data with unknown probability \\(p\\). The posterior distribution 
    is a Beta distribution whose parameters are updated by the observed successes 
    and failures.
    """)
elif analysis_type == "Bayesian Hierarchical Model":
    st.write("""
    **Bayesian hierarchical (multilevel) models** allow partial pooling across groups. 
    They estimate group-specific parameters as draws from a population-level distribution, 
    improving estimates (especially with small sample sizes in each group).
    """)
elif analysis_type == "Bayesian Updating for Mean/Variance":
    st.write("""
    We can place priors on both \\(\\mu\\) (mean) and \\(\\sigma^2\\) (variance) 
    for normally distributed data. As we observe data, the posterior reflects 
    updated beliefs. This can be done incrementally, combining 
    prior → posterior → new prior → posterior, etc.
    """)

########################
# References
########################
st.header("References")
st.write("""
1. **Bayesian Fundamentals**: Gelman A, et al. _Bayesian Data Analysis_, 3rd ed.
2. **PyMC**: [https://www.pymc.io/](https://www.pymc.io/)
3. **ArviZ**: [https://python.arviz.org/](https://python.arviz.org/)
4. **Conjugate Priors**: Carlin BP, Louis TA. _Bayesian Methods for Data Analysis._
5. **Hierarchical Models**: Gelman A, Hill J. _Data Analysis Using Regression and Multilevel/Hierarchical Models._
""")

st.header("Check your understanding")

# Quiz for Bayesian Basics (Beta-Binomial)
if analysis_type == "Bayesian Basics (Beta-Binomial)":
    q1 = st.radio(
        "Which prior distribution is commonly used for a **binomial likelihood**?",
        [
            "Normal",
            "Beta",
            "Poisson",
            "Gamma"
        ]
    )
    if q1 == "Beta":
        st.success("Correct! The **Beta** distribution is the conjugate prior for the binomial likelihood.")
    else:
        st.error("Not quite. The **Beta distribution** is used as a prior for binomial data.")

    q2 = st.radio(
        "If the prior is **Beta(α, β)** and we observe **X successes in n trials**, what is the **posterior distribution**?",
        [
            "Beta(α + X, β + (n - X))",
            "Beta(α * X, β / (n - X))",
            "Normal(α + X, β + (n - X))",
            "Gamma(α + X, β + (n - X))"
        ]
    )
    if q2 == "Beta(α + X, β + (n - X))":
        st.success("Correct! The posterior is **Beta(α + X, β + (n - X))** due to the conjugacy of the Beta-Binomial model.")
    else:
        st.error("Not quite. The correct update rule is **Beta(α + X, β + (n - X))**.")

# Quiz for Bayesian Hierarchical Model
elif analysis_type == "Bayesian Hierarchical Model":
    q3 = st.radio(
        "Why do we use **hierarchical Bayesian models**?",
        [
            "To allow individual groups to borrow strength from the overall population",
            "To make every group's estimate independent",
            "To avoid using priors",
            "To ensure each group has the same estimated effect"
        ]
    )
    if q3 == "To allow individual groups to borrow strength from the overall population":
        st.success("Correct! **Hierarchical models allow partial pooling**, where group estimates are pulled toward the overall mean.")
    else:
        st.error("Not quite. Hierarchical models improve estimates by **allowing partial pooling across groups**.")

    q4 = st.radio(
        "Which distribution is typically used as a **prior for group-level means** in hierarchical models?",
        [
            "Beta",
            "Normal",
            "Exponential",
            "Poisson"
        ]
    )
    if q4 == "Normal":
        st.success("Correct! The **Normal distribution** is commonly used for group-level effects in hierarchical models.")
    else:
        st.error("Not quite. A **Normal distribution** is typically used for group-level effects.")

# Quiz for Bayesian Updating for Mean/Variance
elif analysis_type == "Bayesian Updating for Mean/Variance":
    q5 = st.radio(
        "What happens to the posterior distribution as we observe more data?",
        [
            "It becomes wider",
            "It becomes more concentrated around the true parameter value",
            "It remains the same",
            "It moves randomly"
        ]
    )
    if q5 == "It becomes more concentrated around the true parameter value":
        st.success("Correct! As more data is observed, the posterior **converges to the true parameter value**.")
    else:
        st.error("Not quite. The posterior **narrows and concentrates** with more data.")

    q6 = st.radio(
        "Which prior distribution is commonly used for **unknown variance (σ²)** in Bayesian analysis?",
        [
            "Beta",
            "Gamma",
            "Half-Normal",
            "Poisson"
        ]
    )
    if q6 == "Half-Normal":
        st.success("Correct! The **Half-Normal** or **Inverse-Gamma** priors are commonly used for variance.")
    else:
        st.error("Not quite. **Half-Normal** or **Inverse-Gamma** are standard choices for variance priors.")

st.write("Great job! Keep practicing to master Bayesian Methods. 🚀")

