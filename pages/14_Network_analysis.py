# pages/14_network_analysis.py
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

st.title("Network Analysis in Epidemiology")

# Method selector
method = st.selectbox(
    "Select Analysis Method",
    ["Contact Network Analysis", "Disease Spread Simulation", 
     "Network Centrality Measures", "Cluster Analysis",
     "Network Intervention Strategies"]
)

if method == "Contact Network Analysis":
    st.header("Contact Network Visualization and Analysis")
    
    # Network parameters
    n_individuals = st.slider("Number of Individuals", 20, 200, 50)
    connection_probability = st.slider("Connection Probability", 0.0, 1.0, 0.1)
    initial_infected = st.slider("Initial Infected Count", 1, 10, 1)
    
    # Generate network
    def generate_contact_network(n, p, n_infected):
        # Create random graph
        G = nx.erdos_renyi_graph(n, p)
        
        # Add node attributes
        infection_status = ['Infected' if i < n_infected else 'Susceptible' 
                          for i in range(n)]
        np.random.shuffle(infection_status)
        
        nx.set_node_attributes(G, {i: status 
                                 for i, status in enumerate(infection_status)}, 
                             'status')
        
        return G
    
    G = generate_contact_network(n_individuals, connection_probability, initial_infected)
    
    # Get node positions for visualization
    pos = nx.spring_layout(G, k=1/np.sqrt(n_individuals))
    
    # Create network visualization
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append('red' if G.nodes[node]['status'] == 'Infected' else 'blue')
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=10,
            color=node_colors,
            line_width=2
        )
    ))
    
    fig.update_layout(
        title='Contact Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network metrics
    st.subheader("Network Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Degree", f"{np.mean([d for n, d in G.degree()]):.2f}")
    with col2:
        st.metric("Clustering Coefficient", f"{nx.average_clustering(G):.3f}")
    with col3:
        st.metric("Network Density", f"{nx.density(G):.3f}")

elif method == "Disease Spread Simulation":
    st.header("Disease Spread Simulation")
    
    # Simulation parameters
    n_individuals = st.slider("Population Size", 50, 500, 100)
    n_days = st.slider("Simulation Days", 10, 100, 30)
    transmission_rate = st.slider("Transmission Rate (β)", 0.0, 1.0, 0.3, 
                                help="Probability of disease transmission per contact per day")
    recovery_rate = st.slider("Recovery Rate (γ)", 0.0, 1.0, 0.1,
                            help="Probability of recovery per day")
    initial_infected = st.slider("Initial Infected", 1, 10, 1)
    average_contacts = st.slider("Average Daily Contacts", 1, 20, 5,
                               help="Average number of contacts per person per day")

    # Generate and run SIR simulation on network
    def run_sir_simulation(n_individuals, n_days, beta, gamma, initial_infected, avg_contacts):
        # Initialize states (S: 0, I: 1, R: 2)
        states = np.zeros((n_days, n_individuals), dtype=int)
        
        # Set initial infected
        initial_infected_idx = np.random.choice(n_individuals, initial_infected, replace=False)
        states[0, initial_infected_idx] = 1
        
        # Create contact network (rewiring each day for dynamic contacts)
        def daily_contacts(n, avg_contacts):
            contacts = []
            for i in range(n):
                n_contacts = np.random.poisson(avg_contacts)
                possible_contacts = list(set(range(n)) - {i})
                if n_contacts > 0:
                    contacts.extend([(i, j) for j in 
                                   np.random.choice(possible_contacts, 
                                                  min(n_contacts, len(possible_contacts)), 
                                                  replace=False)])
            return contacts

        # Run simulation
        for day in range(1, n_days):
            # Copy previous day's states
            states[day] = states[day-1].copy()
            
            # Generate today's contacts
            today_contacts = daily_contacts(n_individuals, avg_contacts)
            
            # Process infections
            for i, j in today_contacts:
                if states[day, i] == 1 and states[day, j] == 0:  # I-S contact
                    if np.random.random() < beta:
                        states[day, j] = 1
                elif states[day, j] == 1 and states[day, i] == 0:  # S-I contact
                    if np.random.random() < beta:
                        states[day, i] = 1
            
            # Process recoveries
            infected_idx = np.where(states[day] == 1)[0]
            recoveries = np.random.random(len(infected_idx)) < gamma
            states[day, infected_idx[recoveries]] = 2

        return states

    # Run simulation multiple times
    n_simulations = 10
    all_results = []
    
    progress_bar = st.progress(0)
    for sim in range(n_simulations):
        states = run_sir_simulation(n_individuals, n_days, transmission_rate, 
                                  recovery_rate, initial_infected, average_contacts)
        all_results.append(states)
        progress_bar.progress((sim + 1) / n_simulations)

    # Calculate average trajectories and confidence intervals
    sir_curves = np.array([
        [np.sum(states == i, axis=1) for i in range(3)]
        for states in all_results
    ])
    
    mean_curves = np.mean(sir_curves, axis=0)
    std_curves = np.std(sir_curves, axis=0)
    
    # Create visualization with confidence intervals
    fig = go.Figure()

    # Add traces for S, I, R with confidence intervals
    colors = ['blue', 'red', 'green']
    labels = ['Susceptible', 'Infected', 'Recovered']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        # Mean line
        fig.add_trace(go.Scatter(
            x=list(range(n_days)),
            y=mean_curves[i],
            name=label,
            line=dict(color=color),
            mode='lines'
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=list(range(n_days)) + list(range(n_days))[::-1],
            y=np.concatenate([
                mean_curves[i] + 1.96 * std_curves[i],
                (mean_curves[i] - 1.96 * std_curves[i])[::-1]
            ]),
            fill='toself',
            fillcolor=f'rgba(0,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{label} CI'
        ))

    fig.update_layout(
        title='SIR Model Simulation with 95% Confidence Intervals',
        xaxis_title='Days',
        yaxis_title='Number of Individuals',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display key metrics
    st.subheader("Epidemic Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        peak_time = np.argmax(mean_curves[1])
        st.metric("Peak Day", f"{peak_time}")
        
    with col2:
        peak_infected = int(np.max(mean_curves[1]))
        st.metric("Peak Infected", f"{peak_infected}")
        
    with col3:
        final_recovered = int(mean_curves[2][-1])
        st.metric("Total Infected", f"{final_recovered}")

    # R0 estimation
    st.subheader("Basic Reproduction Number (R₀)")
    estimated_r0 = transmission_rate * average_contacts / recovery_rate
    st.write(f"Estimated R₀: {estimated_r0:.2f}")
    
    if estimated_r0 > 1:
        st.warning("R₀ > 1: Epidemic growth likely")
    else:
        st.success("R₀ ≤ 1: Epidemic containment likely")

    # Parameter sensitivity
    st.subheader("Parameter Impact")
    st.write("""
    - Higher transmission rate (β) increases speed and peak of epidemic
    - Higher recovery rate (γ) reduces duration and peak
    - More initial infected accelerates early spread
    - More average contacts increases transmission opportunities
    """)
    
elif method == "Network Centrality Measures":
    st.header("Network Centrality Analysis")
    
    # Network parameters
    n_nodes = st.slider("Number of Nodes", 20, 100, 50)
    edge_probability = st.slider("Edge Probability", 0.0, 1.0, 0.1)
    
    # Generate network
    G = nx.erdos_renyi_graph(n_nodes, edge_probability)
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
    
    # Create visualization with node sizes based on selected centrality
    centrality_measure = st.selectbox(
        "Select Centrality Measure",
        ["Degree", "Betweenness", "Eigenvector"]
    )
    
    if centrality_measure == "Degree":
        node_sizes = list(degree_centrality.values())
        centrality_values = degree_centrality
    elif centrality_measure == "Betweenness":
        node_sizes = list(betweenness_centrality.values())
        centrality_values = betweenness_centrality
    else:
        node_sizes = list(eigenvector_centrality.values())
        centrality_values = eigenvector_centrality
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G)
    
    # Create network visualization
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=[v * 50 for v in node_sizes],
            color=node_sizes,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=f'{centrality_measure} Centrality')
        )
    ))
    
    fig.update_layout(
        title=f'Network with {centrality_measure} Centrality',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top nodes by centrality
    st.subheader(f"Top 5 Nodes by {centrality_measure} Centrality")
    top_nodes = dict(sorted(centrality_values.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:5])
    
    st.table(pd.DataFrame({
        'Node': list(top_nodes.keys()),
        'Centrality': [f"{v:.3f}" for v in top_nodes.values()]
    }))

elif method == "Cluster Analysis":
    st.header("Network Cluster Analysis")
    
    # Parameters
    n_nodes = st.slider("Number of Nodes", 30, 200, 100)
    n_clusters = st.slider("Number of Clusters", 2, 6, 3)
    p_within = st.slider("Within-Cluster Connection Probability", 0.1, 1.0, 0.3)
    p_between = st.slider("Between-Cluster Connection Probability", 0.0, 0.3, 0.05)
    
    # Generate clustered network
    def generate_clustered_network(n_nodes, n_clusters, p_within, p_between):
        # Divide nodes into clusters
        nodes_per_cluster = n_nodes // n_clusters
        G = nx.Graph()
        
        # Add nodes with cluster attributes
        for i in range(n_nodes):
            cluster = i // nodes_per_cluster
            if cluster >= n_clusters:
                cluster = n_clusters - 1
            G.add_node(i, cluster=cluster)
        
        # Add edges
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if G.nodes[i]['cluster'] == G.nodes[j]['cluster']:
                    if np.random.random() < p_within:
                        G.add_edge(i, j)
                else:
                    if np.random.random() < p_between:
                        G.add_edge(i, j)
        
        return G
    
    G = generate_clustered_network(n_nodes, n_clusters, p_within, p_between)
    
    # Detect communities
    communities = list(nx.community.greedy_modularity_communities(G))
    
    # Assign colors to detected communities
    node_colors = []
    for node in G.nodes():
        for i, community in enumerate(communities):
            if node in community:
                node_colors.append(i)
                break
    
    # Position nodes
    pos = nx.spring_layout(G)
    
    # Create visualization
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=10,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Community')
        )
    ))
    
    fig.update_layout(
        title='Network Communities',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display community statistics
    st.subheader("Community Statistics")
    
    community_sizes = [len(c) for c in communities]
    st.write(f"Number of Communities Detected: {len(communities)}")
    
    # Display community sizes
    community_df = pd.DataFrame({
        'Community': range(len(communities)),
        'Size': community_sizes,
        'Percentage': [size/n_nodes*100 for size in community_sizes]
    })
    
    st.table(community_df.round(2))
    
    # Calculate and display modularity
    modularity = nx.community.modularity(G, communities)
    st.metric("Network Modularity", f"{modularity:.3f}")

elif method == "Network Intervention Strategies":
    st.header("Network Intervention Analysis")
    
    # Parameters
    n_individuals = st.slider("Population Size", 50, 500, 100)
    vaccination_budget = st.slider("Vaccination Budget (%)", 10, 50, 20)
    
    # Generate network
    G = nx.barabasi_albert_graph(n_individuals, 3)
    
    # Intervention strategy selector
    strategy = st.selectbox(
        "Select Intervention Strategy",
        ["Random", "Degree-based", "Betweenness-based", "Community-based"]
    )
    
    # Calculate target nodes based on strategy
    n_vaccinate = int(n_individuals * vaccination_budget / 100)
    
    if strategy == "Random":
        target_nodes = np.random.choice(
            list(G.nodes()), 
            size=n_vaccinate, 
            replace=False
        )
    elif strategy == "Degree-based":
        degree_dict = dict(G.degree())
        target_nodes = sorted(
            degree_dict.keys(),
            key=lambda x: degree_dict[x],
            reverse=True
        )[:n_vaccinate]
    elif strategy == "Betweenness-based":
        betweenness_dict = nx.betweenness_centrality(G)
        target_nodes = sorted(
            betweenness_dict.keys(),
            key=lambda x: betweenness_dict[x],
            reverse=True
        )[:n_vaccinate]
    else:  # Community-based
        communities = list(nx.community.greedy_modularity_communities(G))
        target_nodes = []
        nodes_per_community = n_vaccinate // len(communities)
        for community in communities:
            community_list = list(community)
            target_nodes.extend(
                np.random.choice(
                    community_list,
                    size=min(nodes_per_community, len(community_list)),
                    replace=False
                )
            )
    
    # Visualize network with intervention
    pos = nx.spring_layout(G)
    
    # Create visualization
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in target_nodes:
            node_colors.append('red')
            node_sizes.append(15)
        else:
            node_colors.append('blue')
            node_sizes.append(10)
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line_width=2
        )
    ))
    
    fig.update_layout(
        title=f'Network with {strategy} Intervention Strategy',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display intervention metrics
    st.subheader("Intervention Impact Metrics")
    
    # Calculate network metrics before and after intervention
    G_after = G.copy()
    G_after.remove_nodes_from(target_nodes)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        original_gcc = len(max(nx.connected_components(G), key=len))
        remaining_gcc = len(max(nx.connected_components(G_after), key=len))
        st.metric(
            "Largest Component Reduction",
            f"{((original_gcc - remaining_gcc)/original_gcc*100):.1f}%"
        )
    
    with col2:
        original_path = nx.average_shortest_path_length(G)
        try:
            remaining_path = nx.average_shortest_path_length(G_after)
            path_increase = (remaining_path - original_path)/original_path*100
        except:
            path_increase = float('inf')
        st.metric(
            "Path Length Increase",
            "∞" if path_increase == float('inf') else f"{path_increase:.1f}%"
        )
    
    with col3:
        original_clustering = nx.average_clustering(G)
        remaining_clustering = nx.average_clustering(G_after)
        st.metric(
            "Clustering Change",
            f"{((remaining_clustering - original_clustering)/original_clustering*100):.1f}%"
        )

# Add educational content
st.header("Method Details")

if method == "Contact Network Analysis":
    st.write("""
    Contact network analysis in epidemiology:
    
    1. Network Structure:
    - Nodes represent individuals
    - Edges represent contacts
    - Edge weights can represent contact duration/intensity
    
    2. Applications:
    - Disease transmission modeling
    - Contact tracing
    - Intervention planning
    - Outbreak investigation
    
    3. Key Metrics:
    - Degree distribution
    - Clustering coefficient
    - Network density
    - Path lengths
    """)

elif method == "Disease Spread Simulation":
    st.write("""
    Disease spread simulation on networks:
    
    1. SIR Model Components:
    - Susceptible individuals
    - Infectious individuals
    - Recovered individuals
    
    2. Parameters:
    - Transmission rate
    - Recovery rate
    - Network structure
    - Initial conditions
    
    3. Analysis:
    - Epidemic threshold
    - Final size
    - Peak timing
    - Intervention effects
    """)

elif method == "Network Centrality Measures":
    st.write("""
    Network centrality in epidemiology:
    
    1. Measures:
    - Degree centrality (local connections)
    - Betweenness centrality (pathway control)
    - Eigenvector centrality (influential connections)
    
    2. Applications:
    - Super-spreader identification
    - Vaccination targeting
    - Surveillance planning
    - Resource allocation
    
    3. Interpretation:
    - Risk assessment
    - Intervention prioritization
    - Network vulnerability
    """)

elif method == "Cluster Analysis":
    st.write("""
    Network clustering in epidemiology:
    
    1. Community Detection:
    - Identify natural groupings
    - Assess mixing patterns
    - Evaluate intervention barriers
    
    2. Applications:
    - Contact pattern analysis
    - Outbreak containment
    - Targeted interventions
    - Risk stratification
    
    3. Metrics:
    - Modularity
    - Community sizes
    - Mixing matrices
    """)

elif method == "Network Intervention Strategies":
    st.write("""
    Network-based interventions:
    
    1. Strategies:
    - Random vaccination
    - Targeted vaccination
    - Ring vaccination
    - Community-based approaches
    
    2. Evaluation:
    - Network fragmentation
    - Path length changes
    - Clustering impacts
    - Component analysis
    
    3. Considerations:
    - Resource constraints
    - Implementation feasibility
    - Equity considerations
    - Effectiveness metrics
    """)

# Add references
st.header("Further Reading")
st.write("""
1. Keeling MJ, Eames KTD. Networks and Epidemic Models
2. Newman MEJ. Networks: An Introduction
3. Luke DA, Harris JK. Network Analysis in Public Health
4. Christakis NA, Fowler JH. Social Network Visualization in Epidemiology
""")

st.header("Check your understanding")
if method == "Contact Network Analysis":
    quiz_contact = st.radio(
        "What does a node represent in a contact network?",
        [
            "A disease outbreak",
            "An individual",
            "A geographical region",
            "A simulation timestep"
        ]
    )
    if quiz_contact == "An individual":
        st.success("Correct! Each node in a contact network represents an individual.")
    else:
        st.error("Not quite. Nodes represent individuals in a network.")
        
elif method == "Disease Spread Simulation":
    quiz_spread = st.radio(
        "What is the basic reproduction number (R₀) used for?",
        [
            "To measure the speed of a simulation",
            "To indicate the average number of secondary infections from one case",
            "To measure hospital capacity",
            "To track weather patterns affecting diseases"
        ]
    )
    if quiz_spread == "To indicate the average number of secondary infections from one case":
        st.success("Correct! R₀ measures the expected number of cases generated by an infected individual.")
    else:
        st.error("Not quite. R₀ is a key measure in epidemiology to estimate disease spread.")

elif method == "Network Centrality Measures":
    quiz_centrality = st.radio(
        "Which centrality measure identifies nodes that act as bridges in a network?",
        [
            "Degree centrality",
            "Betweenness centrality",
            "Eigenvector centrality",
            "Clustering coefficient"
        ]
    )
    if quiz_centrality == "Betweenness centrality":
        st.success("Correct! Betweenness centrality measures how often a node acts as a bridge in shortest paths.")
    else:
        st.error("Not quite. Betweenness centrality identifies nodes that serve as connectors in a network.")

elif method == "Cluster Analysis":
    quiz_cluster = st.radio(
        "What does high modularity in a network indicate?",
        [
            "A highly connected network with no distinct clusters",
            "A well-defined community structure",
            "Low interaction between nodes",
            "An unstable network structure"
        ]
    )
    if quiz_cluster == "A well-defined community structure":
        st.success("Correct! High modularity means the network has distinct, well-separated communities.")
    else:
        st.error("Not quite. High modularity suggests strong community structures.")

elif method == "Network Intervention Strategies":
    quiz_intervention = st.radio(
        "Which intervention strategy targets the most connected individuals in a network?",
        [
            "Random vaccination",
            "Ring vaccination",
            "Degree-based vaccination",
            "Community-based intervention"
        ]
    )
    if quiz_intervention == "Degree-based vaccination":
        st.success("Correct! Degree-based vaccination targets highly connected individuals to reduce disease spread.")
    else:
        st.error("Not quite. Degree-based vaccination focuses on individuals with many connections.")
