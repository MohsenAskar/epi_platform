# utils/data_generators.py
import numpy as np
import pandas as pd

def generate_correlated_data(correlation, n_samples=100, random_state=None):
    """Generate correlated data for visualization."""
    if random_state is not None:
        np.random.seed(random_state)
    
    x = np.random.normal(0, 1, n_samples)
    y = correlation * x + np.sqrt(1 - correlation**2) * np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame({
        'Variable 1': x,
        'Variable 2': y
    })

def generate_stratified_data(effect_size, n_samples=100, random_state=None):
    """Generate stratified data with different effect sizes per stratum."""
    if random_state is not None:
        np.random.seed(random_state)
    
    strata = ['A', 'B', 'C']
    data = []
    
    for stratum in strata:
        x = np.random.normal(0, 1, n_samples)
        # Different effect size per stratum
        y = effect_size * (1 + strata.index(stratum)/2) * x + np.random.normal(0, 1, n_samples)
        stratum_data = pd.DataFrame({
            'Variable 1': x,
            'Variable 2': y,
            'Stratum': stratum
        })
        data.append(stratum_data)
    
    return pd.concat(data, ignore_index=True)
