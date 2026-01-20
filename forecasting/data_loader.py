"""
Data loader for volatility surface forecasting pipeline.
Loads SPX surface CSV files (wide format) and reshapes to 3D array format.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple


def load_data(filepath: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Load surface data from wide format CSV and reshape to 3D array.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to surface CSV file. If None, uses SPX_surfaces.csv (combined calls and puts)
        
    Returns:
    --------
    data : np.ndarray, shape (n_dates, n_tau, n_m)
        Implied volatility data
    tau_grid : np.ndarray, shape (n_tau,)
        Tau (maturity) grid values
    m_grid : np.ndarray, shape (n_m,)
        Moneyness grid values
    dates : pd.DatetimeIndex
        Date index
    """
    # Determine filepath
    if filepath is None:
        filepath = 'SPX_surfaces.csv'
    
    print(f"Loading surface data from {filepath}...")
    
    # Read wide format CSV
    df = pd.read_csv(filepath, low_memory=False)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Get dates
    dates = pd.DatetimeIndex(df['date'].values)
    n_dates = len(dates)
    
    # Extract iv_* columns
    iv_columns = [col for col in df.columns if col.startswith('iv_')]
    
    if len(iv_columns) == 0:
        raise ValueError(f"No 'iv_*' columns found in {filepath}")
    
    # Parse column names to extract moneyness and tau values
    # Pattern: iv_{moneyness}_{tau}
    pattern = re.compile(r'iv_([\d.]+)_([\d.]+)')
    
    moneyness_values = []
    tau_values = []
    column_mapping = []
    
    for col in iv_columns:
        match = pattern.match(col)
        if match:
            moneyness = float(match.group(1))
            tau = float(match.group(2))
            moneyness_values.append(moneyness)
            tau_values.append(tau)
            column_mapping.append((moneyness, tau, col))
    
    if len(column_mapping) == 0:
        raise ValueError(f"Could not parse column names in {filepath}. Expected format: iv_{{moneyness}}_{{tau}}")
    
    # Get unique sorted grids
    moneyness_grid = np.sort(np.unique(moneyness_values))
    tau_grid = np.sort(np.unique(tau_values))
    
    n_m = len(moneyness_grid)
    n_tau = len(tau_grid)
    
    print(f"Data shape: {n_dates} dates × {n_tau} tau × {n_m} moneyness")
    print(f"Date range: {dates.min()} to {dates.max()}")
    print(f"Tau range: {tau_grid.min():.4f} to {tau_grid.max():.4f} years")
    print(f"Moneyness range: {moneyness_grid.min():.4f} to {moneyness_grid.max():.4f}")
    
    # Create mapping from (moneyness, tau) to indices
    m_map = {m: i for i, m in enumerate(moneyness_grid)}
    tau_map = {tau: i for i, tau in enumerate(tau_grid)}
    
    # Initialize 3D array: (dates, tau, moneyness)
    data = np.zeros((n_dates, n_tau, n_m))
    
    # Reshape data for each date
    for date_idx in range(n_dates):
        surface = np.full((n_tau, n_m), np.nan)
        
        # Fill surface from wide format columns
        for moneyness, tau, col_name in column_mapping:
            m_idx = m_map[moneyness]
            tau_idx = tau_map[tau]
            value = df.iloc[date_idx][col_name]
            
            # Handle NaN values
            if pd.isna(value):
                surface[tau_idx, m_idx] = np.nan
            else:
                surface[tau_idx, m_idx] = float(value)
        
        data[date_idx] = surface
    
    # Check for missing data
    missing_pct = np.isnan(data).sum() / data.size * 100
    if missing_pct > 0:
        print(f"Warning: {missing_pct:.2f}% missing data")
    
    return data, tau_grid, moneyness_grid, dates


def create_sequences(data: np.ndarray, dates: pd.DatetimeIndex,
                     context_length: int, horizon: int,
                     verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for time-series forecasting.
    
    Parameters:
    -----------
    data : np.ndarray, shape (n_dates, n_tau, n_m)
        Implied volatility data
    dates : pd.DatetimeIndex
        Date index
    context_length : int
        Number of past days to use as input
    horizon : int
        Number of days ahead to forecast
        
    Returns:
    --------
    X : np.ndarray, shape (n_samples, context_length, n_tau, n_m)
        Input sequences
    y : np.ndarray, shape (n_samples, n_tau, n_m)
        Target sequences
    sample_dates : np.ndarray, shape (n_samples,)
        Dates for each sample (target date)
    """
    n_dates, n_tau, n_m = data.shape
    n_samples = n_dates - context_length - horizon + 1
    if n_samples <= 0:
        if verbose:
            print(
                "Not enough data to create sequences: "
                f"n_dates={n_dates}, context_length={context_length}, horizon={horizon}"
            )
        empty_X = np.zeros((0, context_length, n_tau, n_m))
        empty_y = np.zeros((0, n_tau, n_m))
        return empty_X, empty_y, np.array([])
    
    X = np.zeros((n_samples, context_length, n_tau, n_m))
    y = np.zeros((n_samples, n_tau, n_m))
    sample_dates = []
    
    for i in range(n_samples):
        # Input: context_length days before target
        X[i] = data[i:i+context_length]
        
        # Target: horizon days ahead
        target_idx = i + context_length + horizon - 1
        y[i] = data[target_idx]
        sample_dates.append(dates[target_idx])
    
    sample_dates = np.array(sample_dates)
    
    if verbose:
        print(f"Created {n_samples} sequences with context_length={context_length}, horizon={horizon}")
    
    return X, y, sample_dates
