"""
Data loader for volatility surface forecasting pipeline.
Loads SPX_IV_fixed_grid.csv and reshapes to 3D array format.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str = 'SPX_IV_fixed_grid.csv') -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Load fixed grid data and reshape to 3D array.
    
    Parameters:
    -----------
    filepath : str
        Path to SPX_IV_fixed_grid.csv
        
    Returns:
    --------
    data : np.ndarray, shape (n_dates, n_tau, n_logm)
        Implied volatility data
    tau_grid : np.ndarray, shape (n_tau,)
        Tau (maturity) grid values
    logm_grid : np.ndarray, shape (n_logm,)
        Log-moneyness grid values
    dates : pd.DatetimeIndex
        Date index
    """
    print(f"Loading data from {filepath}...")
    
    # Read CSV in chunks to handle large file
    chunks = []
    chunk_size = 100000
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Sort by date, then tau, then log_moneyness
    df = df.sort_values(['date', 'tau', 'log_moneyness'])
    
    # Get unique values
    dates = pd.Series(pd.to_datetime(df['date'].unique())).sort_values().values
    dates = pd.DatetimeIndex(dates)
    tau_grid = np.sort(df['tau'].unique())
    logm_grid = np.sort(df['log_moneyness'].unique())
    
    n_dates = len(dates)
    n_tau = len(tau_grid)
    n_logm = len(logm_grid)
    
    print(f"Data shape: {n_dates} dates × {n_tau} tau × {n_logm} log-moneyness")
    print(f"Date range: {dates.min()} to {dates.max()}")
    print(f"Tau range: {tau_grid.min():.2f} to {tau_grid.max():.2f} years")
    print(f"Log-moneyness range: {logm_grid.min():.2f} to {logm_grid.max():.2f}")
    
    # Reshape to 3D array: (dates, tau, log_moneyness)
    data = np.full((n_dates, n_tau, n_logm), np.nan)
    
    # Create mapping for faster lookup
    tau_map = {tau: i for i, tau in enumerate(tau_grid)}
    logm_map = {logm: i for i, logm in enumerate(logm_grid)}
    date_map = {date: i for i, date in enumerate(dates)}
    
    # Fill array - use 'implied_volatility' column
    for _, row in df.iterrows():
        date_idx = date_map[pd.to_datetime(row['date'])]
        tau_idx = tau_map[row['tau']]
        logm_idx = logm_map[row['log_moneyness']]
        data[date_idx, tau_idx, logm_idx] = row['implied_volatility']
    
    # Check for missing data
    missing_pct = np.isnan(data).sum() / data.size * 100
    if missing_pct > 0:
        print(f"Warning: {missing_pct:.2f}% missing data")
    
    return data, tau_grid, logm_grid, dates


def create_sequences(data: np.ndarray, dates: pd.DatetimeIndex,
                     context_length: int, horizon: int,
                     verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for time-series forecasting.
    
    Parameters:
    -----------
    data : np.ndarray, shape (n_dates, n_tau, n_logm)
        Implied volatility data
    dates : pd.DatetimeIndex
        Date index
    context_length : int
        Number of past days to use as input
    horizon : int
        Number of days ahead to forecast
        
    Returns:
    --------
    X : np.ndarray, shape (n_samples, context_length, n_tau, n_logm)
        Input sequences
    y : np.ndarray, shape (n_samples, n_tau, n_logm)
        Target sequences
    sample_dates : np.ndarray, shape (n_samples,)
        Dates for each sample (target date)
    """
    n_dates, n_tau, n_logm = data.shape
    n_samples = n_dates - context_length - horizon + 1
    if n_samples <= 0:
        if verbose:
            print(
                "Not enough data to create sequences: "
                f"n_dates={n_dates}, context_length={context_length}, horizon={horizon}"
            )
        empty_X = np.zeros((0, context_length, n_tau, n_logm))
        empty_y = np.zeros((0, n_tau, n_logm))
        return empty_X, empty_y, np.array([])
    
    X = np.zeros((n_samples, context_length, n_tau, n_logm))
    y = np.zeros((n_samples, n_tau, n_logm))
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
