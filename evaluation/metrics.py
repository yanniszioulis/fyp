"""
Evaluation metrics for volatility surface forecasting.
"""

import numpy as np
from typing import Dict


def compute_iv_rmse(iv_true: np.ndarray, iv_pred: np.ndarray, 
                   tau: np.ndarray = None) -> float:
    """
    Compute Implied Volatility Root Mean Squared Error.
    
    Parameters:
    -----------
    iv_true : np.ndarray
        True implied volatility values
    iv_pred : np.ndarray
        Predicted implied volatility values
    tau : np.ndarray, optional
        Time to expiration values (not needed for IV, kept for compatibility)
        
    Returns:
    --------
    iv_rmse : float
        IV RMSE
    """
    # Direct IV RMSE (no conversion needed)
    iv_rmse = np.sqrt(np.mean((iv_true - iv_pred) ** 2))
    
    return iv_rmse


def compute_relative_rmse(iv_true: np.ndarray, iv_pred: np.ndarray) -> float:
    """
    Compute relative RMSE (normalized by true values).
    
    Parameters:
    -----------
    iv_true : np.ndarray
        True implied volatility values
    iv_pred : np.ndarray
        Predicted implied volatility values
        
    Returns:
    --------
    relative_rmse : float
        Relative RMSE
    """
    # Avoid division by zero
    iv_true_safe = np.maximum(np.abs(iv_true), 1e-8)
    relative_rmse = np.sqrt(np.mean(((iv_true - iv_pred) / iv_true_safe) ** 2))
    
    return relative_rmse


def compute_mae(iv_true: np.ndarray, iv_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Parameters:
    -----------
    iv_true : np.ndarray
        True implied volatility values
    iv_pred : np.ndarray
        Predicted implied volatility values
        
    Returns:
    --------
    mae : float
        Mean Absolute Error
"""
    return np.mean(np.abs(iv_true - iv_pred))


def compute_all_metrics(iv_true: np.ndarray, iv_pred: np.ndarray,
                       tau: np.ndarray = None) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Parameters:
    -----------
    iv_true : np.ndarray
        True implied volatility values
    iv_pred : np.ndarray
        Predicted implied volatility values
    tau : np.ndarray, optional
        Time to expiration values (not needed for IV, kept for compatibility)
        
    Returns:
    --------
    metrics : dict
        Dictionary of metric names and values
    """
    metrics = {
        'iv_rmse': compute_iv_rmse(iv_true, iv_pred, tau),
        'relative_rmse': compute_relative_rmse(iv_true, iv_pred),
        'mae': compute_mae(iv_true, iv_pred)
    }
    
    return metrics


def compute_metrics_by_maturity(iv_true: np.ndarray, iv_pred: np.ndarray,
                                tau: np.ndarray, tau_grid: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each maturity.
    
    Parameters:
    -----------
    iv_true : np.ndarray, shape (n_samples, n_tau, n_logm)
        True implied volatility
    iv_pred : np.ndarray, shape (n_samples, n_tau, n_logm)
        Predicted implied volatility
    tau : np.ndarray, shape (n_tau,)
        Tau grid values
    tau_grid : np.ndarray, shape (n_tau,)
        Tau grid (same as tau, for consistency)
        
    Returns:
    --------
    metrics_by_tau : dict
        Dictionary mapping tau values to metric dictionaries
    """
    metrics_by_tau = {}
    
    for i, t in enumerate(tau_grid):
        iv_true_tau = iv_true[:, i, :]
        iv_pred_tau = iv_pred[:, i, :]
        
        metrics_by_tau[t] = compute_all_metrics(iv_true_tau, iv_pred_tau)
    
    return metrics_by_tau


def compute_metrics_by_moneyness(iv_true: np.ndarray, iv_pred: np.ndarray,
                                 tau: np.ndarray, logm_grid: np.ndarray,
                                 atm_threshold: float = 0.05) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for ATM and OTM regions.
    
    Parameters:
    -----------
    iv_true : np.ndarray, shape (n_samples, n_tau, n_logm)
        True implied volatility
    iv_pred : np.ndarray, shape (n_samples, n_tau, n_logm)
        Predicted implied volatility
    tau : np.ndarray, shape (n_tau,)
        Tau grid values
    logm_grid : np.ndarray, shape (n_logm,)
        Log-moneyness grid
    atm_threshold : float
        Threshold for ATM (default: 0.05)
        
    Returns:
    --------
    metrics_by_moneyness : dict
        Dictionary with 'atm' and 'otm' keys
    """
    # Find ATM and OTM indices
    atm_mask = np.abs(logm_grid) < atm_threshold
    otm_mask = ~atm_mask
    
    metrics_by_moneyness = {}
    
    # ATM metrics
    iv_true_atm = iv_true[:, :, atm_mask]
    iv_pred_atm = iv_pred[:, :, atm_mask]
    metrics_by_moneyness['atm'] = compute_all_metrics(iv_true_atm, iv_pred_atm)
    
    # OTM metrics
    iv_true_otm = iv_true[:, :, otm_mask]
    iv_pred_otm = iv_pred[:, :, otm_mask]
    metrics_by_moneyness['otm'] = compute_all_metrics(iv_true_otm, iv_pred_otm)
    
    return metrics_by_moneyness
