"""
Rolling window splits for time-series cross-validation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class WindowSplit:
    """Container for a single rolling window split"""
    window_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def create_rolling_windows(dates: pd.DatetimeIndex,
                          window_size_years: float = 10.0,
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.2,
                          shift_months: int = 6) -> List[WindowSplit]:
    """
    Create rolling window splits with specified shift.
    
    Parameters:
    -----------
    dates : pd.DatetimeIndex
        All available dates
    window_size_years : float
        Size of each window in years (default: 10.0)
    train_ratio : float
        Proportion of window for training (default: 0.7)
    val_ratio : float
        Proportion of window for validation (default: 0.1)
    test_ratio : float
        Proportion of window for testing (default: 0.2)
    shift_months : int
        Number of months to shift window forward (default: 6)
        
    Returns:
    --------
    windows : List[WindowSplit]
        List of window splits
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    windows = []
    window_id = 0
    
    # Calculate window size in days (approximate)
    window_size_days = int(window_size_years * 365.25)
    train_size_days = int(window_size_days * train_ratio)
    val_size_days = int(window_size_days * val_ratio)
    test_size_days = int(window_size_days * test_ratio)
    shift_days = int(shift_months * 30.44)  # Approximate days per month
    
    # Start from first date
    current_start = dates[0]
    min_date = dates[0]
    max_date = dates[-1]
    
    while True:
        # Calculate window end
        window_end = current_start + pd.Timedelta(days=window_size_days)
        
        # Check if window fits in available data
        if window_end > max_date:
            break
        
        # Find indices for this window
        window_mask = (dates >= current_start) & (dates < window_end)
        window_indices = np.where(window_mask)[0]
        
        if len(window_indices) == 0:
            break
        
        # Split into train/val/test
        train_end_idx = window_indices[0] + train_size_days
        val_end_idx = train_end_idx + val_size_days
        
        # Find actual indices based on dates
        train_end_date = current_start + pd.Timedelta(days=train_size_days)
        val_end_date = train_end_date + pd.Timedelta(days=val_size_days)
        test_end_date = window_end
        
        train_mask = (dates >= current_start) & (dates < train_end_date)
        val_mask = (dates >= train_end_date) & (dates < val_end_date)
        test_mask = (dates >= val_end_date) & (dates < test_end_date)
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Only create window if we have data in all splits
        if len(train_indices) > 0 and len(val_indices) > 0 and len(test_indices) > 0:
            window = WindowSplit(
                window_id=window_id,
                start_date=current_start,
                end_date=window_end,
                train_start=current_start,
                train_end=train_end_date,
                val_start=train_end_date,
                val_end=val_end_date,
                test_start=val_end_date,
                test_end=test_end_date,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices
            )
            windows.append(window)
            window_id += 1
        
        # Shift window forward
        current_start = current_start + pd.Timedelta(days=shift_days)
        
        # Safety check: don't create windows that start too late
        if current_start + pd.Timedelta(days=window_size_days) > max_date:
            break
    
    print(f"Created {len(windows)} rolling windows")
    for i, w in enumerate(windows):
        print(f"  Window {i+1}: {w.train_start.date()} to {w.test_end.date()} "
              f"(Train: {len(w.train_indices)} days, Val: {len(w.val_indices)} days, "
              f"Test: {len(w.test_indices)} days)")
    
    return windows
