#!/usr/bin/env python3
"""
Create fixed-grid volatility surface using IV with cubic spline interpolation.

Process:
1. Combine SPX_std_option.csv and SPX_surfaces.csv
2. Filter: dates >= 2009-01-01, common to both files, exclude 10-day maturities
3. Keep only OTM options (calls for K>F, puts for K<F)
4. Convert to log-moneyness
5. Stay in IV (implied volatility) - no conversion to total variance
6. For each day: use cubic spline interpolation to create smooth IV surface
7. Sample onto fixed grid: 20 tau × 21 logm = 420 points
8. Output: SPX_IV_fixed_grid.csv

Fixed grid:
- tau: 0.1 to 2.0 years, evenly spaced (0.1 steps) = 20 points
- log_moneyness: -0.5 to 0.5, evenly spaced (0.05 steps) = 21 points
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')


def process_std_options(common_dates):
    """Process standard options file - keep IV, not total variance"""
    print("Processing standard options...")
    
    chunks = []
    chunk_size = 100000
    
    for chunk in pd.read_csv('SPX_std_option.csv', chunksize=chunk_size, low_memory=False):
        # Filter: dates from 2009-01-01 onwards and in common_dates
        chunk = chunk[chunk['date'] >= '2009-01-01'].copy()
        if len(chunk) == 0:
            continue
        chunk = chunk[chunk['date'].isin(common_dates)].copy()
        if len(chunk) == 0:
            continue
        
        # Exclude 10-day maturities
        chunk = chunk[chunk['days'] > 10].copy()
        if len(chunk) == 0:
            continue
            
        # Convert days to years (tau)
        chunk['tau'] = chunk['days'] / 365.25
        
        # Calculate log-moneyness: m = log(K/F)
        valid = (chunk['forward_price'].notna() & 
                 chunk['strike_price'].notna() & 
                 (chunk['forward_price'] > 0) & 
                 (chunk['strike_price'] > 0))
        
        chunk.loc[valid, 'log_moneyness'] = np.log(
            chunk.loc[valid, 'strike_price'] / chunk.loc[valid, 'forward_price']
        )
        
        # Keep IV (implied volatility) - no conversion to total variance
        valid_iv = chunk['impl_volatility'].notna() & (chunk['impl_volatility'] > 0)
        
        # Identify OTM options (or ATM for std_option which only has ATM)
        # Note: Using tolerance for ATM check to handle floating point precision
        chunk['is_otm'] = (
            ((chunk['cp_flag'] == 'C') & (chunk['log_moneyness'] > 0)) |
            ((chunk['cp_flag'] == 'P') & (chunk['log_moneyness'] < 0)) |
            (np.abs(chunk['log_moneyness']) < 1e-10)  # Include ATM for std_option (with tolerance)
        )
        
        # Keep only valid OTM/ATM records with IV
        chunk = chunk[
            chunk['is_otm'] & 
            chunk['log_moneyness'].notna() & 
            valid_iv
        ]
        
        if len(chunk) > 0:
            chunks.append(chunk[['date', 'tau', 'log_moneyness', 'impl_volatility', 
                                'cp_flag', 'forward_price', 'strike_price']])
    
    if chunks:
        result = pd.concat(chunks, ignore_index=True)
        result['source'] = 'std_option'
        return result
    return pd.DataFrame()


def process_surfaces(std_options, common_dates):
    """Process surfaces file - keep IV, not total variance"""
    print("Processing surfaces...")
    
    # Get forward prices lookup from std_options if available
    if len(std_options) > 0:
        forward_lookup = std_options[['date', 'tau', 'forward_price']].drop_duplicates()
        forward_lookup['days'] = (forward_lookup['tau'] * 365.25).round().astype(int)
        forward_lookup = forward_lookup[['date', 'days', 'forward_price']].drop_duplicates()
    else:
        forward_lookup = pd.DataFrame()
    
    chunks = []
    chunk_size = 100000
    
    for chunk in pd.read_csv('SPX_surfaces.csv', chunksize=chunk_size, low_memory=False):
        # Filter: dates from 2009-01-01 onwards and in common_dates
        chunk = chunk[chunk['date'] >= '2009-01-01'].copy()
        if len(chunk) == 0:
            continue
        chunk = chunk[chunk['date'].isin(common_dates)].copy()
        if len(chunk) == 0:
            continue
        
        # Exclude 10-day maturities
        chunk = chunk[chunk['days'] > 10].copy()
        if len(chunk) == 0:
            continue
            
        # Convert days to years
        chunk['tau'] = chunk['days'] / 365.25
        
        # Merge forward prices if available
        if len(forward_lookup) > 0:
            chunk = chunk.merge(forward_lookup, on=['date', 'days'], how='inner')
        else:
            chunk['forward_price'] = np.nan
        
        # Calculate log-moneyness from implied strike
        valid = (chunk['forward_price'].notna() & 
                 chunk['impl_strike'].notna() & 
                 (chunk['forward_price'] > 0) & 
                 (chunk['impl_strike'] > 0))
        
        chunk.loc[valid, 'log_moneyness'] = np.log(
            chunk.loc[valid, 'impl_strike'] / chunk.loc[valid, 'forward_price']
        )
        
        # Keep IV (implied volatility) - no conversion to total variance
        valid_iv = chunk['impl_volatility'].notna() & (chunk['impl_volatility'] > 0)
        
        # Identify OTM (excludes ATM - OTM-only approach)
        # Note: ATM (log_moneyness ≈ 0) is intentionally excluded for surfaces.csv
        chunk['is_otm'] = (
            ((chunk['cp_flag'] == 'C') & (chunk['log_moneyness'] > 0)) |
            ((chunk['cp_flag'] == 'P') & (chunk['log_moneyness'] < 0))
        )
        
        # Keep only valid OTM records with IV
        chunk = chunk[
            chunk['is_otm'] & 
            chunk['log_moneyness'].notna() & 
            valid_iv
        ]
        
        if len(chunk) > 0:
            chunks.append(chunk[['date', 'tau', 'log_moneyness', 'impl_volatility',
                                'cp_flag', 'forward_price', 'impl_strike']])
    
    if chunks:
        result = pd.concat(chunks, ignore_index=True)
        result['source'] = 'surfaces'
        return result
    return pd.DataFrame()


def get_common_dates():
    """Get dates that exist in both files from 2009-01-01 onwards"""
    print("Finding common dates in both files (from 2009-01-01)...")
    
    std_dates = pd.read_csv('SPX_std_option.csv', usecols=['date'], low_memory=False)['date'].unique()
    std_dates = set(std_dates)
    
    surf_dates = pd.read_csv('SPX_surfaces.csv', usecols=['date'], low_memory=False)['date'].unique()
    surf_dates = set(surf_dates)
    
    common = std_dates & surf_dates
    common_2009_plus = sorted([d for d in common if d >= '2009-01-01'])
    
    print(f"  Standard options dates: {len(std_dates):,}")
    print(f"  Surfaces dates: {len(surf_dates):,}")
    print(f"  Common dates (2009-01-01+): {len(common_2009_plus):,}")
    if common_2009_plus:
        print(f"  Date range: {common_2009_plus[0]} to {common_2009_plus[-1]}")
    
    return set(common_2009_plus)


def create_fixed_grid():
    """Create fixed grid: tau × log_moneyness"""
    # Tau: 0.1 to 2.0 years, evenly spaced (0.1 steps) = 20 points
    tau_grid = np.arange(0.1, 2.1, 0.1)
    tau_grid = np.round(tau_grid, 1)  # Round to 1 decimal place to avoid floating point errors
    
    # Log-moneyness: -0.5 to 0.5, evenly spaced (0.05 steps) = 21 points
    logm_grid = np.arange(-0.5, 0.55, 0.05)
    logm_grid = np.round(logm_grid, 2)  # Round to 2 decimal places to avoid floating point errors
    
    # Create meshgrid
    Tau, LogM = np.meshgrid(tau_grid, logm_grid, indexing='ij')
    
    return tau_grid, logm_grid, Tau, LogM


def interpolate_to_fixed_grid(day_data, tau_grid, logm_grid, Tau, LogM):
    """
    Interpolate day's IV data to fixed grid using cubic spline interpolation.
    
    Parameters:
    -----------
    day_data : pd.DataFrame
        Columns: tau, log_moneyness, impl_volatility
    tau_grid : np.ndarray
        Target tau grid
    logm_grid : np.ndarray
        Target log-moneyness grid
    Tau, LogM : np.ndarray
        Meshgrids for target grid
        
    Returns:
    --------
    iv_grid : np.ndarray, shape (len(tau_grid), len(logm_grid))
        Interpolated IV values on fixed grid
    reconstruction_rmse : float
        RMSE when reconstructing original observed points from interpolated grid
    """
    if len(day_data) < 3:
        # Not enough points for cubic interpolation, use linear
        method = 'linear'
    else:
        method = 'cubic'
    
    # Extract observed points
    points = day_data[['tau', 'log_moneyness']].values
    values = day_data['impl_volatility'].values
    
    # Target grid points
    grid_points = np.column_stack([Tau.flatten(), LogM.flatten()])
    
    # Interpolate
    iv_flat = griddata(points, values, grid_points, method=method, fill_value=np.nan)
    
    # Reshape to grid
    iv_grid = iv_flat.reshape(len(tau_grid), len(logm_grid))
    
    # Handle NaN values (extrapolation regions) with linear interpolation
    if np.any(np.isnan(iv_grid)):
        iv_flat_linear = griddata(points, values, grid_points, method='linear', fill_value=np.nan)
        iv_grid_linear = iv_flat_linear.reshape(len(tau_grid), len(logm_grid))
        
        # Fill NaN with linear interpolation
        nan_mask = np.isnan(iv_grid)
        iv_grid[nan_mask] = iv_grid_linear[nan_mask]
        
        # If still NaN, use nearest neighbor
        if np.any(np.isnan(iv_grid)):
            iv_flat_nearest = griddata(points, values, grid_points, method='nearest')
            iv_grid_nearest = iv_flat_nearest.reshape(len(tau_grid), len(logm_grid))
            nan_mask = np.isnan(iv_grid)
            iv_grid[nan_mask] = iv_grid_nearest[nan_mask]
    
    # Compute reconstruction RMSE: interpolate back to original observed points
    # This measures how well the interpolation preserves the original data
    if len(day_data) > 0:
        # Interpolate from fixed grid back to original observed points
        reconstructed = griddata(
            grid_points, iv_flat, points, 
            method='linear', fill_value=np.nan
        )
        
        # Compute RMSE (only for points that were successfully reconstructed)
        valid_mask = ~np.isnan(reconstructed)
        if valid_mask.sum() > 0:
            reconstruction_rmse = np.sqrt(
                np.mean((reconstructed[valid_mask] - values[valid_mask]) ** 2)
            )
        else:
            reconstruction_rmse = np.nan
    else:
        reconstruction_rmse = np.nan
    
    return iv_grid, reconstruction_rmse


def main():
    print("=" * 70)
    print("Creating Fixed-Grid IV Surface with Cubic Spline Interpolation")
    print("=" * 70)
    print("Process:")
    print("  1. Combine SPX_std_option.csv and SPX_surfaces.csv")
    print("  2. Filter: dates >= 2009-01-01, common to both, exclude 10-day maturities")
    print("  3. Keep OTM options only")
    print("  4. Stay in IV (no conversion to total variance)")
    print("  5. Cubic spline interpolation to fixed grid")
    print("  6. Fixed grid: 20 tau × 21 logm = 420 points")
    print("=" * 70)
    
    # Get common dates
    common_dates = get_common_dates()
    
    if len(common_dates) == 0:
        print("ERROR: No common dates found!")
        return
    
    # Process standard options
    std_data = process_std_options(common_dates)
    print(f"\nStandard options (OTM, days > 10): {len(std_data):,} records")
    
    # Process surfaces
    surf_data = process_surfaces(std_data, common_dates)
    print(f"Surfaces (OTM, days > 10): {len(surf_data):,} records")
    
    # Combine
    print("\nCombining datasets...")
    combined = pd.concat([std_data, surf_data], ignore_index=True)
    
    # Remove duplicates (keep std_option if both exist)
    combined = combined.sort_values('source').drop_duplicates(
        subset=['date', 'tau', 'log_moneyness'], 
        keep='first'
    )
    
    print(f"Combined raw data: {len(combined):,} records")
    print(f"  Unique dates: {combined['date'].nunique():,}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Tau range: {combined['tau'].min():.4f} to {combined['tau'].max():.4f} years")
    print(f"  Log-moneyness range: {combined['log_moneyness'].min():.4f} to {combined['log_moneyness'].max():.4f}")
    print(f"  IV range: {combined['impl_volatility'].min():.4f} to {combined['impl_volatility'].max():.4f}")
    
    # Create fixed grid
    print("\nCreating fixed grid...")
    tau_grid, logm_grid, Tau, LogM = create_fixed_grid()
    print(f"  Tau grid: {len(tau_grid)} points ({tau_grid[0]:.1f} to {tau_grid[-1]:.1f} years)")
    print(f"  Log-moneyness grid: {len(logm_grid)} points ({logm_grid[0]:.2f} to {logm_grid[-1]:.2f})")
    print(f"  Total grid points: {len(tau_grid) * len(logm_grid)} = 420")
    
    # Interpolate each day to fixed grid
    print("\nInterpolating to fixed grid (cubic spline)...")
    fixed_grid_data = []
    reconstruction_errors = []
    
    unique_dates = sorted(combined['date'].unique())
    n_dates = len(unique_dates)
    
    for i, date in enumerate(unique_dates):
        if (i + 1) % 100 == 0:
            print(f"  Processing date {i+1}/{n_dates}: {date}")
        
        day_data = combined[combined['date'] == date].copy()
        
        if len(day_data) < 3:
            print(f"  Warning: Date {date} has only {len(day_data)} points, skipping")
            continue
        
        # Interpolate to fixed grid
        iv_grid, recon_rmse = interpolate_to_fixed_grid(day_data, tau_grid, logm_grid, Tau, LogM)
        
        if not np.isnan(recon_rmse):
            reconstruction_errors.append(recon_rmse)
        
        # Create DataFrame for this day
        for tau_idx, tau_val in enumerate(tau_grid):
            for logm_idx, logm_val in enumerate(logm_grid):
                fixed_grid_data.append({
                    'date': date,
                    'tau': round(tau_val, 1),  # Round to 1 decimal place
                    'log_moneyness': round(logm_val, 2),  # Round to 2 decimal places (0.05 steps)
                    'implied_volatility': iv_grid[tau_idx, logm_idx]
                })
    
    # Create final DataFrame
    fixed_grid_df = pd.DataFrame(fixed_grid_data)
    
    # Sort: date -> tau -> log_moneyness
    fixed_grid_df = fixed_grid_df.sort_values(['date', 'tau', 'log_moneyness'])
    fixed_grid_df = fixed_grid_df.reset_index(drop=True)
    
    print(f"\nFixed grid dataset: {len(fixed_grid_df):,} records")
    print(f"  Unique dates: {fixed_grid_df['date'].nunique():,}")
    print(f"  Points per day: {len(fixed_grid_df) // fixed_grid_df['date'].nunique()}")
    print(f"  IV range: {fixed_grid_df['implied_volatility'].min():.4f} to {fixed_grid_df['implied_volatility'].max():.4f}")
    
    # Check for NaN values
    nan_count = fixed_grid_df['implied_volatility'].isna().sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN values in interpolated data ({nan_count/len(fixed_grid_df)*100:.2f}%)")
    
    # Report interpolation reconstruction error
    if len(reconstruction_errors) > 0:
        recon_rmse_mean = np.mean(reconstruction_errors)
        recon_rmse_median = np.median(reconstruction_errors)
        recon_rmse_std = np.std(reconstruction_errors)
        recon_rmse_p95 = np.percentile(reconstruction_errors, 95)
        recon_rmse_p99 = np.percentile(reconstruction_errors, 99)
        
        print(f"\n{'='*70}")
        print("CUBIC SPLINE INTERPOLATION RECONSTRUCTION ERROR")
        print(f"{'='*70}")
        print(f"  Days with reconstruction error: {len(reconstruction_errors)}/{n_dates}")
        print(f"  Mean IV RMSE: {recon_rmse_mean:.6f}")
        print(f"  Median IV RMSE: {recon_rmse_median:.6f}")
        print(f"  Std IV RMSE: {recon_rmse_std:.6f}")
        print(f"  95th percentile: {recon_rmse_p95:.6f}")
        print(f"  99th percentile: {recon_rmse_p99:.6f}")
        print(f"\n  Interpretation:")
        print(f"    This measures how well cubic spline interpolation reconstructs")
        print(f"    the original observed IV values. Lower is better.")
        print(f"    If this is high (>0.01), interpolation may be losing information.")
        print(f"{'='*70}")
    
    # Save
    output_file = 'SPX_IV_fixed_grid.csv'
    fixed_grid_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")
    
    print("\nSample data:")
    print(fixed_grid_df.head(10))
    
    return fixed_grid_df


if __name__ == '__main__':
    try:
        result = main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
