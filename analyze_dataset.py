#!/usr/bin/env python3
"""
Comprehensive dataset analysis for IV surface forecasting.

Analyzes:
1. Temporal trends (mean/std evolution over time)
2. Surface region analysis (which tau/logm regions are most variable)
3. Day-of-week effects
4. Predictability metrics (autocorrelation, variance)
5. Learnability assessment (which parts are easier/harder to predict)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecasting.data_loader import load_data


def compute_daily_statistics(data: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute mean, std, min, max for each day's surface."""
    n_dates, n_tau, n_logm = data.shape
    
    daily_stats = []
    for i in range(n_dates):
        surface = data[i]
        valid = ~np.isnan(surface)
        
        if valid.sum() == 0:
            continue
            
        daily_stats.append({
            'date': dates[i],
            'mean': np.nanmean(surface),
            'std': np.nanstd(surface),
            'min': np.nanmin(surface),
            'max': np.nanmax(surface),
            'median': np.nanmedian(surface),
            'q25': np.nanpercentile(surface, 25),
            'q75': np.nanpercentile(surface, 75),
            'valid_points': valid.sum(),
            'total_points': surface.size
        })
    
    return pd.DataFrame(daily_stats)


def analyze_surface_regions(data: np.ndarray, tau_grid: np.ndarray, 
                           logm_grid: np.ndarray) -> Dict:
    """Analyze variability and statistics by tau and log-moneyness regions."""
    n_dates, n_tau, n_logm = data.shape
    
    # Per-tau statistics
    tau_stats = []
    for i, tau in enumerate(tau_grid):
        tau_data = data[:, i, :]  # (n_dates, n_logm)
        valid = ~np.isnan(tau_data)
        
        tau_stats.append({
            'tau': tau,
            'mean': np.nanmean(tau_data),
            'std': np.nanstd(tau_data),
            'mean_std': np.nanmean(np.nanstd(tau_data, axis=1)),  # Mean of daily stds
            'temporal_std': np.nanstd(np.nanmean(tau_data, axis=1)),  # Std of daily means
            'valid_ratio': valid.sum() / tau_data.size
        })
    
    # Per-logm statistics
    logm_stats = []
    for j, logm in enumerate(logm_grid):
        logm_data = data[:, :, j]  # (n_dates, n_tau)
        valid = ~np.isnan(logm_data)
        
        logm_stats.append({
            'logm': logm,
            'mean': np.nanmean(logm_data),
            'std': np.nanstd(logm_data),
            'mean_std': np.nanmean(np.nanstd(logm_data, axis=1)),
            'temporal_std': np.nanstd(np.nanmean(logm_data, axis=1)),
            'valid_ratio': valid.sum() / logm_data.size
        })
    
    # Per-point statistics (full surface)
    point_stats = []
    for i in range(n_tau):
        for j in range(n_logm):
            point_series = data[:, i, j]
            valid = ~np.isnan(point_series)
            
            if valid.sum() < 10:  # Skip if too few valid points
                continue
            
            point_stats.append({
                'tau_idx': i,
                'logm_idx': j,
                'tau': tau_grid[i],
                'logm': logm_grid[j],
                'mean': np.nanmean(point_series),
                'std': np.nanstd(point_series),
                'cv': np.nanstd(point_series) / (np.nanmean(point_series) + 1e-8),  # Coefficient of variation
                'autocorr_1': self_autocorr(point_series[valid], lag=1) if valid.sum() > 1 else np.nan,
                'autocorr_5': self_autocorr(point_series[valid], lag=5) if valid.sum() > 5 else np.nan,
                'autocorr_21': self_autocorr(point_series[valid], lag=21) if valid.sum() > 21 else np.nan,
            })
    
    return {
        'tau_stats': pd.DataFrame(tau_stats),
        'logm_stats': pd.DataFrame(logm_stats),
        'point_stats': pd.DataFrame(point_stats)
    }


def self_autocorr(x: np.ndarray, lag: int = 1) -> float:
    """Compute autocorrelation at given lag."""
    if len(x) <= lag:
        return np.nan
    x_centered = x - np.mean(x)
    if np.std(x_centered) == 0:
        return np.nan
    corr = np.corrcoef(x_centered[:-lag], x_centered[lag:])[0, 1]
    return corr if not np.isnan(corr) else 0.0


def analyze_day_of_week_effects(data: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Analyze day-of-week patterns."""
    daily_stats = compute_daily_statistics(data, dates)
    daily_stats['dow'] = daily_stats['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    daily_stats['dow_name'] = daily_stats['date'].dt.day_name()
    
    dow_summary = daily_stats.groupby('dow').agg({
        'mean': ['mean', 'std', 'count'],
        'std': ['mean', 'std'],
    }).round(6)
    
    return daily_stats, dow_summary


def analyze_persistence_by_dow(data: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Analyze persistence RMSE by day of week and transitions.
    Persistence: predict day t using day t-1.
    """
    n_dates, n_tau, n_logm = data.shape
    
    persistence_errors = []
    
    for i in range(1, n_dates):  # Start from 1 since we need previous day
        prev_surface = data[i-1]  # Day t-1
        curr_surface = data[i]     # Day t (target)
        
        # Skip if either has too many NaNs
        prev_valid = ~np.isnan(prev_surface)
        curr_valid = ~np.isnan(curr_surface)
        both_valid = prev_valid & curr_valid
        
        if both_valid.sum() < 100:  # Need enough valid points
            continue
        
        # Persistence prediction: use previous day
        error = curr_surface - prev_surface
        rmse = np.sqrt(np.nanmean(error[both_valid] ** 2))
        
        # Get day of week info
        try:
            curr_date = dates[i]
            prev_date = dates[i-1]
            
            # Calculate gap (weekend = 3 days, normal = 1 day)
            gap_days = (curr_date - prev_date).days
        except (IndexError, TypeError, AttributeError) as e:
            # Skip if date access fails
            continue
        
        persistence_errors.append({
            'date': curr_date,
            'prev_date': prev_date,
            'dow': curr_date.dayofweek,
            'dow_name': curr_date.day_name(),
            'prev_dow': prev_date.dayofweek,
            'prev_dow_name': prev_date.day_name(),
            'gap_days': gap_days,
            'is_weekend': gap_days > 1,  # Weekend transition
            'transition': f"{prev_date.day_name()[:3]}->{curr_date.day_name()[:3]}",
            'rmse': rmse
        })
    
    return pd.DataFrame(persistence_errors)


def analyze_predictability(data: np.ndarray, tau_grid: np.ndarray, 
                          logm_grid: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Analyze predictability metrics for each surface point."""
    n_dates, n_tau, n_logm = data.shape
    
    predictability = []
    
    for i in range(n_tau):
        for j in range(n_logm):
            series = data[:, i, j]
            valid = ~np.isnan(series)
            
            if valid.sum() < 50:  # Need enough data
                continue
            
            series_clean = series[valid]
            
            # Persistence error (how much does it change day-to-day?)
            if len(series_clean) > 1:
                daily_changes = np.diff(series_clean)
                persistence_rmse = np.sqrt(np.mean(daily_changes ** 2))
            else:
                persistence_rmse = np.nan
            
            # Autocorrelations at different lags
            ac1 = self_autocorr(series_clean, lag=1)
            ac5 = self_autocorr(series_clean, lag=5)
            ac21 = self_autocorr(series_clean, lag=21)
            
            # Trend strength (linear trend)
            if len(series_clean) > 10:
                x = np.arange(len(series_clean))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series_clean)
                trend_strength = abs(r_value)
            else:
                trend_strength = np.nan
            
            # Volatility clustering (GARCH-like: high vol followed by high vol)
            if len(series_clean) > 10:
                returns = np.diff(series_clean) / (series_clean[:-1] + 1e-8)
                vol_clustering = self_autocorr(np.abs(returns), lag=1) if len(returns) > 1 else np.nan
            else:
                vol_clustering = np.nan
            
            predictability.append({
                'tau': tau_grid[i],
                'logm': logm_grid[j],
                'tau_idx': i,
                'logm_idx': j,
                'mean': np.nanmean(series_clean),
                'std': np.nanstd(series_clean),
                'persistence_rmse': persistence_rmse,
                'autocorr_1': ac1,
                'autocorr_5': ac5,
                'autocorr_21': ac21,
                'trend_strength': trend_strength,
                'vol_clustering': vol_clustering,
                'learnability_score': (ac1 + ac5) / 2 if not np.isnan(ac1) and not np.isnan(ac5) else np.nan
            })
    
    return pd.DataFrame(predictability)


def create_visualizations(data: np.ndarray, dates: pd.DatetimeIndex,
                         tau_grid: np.ndarray, logm_grid: np.ndarray,
                         output_dir: str = 'results/analysis'):
    """Create comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Daily statistics over time
    daily_stats = compute_daily_statistics(data, dates)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(daily_stats['date'], daily_stats['mean'], alpha=0.7, linewidth=0.5)
    axes[0].set_title('Daily Mean IV Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mean IV')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(daily_stats['date'], daily_stats['std'], alpha=0.7, linewidth=0.5, color='orange')
    axes[1].set_title('Daily Std IV Over Time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Std IV')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(daily_stats['date'], daily_stats['max'] - daily_stats['min'], 
                 alpha=0.7, linewidth=0.5, color='green')
    axes[2].set_title('Daily Range (Max - Min) Over Time', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Range')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/daily_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Surface region analysis
    region_stats = analyze_surface_regions(data, tau_grid, logm_grid)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean IV by tau
    axes[0, 0].plot(region_stats['tau_stats']['tau'], region_stats['tau_stats']['mean'], 'o-')
    axes[0, 0].set_title('Mean IV by Maturity (Tau)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Tau (years)')
    axes[0, 0].set_ylabel('Mean IV')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std IV by tau
    axes[0, 1].plot(region_stats['tau_stats']['tau'], region_stats['tau_stats']['std'], 'o-', color='orange')
    axes[0, 1].set_title('Std IV by Maturity (Tau)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Tau (years)')
    axes[0, 1].set_ylabel('Std IV')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean IV by log-moneyness
    axes[1, 0].plot(region_stats['logm_stats']['logm'], region_stats['logm_stats']['mean'], 'o-')
    axes[1, 0].set_title('Mean IV by Log-Moneyness', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Log-Moneyness')
    axes[1, 0].set_ylabel('Mean IV')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Std IV by log-moneyness
    axes[1, 1].plot(region_stats['logm_stats']['logm'], region_stats['logm_stats']['std'], 'o-', color='orange')
    axes[1, 1].set_title('Std IV by Log-Moneyness', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Log-Moneyness')
    axes[1, 1].set_ylabel('Std IV')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/surface_regions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Day-of-week effects
    daily_stats_dow, dow_summary = analyze_day_of_week_effects(data, dates)
    
    # 3b. Persistence RMSE by day of week
    persistence_dow = analyze_persistence_by_dow(data, dates)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = daily_stats_dow.groupby('dow_name')['mean'].agg(['mean', 'std', 'count'])
    # Only include days that exist in data
    dow_stats = dow_stats.reindex([d for d in dow_order if d in dow_stats.index])
    
    if len(dow_stats) > 0:
        axes[0].bar(range(len(dow_stats)), dow_stats['mean'], yerr=dow_stats['std'], 
                    capsize=5, alpha=0.7, color='steelblue')
        axes[0].set_xticks(range(len(dow_stats)))
        axes[0].set_xticklabels(dow_stats.index, rotation=45, ha='right')
        axes[0].set_title('Mean IV by Day of Week', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Mean IV')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        dow_std = daily_stats_dow.groupby('dow_name')['std'].mean()
        dow_std = dow_std.reindex([d for d in dow_order if d in dow_std.index])
        
        axes[1].bar(range(len(dow_std)), dow_std.values, alpha=0.7, color='orange')
        axes[1].set_xticks(range(len(dow_std)))
        axes[1].set_xticklabels(dow_std.index, rotation=45, ha='right')
        axes[1].set_title('Mean Daily Std by Day of Week', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mean Daily Std')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/day_of_week_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3b. Persistence RMSE by day of week and transitions
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Persistence RMSE by target day of week
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pers_by_dow = persistence_dow.groupby('dow_name')['rmse'].agg(['mean', 'std', 'count'])
    pers_by_dow = pers_by_dow.reindex([d for d in dow_order if d in pers_by_dow.index])
    
    axes[0, 0].bar(range(len(pers_by_dow)), pers_by_dow['mean'], 
                   yerr=pers_by_dow['std'], capsize=5, alpha=0.7, color='steelblue')
    axes[0, 0].set_xticks(range(len(pers_by_dow)))
    axes[0, 0].set_xticklabels(pers_by_dow.index, rotation=45, ha='right')
    axes[0, 0].set_title('Persistence RMSE by Target Day of Week', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Weekend vs weekday transitions
    weekend_pers = persistence_dow[persistence_dow['is_weekend']]['rmse']
    weekday_pers = persistence_dow[~persistence_dow['is_weekend']]['rmse']
    
    axes[0, 1].boxplot([weekday_pers.values, weekend_pers.values], 
                       labels=['Weekday\n(1 day gap)', 'Weekend\n(>1 day gap)'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[0, 1].set_title('Persistence RMSE: Weekend vs Weekday Transitions', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Specific transitions
    transition_stats = persistence_dow.groupby('transition')['rmse'].agg(['mean', 'std', 'count']).sort_values('mean')
    # Show top 10 most common transitions
    top_transitions = transition_stats.nlargest(10, 'count')
    
    axes[1, 0].barh(range(len(top_transitions)), top_transitions['mean'],
                    xerr=top_transitions['std'], capsize=3, alpha=0.7, color='coral')
    axes[1, 0].set_yticks(range(len(top_transitions)))
    axes[1, 0].set_yticklabels(top_transitions.index)
    axes[1, 0].set_title('Persistence RMSE by Transition (Top 10 Most Common)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('RMSE')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Gap days analysis
    gap_stats = persistence_dow.groupby('gap_days')['rmse'].agg(['mean', 'std', 'count'])
    gap_stats = gap_stats[gap_stats['count'] >= 10]  # Only show gaps with enough samples
    
    axes[1, 1].scatter(gap_stats.index, gap_stats['mean'], 
                      s=gap_stats['count']*2, alpha=0.6, color='purple')
    axes[1, 1].errorbar(gap_stats.index, gap_stats['mean'], 
                        yerr=gap_stats['std'], fmt='none', alpha=0.3, color='purple')
    axes[1, 1].set_title('Persistence RMSE by Gap Days', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Gap Days (1=consecutive, 3=weekend)')
    axes[1, 1].set_ylabel('Mean RMSE')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Consecutive')
    axes[1, 1].axvline(x=3, color='red', linestyle='--', alpha=0.5, label='Weekend')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/persistence_by_dow.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Predictability heatmaps
    predictability = analyze_predictability(data, tau_grid, logm_grid, dates)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Autocorrelation at lag 1
    ac1_matrix = predictability.pivot(index='tau', columns='logm', values='autocorr_1')
    im1 = axes[0, 0].imshow(ac1_matrix.values, aspect='auto', cmap='RdYlGn', 
                           extent=[logm_grid.min(), logm_grid.max(), tau_grid.max(), tau_grid.min()])
    axes[0, 0].set_title('Autocorrelation (lag=1) - Higher = More Predictable', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Log-Moneyness')
    axes[0, 0].set_ylabel('Tau (years)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Persistence RMSE (lower = more predictable)
    pers_matrix = predictability.pivot(index='tau', columns='logm', values='persistence_rmse')
    im2 = axes[0, 1].imshow(pers_matrix.values, aspect='auto', cmap='RdYlGn_r',
                           extent=[logm_grid.min(), logm_grid.max(), tau_grid.max(), tau_grid.min()])
    axes[0, 1].set_title('Persistence RMSE - Lower = More Predictable', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Log-Moneyness')
    axes[0, 1].set_ylabel('Tau (years)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Learnability score
    learn_matrix = predictability.pivot(index='tau', columns='logm', values='learnability_score')
    im3 = axes[1, 0].imshow(learn_matrix.values, aspect='auto', cmap='RdYlGn',
                           extent=[logm_grid.min(), logm_grid.max(), tau_grid.max(), tau_grid.min()])
    axes[1, 0].set_title('Learnability Score (avg autocorr) - Higher = More Learnable', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Log-Moneyness')
    axes[1, 0].set_ylabel('Tau (years)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Coefficient of variation
    cv_matrix = predictability.pivot(index='tau', columns='logm', values='std') / \
                predictability.pivot(index='tau', columns='logm', values='mean')
    im4 = axes[1, 1].imshow(cv_matrix.values, aspect='auto', cmap='viridis',
                           extent=[logm_grid.min(), logm_grid.max(), tau_grid.max(), tau_grid.min()])
    axes[1, 1].set_title('Coefficient of Variation (Std/Mean)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Log-Moneyness')
    axes[1, 1].set_ylabel('Tau (years)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predictability_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Summary statistics table
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\n1. TEMPORAL STATISTICS:")
    print(f"   Total days: {len(daily_stats):,}")
    print(f"   Date range: {daily_stats['date'].min().date()} to {daily_stats['date'].max().date()}")
    print(f"   Overall mean IV: {daily_stats['mean'].mean():.4f}")
    print(f"   Overall std IV: {daily_stats['std'].mean():.4f}")
    print(f"   Mean daily range: {daily_stats['max'].mean() - daily_stats['min'].mean():.4f}")
    
    print(f"\n2. SURFACE REGIONS:")
    print(f"   Tau range: {tau_grid.min():.2f} to {tau_grid.max():.2f} years ({len(tau_grid)} points)")
    print(f"   Logm range: {logm_grid.min():.2f} to {logm_grid.max():.2f} ({len(logm_grid)} points)")
    print(f"   Total surface points: {len(tau_grid) * len(logm_grid)}")
    
    print(f"\n3. DAY-OF-WEEK EFFECTS:")
    if len(dow_summary) > 0:
        print(dow_summary)
    else:
        print("   (No day-of-week data available)")
    
    print(f"\n3b. PERSISTENCE RMSE BY DAY OF WEEK:")
    if len(persistence_dow) > 0:
        pers_by_dow_summary = persistence_dow.groupby('dow_name')['rmse'].agg(['mean', 'std', 'count']).round(6)
        print(pers_by_dow_summary)
        
        # Weekend vs weekday
        weekend_rmse = persistence_dow[persistence_dow['is_weekend']]['rmse'].mean()
        weekday_rmse = persistence_dow[~persistence_dow['is_weekend']]['rmse'].mean()
        print(f"\n   Weekend transitions (Fri->Mon, etc.): RMSE = {weekend_rmse:.6f}")
        print(f"   Weekday transitions (Mon->Tue, etc.): RMSE = {weekday_rmse:.6f}")
        print(f"   Weekend is {weekend_rmse/weekday_rmse:.2f}x worse than weekday")
        
        # Specific transitions
        print(f"\n   Key Transitions:")
        key_transitions = ['Fri->Mon', 'Mon->Tue', 'Thu->Fri', 'Tue->Wed', 'Wed->Thu']
        for trans in key_transitions:
            trans_data = persistence_dow[persistence_dow['transition'] == trans]
            if len(trans_data) > 0:
                print(f"     {trans}: RMSE = {trans_data['rmse'].mean():.6f} (n={len(trans_data)})")
    else:
        print("   (No persistence data available)")
    
    print(f"\n4. PREDICTABILITY INSIGHTS:")
    print(f"   Mean autocorr (lag=1): {predictability['autocorr_1'].mean():.4f}")
    print(f"   Mean autocorr (lag=5): {predictability['autocorr_5'].mean():.4f}")
    print(f"   Mean autocorr (lag=21): {predictability['autocorr_21'].mean():.4f}")
    print(f"   Mean persistence RMSE: {predictability['persistence_rmse'].mean():.4f}")
    
    # Most/least learnable regions
    top_learnable = predictability.nlargest(10, 'learnability_score')
    print(f"\n   Top 10 Most Learnable Points:")
    for _, row in top_learnable.iterrows():
        print(f"     tau={row['tau']:.2f}, logm={row['logm']:.2f}, score={row['learnability_score']:.4f}")
    
    bottom_learnable = predictability.nsmallest(10, 'learnability_score')
    print(f"\n   Bottom 10 Least Learnable Points:")
    for _, row in bottom_learnable.iterrows():
        print(f"     tau={row['tau']:.2f}, logm={row['logm']:.2f}, score={row['learnability_score']:.4f}")
    
    # Save dataframes
    daily_stats.to_csv(f'{output_dir}/daily_statistics.csv', index=False)
    region_stats['tau_stats'].to_csv(f'{output_dir}/tau_statistics.csv', index=False)
    region_stats['logm_stats'].to_csv(f'{output_dir}/logm_statistics.csv', index=False)
    predictability.to_csv(f'{output_dir}/predictability_analysis.csv', index=False)
    daily_stats_dow.to_csv(f'{output_dir}/daily_stats_with_dow.csv', index=False)
    persistence_dow.to_csv(f'{output_dir}/persistence_by_dow.csv', index=False)
    
    print(f"\n5. OUTPUTS SAVED:")
    print(f"   Plots: {output_dir}/")
    print(f"   CSV files: {output_dir}/")
    print("=" * 70)


def main():
    print("=" * 70)
    print("COMPREHENSIVE IV SURFACE DATASET ANALYSIS")
    print("=" * 70)
    
    # Load data
    data, tau_grid, logm_grid, dates = load_data('SPX_IV_fixed_grid.csv')
    
    print(f"\nData loaded: {data.shape}")
    print(f"Missing data: {np.isnan(data).sum():,} points ({np.isnan(data).sum() / data.size * 100:.2f}%)")
    
    # Create visualizations and analysis
    create_visualizations(data, dates, tau_grid, logm_grid, output_dir='results/analysis')
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
