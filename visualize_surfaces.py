#!/usr/bin/env python3
"""
Visualize raw option points and interpolated volatility surfaces.
Selects one random date and compares two surface files side by side.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import re

# Configure matplotlib to use LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


def load_surface_data(filepath):
    """Load interpolated surface data from CSV"""
    print(f"Loading surface data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    iv_columns = [col for col in df.columns if col.startswith('iv_')]
    pattern = re.compile(r'iv_([\d.]+)_([\d.]+)')
    
    moneyness_values = []
    tau_values = []
    
    for col in iv_columns:
        match = pattern.match(col)
        if match:
            moneyness = float(match.group(1))
            tau = float(match.group(2))
            moneyness_values.append(moneyness)
            tau_values.append(tau)
    
    moneyness_grid = np.sort(np.unique(moneyness_values))
    tau_grid = np.sort(np.unique(tau_values))
    
    return df, moneyness_grid, tau_grid


def load_raw_options(options_file, price_file, date):
    """Load raw option points for a specific date with OTM filtering (same as preprocessing)"""
    print(f"Loading raw option data for {date.date()}...")
    
    price_df = pd.read_csv(price_file)
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_dict = dict(zip(price_df['date'], (price_df['high'] + price_df['low']) / 2))
    
    chunks = []
    chunk_size = 1000000
    for chunk in pd.read_csv(options_file, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
    options_df = pd.concat(chunks, ignore_index=True)
    
    options_df['date'] = pd.to_datetime(options_df['date'])
    options_df['exdate'] = pd.to_datetime(options_df['exdate'])
    day_options = options_df[options_df['date'] == date].copy()
    
    if len(day_options) == 0:
        return None, None, None, None, None, None
    
    if date not in price_dict:
        return None, None, None, None, None, None
    
    underlying_price = price_dict[date]
    
    day_options['strike'] = day_options['strike_price'] / 1000.0
    day_options['moneyness'] = day_options['strike'] / underlying_price
    day_options['tau'] = (day_options['exdate'] - day_options['date']).dt.days / 365.0
    
    ttm_filter = (day_options['tau'] > 0) & (day_options['tau'] <= 1.5)
    day_options = day_options[ttm_filter].copy()
    
    moneyness_filter = (day_options['moneyness'] >= 0.85) & (day_options['moneyness'] <= 1.15)
    day_options = day_options[moneyness_filter].copy()
    
    day_options = day_options[day_options['volume'] > 0].copy()
    
    valid_mask = ~pd.isna(day_options['impl_volatility'])
    day_options = day_options[valid_mask].copy()
    
    atm_threshold = 0.01
    
    calls = day_options[day_options['cp_flag'] == 'C'].copy()
    puts = day_options[day_options['cp_flag'] == 'P'].copy()
    
    calls_otm = calls[calls['moneyness'] > 1.0].copy()
    calls_atm = calls[np.abs(calls['moneyness'] - 1.0) < atm_threshold].copy()
    calls_filtered = pd.concat([calls_otm, calls_atm]).drop_duplicates()
    
    puts_otm = puts[puts['moneyness'] < 1.0].copy()
    puts_atm = puts[np.abs(puts['moneyness'] - 1.0) < atm_threshold].copy()
    puts_filtered = pd.concat([puts_otm, puts_atm]).drop_duplicates()
    
    calls_moneyness = calls_filtered['moneyness'].values
    calls_tau = calls_filtered['tau'].values
    calls_iv = calls_filtered['impl_volatility'].values
    
    puts_moneyness = puts_filtered['moneyness'].values
    puts_tau = puts_filtered['tau'].values
    puts_iv = puts_filtered['impl_volatility'].values
    
    return calls_moneyness, calls_tau, calls_iv, puts_moneyness, puts_tau, puts_iv


def extract_surface_data(df, date, moneyness_grid, tau_grid):
    """Extract surface data for a given date"""
    date_row = df[df['date'] == date]
    if len(date_row) == 0:
        print(f"Warning: Date {date.date()} not found in surface data. Using closest date.")
        date_row = df.iloc[(df['date'] - date).abs().argsort()[:1]]
        date = date_row['date'].values[0]
        date = pd.to_datetime(date)
    
    surface_data = np.zeros((len(tau_grid), len(moneyness_grid)))
    
    iv_columns = [col for col in date_row.columns if col.startswith('iv_')]
    pattern = re.compile(r'iv_([\d.]+)_([\d.]+)')
    
    col_map = {}
    for col in iv_columns:
        match = pattern.match(col)
        if match:
            m = float(match.group(1))
            t = float(match.group(2))
            col_map[(m, t)] = col
    
    for i, tau in enumerate(tau_grid):
        for j, moneyness in enumerate(moneyness_grid):
            if (moneyness, tau) in col_map:
                col_name = col_map[(moneyness, tau)]
                surface_data[i, j] = date_row[col_name].values[0]
    
    return surface_data, date


def plot_single_surface(ax, surface_data, moneyness_grid, tau_grid, calls_m, calls_t, calls_iv, 
                        puts_m, puts_t, puts_iv, date, title_suffix=None):
    """Plot a single surface on given axes"""
    M, T = np.meshgrid(moneyness_grid, tau_grid)
    
    ax.plot_surface(M, T, surface_data, alpha=0.5, cmap='viridis', edgecolor='none', linewidth=0.1)
    
    if calls_m is not None and len(calls_m) > 0:
        ax.scatter(calls_m, calls_t, calls_iv, c='red', s=30, alpha=0.8, 
                  label='Calls', edgecolors='darkred', linewidths=0.5)
    
    if puts_m is not None and len(puts_m) > 0:
        ax.scatter(puts_m, puts_t, puts_iv, c='blue', s=30, alpha=0.8,
                  label='Puts', edgecolors='darkblue', linewidths=0.5)
    
    ax.set_xlabel(r'Moneyness $(K/S)$', fontsize=16)
    ax.set_ylabel(r'Time to Maturity (years)', fontsize=16)
    ax.set_zlabel(r'Implied Volatility', fontsize=16)
    title = str(pd.to_datetime(date).date())
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.view_init(elev=25, azim=45)
    
    ax.legend(loc='upper left', fontsize=14)


def plot_surface_file_comparison(surface_file_left, surface_file_right, options_file, price_file, date, output_file):
    df_l, m_grid_l, t_grid_l = load_surface_data(surface_file_left)
    df_r, m_grid_r, t_grid_r = load_surface_data(surface_file_right)
    
    fig = plt.figure(figsize=(22, 9))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.0)  # No spacing between plots
    
    calls_m, calls_t, calls_iv, puts_m, puts_t, puts_iv = load_raw_options(options_file, price_file, date)
    
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surface_l, actual_date_l = extract_surface_data(df_l, date, m_grid_l, t_grid_l)
    plot_single_surface(
        ax1,
        surface_l,
        m_grid_l,
        t_grid_l,
        calls_m,
        calls_t,
        calls_iv,
        puts_m,
        puts_t,
        puts_iv,
        actual_date_l,
        title_suffix=os.path.basename(surface_file_left),
    )
    
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    surface_r, actual_date_r = extract_surface_data(df_r, date, m_grid_r, t_grid_r)
    plot_single_surface(
        ax2,
        surface_r,
        m_grid_r,
        t_grid_r,
        calls_m,
        calls_t,
        calls_iv,
        puts_m,
        puts_t,
        puts_iv,
        actual_date_r,
        title_suffix=os.path.basename(surface_file_right),
    )
    
    plt.tight_layout(pad=-10)  # Extra tight spacing
    
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully!")
    plt.show()

def _resolve_prev_surface_path():
    candidates = ["SPX_surfaces_prev.csv", "SPX_surfaces.prev.csv"]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def main():
    surface_file_left = 'SPX_surfaces.csv'
    surface_file_right = _resolve_prev_surface_path()
    options_file = os.path.join('data_prep', 'SPX_options.csv')
    price_file = os.path.join('data_prep', 'SPX_price.csv')
    
    if not os.path.exists(surface_file_left):
        print(f"Error: {surface_file_left} not found!")
        return
    if not os.path.exists(surface_file_right):
        print(f"Error: {surface_file_right} not found!")
        return
    
    if not os.path.exists(options_file):
        print(f"Error: {options_file} not found!")
        return
    
    if not os.path.exists(price_file):
        print(f"Error: {price_file} not found!")
        return
    
    print("Loading surface data to find available dates...")
    df_l, _, _ = load_surface_data(surface_file_left)
    df_r, _, _ = load_surface_data(surface_file_right)
    dates_l = set(pd.to_datetime(df_l['date']).dt.normalize().unique())
    dates_r = set(pd.to_datetime(df_r['date']).dt.normalize().unique())
    common_dates = np.array(sorted(dates_l.intersection(dates_r)))
    
    common_dates = np.array([d for d in common_dates if pd.to_datetime(d).year == 2009])
    
    if len(common_dates) < 1:
        print("Error: No common dates found between the two surface files (from 2009 onwards).")
        return
    
    print(f"Found {len(common_dates)} common dates from 2009 onwards")
    
    rng = np.random.default_rng()
    date = pd.to_datetime(rng.choice(common_dates))
    
    print(f"\nSelected random date: {date.date()}")
    
    date_str = date.strftime("%Y%m%d")
    output_file = f'surface_comp_{date_str}_current_vs_prev.pdf'
    
    plot_surface_file_comparison(
        surface_file_left=surface_file_left,
        surface_file_right=surface_file_right,
        options_file=options_file,
        price_file=price_file,
        date=date,
        output_file=output_file,
    )
    
    print(f"\nVisualization complete! Output saved to: {output_file}")


if __name__ == "__main__":
    main()
