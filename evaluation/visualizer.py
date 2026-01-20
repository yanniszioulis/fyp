"""
Visualization tools for forecasting results.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import os


def load_results(results_file: str) -> pd.DataFrame:
    """
    Load model results from a JSON file into a DataFrame.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    rows = []
    for result in results:
        row = {
            'window_id': result['window_id'],
            'context_length': result['context_length'],
            'horizon': result['horizon'],
            'iv_rmse': result['metrics']['iv_rmse'],
            'relative_rmse': result['metrics']['relative_rmse'],
            'mae': result['metrics']['mae'],
            'n_samples': result['n_samples'],
            'test_start': result['test_start'],
            'test_end': result['test_end']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def load_persistence_results(results_file: str = 'results/metrics/persistence_results.json') -> pd.DataFrame:
    """Backwards-compatible wrapper for persistence results."""
    return load_results(results_file)


def plot_metrics_by_window(df: pd.DataFrame,
                           context_length: Optional[int] = None,
                           horizon: Optional[int] = None,
                           metric_type: str = 'line',
                           model_label: str = 'Model',
                           save_path: Optional[str] = None,
                           figsize: tuple = (12, 6)):
    """
    Plot error metrics by window.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    context_length : int, optional
        Filter by specific context length (if None, uses all)
    horizon : int, optional
        Filter by specific horizon (if None, uses all)
    metric_type : str
        'line' for line plot, 'bar' for bar chart
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Filter data if needed
    plot_df = df.copy()
    if context_length is not None:
        plot_df = plot_df[plot_df['context_length'] == context_length]
    if horizon is not None:
        plot_df = plot_df[plot_df['horizon'] == horizon]
    
    if len(plot_df) == 0:
        print("No data to plot after filtering")
        return
    
    # Group by window_id and compute mean if multiple configs per window
    plot_df = plot_df.groupby('window_id').agg({
        'iv_rmse': 'mean',
        'relative_rmse': 'mean',
        'mae': 'mean',
        'n_samples': 'sum'
    }).reset_index()
    
    # Sort by window_id
    plot_df = plot_df.sort_values('window_id')
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    x = plot_df['window_id']
    width = 0.25 if metric_type == 'bar' else 0
    
    if metric_type == 'bar':
        # Bar chart
        x_pos = np.arange(len(x))
        ax.bar(x_pos - width, plot_df['iv_rmse'], width, label='IV RMSE', alpha=0.8)
        ax.bar(x_pos, plot_df['relative_rmse'], width, label='Relative RMSE', alpha=0.8)
        ax.bar(x_pos + width, plot_df['mae'], width, label='MAE', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Window {w}' for w in x], rotation=45, ha='right')
    else:
        # Line plot
        ax.plot(x, plot_df['iv_rmse'], marker='o', label='IV RMSE', linewidth=2, markersize=8)
        ax.plot(x, plot_df['relative_rmse'], marker='s', label='Relative RMSE', linewidth=2, markersize=8)
        ax.plot(x, plot_df['mae'], marker='^', label='MAE', linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Window {w}' for w in x], rotation=45, ha='right')
    
    ax.set_xlabel('Window ID', fontsize=12)
    ax.set_ylabel('Error Metric Value', fontsize=12)
    
    # Title
    title_parts = [f'{model_label}: Error Metrics by Window']
    if context_length is not None:
        title_parts.append(f'Context: {context_length} days')
    if horizon is not None:
        title_parts.append(f'Horizon: {horizon} days')
    ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close figure instead of showing


def plot_all_configurations(df: pd.DataFrame,
                           save_dir: str = 'results/plots',
                           metric_type: str = 'line',
                           model_id: str = 'model',
                           model_label: str = 'Model'):
    """
    Plot metrics for all context_length and horizon combinations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    save_dir : str
        Directory to save plots
    metric_type : str
        'line' or 'bar'
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    context_lengths = sorted(df['context_length'].unique())
    horizons = sorted(df['horizon'].unique())
    
    for context_length in context_lengths:
        for horizon in horizons:
            save_path = os.path.join(
                save_dir,
                f'{model_id}_metrics_c{context_length}_h{horizon}_{metric_type}.png'
            )
            plot_metrics_by_window(
                df,
                context_length=context_length,
                horizon=horizon,
                metric_type=metric_type,
                model_label=model_label,
                save_path=save_path
            )


def plot_summary_comparison(df: pd.DataFrame,
                           model_label: str = 'Model',
                           save_path: Optional[str] = None,
                           figsize: tuple = (15, 10)):
    """
    Create a comprehensive summary plot with subplots for different configurations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    context_lengths = sorted(df['context_length'].unique())
    horizons = sorted(df['horizon'].unique())
    
    n_configs = len(context_lengths) * len(horizons)
    n_cols = len(horizons)
    n_rows = len(context_lengths)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    sns.set_style("whitegrid")
    
    for i, context_length in enumerate(context_lengths):
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            
            # Filter data
            plot_df = df[(df['context_length'] == context_length) & 
                        (df['horizon'] == horizon)].copy()
            
            if len(plot_df) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Context: {context_length}d, Horizon: {horizon}d', fontsize=10)
                continue
            
            # Group by window
            plot_df = plot_df.groupby('window_id').agg({
                'iv_rmse': 'mean',
                'relative_rmse': 'mean',
                'mae': 'mean'
            }).reset_index().sort_values('window_id')
            
            x = plot_df['window_id']
            ax.plot(x, plot_df['iv_rmse'], marker='o', label='IV RMSE', linewidth=2, markersize=6)
            ax.plot(x, plot_df['relative_rmse'], marker='s', label='Relative RMSE', linewidth=2, markersize=6)
            ax.plot(x, plot_df['mae'], marker='^', label='MAE', linewidth=2, markersize=6)
            
            ax.set_title(f'Context: {context_length}d, Horizon: {horizon}d', fontsize=10, fontweight='bold')
            ax.set_xlabel('Window ID', fontsize=9)
            if j == 0:
                ax.set_ylabel('Error Metric', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([f'W{w}' for w in x], fontsize=8)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_label}: Error Metrics Across All Configurations',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to {save_path}")
    
    plt.close()  # Close figure instead of showing


def plot_model_comparison_summary(model_dfs: Dict[str, pd.DataFrame],
                                  model_labels: Dict[str, str],
                                  save_path: Optional[str] = None,
                                  figsize: tuple = (15, 10)):
    """
    Plot IV RMSE comparisons across models in a context/horizon grid.
    """
    # Configure matplotlib to use LaTeX
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    
    # Determine grid from first model
    first_df = next(iter(model_dfs.values()))
    context_lengths = sorted(first_df['context_length'].unique())
    horizons = sorted(first_df['horizon'].unique())
    
    n_cols = len(horizons)
    n_rows = len(context_lengths)
    
    # Don't share y-axis globally - each column will have its own scaling
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=False)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    sns.set_style("whitegrid")
    
    # Store y-values per column to compute column-specific y-limits
    column_y_values = {j: [] for j in range(n_cols)}
    
    # Print header
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    for i, context_length in enumerate(context_lengths):
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            
            # Print config header
            print(f"\nConfig: Context={context_length}d, Horizon={horizon}d")
            print("-" * 80)
            
            for model_id, df in model_dfs.items():
                plot_df = df[
                    (df['context_length'] == context_length) &
                    (df['horizon'] == horizon)
                ].copy()
                
                if len(plot_df) == 0:
                    continue
                
                plot_df = plot_df.groupby('window_id').agg({
                    'iv_rmse': 'mean'
                }).reset_index().sort_values('window_id')
                
                x = plot_df['window_id']
                label = model_labels.get(model_id, model_id)
                ax.plot(x, plot_df['iv_rmse'], marker='o', linewidth=2, markersize=5, label=label)
                
                # Collect y-values for this column
                column_y_values[j].extend(plot_df['iv_rmse'].values)
                
                # Print results for this model
                print(f"\n{label}:")
                for _, row in plot_df.iterrows():
                    window_id = int(row['window_id'])
                    iv_rmse = row['iv_rmse']
                    print(f"  W{window_id+1}: {iv_rmse:.6f}")
                # Print summary stats
                mean_rmse = plot_df['iv_rmse'].mean()
                std_rmse = plot_df['iv_rmse'].std()
                min_rmse = plot_df['iv_rmse'].min()
                max_rmse = plot_df['iv_rmse'].max()
                print(f"  Mean: {mean_rmse:.6f}, Std: {std_rmse:.6f}, Min: {min_rmse:.6f}, Max: {max_rmse:.6f}")
            
            ax.set_title(rf'Context: {context_length}d, Horizon: {horizon}d', 
                        fontsize=12, fontweight='bold', usetex=True)
            ax.set_xlabel(r'Window ID', fontsize=11, usetex=True)
            if j == 0:
                ax.set_ylabel(r'IV RMSE', fontsize=11, usetex=True)
            ax.set_xticks(x)
            # Add +1 to window IDs for display
            ax.set_xticklabels([rf'$W_{{{w+1}}}$' for w in x], fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # Set same y-axis limits for each column
    for j in range(n_cols):
        if column_y_values[j]:
            y_min = min(column_y_values[j])
            y_max = max(column_y_values[j])
            # Add small padding
            y_range = y_max - y_min
            y_padding = y_range * 0.05
            for i in range(n_rows):
                axes[i, j].set_ylim(y_min - y_padding, y_max + y_padding)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        # Convert labels to LaTeX format - escape underscores
        latex_labels = [label.replace('_', r'\_') for label in labels]
        fig.legend(handles, latex_labels, loc='upper right', ncol=1, fontsize=11, prop={'family': 'serif'})
    
    fig.suptitle(r'Model Comparison: IV RMSE Across All Configurations',
                 fontsize=16, fontweight='bold', y=0.995, usetex=True)
    plt.tight_layout()
    
    if save_path:
        # Change extension to PDF if not already
        if save_path.endswith('.png'):
            save_path = save_path[:-4] + '.pdf'
        elif not save_path.endswith('.pdf'):
            save_path = save_path + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()


def visualize_results(results_file: str,
                      model_id: str,
                      model_label: Optional[str] = None,
                      save_dir: str = 'results/plots'):
    """Create visualizations for a given model's results file."""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    model_label = model_label or model_id.replace('_', ' ').title()
    
    print("Loading results...")
    df = load_results(results_file)
    print(f"Loaded {len(df)} result rows")
    print(f"Windows: {sorted(df['window_id'].unique())}")
    print(f"Context lengths: {sorted(df['context_length'].unique())}")
    print(f"Horizons: {sorted(df['horizon'].unique())}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nCreating overall summary plot...")
    plot_metrics_by_window(
        df,
        metric_type='line',
        model_label=model_label,
        save_path=os.path.join(save_dir, f'{model_id}_overall_line.png')
    )
    
    plot_metrics_by_window(
        df,
        metric_type='bar',
        model_label=model_label,
        save_path=os.path.join(save_dir, f'{model_id}_overall_bar.png')
    )
    
    print("\nCreating plots for each configuration...")
    plot_all_configurations(
        df,
        metric_type='line',
        model_id=model_id,
        model_label=model_label,
        save_dir=save_dir
    )
    plot_all_configurations(
        df,
        metric_type='bar',
        model_id=model_id,
        model_label=model_label,
        save_dir=save_dir
    )
    
    print("\nCreating comprehensive summary plot...")
    plot_summary_comparison(
        df,
        model_label=model_label,
        save_path=os.path.join(save_dir, f'{model_id}_summary_all_configs.png')
    )
    
    print("\nAll visualizations created!")


def main():
    """Default entry point: visualize persistence results."""
    visualize_results(
        results_file='results/metrics/persistence_results.json',
        model_id='persistence',
        model_label='Persistence Model'
    )


if __name__ == '__main__':
    main()
