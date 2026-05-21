"""
Plot the per-day fit error (MSE) of the gridded IV surface against the raw
OptionMetrics quotes it was smoothed from - a single line over the dataset
period. Reads the fit_diagnostics.csv written by preprocess_optionmetrics.py.

Usage:
    python _data_prep/plot_fit_error.py _data_prep/out_hm_8e-4/fit_diagnostics.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Plot daily surface-vs-quote MSE")
    p.add_argument("csvs", nargs="+", help="fit_diagnostics.csv file(s)")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(13, 5))
    for path in args.csvs:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        mse = df["rmse"].to_numpy() ** 2          # MSE of quoted IV vs surface
        label = f"{Path(path).parent.name}  (mean {mse.mean():.2e})"
        ax.plot(df["date"], mse, lw=0.8, label=label)

    ax.set_xlabel("date")
    ax.set_ylabel(r"daily MSE  (quoted IV vs surface, IV$^2$)")
    ax.set_title("Surface fit error vs quoted options - daily MSE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = Path(args.csvs[0]).with_name("fit_error_mse.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
