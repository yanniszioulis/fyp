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

# LaTeX rendering for thesis figures. Requires a TeX install on PATH
# (MacTeX / TeX Live). If LaTeX is unavailable, set usetex=False to fall
# back to mathtext with a serif family.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
})


def main():
    p = argparse.ArgumentParser(description="Plot daily surface-vs-quote MSE")
    p.add_argument("csvs", nargs="+", help="fit_diagnostics.csv file(s)")
    p.add_argument("--out", default=None,
                   help="output PDF path (default: alongside first CSV)")
    args = p.parse_args()

    end_date = pd.Timestamp("2023-12-29")  # in-sample period cut-off
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    for path in args.csvs:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= end_date].sort_values("date")
        mse = df["rmse"].to_numpy() ** 2          # MSE of quoted IV vs surface
        ax.plot(df["date"], mse, lw=0.7, color="#1f77b4")
        print(f"{Path(path).parent.name}: mean MSE = {mse.mean():.3e}  "
              f"(through {end_date.date()})")

    ax.set_xlabel("Date")
    ax.set_ylabel(r"Daily MSE  ($\sigma^2$)")
    ax.set_title(r"Surface Fit Error vs.\ Quoted Options", fontsize=15)
    ax.grid(alpha=0.3)
    ax.margins(x=0.01)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.csvs[0]).with_name("fit_error_mse.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
