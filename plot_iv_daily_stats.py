"""Plot per-day mean and std of SPX IV across the surface, in level and logdiff space."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "SPX_surfaces.csv"
OUT_PATH = Path(__file__).parent / "iv_daily_stats.png"


def main():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    iv_cols = [c for c in df.columns if re.match(r"^iv_-?\d*\.?\d+_-?\d*\.?\d+$", c)]
    print(f"Loaded {len(df)} days, {len(iv_cols)} IV cells")

    iv = df[iv_cols].to_numpy()
    level_mean = iv.mean(axis=1)
    level_std = iv.std(axis=1)

    logdiff = np.diff(np.log(iv), axis=0)
    logdiff_mean = logdiff.mean(axis=1)
    logdiff_std = logdiff.std(axis=1)

    dates = df["date"]
    dates_ld = dates.iloc[1:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    axes[0, 0].plot(dates, level_mean, color="C0", lw=0.8)
    axes[0, 0].set_title("Level: cross-surface mean per day")
    axes[0, 0].set_ylabel("mean IV")

    axes[0, 1].plot(dates, level_std, color="C1", lw=0.8)
    axes[0, 1].set_title("Level: cross-surface std per day")
    axes[0, 1].set_ylabel("std IV")

    axes[1, 0].plot(dates_ld, logdiff_mean, color="C2", lw=0.8)
    axes[1, 0].set_title("Logdiff: cross-surface mean per day")
    axes[1, 0].set_ylabel("mean Δlog IV")
    axes[1, 0].set_xlabel("date")

    axes[1, 1].plot(dates_ld, logdiff_std, color="C3", lw=0.8)
    axes[1, 1].set_title("Logdiff: cross-surface std per day")
    axes[1, 1].set_ylabel("std Δlog IV")
    axes[1, 1].set_xlabel("date")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)

    fig.suptitle(f"SPX IV daily cross-surface stats ({dates.iloc[0].date()} → {dates.iloc[-1].date()})")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
