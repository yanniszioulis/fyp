"""Plot 3 randomly selected SPX IV surfaces from SPX_surfaces.csv as 3D graphs."""

import argparse
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "SPX_surfaces.csv"
N_SURFACES = 3
SEED = None  # set an int for reproducibility


def parse_iv_columns(columns):
    pattern = re.compile(r"^iv_(-?\d*\.?\d+)_(-?\d*\.?\d+)$")
    parsed = []
    for c in columns:
        m = pattern.match(c)
        if m:
            parsed.append((c, float(m.group(1)), float(m.group(2))))
    moneyness = sorted({p[1] for p in parsed})
    taus = sorted({p[2] for p in parsed})
    return parsed, moneyness, taus


def surface_grid(values, parsed, moneyness, taus):
    m_idx = {v: i for i, v in enumerate(moneyness)}
    t_idx = {v: i for i, v in enumerate(taus)}
    grid = np.full((len(taus), len(moneyness)), np.nan)
    for col, m, t in parsed:
        grid[t_idx[t], m_idx[m]] = values[col]
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdiff", type=int, default=0, choices=[0, 1],
                    help="0 = plot raw IV surfaces (default); "
                         "1 = plot log(IV)[t]-log(IV)[t-1] surfaces")
    args = ap.parse_args()
    use_logdiff = bool(args.logdiff)

    df = pd.read_csv(CSV_PATH)
    parsed, moneyness, taus = parse_iv_columns(df.columns)
    iv_cols = [c for c, _, _ in parsed]
    print(f"Loaded {len(df)} days, {len(moneyness)} moneyness x {len(taus)} taus")

    if use_logdiff:
        iv = df[iv_cols].to_numpy()
        if not np.all(iv > 0):
            raise ValueError("logdiff requires all IV > 0")
        ld = np.diff(np.log(iv), axis=0)
        data_df = pd.DataFrame(ld, columns=iv_cols)
        data_df["date"] = df["date"].iloc[1:].to_numpy()
        data_df["prev_date"] = df["date"].iloc[:-1].to_numpy()
        data_df = data_df.reset_index(drop=True)
        zlabel = "log(IV)[t] - log(IV)[t-1]"
        out_name = "random_logdiffs.png"
    else:
        data_df = df
        zlabel = "IV"
        out_name = "random_surfaces.png"

    rng = random.Random(SEED)
    indices = rng.sample(range(len(data_df)), N_SURFACES)

    M, T = np.meshgrid(np.array(moneyness), np.array(taus))

    fig = plt.figure(figsize=(6 * N_SURFACES, 6))
    for i, idx in enumerate(indices, start=1):
        row = data_df.iloc[idx]
        Z = surface_grid(row, parsed, moneyness, taus)
        ax = fig.add_subplot(1, N_SURFACES, i, projection="3d")
        ax.plot_surface(M, T, Z, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_xlabel("log-moneyness")
        ax.set_ylabel("tau (years)")
        ax.set_zlabel(zlabel)
        if use_logdiff:
            ax.set_title(f"{row['prev_date']} -> {row['date']}")
        else:
            ax.set_title(f"{row['date']}  (forward={row['forward_ref']:.2f})")

    plt.tight_layout()
    out_path = Path(__file__).parent / out_name
    plt.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
