"""Plot 3 randomly selected SPX IV surfaces from SPX_surfaces.csv as 3D graphs."""

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


def surface_grid(row, parsed, moneyness, taus):
    m_idx = {v: i for i, v in enumerate(moneyness)}
    t_idx = {v: i for i, v in enumerate(taus)}
    grid = np.full((len(taus), len(moneyness)), np.nan)
    for col, m, t in parsed:
        grid[t_idx[t], m_idx[m]] = row[col]
    return grid


def main():
    df = pd.read_csv(CSV_PATH)
    parsed, moneyness, taus = parse_iv_columns(df.columns)
    print(f"Loaded {len(df)} days, {len(moneyness)} moneyness x {len(taus)} taus")

    rng = random.Random(SEED)
    indices = rng.sample(range(len(df)), N_SURFACES)

    M, T = np.meshgrid(np.array(moneyness), np.array(taus))

    fig = plt.figure(figsize=(6 * N_SURFACES, 6))
    for i, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        Z = surface_grid(row, parsed, moneyness, taus)
        ax = fig.add_subplot(1, N_SURFACES, i, projection="3d")
        ax.plot_surface(M, T, Z, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_xlabel("log-moneyness")
        ax.set_ylabel("tau (years)")
        ax.set_zlabel("IV")
        ax.set_title(f"{row['date']}  (forward={row['forward_ref']:.2f})")

    plt.tight_layout()
    out_path = Path(__file__).parent / "random_surfaces.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
