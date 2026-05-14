"""3D plot of the mean and standard-deviation IV surface across the dataset."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "SPX_surfaces.csv"
OUT_PATH = Path(__file__).parent / "iv_mean_std_surfaces.png"


def parse_iv_columns(columns):
    pattern = re.compile(r"^iv_(-?\d*\.?\d+)_(-?\d*\.?\d+)$")
    parsed = []
    for c in columns:
        m = pattern.match(c)
        if m:
            parsed.append((c, float(m.group(1)), float(m.group(2))))
    moneyness = sorted({p[1] for p in parsed})
    taus      = sorted({p[2] for p in parsed})
    return parsed, moneyness, taus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(CSV_PATH))
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    parsed, moneyness, taus = parse_iv_columns(df.columns)
    iv_cols = [c for c, _, _ in parsed]
    print(f"Loaded {len(df)} days, {len(moneyness)} moneyness x {len(taus)} taus")

    iv = df[iv_cols].to_numpy(dtype=float)
    mean_flat = iv.mean(axis=0)
    std_flat  = iv.std(axis=0, ddof=0)

    m_idx = {v: i for i, v in enumerate(moneyness)}
    t_idx = {v: i for i, v in enumerate(taus)}
    Z_mean = np.empty((len(taus), len(moneyness)))
    Z_std  = np.empty((len(taus), len(moneyness)))
    for k, (_, m, t) in enumerate(parsed):
        Z_mean[t_idx[t], m_idx[m]] = mean_flat[k]
        Z_std [t_idx[t], m_idx[m]] = std_flat[k]

    M, T = np.meshgrid(np.array(moneyness), np.array(taus))

    fig = plt.figure(figsize=(14, 6))
    panels = [
        ("Mean IV surface",   Z_mean, "viridis"),
        ("Std. dev. IV surface", Z_std,  "magma"),
    ]
    for i, (title, Z, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        surf = ax.plot_surface(M, T, Z, cmap=cmap, edgecolor="none", alpha=0.95)
        ax.set_xlabel("log-moneyness")
        ax.set_ylabel("tau (years)")
        ax.set_zlabel("IV")
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.05)

    fig.suptitle(
        f"SPX IV surface — cross-day statistics over {len(df):,} days "
        f"({df['date'].iloc[0]} → {df['date'].iloc[-1]})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved {args.out}")

    # quick numeric summary
    print(f"Mean IV   range: {Z_mean.min():.4f}  ..  {Z_mean.max():.4f}")
    print(f"StdDev IV range: {Z_std.min():.4f}  ..  {Z_std.max():.4f}")


if __name__ == "__main__":
    main()
