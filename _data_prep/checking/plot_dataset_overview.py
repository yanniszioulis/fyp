"""Three dataset-overview plots produced by one script:

1. Time series of the daily IV mean and the daily intra-surface std (1x2).
2. Top-3 PCA eigensurfaces of the daily IV surfaces (1x3, 3D).
3. Top-3 PCA eigensurfaces of the daily IV differences (1x3, 3D).
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "SPX_surfaces.csv"
OUT_DIR  = Path(__file__).parent


def parse_iv_columns(columns):
    pat = re.compile(r"^iv_(-?\d*\.?\d+)_(-?\d*\.?\d+)$")
    parsed = []
    for c in columns:
        m = pat.match(c)
        if m:
            parsed.append((c, float(m.group(1)), float(m.group(2))))
    moneyness = sorted({p[1] for p in parsed})
    taus      = sorted({p[2] for p in parsed})
    return parsed, moneyness, taus


def to_grid(flat, parsed, moneyness, taus):
    m_idx = {v: i for i, v in enumerate(moneyness)}
    t_idx = {v: i for i, v in enumerate(taus)}
    Z = np.empty((len(taus), len(moneyness)))
    for k, (_, m, t) in enumerate(parsed):
        Z[t_idx[t], m_idx[m]] = flat[k]
    return Z


def pca(X):
    """Centred PCA via SVD. Returns (components [k, p], explained_var_ratio [k])."""
    mu = X.mean(axis=0)
    Xc = X - mu
    # economy SVD: Xc = U S Vt;  Vt rows are principal axes
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_total = (S ** 2).sum()
    return Vt, (S ** 2) / var_total


def plot_eigensurfaces(components, ratios, parsed, moneyness, taus, title, out_path):
    M, T = np.meshgrid(np.array(moneyness), np.array(taus))

    # Sign convention: largest-magnitude entry positive (PCA sign is arbitrary).
    Zs = []
    for i in range(3):
        Z = to_grid(components[i], parsed, moneyness, taus)
        if Z.flat[np.argmax(np.abs(Z))] < 0:
            Z = -Z
        Zs.append(Z)

    # Shared symmetric colour scale across all three panels so same-sign
    # surfaces are visually obvious (all one colour) and sign-changing
    # surfaces show both reds and blues at the same intensity.
    vmax = max(np.abs(Z).max() for Z in Zs)
    norm = plt.matplotlib.colors.Normalize(vmin=-vmax, vmax=+vmax)
    cmap = "coolwarm"

    fig = plt.figure(figsize=(18, 5.5))
    surf_handle = None
    for i, Z in enumerate(Zs):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        surf_handle = ax.plot_surface(M, T, Z, cmap=cmap, norm=norm,
                                      edgecolor="none", alpha=0.95)
        ax.set_xlabel("log-moneyness")
        ax.set_ylabel("tau (years)")
        ax.set_zlabel("loading")
        ax.set_zlim(-vmax, +vmax)
        ax.set_title(f"PC{i+1} — {ratios[i]*100:.1f}% var")

    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.66])
    fig.colorbar(surf_handle, cax=cbar_ax, label="loading")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(CSV_PATH))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["date"])
    parsed, moneyness, taus = parse_iv_columns(df.columns)
    iv_cols = [c for c, _, _ in parsed]
    print(f"Loaded {len(df)} days, {len(moneyness)} moneyness x {len(taus)} taus")

    iv = df[iv_cols].to_numpy(dtype=float)              # (T, P)
    dates = df["date"].to_numpy()

    # ---------- 1) daily mean and intra-surface std ----------
    daily_mean = iv.mean(axis=1)
    daily_std  = iv.std(axis=1, ddof=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    axes[0].plot(dates, daily_mean, lw=0.6, color="#1f77b4")
    axes[0].set_title("Daily IV mean (avg across grid cells)")
    axes[0].set_ylabel("mean IV")
    axes[1].plot(dates, daily_std, lw=0.6, color="#d62728")
    axes[1].set_title("Daily intra-surface std (across grid cells)")
    axes[1].set_ylabel("std IV")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("date")
    fig.suptitle(f"SPX IV surface — daily summary stats "
                 f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}, "
                 f"{len(df):,} days)", y=1.02)
    fig.tight_layout()
    p1 = out_dir / "iv_daily_mean_std.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    print(f"Saved {p1}")

    # ---------- 2) PCA on daily IV surfaces ----------
    Vt_lvl, ratios_lvl = pca(iv)
    plot_eigensurfaces(Vt_lvl, ratios_lvl, parsed, moneyness, taus,
                       title="Top-3 PCA eigensurfaces — daily IV levels",
                       out_path=out_dir / "iv_pca_levels.png")
    print(f"  cumulative var (top3): {ratios_lvl[:3].sum()*100:.2f}%")

    # ---------- 3) PCA on daily IV differences ----------
    diffs = np.diff(iv, axis=0)
    Vt_d, ratios_d = pca(diffs)
    plot_eigensurfaces(Vt_d, ratios_d, parsed, moneyness, taus,
                       title="Top-3 PCA eigensurfaces — daily IV differences",
                       out_path=out_dir / "iv_pca_diffs.png")
    print(f"  cumulative var (top3): {ratios_d[:3].sum()*100:.2f}%")


if __name__ == "__main__":
    main()
