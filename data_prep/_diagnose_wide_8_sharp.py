"""
Read the full wide_8_sharp output and report:
  - per-year stats: count of dates, ffill share, mean cross-surface std, PCA top-1/3 of daily diffs
  - global PCA top-1/3/5 on diffs
  - worst-ffill dates (top 20 by share)
  - dates with no surface produced (if any)
  - level distribution checks (min/max/mean) per year

Also writes plots (under _wide_8_sharp_full/):
  - ffill_per_day.png       : count of ffill cells per day over time
  - cross_std_per_day.png   : mean cross-surface std per day over time
  - sample_surfaces.png     : surfaces on a stress day (vol-spike), normal day,
                              and a quiet day, side by side
"""
import os
from os.path import join

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "_wide_8_sharp_full"
N = 8

surf_csv  = join(OUT, "SPX_surfaces.csv")
ffill_csv = join(OUT, "daily_ffill.csv")

print(f"Reading {surf_csv} ...")
df = pd.read_csv(surf_csv)
df["date"] = pd.to_datetime(df["date"])
ff  = pd.read_csv(ffill_csv)
ff["date"] = pd.to_datetime(ff["date"])

iv = df.iloc[:, 2:].values.reshape(-1, N, N)
T  = iv.shape[0]
print(f"surfaces shape: {iv.shape}")

# ---- daily ffill share ---------------------------------------------------
ff["ffill_share"] = ff["n_ffill_cells"] / ff["n_total_cells"]
no_surf = ff[~ff["surface_produced"]]
print(f"\nDates with NO surface produced: {len(no_surf)}")
if len(no_surf):
    print(no_surf.head(20).to_string(index=False))

print(f"\nWorst 20 dates by ffill share:")
print(ff.nlargest(20, "ffill_share")[["date","n_obs_used","n_ffill_cells","ffill_share"]].to_string(index=False))

# ---- per-year stats ------------------------------------------------------
def adj_corr(S, axis):
    a = np.take(S, np.arange(S.shape[axis]-1), axis=axis)
    b = np.take(S, np.arange(1, S.shape[axis]), axis=axis)
    a = a.reshape(a.shape[0], -1); b = b.reshape(b.shape[0], -1)
    am = a - a.mean(0); bm = b - b.mean(0)
    num = (am*bm).mean(0); den = a.std(0)*b.std(0) + 1e-12
    return float((num/den).mean())

def pca_diffs(S):
    d = np.diff(S, axis=0)
    Xd = d.reshape(d.shape[0], -1); Xd = Xd - Xd.mean(0)
    if Xd.shape[0] < 3:
        return float('nan'), float('nan'), float('nan')
    Sv = np.linalg.svd(Xd, compute_uv=False)
    cum = np.cumsum(Sv**2) / np.sum(Sv**2)
    return float(cum[0]), float(cum[2] if len(cum)>=3 else cum[-1]), float(cum[4] if len(cum)>=5 else cum[-1])

df["year"] = df["date"].dt.year
ff["year"] = ff["date"].dt.year
years = sorted(df["year"].unique())

rows = []
for y in years:
    mask = (df["year"] == y).values
    Sy = iv[mask]
    ffy = ff[ff["year"] == y]
    p1, p3, p5 = pca_diffs(Sy)
    rows.append({
        "year": y,
        "n_dates": int(mask.sum()),
        "ffill_share_mean": float(ffy["ffill_share"].mean()),
        "ffill_share_max":  float(ffy["ffill_share"].max()),
        "n_dates_ffill_gt_5pct": int((ffy["ffill_share"] > 0.05).sum()),
        "iv_mean": float(Sy.mean()),
        "iv_std":  float(Sy.std()),
        "cross_std_per_day": float(Sy.reshape(Sy.shape[0], -1).std(axis=1).mean()),
        "diffs_adj_corr_m":   adj_corr(np.diff(Sy, axis=0), 2) if Sy.shape[0] > 1 else float('nan'),
        "diffs_adj_corr_tau": adj_corr(np.diff(Sy, axis=0), 1) if Sy.shape[0] > 1 else float('nan'),
        "pca_diffs_top1":  p1, "pca_diffs_top3": p3, "pca_diffs_top5": p5,
    })

stats = pd.DataFrame(rows)
print("\nPer-year diagnostics:")
print(stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# global
d_all = np.diff(iv, axis=0)
Xd = d_all.reshape(d_all.shape[0], -1); Xd = Xd - Xd.mean(0)
Sv = np.linalg.svd(Xd, compute_uv=False)
cum = np.cumsum(Sv**2)/np.sum(Sv**2)
print(f"\nGLOBAL PCA on diffs: top1/3/5/10 = "
      f"{cum[0]*100:.2f}% / {cum[2]*100:.2f}% / {cum[4]*100:.2f}% / {cum[9]*100:.2f}%")
print(f"GLOBAL DIFFS adj corr (m / tau): "
      f"{adj_corr(d_all, 2):.4f} / {adj_corr(d_all, 1):.4f}")
print(f"GLOBAL ffill share: {ff['ffill_share'].mean()*100:.3f}% mean, "
      f"{ff['ffill_share'].max()*100:.2f}% max")

# ---- plots ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(ff["date"], ff["n_ffill_cells"], lw=0.6)
ax.set_title("ffill cells per day (out of 64)")
ax.set_xlabel("date"); ax.set_ylabel("# ffill cells")
plt.tight_layout()
plt.savefig(join(OUT, "ffill_per_day.png"), dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(10, 3))
day_std = iv.reshape(T, -1).std(axis=1)
ax.plot(df["date"], day_std, lw=0.6, color="C2")
ax.set_title("cross-surface IV std per day (proxy for surface dispersion)")
ax.set_xlabel("date"); ax.set_ylabel("std of 64 IV cells")
plt.tight_layout()
plt.savefig(join(OUT, "cross_std_per_day.png"), dpi=120)
plt.close()

# sample surfaces: a stress day, a normal day, a quiet day
day_std_series = pd.Series(day_std, index=df["date"])
stress_dates = [
    pd.Timestamp("2020-03-16"),  # COVID vol spike
    pd.Timestamp("2018-09-04"),  # mid-2018, "normal"
    pd.Timestamp("2017-08-08"),  # 2017 quiet regime
]
m_grid = np.linspace(-0.20, 0.20, N)
t_grid = np.exp(np.linspace(np.log(0.04), np.log(1.0), N))

fig = plt.figure(figsize=(15, 5))
for i, d in enumerate(stress_dates):
    # find nearest available date
    diff = (df["date"] - d).abs()
    idx = int(diff.idxmin())
    surf = iv[idx]
    n_ff = int(ff.iloc[idx]["n_ffill_cells"])
    ax = fig.add_subplot(1, 3, i+1, projection="3d")
    M, Tg = np.meshgrid(m_grid, t_grid)
    ax.plot_surface(M, Tg, surf, cmap="viridis", edgecolor="k", lw=0.3)
    ax.set_title(f"{df.iloc[idx]['date'].date()}  (ffill={n_ff}/64)")
    ax.set_xlabel("log-moneyness"); ax.set_ylabel("tau"); ax.set_zlabel("IV")
plt.tight_layout()
plt.savefig(join(OUT, "sample_surfaces.png"), dpi=120)
plt.close()

print("\nWrote plots:")
for p in ("ffill_per_day.png", "cross_std_per_day.png", "sample_surfaces.png"):
    print("  -", join(OUT, p))
