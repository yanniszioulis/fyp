#!/usr/bin/env python3
"""
diag_factors.py — PCA + factor-dynamics linearity diagnostic.

Establishes *why* a linear forecaster is hard to beat on this surface,
in three steps. All on the log-IV level series, split at the same
train_end / val_end row indices the surface models use.

  1. PCA on the training-period surface (log-IV *levels*, not the
     per-channel-standardized series — standardizing inflates cross-cell
     correlation and artificially collapses the variance into PC1).
     Reports cumulative explained variance: how many factors F carry the
     surface (the level / slope / curvature decomposition).

  2. Per-PC persistence — variance share (EVR) and lag-1 autocorrelation
     of each leading PC. Shows how concentrated the surface is in the
     dominant level factor and how close each factor is to a random walk.

  3. Linear vs nonlinear factor dynamics — horizon-matched direct
     prediction of the top-F factor scores at h in {1,5,10,21}. Linear
     (ridge) and nonlinear (gradient boosting) use the *same* p-lag
     inputs, so the only difference is the function class. If the
     nonlinear model does not beat the linear one out-of-sample, there
     is little nonlinear autoregressive signal for a deep model to find.
     A persistence baseline (predict last observed score) is included.

R^2 in step 3 is pooled across the F factors: 1 - sum_f SSE_f /
sum_f SST_f, with SST around the per-factor test mean.

Outputs (under _test_results/63_<P>/diagnostics/):
  - factor_pca.csv            per-PC: EVR, cumulative EVR, lag-1 autocorr
  - factor_lin_vs_nonlin.csv  persistence / linear / nonlinear R^2 per horizon
  - factor_diagnostic.png     3-panel summary figure

Usage:
    python diag_factors.py --pred_len 21
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

from train import LOOKBACK, ROOT, load_dataset

HORIZONS    = (1, 5, 10, 21)
N_PC_SHOW   = 10      # PCs reported in the per-PC forecastability table
P_LAGS      = 5       # lag window fed to both the linear and nonlinear maps
RIDGE_ALPHA = 1.0


def fit_pca(train_rows: np.ndarray):
    """SVD-PCA on the centred training rows. Returns the train-row mean,
    the component matrix Vt (rows = PCs) and the explained-variance ratio."""
    mu = train_rows.mean(axis=0)
    _, sv, vt = np.linalg.svd(train_rows - mu, full_matrices=False)
    evr = (sv ** 2) / (sv ** 2).sum()
    return mu, vt, evr


def make_xy(Z: np.ndarray, F: int, p: int, h: int, lo: int, hi: int,
            keep: np.ndarray | None = None):
    """Horizon-matched pairs whose *target* index lies in [lo, hi).

    Input is the p most-recent score vectors ending at the decision time
    t = target - h (flattened to p*F); target is the F-vector at `target`.
    `keep`, if given, is a boolean mask over target indices — a pair is
    emitted only where keep[target] is True (used to restrict the OOS
    evaluation to one calendar year).
    """
    X, y = [], []
    for tgt in range(lo, hi):
        if keep is not None and not keep[tgt]:
            continue
        t = tgt - h
        if t - p + 1 < 0:
            continue
        X.append(Z[t - p + 1 : t + 1, :F].reshape(-1))
        y.append(Z[tgt, :F])
    return np.asarray(X), np.asarray(y)


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance-weighted R^2 across factors, around the test mean."""
    sse = ((y_true - y_pred) ** 2).sum()
    sst = ((y_true - y_true.mean(axis=0)) ** 2).sum()
    return float(1.0 - sse / sst)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_len", type=int, default=21,
                    choices=(5, 10, 21, 42, 63))
    ap.add_argument("--csv_path",
                    default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--data_end", default="2023-12-29")
    ap.add_argument("--year", type=int, default=None,
                    help="restrict the OOS evaluation to one calendar "
                         "year, e.g. 2023 (training pairs are unaffected)")
    args = ap.parse_args()

    data_end = None if args.data_end.lower() == "none" else args.data_end
    data = load_dataset(args.csv_path, 0.7, 0.1, LOOKBACK, args.pred_len,
                        data_end=data_end)
    # Recover log-IV *levels* — PCA on the structural surface, not on the
    # per-channel-standardized model-input space.
    sc = data["scaler"]
    log_iv = data["scaled_log_iv"] * sc["std"] + sc["mean"]   # [N, C]
    train_end = data["rows"]["train_end"]
    val_end   = data["rows"]["val_end"]
    N, C = log_iv.shape
    horizons = [h for h in HORIZONS if h <= args.pred_len]

    # Optional restriction of the OOS evaluation to one calendar year (a
    # test pair is kept by the date of its forecast target). Training
    # pairs are unaffected — only the evaluation window narrows. The full
    # test period is COVID-dominated (2020).
    year_keep, period, tag = None, "full test period", ""
    if args.year is not None:
        row_years = pd.DatetimeIndex(data["dates_iso"]).year.to_numpy()
        year_keep = row_years == args.year
        if not year_keep[val_end:].any():
            raise SystemExit(f"no test rows land in {args.year}")
        period, tag = str(args.year), f"_{args.year}"

    # ── 1) PCA on log-IV levels ──────────────────────────────────────
    mu, vt, evr = fit_pca(log_iv[:train_end])
    cum = np.cumsum(evr)
    F95 = int(np.argmax(cum >= 0.95) + 1)
    F99 = int(np.argmax(cum >= 0.99) + 1)
    Z = (log_iv - mu) @ vt.T                   # [N, C] PC scores

    # ── 2) Per-PC persistence ────────────────────────────────────────
    pc_recs = []
    for j in range(min(N_PC_SHOW, C)):
        ztr = Z[:train_end, j]
        ac1 = float(np.corrcoef(ztr[:-1], ztr[1:])[0, 1])
        pc_recs.append({"pc": j + 1, "evr": float(evr[j]),
                        "cum_evr": float(cum[j]), "lag1_autocorr": ac1})

    # ── 3) Linear vs nonlinear factor dynamics (horizon-matched) ─────
    F = max(2, F95)
    lvn_recs = []
    for h in horizons:
        Xtr, ytr = make_xy(Z, F, P_LAGS, h, P_LAGS + h, train_end)
        Xte, yte = make_xy(Z, F, P_LAGS, h, val_end, N, keep=year_keep)
        r2_pers = pooled_r2(yte, Xte[:, -F:])  # last input block = last score
        r2_lin  = pooled_r2(yte, Ridge(alpha=RIDGE_ALPHA)
                            .fit(Xtr, ytr).predict(Xte))
        pred_nl = np.column_stack([
            GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      random_state=0)
            .fit(Xtr, ytr[:, k]).predict(Xte)
            for k in range(F)])
        r2_nl = pooled_r2(yte, pred_nl)
        lvn_recs.append({"horizon": h, "r2_persistence": r2_pers,
                         "r2_linear": r2_lin, "r2_nonlinear": r2_nl,
                         "nonlin_minus_lin": r2_nl - r2_lin})

    # ── Write CSVs ───────────────────────────────────────────────────
    out_dir = os.path.join(ROOT, "_test_results",
                           f"{LOOKBACK}_{args.pred_len}", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    pc_df  = pd.DataFrame(pc_recs)
    lvn_df = pd.DataFrame(lvn_recs)
    pc_df.to_csv(os.path.join(out_dir, "factor_pca.csv"), index=False)
    lvn_df.to_csv(os.path.join(out_dir, f"factor_lin_vs_nonlin{tag}.csv"),
                  index=False)

    # ── Print summary ────────────────────────────────────────────────
    bar = "=" * 78
    print(f"\n{bar}\nFACTOR DIAGNOSTIC — pred_len={args.pred_len}, "
          f"{period}, log-IV level space\n{bar}")
    print(f"\nPCA on log-IV levels: {F95} PCs explain >=95% of variance "
          f"(cum={cum[F95-1]:.4f}); {F99} PCs explain >=99%.")
    print(f"linear-vs-nonlinear test uses F={F} factors, "
          f"{P_LAGS}-lag inputs.\n")
    print("per-PC structure (variance share and lag-1 autocorrelation):")
    print(pc_df.to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))
    print("\nlinear vs nonlinear factor dynamics (pooled OOS R^2):")
    print(lvn_df.to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    # ── 3-panel figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(pc_df["pc"], pc_df["cum_evr"], "o-")
    ax[0].axhline(0.95, ls="--", c="grey", lw=1)
    ax[0].set_xlabel("number of PCs")
    ax[0].set_ylabel("cumulative explained variance")
    ax[0].set_title(f"PCA on log-IV levels — {F95} PCs reach 95%")
    ax[0].set_ylim(0, 1.02)

    ax[1].bar(pc_df["pc"], pc_df["lag1_autocorr"])
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("principal component")
    ax[1].set_ylabel("lag-1 autocorrelation")
    ax[1].set_title("per-PC persistence (all factors near unit-root)")

    x = np.arange(len(horizons))
    w = 0.27
    ax[2].bar(x - w, lvn_df["r2_persistence"], w, label="persistence")
    ax[2].bar(x,     lvn_df["r2_linear"],      w, label="linear (ridge)")
    ax[2].bar(x + w, lvn_df["r2_nonlinear"],   w, label="nonlinear (GBR)")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"h+{h}" for h in horizons])
    ax[2].set_xlabel("forecast horizon")
    ax[2].set_ylabel("pooled out-of-sample R^2")
    ax[2].set_title("factor dynamics: linear vs nonlinear")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"factor_diagnostic{tag}.png"), dpi=150)
    plt.close(fig)

    print(f"\nwritten to {os.path.relpath(out_dir, ROOT)}/")
    print(f"  factor_pca.csv, factor_lin_vs_nonlin{tag}.csv, "
          f"factor_diagnostic{tag}.png")


if __name__ == "__main__":
    main()
