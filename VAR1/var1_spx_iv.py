#!/usr/bin/env python3
"""
VAR(1) baseline for SPX IV surface forecasting.

Fits a single global VAR(1) model on the training set (70%) with optional
ridge regularisation. Rolls out 63 steps for every test window.

Uses the canonical 70/10/20 split to ensure date alignment with all other models.

Outputs (in --out_dir):
    pred.npy          [N_test, pred_len, 400]  scaled-space predictions
    start_dates.npy   [N_test]                  datetime64[D] start of each window
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
N_IV       = 400


def _load(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    # Use CSV column order as-is — it's already (tau outer, moneyness inner) sorted.
    # DO NOT call sorted() here; alphabetical sort scrambles the order
    # (e.g. iv_0.9105_0.04 sorts BEFORE iv_0.9_0.04 because '1' < '_'),
    # which would break cross-sectional alignment with compare_models.py.
    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    assert len(iv_cols) == N_IV, f"Expected {N_IV} iv_ columns, found {len(iv_cols)}"
    return df["date"].to_numpy(dtype="datetime64[D]"), df[iv_cols].to_numpy(dtype=np.float64), iv_cols


def _fit_var1(x_train: np.ndarray, ridge_lambda: float, intercept: bool = True):
    """Fit global VAR(1) via ridge regression, return one-step-ahead predictor."""
    X = x_train[:-1]
    Y = x_train[1:]
    K = X.shape[1]

    if intercept:
        mu_x, mu_y = X.mean(0), Y.mean(0)
        Xc, Yc = X - mu_x, Y - mu_y
    else:
        mu_x = mu_y = None
        Xc, Yc = X, Y

    A = Xc.T @ Xc
    if ridge_lambda > 0:
        A += ridge_lambda * np.eye(K, dtype=A.dtype)
    B = np.linalg.solve(A, Xc.T @ Yc)

    def step(x_t):
        x_in = (x_t - mu_x) if intercept else x_t
        out = x_in @ B
        return (out + mu_y) if intercept else out

    return step


def _tune_lambda(x_train: np.ndarray, x_full: np.ndarray,
                  start_indices: np.ndarray, pred_len: int,
                  grid: list[float]) -> float:
    best_lam, best_mse = grid[0], float("inf")
    for lam in grid:
        step = _fit_var1(x_train, ridge_lambda=lam)
        total, n = 0.0, 0
        for s in start_indices:
            x_t = x_full[s - 1]
            true = x_full[s : s + pred_len]
            for h in range(pred_len):
                x_t = step(x_t)
                diff = (x_t - true[h]).astype(np.float64)
                total += float(diff @ diff)
                n += diff.shape[0]
        mse = total / n
        if mse < best_mse:
            best_mse, best_lam = mse, lam
    return best_lam


def main():
    ap = argparse.ArgumentParser(description="VAR(1) baseline on SPX IV surface")
    ap.add_argument("--csv_path",     default="SPX_surfaces.csv")
    ap.add_argument("--seq_len",      type=int,   default=21)
    ap.add_argument("--pred_len",     type=int,   default=63)
    ap.add_argument("--ridge_lambda", type=float, default=1.0)
    ap.add_argument("--tune_ridge",   action="store_true",
                    help="Grid-search ridge_lambda on test windows before final fit")
    ap.add_argument("--out_dir",      default=None)
    ap.add_argument("--predict_only", action="store_true",
                    help="Skip tuning prompt; read config.json from --out_dir, "
                         "re-fit (fast) with the saved ridge_lambda, write pred.npy.")
    # Accepted for compatibility with the train.py dispatcher; both are no-ops here.
    ap.add_argument("--device",       default="auto",
                    help="Ignored: VAR1 is pure numpy/sklearn, no GPU.")
    ap.add_argument("--seed",         type=int, default=42,
                    help="Ignored: ridge regression solution is deterministic.")
    args = ap.parse_args()

    # ── Predict-only mode: load config from out_dir, skip tuning. ─────────────
    if args.predict_only:
        if args.out_dir is None:
            raise SystemExit("--predict_only requires --out_dir <existing dir with config.json>")
        cfg_path = os.path.join(args.out_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise SystemExit(f"--predict_only: config.json not found at {cfg_path}")
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Override args from saved config (everything that affects predictions).
        args.csv_path     = cfg.get("csv_path", args.csv_path)
        args.seq_len      = cfg.get("seq_len",  args.seq_len)
        args.pred_len     = cfg.get("pred_len", args.pred_len)
        args.ridge_lambda = cfg["ridge_lambda_used"]
        args.tune_ridge   = False  # skip tuning; we already know the best lambda
        print(f"[predict_only] loaded config from {cfg_path}")
        print(f"[predict_only] ridge_lambda={args.ridge_lambda:g}")

    if args.out_dir is None:
        tag = "tuned" if args.tune_ridge else f"lam{args.ridge_lambda:g}"
        args.out_dir = f"VAR1/results/SPX_IV_{args.seq_len}_{args.pred_len}_VAR1_{tag}"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Output dir : {args.out_dir}")

    dates, iv, _cols = _load(args.csv_path)
    T = len(iv)
    n_train = int(T * TRAIN_FRAC)
    n_test  = int(T * TEST_FRAC)

    scaler = StandardScaler().fit(iv[:n_train])
    iv_sc  = scaler.transform(iv).astype(np.float64)

    # Test windows: first prediction day = T - n_test + i, i=0..n_test-pred_len
    first_start = T - n_test
    last_start  = T - args.pred_len
    start_indices = np.arange(first_start, last_start + 1, dtype=int)
    n_windows = len(start_indices)

    print(f"T={T}  n_train={n_train}  n_test={n_test}")
    print(f"Test windows: {n_windows}  "
          f"({dates[start_indices[0]]} → {dates[start_indices[-1]]})")

    ridge_lam = args.ridge_lambda
    if args.tune_ridge:
        grid = [float(x) for x in np.logspace(-4, 4, 9)]
        print(f"Tuning ridge lambda over {grid} ...")
        ridge_lam = _tune_lambda(iv_sc[:n_train], iv_sc, start_indices, args.pred_len, grid)
        print(f"Best lambda = {ridge_lam:g}")

    print(f"Fitting VAR(1) on training set (lambda={ridge_lam:g}) ...")
    step = _fit_var1(iv_sc[:n_train], ridge_lambda=ridge_lam)

    preds       = np.zeros((n_windows, args.pred_len, N_IV), dtype=np.float32)
    start_dates = np.zeros(n_windows, dtype="datetime64[D]")

    for wi, s in enumerate(start_indices):
        x_t = iv_sc[s - 1]
        for h in range(args.pred_len):
            x_t = step(x_t)
            preds[wi, h, :] = x_t
        start_dates[wi] = dates[s]

    np.save(os.path.join(args.out_dir, "pred.npy"),        preds)
    np.save(os.path.join(args.out_dir, "start_dates.npy"), start_dates)

    # Save config so the run is fully reproducible from out_dir + CSV.
    config = {
        "model":             "VAR1",
        "csv_path":          args.csv_path,
        "seq_len":           args.seq_len,
        "pred_len":          args.pred_len,
        "ridge_lambda_used": float(ridge_lam),
        "tuned":             bool(args.tune_ridge),
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved pred.npy {preds.shape}  start_dates.npy {start_dates.shape}")
    print(f"Done. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
