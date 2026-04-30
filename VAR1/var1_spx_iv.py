#!/usr/bin/env python3
"""
VAR(1) baseline for SPX IV surface forecasting.

Fits a single global VAR(1) model on the training set (70%) by ordinary
least squares — no regularisation, no hyperparameter tuning. Rolls out
63 steps for every test window.

Uses the canonical 70/10/20 split to ensure date alignment with all other models.

Outputs (in --out_dir):
    pred.npy          [N_test, pred_len, 400]  scaled-space predictions
    start_dates.npy   [N_test]                  datetime64[D] start of each window
    config.json       {model, csv_path, seq_len, pred_len}
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


def _fit_var1(x_train: np.ndarray):
    """
    Fit global VAR(1) by ordinary least squares with intercept.
        x_{t+1} = x_t @ B + b
    Solved via lstsq for numerical robustness when K is large.
    Returns a one-step-ahead predictor `step(x) -> x_next`.
    """
    X = x_train[:-1]
    Y = x_train[1:]

    mu_x = X.mean(0)
    mu_y = Y.mean(0)
    Xc = X - mu_x
    Yc = Y - mu_y

    # B is K×K; lstsq handles rank-deficiency gracefully if K ≈ N.
    B, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)

    def step(x_t):
        return ((x_t - mu_x) @ B) + mu_y

    return step


def main():
    ap = argparse.ArgumentParser(description="VAR(1) baseline on SPX IV surface (plain OLS)")
    ap.add_argument("--csv_path",     default="SPX_surfaces.csv")
    ap.add_argument("--seq_len",      type=int, default=21)
    ap.add_argument("--pred_len",     type=int, default=63)
    ap.add_argument("--out_dir",      default=None)
    ap.add_argument("--predict_only", action="store_true",
                    help="Skip the run banner; read config.json from --out_dir, "
                         "re-fit (fast), write pred.npy.")
    # Accepted for compatibility with the train.py dispatcher; both are no-ops here.
    ap.add_argument("--device",       default="auto",
                    help="Ignored: VAR1 is pure numpy/sklearn, no GPU.")
    ap.add_argument("--seed",         type=int, default=42,
                    help="Ignored: OLS solution is deterministic.")
    args = ap.parse_args()

    if args.predict_only:
        if args.out_dir is None:
            raise SystemExit("--predict_only requires --out_dir <existing dir with config.json>")
        cfg_path = os.path.join(args.out_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise SystemExit(f"--predict_only: config.json not found at {cfg_path}")
        with open(cfg_path) as f:
            cfg = json.load(f)
        args.csv_path = cfg.get("csv_path", args.csv_path)
        args.seq_len  = cfg.get("seq_len",  args.seq_len)
        args.pred_len = cfg.get("pred_len", args.pred_len)
        print(f"[predict_only] loaded config from {cfg_path}")

    if args.out_dir is None:
        args.out_dir = f"VAR1/results/SPX_IV_{args.seq_len}_{args.pred_len}_VAR1"
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

    print("Fitting VAR(1) on training set by OLS ...")
    step = _fit_var1(iv_sc[:n_train])

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

    config = {
        "model":    "VAR1",
        "csv_path": args.csv_path,
        "seq_len":  args.seq_len,
        "pred_len": args.pred_len,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved pred.npy {preds.shape}  start_dates.npy {start_dates.shape}")
    print(f"Done. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
