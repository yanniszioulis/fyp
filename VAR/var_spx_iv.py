#!/usr/bin/env python3
"""
VAR(p) baseline for SPX IV surface forecasting with automatic lag selection.

For each candidate p in {1, …, max_lag} fits a global VAR(p) by ordinary
least squares with intercept on the training set (70%), computes the
information criteria

    AIC(p)  = log|Σ̂(p)| + (2 / T_eff)             · p · K²
    BIC(p)  = log|Σ̂(p)| + (log T_eff / T_eff)     · p · K²
    HQIC(p) = log|Σ̂(p)| + (2 log log T_eff / T_eff)· p · K²

(Lütkepohl, *New Introduction to Multiple Time Series Analysis*, 2005,
chapter 4.3) where Σ̂(p) is the ML residual covariance and K is the number
of features. Picks the p that minimises the chosen `--criterion`, then
rolls out `pred_len` steps for every test window using the last p
observations as the initial state.

Dataset selection:
    --dataset full      use the full CSV (default).
    --dataset precovid  slice the CSV to date <= 2019-12-31 before splitting.

Outputs (in --out_dir; default
`VAR/results/{dataset}_SPX_IV_{seq_len}_{pred_len}_VAR_p{selected}_{criterion}`):
    pred.npy          [N_test, pred_len, n_iv]  scaled-space predictions  (gitignored)
    start_dates.npy   [N_test]                  datetime64[D] start of each window
    config.json       full record incl. ic_table sweep over all candidate p
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN_FRAC          = 0.70
TEST_FRAC           = 0.20
PRECOVID_END        = "2019-12-31"
DATASET_CHOICES     = ["full", "precovid"]
TARGET_SPACE_CHOICES = ["level", "logdiff"]
CRITERION_CHOICES   = ["aic", "bic", "hqic"]


def _slice_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "full":
        return df
    if dataset == "precovid":
        end = pd.Timestamp(PRECOVID_END)
        return df[df["date"] <= end].reset_index(drop=True)
    raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASET_CHOICES}")


def _apply_target_space(iv_raw: np.ndarray, dates_full: np.ndarray, target_space: str):
    """
    Returns (data, dates) for the chosen target space.
      level   → data = iv_raw, dates = dates_full
      logdiff → data = log(iv)[1:] - log(iv)[:-1], dates = dates_full[1:]
    For logdiff, asserts iv > 0 (preprocessing guarantees this).
    """
    if target_space == "level":
        return iv_raw, dates_full
    if target_space == "logdiff":
        if (iv_raw <= 0).any():
            raise ValueError("logdiff target_space requires all IV > 0")
        log_iv = np.log(iv_raw)
        return log_iv[1:] - log_iv[:-1], dates_full[1:]
    raise ValueError(f"Unknown target_space {target_space!r}; choose from {TARGET_SPACE_CHOICES}")


def _load(csv_path: str, dataset: str, target_space: str):
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = _slice_dataset(df, dataset)
    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    if not iv_cols:
        raise ValueError(f"No iv_* columns found in {csv_path}")
    iv_raw = df[iv_cols].to_numpy(dtype=np.float64)
    dates_full = df["date"].to_numpy(dtype="datetime64[D]")
    data, dates = _apply_target_space(iv_raw, dates_full, target_space)
    return dates, data, iv_cols


# ── VAR(p) fit ──────────────────────────────────────────────────────────────

def _fit_var_p(X: np.ndarray, p: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Fit VAR(p) by OLS with intercept on `X` of shape [T, K].

    Model: x_t = c + Σ_{i=1..p} A_i x_{t-i} + ε_t

    Returns:
        c        intercept [K]
        A_list   list of p coefficient matrices each [K, K], A_list[i-1] is A_i
        E        residual matrix [T-p, K]
    """
    T, K = X.shape
    T_eff = T - p
    if T_eff <= K * p:
        raise ValueError(
            f"VAR({p}) needs T_eff > K*p (have T_eff={T_eff}, K*p={K*p}); "
            f"reduce max_lag."
        )
    Z = np.empty((T_eff, p * K), dtype=np.float64)
    for i in range(p):
        # block i corresponds to lag (i+1).
        # Target row t (0..T_eff-1) is X[p+t]; lag-(i+1) is X[p+t-(i+1)] = X[p-1-i+t].
        Z[:, i * K : (i + 1) * K] = X[p - 1 - i : T_eff + p - 1 - i]
    Y = X[p:]
    Z_full = np.hstack([Z, np.ones((T_eff, 1))])
    coef, *_ = np.linalg.lstsq(Z_full, Y, rcond=None)        # [pK+1, K]
    A_stacked = coef[:-1]                                     # [pK, K]
    c         = coef[-1]                                      # [K]
    A_list    = [A_stacked[i * K : (i + 1) * K] for i in range(p)]
    E         = Y - Z_full @ coef
    return c, A_list, E


def _info_criteria(E: np.ndarray, p: int, K: int) -> dict:
    """Compute logdet of ML residual cov + AIC/BIC/HQIC for VAR(p)."""
    T_eff = E.shape[0]
    Sigma = (E.T @ E) / T_eff
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        # Numerical degeneracy — add a tiny ridge and retry.
        Sigma = Sigma + 1e-8 * np.eye(K)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            raise ValueError(f"residual cov matrix not positive definite at p={p}")
    n_params = p * K * K
    aic  = logdet + (2.0                       * n_params) / T_eff
    bic  = logdet + (np.log(T_eff)             * n_params) / T_eff
    hqic = logdet + (2.0 * np.log(np.log(T_eff)) * n_params) / T_eff
    return {"logdet": float(logdet), "aic": float(aic),
            "bic": float(bic),       "hqic": float(hqic),
            "T_eff": int(T_eff),     "n_params": int(n_params)}


def _make_step_window_fn(c: np.ndarray, A_list: list[np.ndarray]):
    """Return a fn that consumes a [p, K] window (lag-1 = window[-1]) and emits x_next."""
    p = len(A_list)
    def step(window: np.ndarray) -> np.ndarray:
        x_next = c.copy()
        for i in range(p):
            x_next += window[p - 1 - i] @ A_list[i]
        return x_next
    return step


# ── Misc ────────────────────────────────────────────────────────────────────

def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _build_config(args, n_iv: int, n_train: int, n_val: int, n_test: int,
                  train_end_date: str, ic_table: dict, selected_lag: int) -> dict:
    return {
        "model":           "VAR",
        "dataset":         args.dataset,
        "target_space":    args.target_space,
        "csv_path":        args.csv_path,
        "seq_len":         args.seq_len,
        "pred_len":        args.pred_len,
        "n_iv":            n_iv,
        "n_train":         n_train,
        "n_val":           n_val,
        "n_test":          n_test,
        "train_end_date":  train_end_date,
        "seed":            args.seed,
        "loss":            "ols",
        "criterion":       args.criterion,
        "max_lag":         args.max_lag,
        "selected_lag":    selected_lag,
        "ic_table":        ic_table,
        "git_commit":      _git_commit(),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VAR(p) with AIC/BIC lag selection on SPX IV surface")
    ap.add_argument("--csv_path",     default="SPX_surfaces.csv")
    ap.add_argument("--dataset",      default="full", choices=DATASET_CHOICES,
                    help="full = entire CSV; precovid = dates <= 2019-12-31")
    ap.add_argument("--target_space", default="level", choices=TARGET_SPACE_CHOICES,
                    help="level = train on raw IV (default); logdiff = train on "
                         "log(IV)[1:]-log(IV)[:-1]. logdiff loses one day at the front.")
    ap.add_argument("--seq_len",      type=int, default=21)
    ap.add_argument("--pred_len",     type=int, default=63)
    ap.add_argument("--max_lag",      type=int, default=6,
                    help="Largest p considered in the AIC/BIC sweep (must satisfy "
                         "T_eff > K·p; with K=170 and T_train≈1937, p<11 is safe).")
    ap.add_argument("--criterion",    default="bic", choices=CRITERION_CHOICES,
                    help="Information criterion used to pick p (default bic).")
    ap.add_argument("--out_dir",      default=None)
    ap.add_argument("--predict_only", action="store_true",
                    help="Skip the run banner; read config.json from --out_dir, "
                         "refit only at config.selected_lag (fast), write pred.npy.")
    ap.add_argument("--device",       default="auto",
                    help="Ignored: VAR is pure numpy/sklearn, no GPU.")
    ap.add_argument("--seed",         type=int, default=42,
                    help="Ignored: OLS is deterministic.")
    args = ap.parse_args()

    # predict_only: read config and short-circuit the sweep.
    forced_lag: int | None = None
    if args.predict_only:
        if args.out_dir is None:
            raise SystemExit("--predict_only requires --out_dir <existing dir with config.json>")
        cfg_path = os.path.join(args.out_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise SystemExit(f"--predict_only: config.json not found at {cfg_path}")
        with open(cfg_path) as f:
            cfg = json.load(f)
        args.csv_path     = cfg.get("csv_path",     args.csv_path)
        args.dataset      = cfg.get("dataset",      args.dataset)
        args.target_space = cfg.get("target_space", args.target_space)
        args.seq_len      = cfg.get("seq_len",      args.seq_len)
        args.pred_len     = cfg.get("pred_len",     args.pred_len)
        args.criterion    = cfg.get("criterion",    args.criterion)
        args.max_lag      = cfg.get("max_lag",      args.max_lag)
        forced_lag        = int(cfg["selected_lag"])
        print(f"[predict_only] loaded config from {cfg_path} (selected_lag={forced_lag})")

    # Defer out_dir choice until selected_lag is known so it lands in the name.
    print(f"Dataset    : {args.dataset}")
    print(f"Target     : {args.target_space}")

    dates, iv, iv_cols = _load(args.csv_path, args.dataset, args.target_space)
    n_iv = len(iv_cols)
    T = len(iv)
    n_train = int(T * TRAIN_FRAC)
    n_test  = int(T * TEST_FRAC)
    n_val   = T - n_train - n_test
    train_end_date = str(dates[n_train - 1])

    scaler = StandardScaler().fit(iv[:n_train])
    iv_sc  = scaler.transform(iv).astype(np.float64)

    print(f"T={T}  n_train={n_train}  n_val={n_val}  n_test={n_test}  n_iv={n_iv}")
    print(f"Train ends {train_end_date}")

    # ── Lag sweep ──
    if forced_lag is not None:
        candidate_ps = [forced_lag]
    else:
        candidate_ps = list(range(1, args.max_lag + 1))

    ic_table: dict = {}
    fits: dict     = {}
    print(f"Fitting VAR(p) for p in {candidate_ps} ...")
    print(f"  {'p':>3}  {'logdet':>12}  {'aic':>12}  {'bic':>12}  {'hqic':>12}  {'T_eff':>6}")
    for p in candidate_ps:
        try:
            c, A_list, E = _fit_var_p(iv_sc[:n_train], p)
        except ValueError as err:
            print(f"  p={p}: skipped — {err}")
            continue
        ic = _info_criteria(E, p, n_iv)
        ic_table[str(p)] = ic
        fits[p] = (c, A_list)
        print(f"  {p:>3}  {ic['logdet']:>+12.4f}  {ic['aic']:>+12.4f}  "
              f"{ic['bic']:>+12.4f}  {ic['hqic']:>+12.4f}  {ic['T_eff']:>6}")

    if not fits:
        raise SystemExit("All VAR(p) fits failed — reduce --max_lag.")

    if forced_lag is not None:
        selected_lag = forced_lag
        print(f"[predict_only] using selected_lag={selected_lag} from config")
    else:
        selected_lag = min(fits.keys(), key=lambda p: ic_table[str(p)][args.criterion])
        print(f"Selected lag p={selected_lag} by {args.criterion.upper()}")

    if args.out_dir is None:
        args.out_dir = (f"VAR/results/"
                        f"{args.dataset}_{args.target_space}_SPX_IV_"
                        f"{args.seq_len}_{args.pred_len}_VAR_"
                        f"p{selected_lag}_{args.criterion}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir : {args.out_dir}")

    # ── Rollout with the chosen p ──
    c, A_list = fits[selected_lag]
    step = _make_step_window_fn(c, A_list)
    p = selected_lag

    first_start = T - n_test
    last_start  = T - args.pred_len
    start_indices = np.arange(first_start, last_start + 1, dtype=int)
    n_windows = len(start_indices)
    if first_start - p < 0:
        raise SystemExit(f"selected_lag={p} too large for first test window "
                         f"(need {p} obs before s={first_start})")
    print(f"Test windows: {n_windows}  "
          f"({dates[start_indices[0]]} → {dates[start_indices[-1]]})")

    preds       = np.zeros((n_windows, args.pred_len, n_iv), dtype=np.float32)
    start_dates = np.zeros(n_windows, dtype="datetime64[D]")

    for wi, s in enumerate(start_indices):
        window = iv_sc[s - p : s].copy()      # [p, K], lag-1 is window[-1]
        for h in range(args.pred_len):
            x_next = step(window)
            preds[wi, h, :] = x_next
            window = np.vstack([window[1:], x_next[None]])
        start_dates[wi] = dates[s]

    np.save(os.path.join(args.out_dir, "pred.npy"),        preds)
    np.save(os.path.join(args.out_dir, "start_dates.npy"), start_dates)

    config = _build_config(args, n_iv, n_train, n_val, n_test, train_end_date,
                           ic_table, selected_lag)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved pred.npy {preds.shape}  start_dates.npy {start_dates.shape}")
    print(f"Done. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
