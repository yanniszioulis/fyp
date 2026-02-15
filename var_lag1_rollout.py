#!/usr/bin/env python3
"""
Lag-1 VAR rollout on SPX implied-vol surface vectors.

- Loads /Users/yanniszioulis/Documents/FYP_real/fyp/SPX_surfaces.csv
- Uses the final 20% of days as a test region (by date order in the file)
- Defines a "test day" as the first day of the prediction horizon
- For each test start day t:
    - context = previous 21 days (t-21 ... t-1)
    - fit VAR(1) on context (lag=1), coefficients fixed
    - roll out 63 steps (t ... t+62)

Saves outputs under ./var_lag1_results/:
- pred.npy: [n_test_windows, 63, n_iv_features]
- true.npy: [n_test_windows, 63, n_iv_features]
- start_dates.npy: [n_test_windows] (datetime64[D])
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    csv_path: str
    context_len: int
    horizon_len: int
    test_frac: float
    ridge_lambda: float
    tune_ridge: bool
    lambda_grid: list[float]
    tune_max_windows: int | None
    scale: bool
    max_windows: int | None
    verbose: bool
    debug_windows: int
    out_dir: str


def load_iv_matrix(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    iv_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("iv_")]
    if not iv_cols:
        raise ValueError("No iv_* columns found.")
    dates = df["date"].to_numpy(dtype="datetime64[D]")
    x = df[iv_cols].to_numpy(dtype=np.float64, copy=True)
    return dates, x, iv_cols


def standardize_train_only(x: np.ndarray, train_end_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x[:train_end_idx].mean(axis=0)
    sigma = x[:train_end_idx].std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    xz = (x - mu) / sigma
    return xz, mu, sigma


def fit_var1_stepper(context: np.ndarray, ridge_lambda: float, intercept: bool = True):
    """
    Fit VAR(1) in dual form with ridge.

    context: [T, k], T=context_len
    Returns: step(x)->x_next, where x is [k].
    """
    if context.shape[0] < 2:
        raise ValueError("Need at least 2 points to fit VAR(1).")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be >= 0.")

    X = context[:-1]  # [n, k]
    Y = context[1:]  # [n, k]
    n = X.shape[0]

    if intercept:
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)
        Xc = X - mu_x
        Yc = Y - mu_y
    else:
        mu_x = None
        mu_y = None
        Xc = X
        Yc = Y

    G = Xc @ Xc.T  # [n, n]
    if ridge_lambda > 0:
        G = G + ridge_lambda * np.eye(n, dtype=G.dtype)

    YT = Yc.T  # [k, n]

    chol = None
    if ridge_lambda > 0:
        try:
            chol = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            chol = None

    def step(x_t: np.ndarray) -> np.ndarray:
        if intercept:
            x_in = x_t - mu_x
        else:
            x_in = x_t
        v = Xc @ x_in  # [n]
        if chol is not None:
            y = np.linalg.solve(chol, v)
            w = np.linalg.solve(chol.T, y)
        else:
            w = np.linalg.solve(G, v)  # [n]
        out = YT @ w  # [k]
        if intercept:
            out = out + mu_y
        return out

    return step


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    diff = pred.astype(np.float64, copy=False) - true.astype(np.float64, copy=False)
    return float(np.mean(diff * diff))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    diff = pred.astype(np.float64, copy=False) - true.astype(np.float64, copy=False)
    return float(np.mean(np.abs(diff)))


def rse(pred: np.ndarray, true: np.ndarray) -> float:
    pred64 = pred.astype(np.float64, copy=False)
    true64 = true.astype(np.float64, copy=False)
    diff = true64 - pred64
    num = np.sqrt(np.sum(diff * diff))
    denom_diff = true64 - true64.mean()
    den = np.sqrt(np.sum(denom_diff * denom_diff))
    return float(num / den)


def per_horizon_metrics(pred: np.ndarray, true: np.ndarray):
    h = pred.shape[1]
    out = {
        "mse": np.zeros(h, dtype=np.float64),
        "mae": np.zeros(h, dtype=np.float64),
        "rse": np.zeros(h, dtype=np.float64),
    }
    for i in range(h):
        out["mse"][i] = mse(pred[:, i : i + 1, :], true[:, i : i + 1, :])
        out["mae"][i] = mae(pred[:, i : i + 1, :], true[:, i : i + 1, :])
        out["rse"][i] = rse(pred[:, i : i + 1, :], true[:, i : i + 1, :])
    return out


def _fmt_date(d: np.datetime64) -> str:
    return str(d.astype("datetime64[D]"))


def _parse_lambda_grid(s: str | None) -> list[float]:
    if s is None or not str(s).strip():
        grid = np.logspace(-6, 4, 11)  # 1e-6 ... 1e4
        return [float(x) for x in grid]
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        out.append(float(p))
    if any(x < 0 for x in out):
        raise ValueError("lambda_grid values must be >= 0.")
    return out


def _score_lambda(
    x: np.ndarray,
    start_indices: np.ndarray,
    *,
    context_len: int,
    horizon_len: int,
    ridge_lambda: float,
    intercept: bool,
) -> float:
    sse = 0.0
    n = 0
    for s in start_indices:
        context = x[s - context_len : s]
        step = fit_var1_stepper(context, ridge_lambda=ridge_lambda, intercept=intercept)
        x_t = context[-1]
        true = x[s : s + horizon_len]
        for h in range(horizon_len):
            x_t = step(x_t)
            diff = x_t.astype(np.float64, copy=False) - true[h].astype(np.float64, copy=False)
            sse += float(diff @ diff)
            n += diff.shape[0]
    return sse / n


def main():
    parser = argparse.ArgumentParser(description="VAR(1) context-fit rollout on SPX IV vectors.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/Users/yanniszioulis/Documents/FYP_real/fyp/SPX_surfaces.csv",
    )
    parser.add_argument("--context_len", type=int, default=21)
    parser.add_argument("--horizon_len", type=int, default=63)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--ridge_lambda", type=float, default=1.0)
    parser.add_argument("--intercept", action="store_true", default=True)
    parser.add_argument("--no_intercept", action="store_true", default=False)
    parser.add_argument(
        "--tune_ridge",
        action="store_true",
        default=False,
        help="Grid-search ridge_lambda to minimize global MSE on the test windows.",
    )
    parser.add_argument(
        "--lambda_grid",
        type=str,
        default=None,
        help="Comma-separated ridge lambdas. Default: logspace(1e-6..1e4).",
    )
    parser.add_argument(
        "--tune_max_windows",
        type=int,
        default=None,
        help="Optionally tune on only the first N test windows (faster).",
    )
    parser.add_argument("--scale", action="store_true", default=True)
    parser.add_argument("--no_scale", action="store_true", default=False)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument(
        "--debug_windows",
        type=int,
        default=3,
        help="How many early test windows to print deeper diagnostics for.",
    )
    parser.add_argument("--out_dir", type=str, default="var_lag1_results")
    args = parser.parse_args()

    cfg = Config(
        csv_path=args.csv_path,
        context_len=int(args.context_len),
        horizon_len=int(args.horizon_len),
        test_frac=float(args.test_frac),
        ridge_lambda=float(args.ridge_lambda),
        tune_ridge=bool(args.tune_ridge),
        lambda_grid=_parse_lambda_grid(args.lambda_grid),
        tune_max_windows=None if args.tune_max_windows is None else int(args.tune_max_windows),
        scale=bool(args.scale) and (not bool(args.no_scale)),
        max_windows=None if args.max_windows is None else int(args.max_windows),
        verbose=bool(args.verbose) and (not bool(args.quiet)),
        debug_windows=int(args.debug_windows),
        out_dir=str(args.out_dir),
    )

    dates, x, iv_cols = load_iv_matrix(cfg.csv_path)
    N, K = x.shape

    if not (0.0 < cfg.test_frac < 1.0):
        raise ValueError("test_frac must be in (0, 1).")
    if cfg.context_len < 1:
        raise ValueError("context_len must be >= 1.")
    if cfg.horizon_len < 1:
        raise ValueError("horizon_len must be >= 1.")
    if N < cfg.context_len + cfg.horizon_len:
        raise ValueError("Not enough rows for context+horizon.")

    split_idx = int(np.floor(N * (1.0 - cfg.test_frac)))
    split_idx = max(split_idx, cfg.context_len)

    use_intercept = bool(args.intercept) and (not bool(args.no_intercept))

    if cfg.verbose:
        print("=== Data ===")
        print(f"csv_path: {cfg.csv_path}")
        print(f"rows: {N}, iv_features: {K}")
        print(f"date_range: {_fmt_date(dates[0])} .. {_fmt_date(dates[-1])}")
        print(f"context_len: {cfg.context_len}, horizon_len: {cfg.horizon_len}, lag: 1")
        print(f"test_frac: {cfg.test_frac:.3f} (final {cfg.test_frac*100:.1f}%)")
        print(f"ridge_lambda: {cfg.ridge_lambda:g}{' (tune_ridge=True)' if cfg.tune_ridge else ''}")
        print(f"intercept: {use_intercept}")
        print(f"scale(train_only): {cfg.scale}")
        print(f"split_idx: {split_idx} (split_date={_fmt_date(dates[split_idx])})")

    if cfg.scale:
        x, mu, sigma = standardize_train_only(x, train_end_idx=split_idx)
        if cfg.verbose:
            print("\n=== Scaling diagnostics (train-only Standardization) ===")
            print(
                f"mu: mean={float(mu.mean()):.6f}, std={float(mu.std()):.6f}, "
                f"min={float(mu.min()):.6f}, max={float(mu.max()):.6f}"
            )
            print(
                f"sigma: mean={float(sigma.mean()):.6f}, std={float(sigma.std()):.6f}, "
                f"min={float(sigma.min()):.6f}, max={float(sigma.max()):.6f}"
            )

    first_start = split_idx
    last_start = N - cfg.horizon_len  # inclusive index for start day
    if last_start < first_start:
        raise ValueError("No valid test start days with full horizon inside dataset.")

    start_indices = np.arange(first_start, last_start + 1, dtype=int)
    if cfg.max_windows is not None:
        start_indices = start_indices[: cfg.max_windows]

    nW = len(start_indices)
    if cfg.verbose:
        print("\n=== Windowing ===")
        print(f"test_start_index_range: [{first_start}, {last_start}] (inclusive)")
        print(f"n_test_windows: {nW}")
        if nW > 0:
            print(f"first_test_start_date: {_fmt_date(dates[start_indices[0]])}")
            print(f"last_test_start_date:  {_fmt_date(dates[start_indices[-1]])}")

    ridge_lambda_run = cfg.ridge_lambda
    if cfg.tune_ridge:
        tune_indices = start_indices
        if cfg.tune_max_windows is not None:
            tune_indices = tune_indices[: cfg.tune_max_windows]
        if cfg.verbose:
            print("\n=== Ridge tuning (minimize global MSE) ===")
            print(f"tuning_windows: {len(tune_indices)}")
            print(f"lambda_grid: {cfg.lambda_grid}")
        best_lam = None
        best_mse = None
        mses = []
        for lam in cfg.lambda_grid:
            m = _score_lambda(
                x,
                tune_indices,
                context_len=cfg.context_len,
                horizon_len=cfg.horizon_len,
                ridge_lambda=float(lam),
                intercept=use_intercept,
            )
            mses.append((float(lam), float(m)))
            if best_mse is None or m < best_mse:
                best_mse = m
                best_lam = float(lam)
        ridge_lambda_run = float(best_lam)
        if cfg.verbose:
            mses_sorted = sorted(mses, key=lambda t: t[1])
            print("top_lambda_by_mse:")
            for lam, m in mses_sorted[: min(8, len(mses_sorted))]:
                print(f"  lambda={lam:g}  mse={m:.10f}")
            print(f"selected_ridge_lambda: {ridge_lambda_run:g} (mse={float(best_mse):.10f})")

    preds = np.zeros((nW, cfg.horizon_len, K), dtype=np.float32)
    trues = np.zeros((nW, cfg.horizon_len, K), dtype=np.float32)
    start_dates = np.zeros((nW,), dtype="datetime64[D]")

    for wi, s in enumerate(start_indices):
        context = x[s - cfg.context_len : s]  # [21, K]
        step = fit_var1_stepper(context, ridge_lambda=ridge_lambda_run, intercept=use_intercept)

        x_t = context[-1]
        if cfg.verbose and wi < cfg.debug_windows:
            print("\n--- VAR window debug ---")
            print(f"window_idx: {wi}")
            print(f"start_day_index: {s} (start_date={_fmt_date(dates[s])})")
            print(f"context_date_range: {_fmt_date(dates[s-cfg.context_len])} .. {_fmt_date(dates[s-1])}")
            print(f"context_last_vector_l2: {float(np.linalg.norm(context[-1])):.6f}")

            X = context[:-1]
            if use_intercept:
                X = X - X.mean(axis=0)
            G = X @ X.T
            if ridge_lambda_run > 0:
                G = G + ridge_lambda_run * np.eye(G.shape[0], dtype=G.dtype)
            try:
                cond = float(np.linalg.cond(G))
            except Exception:
                cond = float("nan")
            print(f"G_shape: {G.shape}, cond(G): {cond:.3e}")

            sample_next = step(context[-1])
            growth = float(np.linalg.norm(sample_next) / (np.linalg.norm(context[-1]) + 1e-12))
            print(f"one_step_growth_l2: {growth:.6f}")

        for h in range(cfg.horizon_len):
            x_t = step(x_t)
            preds[wi, h, :] = x_t

        trues[wi, :, :] = x[s : s + cfg.horizon_len].astype(np.float32, copy=False)
        start_dates[wi] = dates[s]

    os.makedirs(cfg.out_dir, exist_ok=True)
    np.save(os.path.join(cfg.out_dir, "pred.npy"), preds)
    np.save(os.path.join(cfg.out_dir, "true.npy"), trues)
    np.save(os.path.join(cfg.out_dir, "start_dates.npy"), start_dates)

    if cfg.verbose:
        print("\n=== Prediction diagnostics ===")
        finite = np.isfinite(preds)
        print(
            f"preds finite: {int(finite.sum())}/{preds.size} "
            f"(nan={int(np.isnan(preds).sum())}, inf={int(np.isinf(preds).sum())})"
        )
        if finite.any():
            abs_vals = np.abs(preds[finite])
            print(f"preds absmax: {float(abs_vals.max()):.6e}")
            for q in [50, 90, 99, 99.9]:
                print(f"preds abs_percentile_{q:g}: {float(np.percentile(abs_vals, q)):.6e}")

        print("\n=== Metrics (overall, across all windows/horizons/features) ===")
        print(f"MSE: {mse(preds, trues):.10f}")
        print(f"MAE: {mae(preds, trues):.10f}")
        print(f"RSE: {rse(preds, trues):.10f}")

        ph = per_horizon_metrics(preds, trues)
        horizons_to_show = [1, 5, 10, 21, 42, 63]
        print("\n=== Metrics (per-horizon, averaged across windows/features) ===")
        print(f"{'Horizon':<10} {'MSE':>16} {'MAE':>16} {'RSE':>16}")
        for h in horizons_to_show:
            if 1 <= h <= cfg.horizon_len:
                i = h - 1
                print(f"t+{h:<7} {ph['mse'][i]:>16.10f} {ph['mae'][i]:>16.10f} {ph['rse'][i]:>16.10f}")


if __name__ == "__main__":
    main()

