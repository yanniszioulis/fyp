#!/usr/bin/env python3
"""
Statistical significance + outlier robustness for SPX IV forecasting models.

Reuses the `compare_models` pipeline (same `build_reference`, model registry,
loaders, date alignment) — no model retraining. For every registered model with
an existing `pred.npy`, reports:

  1. Per-window MSE distribution (mean, median, p-quantiles, max).
  2. Trimmed MSE — per-model trim (drop each model's own worst k% windows)
     and common-window trim (rank windows by median MSE across models, drop
     top k% from every model). k ∈ {1, 5, 10}.
  3. Top-10 worst common days with each model's per-window MSE.
  4. Diebold-Mariano test vs Persist on the per-window mean MSE (HAC variance
     with Bartlett kernel, bandwidth = pred_len since consecutive windows share
     pred_len-1 of pred_len forecast days).
  5. Per-horizon DM tests at t+{1, 5, 10, 21, 42, 63}. Same H₀ (equal MSE) but
     bandwidth = ceil(4·(N/100)^(2/9)) (Newey-West rule; per-horizon losses
     don't share days, only market autocorrelation).
  6. Holm-Bonferroni adjustment within each row of (4) and (5): family = the
     ~7 model-vs-Persist tests at that horizon condition.

DM sign convention: d = loss_Persist − loss_model. Positive DM stat means the
model significantly outperforms persistence.

Outputs (in --out_dir):
    {dataset}_sl{}_pl{}_significance.csv   distribution + trimmed table
    {dataset}_sl{}_pl{}_dm_tests.csv       DM stats + p-values + Holm
    {dataset}_sl{}_pl{}_worst_days.csv     top 10 worst common days
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from compare_models import (
    DATASET_CHOICES, DEFAULT_PRED_LEN, DEFAULT_SEQ_LEN, LOADERS,
    REPORT_HORIZONS, TARGET_SPACE_CHOICES, _maybe_regen, align_to_ref,
    build_models, build_reference,
)

TRIM_LEVELS = [0.01, 0.05, 0.10]


# ─── HAC variance + DM test ───────────────────────────────────────────────────

def _hac_variance(d: np.ndarray, L: int) -> float:
    """
    Newey-West HAC long-run variance estimator with Bartlett kernel weights.

    HAC_var(d) = γ_0 + 2 · Σ_{l=1..L} (1 - l/(L+1)) · γ_l

    γ_l is the lag-l autocovariance of `d` (biased estimator with divisor N).
    Returns 0 if all elements of d are equal (degenerate case; caller handles NaN).
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    dc = d - d.mean()
    g0 = float(np.mean(dc * dc))
    var = g0
    for l in range(1, min(L, n - 1) + 1):
        gl = float(np.mean(dc[l:] * dc[:-l]))
        w  = 1.0 - l / (L + 1)
        var += 2.0 * w * gl
    return max(var, 0.0)   # truncate negative HAC estimates at 0


def _dm_test(diff: np.ndarray, hac_lag: int) -> tuple[float, float]:
    """
    Two-sided Diebold-Mariano test on a loss differential array.

    Convention (caller's responsibility): pass diff = loss_baseline - loss_model
    so that positive DM stat ↔ model significantly better than baseline.

    Returns (dm_stat, p_value). NaN/NaN if variance is degenerate.
    """
    n = len(diff)
    var_d = _hac_variance(diff, hac_lag)
    if var_d <= 0 or n < 2:
        return float("nan"), float("nan")
    se = math.sqrt(var_d / n)
    if se == 0:
        return float("nan"), float("nan")
    dm = float(np.mean(diff)) / se
    p  = 2.0 * (1.0 - norm.cdf(abs(dm)))
    return dm, p


def _holm_adjust(pvals: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down adjustment over a family of K tests.
    NaN p-values are passed through unchanged. Output is monotone non-decreasing
    in the order of ascending raw p-value.
    """
    arr   = np.array(pvals, dtype=np.float64)
    valid = ~np.isnan(arr)
    if not valid.any():
        return list(arr)
    idx_valid = np.where(valid)[0]
    sub = arr[idx_valid]
    K   = len(sub)
    order = np.argsort(sub)
    adj_sub = np.empty(K)
    prev = 0.0
    for i, j in enumerate(order):
        a = min(1.0, sub[j] * (K - i))
        prev = max(prev, a)
        adj_sub[j] = prev
    out = arr.copy()
    out[idx_valid] = adj_sub
    return list(out)


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_all_preds(dataset: str, target_space: str, seq_len: int, pred_len: int,
                   csv_path: str):
    """
    Build reference and load every registered model's pred.npy. Returns:
        ref_dates  [N]
        trues      [N, pred_len, n_iv]
        preds      dict[name → [N, pred_len, n_iv]]   includes 'Persist(ref)'
        meta       reference meta dict
    """
    print(f"Building reference (dataset={dataset}, target_space={target_space}, "
          f"seq_len={seq_len}, pred_len={pred_len})...")
    trues, persist, ref_dates, meta = build_reference(
        csv_path, dataset, target_space, seq_len, pred_len)
    n_iv = meta["n_iv"]
    print(f"  T={meta['T']}, n_iv={n_iv}, test windows={meta['n_test']}, "
          f"{meta['date_range']}")

    preds = {"Persist(ref)": persist}

    for spec in build_models(dataset, target_space, seq_len, pred_len):
        name = spec["name"]
        pred_path = _maybe_regen(spec, name)
        if pred_path is None:
            print(f"  {name}: no result dir — skipped")
            continue
        sibling_dates = os.path.join(os.path.dirname(pred_path), "start_dates.npy")
        try:
            pred = LOADERS[spec["loader"]](pred_path, n_iv)
        except Exception as e:
            print(f"  {name}: load error — {e}")
            continue
        if os.path.exists(sibling_dates):
            pred_dates = np.load(sibling_dates).astype("datetime64[D]")
            try:
                pred = align_to_ref(pred, pred_dates, ref_dates, name)
            except ValueError as e:
                print(f"  {name}: date alignment failed — {e}")
                continue
        else:
            n = min(len(pred), len(trues))
            pred = pred[:n]
        if pred.shape != trues.shape:
            print(f"  {name}: shape mismatch {pred.shape} vs {trues.shape} — skipped")
            continue
        preds[name] = pred
        print(f"  {name}: OK")

    return ref_dates, trues, preds, meta


# ─── Per-window MSE ───────────────────────────────────────────────────────────

def per_window_mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """[N, pred_len, n_iv] → [N] mean squared error over horizons × features per window."""
    err = pred - true
    return np.mean(err * err, axis=(1, 2))


def per_window_horizon_mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """[N, pred_len, n_iv] → [N, pred_len] mean over features at each horizon."""
    err = pred - true
    return np.mean(err * err, axis=2)


# ─── Distribution + trimmed tables ────────────────────────────────────────────

def distribution_stats(pwm: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, x in pwm.items():
        rows.append({
            "model":  name,
            "n":      len(x),
            "mean":   float(np.mean(x)),
            "median": float(np.median(x)),
            "std":    float(np.std(x, ddof=1)),
            "p25":    float(np.percentile(x, 25)),
            "p75":    float(np.percentile(x, 75)),
            "p95":    float(np.percentile(x, 95)),
            "p99":    float(np.percentile(x, 99)),
            "max":    float(np.max(x)),
        })
    return pd.DataFrame(rows)


def trimmed_per_model(pwm: dict[str, np.ndarray], levels: list[float]) -> pd.DataFrame:
    """
    For each model, drop its OWN worst k% windows (by per-window MSE) and
    recompute the mean. Each model's trim mask is independent.
    """
    rows = []
    for name, x in pwm.items():
        n = len(x)
        row = {"model": name, "full_mean": float(np.mean(x))}
        for k in levels:
            keep = max(1, n - int(np.ceil(n * k)))
            row[f"perModel_trim{int(k*100)}pct"] = float(np.mean(np.sort(x)[:keep]))
        rows.append(row)
    return pd.DataFrame(rows)


def trimmed_common(pwm: dict[str, np.ndarray], levels: list[float]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Rank windows by MEDIAN MSE across models (descending = worst first). Drop
    the top k% of windows globally; apply the same mask to every model.

    Returns (df, median_per_window_array).
    """
    names = list(pwm.keys())
    M = np.stack([pwm[n] for n in names], axis=0)   # [n_models, N]
    median_per_window = np.median(M, axis=0)        # [N]
    order_worst = np.argsort(-median_per_window)    # descending
    n = M.shape[1]

    rows = []
    for name in names:
        x = pwm[name]
        row = {"model": name, "full_mean": float(np.mean(x))}
        for k in levels:
            n_drop = int(np.ceil(n * k))
            drop_idx = set(order_worst[:n_drop].tolist())
            keep_mask = np.array([i not in drop_idx for i in range(n)])
            row[f"commonTrim{int(k*100)}pct"] = float(np.mean(x[keep_mask]))
        rows.append(row)
    return pd.DataFrame(rows), median_per_window


def worst_common_days(pwm: dict[str, np.ndarray], dates: np.ndarray,
                       median_per_window: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """Top-N worst windows ranked by median MSE; row per window, column per model."""
    order = np.argsort(-median_per_window)[:top_n]
    rows = []
    for rank, idx in enumerate(order, start=1):
        row = {
            "rank":       rank,
            "date":       str(dates[idx]),
            "median_mse": float(median_per_window[idx]),
        }
        for name, x in pwm.items():
            row[name] = float(x[idx])
        rows.append(row)
    return pd.DataFrame(rows)


# ─── DM tests ────────────────────────────────────────────────────────────────

def dm_vs_persist(pwm: dict[str, np.ndarray],
                   pwm_per_horizon: dict[str, np.ndarray],
                   pred_len: int) -> pd.DataFrame:
    """
    Run DM test of every model vs Persist on (a) overall per-window MSE and
    (b) per-window per-horizon MSE at every horizon in REPORT_HORIZONS.

    Sign: d = loss_persist - loss_model → positive DM = model better than Persist.
    HAC bandwidth: pred_len for overall; Newey-West rule for per-horizon.
    Holm-Bonferroni applied per-row (within each horizon condition).
    """
    if "Persist(ref)" not in pwm:
        raise ValueError("Persist(ref) missing from preds")

    persist_overall = pwm["Persist(ref)"]
    persist_per_h   = pwm_per_horizon["Persist(ref)"]

    n_overall = len(persist_overall)
    nw_lag    = max(1, int(math.ceil(4.0 * (n_overall / 100.0) ** (2.0 / 9.0))))

    other_models = [n for n in pwm.keys() if n != "Persist(ref)"]

    # Family per horizon condition: collect raw p-values, then Holm-adjust.
    rows: list[dict] = []

    # Overall
    overall_records: list[dict] = []
    for name in other_models:
        d = persist_overall - pwm[name]
        dm, p = _dm_test(d, hac_lag=pred_len)
        overall_records.append({
            "model":          name,
            "horizon":        "overall",
            "hac_bandwidth":  pred_len,
            "n":              n_overall,
            "mean_diff":      float(np.mean(d)),
            "dm_stat":        dm,
            "p_value":        p,
        })
    holm = _holm_adjust([r["p_value"] for r in overall_records])
    for r, ph in zip(overall_records, holm):
        r["p_value_holm"] = ph
        rows.append(r)

    # Per horizon
    for h in REPORT_HORIZONS:
        idx = h - 1
        recs: list[dict] = []
        for name in other_models:
            d = persist_per_h[:, idx] - pwm_per_horizon[name][:, idx]
            dm, p = _dm_test(d, hac_lag=nw_lag)
            recs.append({
                "model":         name,
                "horizon":       f"t+{h}",
                "hac_bandwidth": nw_lag,
                "n":             n_overall,
                "mean_diff":     float(np.mean(d)),
                "dm_stat":       dm,
                "p_value":       p,
            })
        holm = _holm_adjust([r["p_value"] for r in recs])
        for r, ph in zip(recs, holm):
            r["p_value_holm"] = ph
            rows.append(r)

    return pd.DataFrame(rows)


# ─── Display ─────────────────────────────────────────────────────────────────

def _print_df(title: str, df: pd.DataFrame, float_cols: list[str] = None,
              int_cols: list[str] = None, max_col_width: int = 16):
    print(f"\n{title}")
    print("-" * len(title))
    formatted = df.copy()
    if float_cols:
        for c in float_cols:
            if c in formatted.columns:
                formatted[c] = formatted[c].apply(
                    lambda v: ("nan" if v != v else f"{v:.6f}")
                )
    if int_cols:
        for c in int_cols:
            if c in formatted.columns:
                formatted[c] = formatted[c].apply(
                    lambda v: "" if v != v else f"{int(v)}"
                )
    with pd.option_context("display.max_columns", None,
                            "display.width", 200,
                            "display.max_colwidth", max_col_width):
        print(formatted.to_string(index=False))


def _annotate_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'sig' column with stars based on Holm-adjusted p-value."""
    out = df.copy()
    def stars(p):
        if p != p: return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""
    out["sig"] = out["p_value_holm"].apply(stars)
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DM significance + trimmed MSE for SPX IV models")
    ap.add_argument("--csv_path", default="SPX_surfaces.csv")
    ap.add_argument("--dataset",  default="full", choices=DATASET_CHOICES)
    ap.add_argument("--target_space", default="level", choices=TARGET_SPACE_CHOICES,
                    help="level or logdiff. Persist baseline becomes "
                         "predict-zero-change in logdiff space.")
    ap.add_argument("--seq_len",  type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--pred_len", type=int, default=DEFAULT_PRED_LEN)
    ap.add_argument("--out_dir",  default="comparison_results")
    args = ap.parse_args()

    ref_dates, trues, preds, meta = load_all_preds(
        args.dataset, args.target_space, args.seq_len, args.pred_len, args.csv_path,
    )
    if len(preds) <= 1:
        print("\nNo non-baseline models loaded — nothing to test.")
        return

    # Per-window losses
    pwm           = {name: per_window_mse(p, trues)         for name, p in preds.items()}
    pwm_per_horiz = {name: per_window_horizon_mse(p, trues) for name, p in preds.items()}

    # 1. Distribution
    df_dist = distribution_stats(pwm)
    _print_df("Per-window MSE distribution",
              df_dist,
              float_cols=["mean", "median", "std", "p25", "p75", "p95", "p99", "max"],
              int_cols=["n"])

    # 2. Trimmed MSE
    df_trim_pm = trimmed_per_model(pwm, TRIM_LEVELS)
    _print_df("Trimmed MSE — per-model trim (each model's own worst k%)",
              df_trim_pm,
              float_cols=["full_mean"] + [f"perModel_trim{int(k*100)}pct" for k in TRIM_LEVELS])

    df_trim_co, median_pw = trimmed_common(pwm, TRIM_LEVELS)
    _print_df("Trimmed MSE — common-window trim (rank by median MSE across models)",
              df_trim_co,
              float_cols=["full_mean"] + [f"commonTrim{int(k*100)}pct" for k in TRIM_LEVELS])

    # 3. Worst common days
    df_worst = worst_common_days(pwm, ref_dates, median_pw, top_n=10)
    _print_df("Top-10 worst common days (ranked by median MSE across models)",
              df_worst,
              float_cols=["median_mse"] + [n for n in pwm.keys()],
              int_cols=["rank"], max_col_width=20)

    # 4+5. DM tests
    df_dm = dm_vs_persist(pwm, pwm_per_horiz, args.pred_len)
    df_dm_show = _annotate_significance(df_dm)
    _print_df("Diebold-Mariano vs Persist  (positive DM ↔ model beats Persist)",
              df_dm_show,
              float_cols=["mean_diff", "dm_stat", "p_value", "p_value_holm"],
              int_cols=["hac_bandwidth", "n"])

    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05 (Holm-adjusted within horizon family).")

    # ── Save ──
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.target_space}_sl{args.seq_len}_pl{args.pred_len}"

    sig_path   = os.path.join(args.out_dir, f"{tag}_significance.csv")
    dm_path    = os.path.join(args.out_dir, f"{tag}_dm_tests.csv")
    worst_path = os.path.join(args.out_dir, f"{tag}_worst_days.csv")

    # Combine distribution + trims into one significance file (one row per model)
    df_sig = (df_dist
              .merge(df_trim_pm.drop(columns=["full_mean"]), on="model", how="left")
              .merge(df_trim_co.drop(columns=["full_mean"]), on="model", how="left"))
    df_sig.to_csv(sig_path, index=False)
    df_dm.to_csv(dm_path, index=False)
    df_worst.to_csv(worst_path, index=False)

    print(f"\nSaved to {args.out_dir}/")
    print(f"  {os.path.basename(sig_path)}")
    print(f"  {os.path.basename(dm_path)}")
    print(f"  {os.path.basename(worst_path)}")


if __name__ == "__main__":
    main()
