#!/usr/bin/env python3
"""
Benchmark comparison of SPX IV surface forecasting models.

Ground truth is built directly from SPX_surfaces.csv using the canonical
70/10/20 train/val/test split (same formula as DynGWN and HOT).

All predictions are expected in scaled space (StandardScaler fit on train).
Models with start_dates.npy are aligned by date to the reference test windows.
Models without dates (PatchTST, Persistence) are assumed to align sequentially
from reference window 0 — see WARNING printed at runtime if T differs.

Metrics (all in scaled space):
  mse, rmse, mae, rse   standard regression metrics
  bias                  mean signed error; positive = over-prediction
  da                    directional accuracy: fraction where sign(pred)==sign(true)
  ic_mean, ic_std       Spearman rank-IC averaged across (window, step) over 400 features

Outputs saved to --out_dir (default: comparison_results/):
  YYYYMMDD_HHMMSS.csv   timestamped overall metrics
  latest.csv            always overwritten with most recent run
  latest_horizons.csv   per-horizon ic/mse/mae for every model
  latest_summary.json   machine-readable summary
"""

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# ─── Constants ────────────────────────────────────────────────────────────────

SEQ_LEN = 21
PRED_LEN = 63
TRAIN_FRAC = 0.70
TEST_FRAC = 0.20
N_IV = 400
REPORT_HORIZONS = [1, 5, 10, 21, 42, 63]

# ─── Model registry ───────────────────────────────────────────────────────────
# Glob patterns allowed. _find_latest() picks the most recently modified match.
# loader: "flat" expects [N, pred_len, 400]; "hot" expects [N, H_mono, W_tau, pred_len].

MODELS = [
    {
        "name": "PatchTST",
        # New canonical path first, then legacy PatchTST-main fallback.
        "pred":  ["PatchTST/results/SPX_IV_21_63_PatchTST_*/pred.npy",
                  "PatchTST-main/PatchTST_supervised/results/"
                  "SPX_IV_21_63_PatchTST_custom_ftM_sl21_ll0_pl63_*/pred.npy"],
        "dates": ["PatchTST/results/SPX_IV_21_63_PatchTST_*/start_dates.npy",
                  None],
        "loader": "flat",
    },
    {
        "name": "Persistence",
        "pred":  "PatchTST-main/PatchTST_supervised/results/"
                 "SPX_IV_21_63_Persistence_*/pred.npy",
        "dates": None,
        "loader": "flat",
    },
    {
        "name": "VAR1",
        # New canonical path first, then legacy fallback.
        "pred":  ["VAR1/results/SPX_IV_21_63_VAR1_*/pred.npy",
                  "var_lag1_results/pred.npy"],
        "dates": ["VAR1/results/SPX_IV_21_63_VAR1_*/start_dates.npy",
                  "var_lag1_results/start_dates.npy"],
        "loader": "flat",
    },
    {
        "name": "HOT(product)",
        "pred":  "HOT/results/SPX_IV_21_63_HOT_tensor_*_kronecker_product/pred.npy",
        "dates": "HOT/results/SPX_IV_21_63_HOT_tensor_*_kronecker_product/start_dates.npy",
        "loader": "hot",
    },
    {
        "name": "HOT(sum)",
        "pred":  "HOT/results/SPX_IV_21_63_HOT_tensor_*_kronecker_sum/pred.npy",
        "dates": "HOT/results/SPX_IV_21_63_HOT_tensor_*_kronecker_sum/start_dates.npy",
        "loader": "hot",
    },
    {
        "name": "DynGWN",
        "pred":  "DynGWN/results/SPX_IV_21_63_DynGWN_*/pred.npy",
        "dates": "DynGWN/results/SPX_IV_21_63_DynGWN_*/start_dates.npy",
        "loader": "flat",
    },
    {
        "name": "DLinear",
        "pred":  "DLinear/results/SPX_IV_21_63_DLinear_*/pred.npy",
        "dates": "DLinear/results/SPX_IV_21_63_DLinear_*/start_dates.npy",
        "loader": "flat",
    },
]


# ─── Ground truth ─────────────────────────────────────────────────────────────

def build_reference(csv_path: str, seq_len: int, pred_len: int):
    """
    Returns:
      trues      [N, pred_len, 400] ground truth in scaled space
      persist    [N, pred_len, 400] naive persistence forecast (last observed repeated)
      ref_dates  [N] datetime64[D] start date of each prediction window
      meta       dict of split statistics
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    if len(iv_cols) != N_IV:
        raise ValueError(f"Expected {N_IV} iv_ columns, found {len(iv_cols)}")

    T = len(df)
    num_train = int(T * TRAIN_FRAC)
    num_test = int(T * TEST_FRAC)
    num_val = T - num_train - num_test

    # Same border formula used by HOT and DynGWN
    train_end = num_train
    test_start = T - num_test - seq_len

    iv_np = df[iv_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(iv_np[:train_end])
    iv_scaled = scaler.transform(iv_np).astype(np.float32)

    test_slice = iv_scaled[test_start:]          # [num_test + seq_len, 400]
    all_dates = df["date"].to_numpy(dtype="datetime64[D]")
    test_date_slice = all_dates[test_start:]

    n = len(test_slice) - seq_len - pred_len + 1
    trues   = np.empty((n, pred_len, N_IV), dtype=np.float32)
    persist = np.empty((n, pred_len, N_IV), dtype=np.float32)
    ref_dates = np.empty(n, dtype="datetime64[D]")

    for i in range(n):
        trues[i]   = test_slice[i + seq_len : i + seq_len + pred_len]
        persist[i] = test_slice[i + seq_len - 1]   # broadcast last observed value
        ref_dates[i] = test_date_slice[i + seq_len]

    meta = {
        "T": T, "num_train": num_train, "num_val": num_val, "num_test": num_test,
        "n_test": n, "seq_len": seq_len, "pred_len": pred_len,
        "date_range": f"{ref_dates[0]} → {ref_dates[-1]}",
    }
    return trues, persist, ref_dates, meta


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_flat(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != N_IV:
        raise ValueError(f"Expected [N, pred, {N_IV}], got {arr.shape}")
    return arr


def load_hot(path: str) -> np.ndarray:
    """
    HOT saves [N, H_mono=20, W_tau=20, pred_len=63].
    CSV column order is (tau outer, moneyness inner), i.e. column k = i_tau*20 + i_mono.
    Since HOT uses H=moneyness, W=tau, F-order reshape produces
      vec[i_mono + 20*i_tau] = tensor[i_mono, i_tau]  which equals column k. ✓
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected HOT [N, H, W, pred], got {arr.shape}")
    n, h, w, p = arr.shape
    if h * w != N_IV:
        raise ValueError(f"HOT grid {h}×{w}={h*w} ≠ {N_IV}")
    arr = arr.transpose(0, 3, 1, 2)            # [N, pred, H_mono, W_tau]
    return arr.reshape(n, p, h * w, order="F") # [N, pred, 400] in CSV column order


LOADERS = {"flat": load_flat, "hot": load_hot}


# ─── Date alignment ───────────────────────────────────────────────────────────

def align_to_ref(pred: np.ndarray, pred_dates: np.ndarray,
                 ref_dates: np.ndarray, name: str) -> np.ndarray:
    pred_dates = pred_dates.astype("datetime64[D]")
    date_to_idx = {d: i for i, d in enumerate(pred_dates)}
    indices, missing = [], []
    for d in ref_dates:
        i = date_to_idx.get(d)
        if i is None:
            missing.append(str(d))
        else:
            indices.append(i)
    if missing:
        raise ValueError(
            f"{name}: {len(missing)}/{len(ref_dates)} reference dates not in pred "
            f"(first missing: {missing[0]})"
        )
    return pred[np.array(indices)]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    pred, true: [N, pred_len, 400] in scaled space.

    Returns scalar metrics plus per-horizon arrays (keyed with '_' prefix).
    """
    diff    = pred - true
    sq      = diff * diff
    abs_d   = np.abs(diff)

    mse_v  = float(np.mean(sq))
    rmse_v = float(np.sqrt(mse_v))
    mae_v  = float(np.mean(abs_d))

    t64 = true.astype(np.float64)
    p64 = pred.astype(np.float64)
    rse_v = float(
        np.sqrt(np.sum((t64 - p64) ** 2))
        / np.sqrt(np.sum((t64 - t64.mean()) ** 2))
    )

    bias_v = float(np.mean(diff))   # positive = model over-predicts
    da_v   = float(np.mean(np.sign(pred) == np.sign(true)))  # above/below hist mean

    # Spearman IC across 400 features, per (window, step)
    N, P, F = pred.shape
    ic = np.full((N, P), np.nan)
    for i in range(N):
        for t in range(P):
            r, _ = spearmanr(pred[i, t], true[i, t])
            ic[i, t] = r if np.isfinite(r) else np.nan

    ic_mean_v = float(np.nanmean(ic))
    ic_std_v  = float(np.nanstd(ic))

    per_h_mse = np.mean(sq,    axis=(0, 2))   # [pred_len]
    per_h_mae = np.mean(abs_d, axis=(0, 2))   # [pred_len]
    per_h_ic  = np.nanmean(ic, axis=0)        # [pred_len]

    return {
        "mse": mse_v, "rmse": rmse_v, "mae": mae_v, "rse": rse_v,
        "bias": bias_v, "da": da_v,
        "ic_mean": ic_mean_v, "ic_std": ic_std_v,
        "_per_h_mse": per_h_mse,
        "_per_h_mae": per_h_mae,
        "_per_h_ic":  per_h_ic,
    }


# ─── Display ──────────────────────────────────────────────────────────────────

OVERALL_COLS = ["mse", "rmse", "mae", "rse", "bias", "da", "ic_mean", "ic_std"]
# fmt strings: optional leading '+' for sign flag (placed before width at render time)
COL_FMT = {
    "mse": ".6f", "rmse": ".6f", "mae": ".6f", "rse": ".6f",
    "bias": "+.6f", "da": ".4f", "ic_mean": "+.4f", "ic_std": ".4f",
}


def _fmt(val: float, cw: int, fmt: str) -> str:
    """Format val right-aligned in width cw, honouring optional leading '+' sign flag."""
    if fmt.startswith("+"):
        return format(val, f">+{cw}{fmt[1:]}")
    return format(val, f">{cw}{fmt}")


def _col_width(results: list[dict]) -> int:
    return max(14, max(len(r["name"]) for r in results) + 2)


def print_summary(results: list[dict]):
    cw = _col_width(results)
    rw = 10
    names = [r["name"] for r in results]

    sep   = "=" * (rw + 1 + cw * len(names))
    hdash = "-" * (rw + 1 + cw * len(names))

    print(f"\n{sep}")
    print(f"{'Metric':<{rw}} " + "".join(f"{n:>{cw}}" for n in names))
    print(sep)
    for m in OVERALL_COLS:
        fmt = COL_FMT[m]
        row = f"{m:<{rw}} " + "".join(_fmt(r[m], cw, fmt) for r in results)
        print(row)
    print(sep)

    # Per-horizon IC table
    n_ref = results[0]["n_windows"]
    print(f"\nPer-horizon Spearman IC  (N={n_ref} reference windows):")
    print(f"{'h':<8}" + "".join(f"{n:>{cw}}" for n in names))
    print(hdash)
    for h in REPORT_HORIZONS:
        idx = h - 1
        row = f"t+{h:<6}" + "".join(_fmt(r["_per_h_ic"][idx], cw, "+.4f") for r in results)
        print(row)

    # Per-horizon MSE table
    print(f"\nPer-horizon MSE:")
    print(f"{'h':<8}" + "".join(f"{n:>{cw}}" for n in names))
    print(hdash)
    for h in REPORT_HORIZONS:
        idx = h - 1
        row = f"t+{h:<6}" + "".join(_fmt(r["_per_h_mse"][idx], cw, ".6f") for r in results)
        print(row)
    print()


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Overall metrics CSV (one row per model)
    scalar_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ]
    df_overall = pd.DataFrame(scalar_rows)
    df_overall.to_csv(os.path.join(out_dir, f"{ts}.csv"), index=False)
    df_overall.to_csv(os.path.join(out_dir, "latest.csv"), index=False)

    # Per-horizon CSV (one row per model × horizon)
    h_rows = []
    for r in results:
        for h in range(1, PRED_LEN + 1):
            h_rows.append({
                "model":   r["name"],
                "horizon": h,
                "ic":      r["_per_h_ic"][h - 1],
                "mse":     r["_per_h_mse"][h - 1],
                "mae":     r["_per_h_mae"][h - 1],
            })
    pd.DataFrame(h_rows).to_csv(os.path.join(out_dir, "latest_horizons.csv"), index=False)

    # JSON summary
    summary = {
        "timestamp": ts,
        "models": {
            r["name"]: {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        },
    }
    with open(os.path.join(out_dir, "latest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to {out_dir}/  ({ts}.csv + latest.*)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def _resolve_path(pattern) -> str | None:
    """Resolve a glob pattern (or list of patterns) to the most-recently-modified match."""
    patterns = pattern if isinstance(pattern, list) else [pattern]
    for p in patterns:
        if p is None:
            continue
        if "*" not in p:
            if os.path.exists(p):
                return p
        else:
            matches = glob.glob(p)
            if matches:
                return max(matches, key=os.path.getmtime)
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare SPX IV forecasting models")
    parser.add_argument("--csv_path", default="SPX_surfaces.csv")
    parser.add_argument("--seq_len",  type=int, default=SEQ_LEN)
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--out_dir",  default="comparison_results")
    args = parser.parse_args()

    print("Building reference ground truth...")
    trues, persist, ref_dates, meta = build_reference(
        args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len
    )
    print(f"  T={meta['T']}, test windows={meta['n_test']}, {meta['date_range']}")

    results = []

    # Reference persistence (built from CSV, always available)
    m = compute_metrics(persist, trues)
    m["name"] = "Persist(ref)"
    m["n_windows"] = meta["n_test"]
    m["source"] = "built-in"
    results.append(m)
    print(f"  Persist(ref): OK ({meta['n_test']} windows)")

    for spec in MODELS:
        name = spec["name"]
        pred_path  = _resolve_path(spec["pred"])
        dates_path = _resolve_path(spec["dates"])

        if pred_path is None:
            print(f"  {name}: no results found — skipping")
            continue

        try:
            pred = LOADERS[spec["loader"]](pred_path)
        except Exception as e:
            print(f"  {name}: load error — {e}")
            continue

        if dates_path is not None:
            pred_dates = np.load(dates_path).astype("datetime64[D]")
            try:
                pred = align_to_ref(pred, pred_dates, ref_dates, name)
            except ValueError as e:
                print(f"  {name}: date alignment failed — {e}")
                continue
            n_windows = len(ref_dates)
            true_aligned = trues
        else:
            # No dates saved — sequential alignment assumption
            n_windows = min(len(pred), len(trues))
            if n_windows < len(trues):
                print(
                    f"  {name}: WARNING — no start_dates.npy; assuming sequential "
                    f"alignment from window 0 (model has {len(pred)}, ref has {len(trues)}). "
                    f"Dates may be misaligned if model was run on a different CSV."
                )
            pred = pred[:n_windows]
            true_aligned = trues[:n_windows]

        if pred.shape[1:] != (args.pred_len, N_IV):
            print(f"  {name}: unexpected pred shape {pred.shape} — skipping")
            continue

        m = compute_metrics(pred, true_aligned)
        m["name"] = name
        m["n_windows"] = n_windows
        m["source"] = pred_path
        results.append(m)
        print(f"  {name}: OK ({n_windows} windows, {pred_path})")

    if len(results) == 1:
        print("\nOnly baseline available — nothing to compare.")
        return

    print_summary(results)
    save_results(results, args.out_dir)


if __name__ == "__main__":
    main()
