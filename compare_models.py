#!/usr/bin/env python3
"""
Compare PatchTST vs HOT(tensor) vs Persistence on SPX surfaces.

All comparisons are done on the IV grid only (400 features), in the scaled space
using the same split + scaling as PatchTST Dataset_Custom:
- train = first 70%
- test  = last 20%
- val   = remainder
- borders for val/test include a -seq_len overlap

Models:
- PatchTST: expects pred.npy with shape [N, pred_len, 401] (underlying_price + 400 IVs)
- Persistence: same shape/order as PatchTST
- HOT: expects pred.npy with shape [N, H, W, pred_len] and start_dates.npy with dates
       (H=20, W=20 so H*W=400 IVs). Saved from HOT pipeline.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        corr, _ = spearmanr(x, y)
        return float(corr)
    except Exception:
        rx = pd.Series(x).rank(method="average").to_numpy()
        ry = pd.Series(y).rank(method="average").to_numpy()
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        if denom == 0:
            return float("nan")
        return float((rx * ry).sum() / denom)


def compute_ic(pred: np.ndarray, true: np.ndarray):
    n_samples, n_steps, _ = pred.shape
    ic_values = np.zeros((n_samples, n_steps), dtype=np.float64)
    for i in range(n_samples):
        for t in range(n_steps):
            ic_values[i, t] = _spearman_corr(pred[i, t, :], true[i, t, :])
    overall_ic = np.nanmean(ic_values)
    overall_std = np.nanstd(ic_values)
    per_horizon_ic = np.nanmean(ic_values, axis=0)
    return overall_ic, overall_std, per_horizon_ic


def mse(pred, true):
    diff = pred.astype(np.float64, copy=False) - true.astype(np.float64, copy=False)
    return float(np.mean(diff * diff))


def mae(pred, true):
    diff = pred.astype(np.float64, copy=False) - true.astype(np.float64, copy=False)
    return float(np.mean(np.abs(diff)))


def rse(pred, true):
    pred64 = pred.astype(np.float64, copy=False)
    true64 = true.astype(np.float64, copy=False)
    diff = true64 - pred64
    num = np.sqrt(np.sum(diff * diff))
    denom_diff = true64 - true64.mean()
    den = np.sqrt(np.sum(denom_diff * denom_diff))
    return float(num / den)


def build_patchtst_scaled_truth_iv(
    csv_path: str,
    *,
    seq_len: int,
    pred_len: int,
):
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    iv_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("iv_")]
    if not iv_cols:
        raise ValueError("No iv_* columns found in CSV.")

    T = len(df)
    num_train = int(T * 0.7)
    num_test = int(T * 0.2)
    num_val = T - num_train - num_test

    border1s = [0, num_train - seq_len, T - num_test - seq_len]
    border2s = [num_train, num_train + num_val, T]

    scaler = StandardScaler()
    scaler.fit(df.loc[border1s[0] : border2s[0] - 1, iv_cols].to_numpy(dtype=np.float32, copy=False))
    iv_scaled = scaler.transform(df[iv_cols].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)

    tb1, tb2 = border1s[2], border2s[2]
    test_iv = iv_scaled[tb1:tb2]  # includes seq_len overlap

    n_test_samples = len(test_iv) - seq_len - pred_len + 1
    if n_test_samples <= 0:
        raise ValueError("Not enough test data to build windows.")

    trues = np.zeros((n_test_samples, pred_len, len(iv_cols)), dtype=np.float32)
    start_dates = np.zeros((n_test_samples,), dtype="datetime64[D]")
    all_dates = df["date"].to_numpy(dtype="datetime64[D]")

    for i in range(n_test_samples):
        trues[i] = test_iv[i + seq_len : i + seq_len + pred_len]
        start_dates[i] = all_dates[tb1 + i + seq_len]

    meta = {
        "T": T,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "border1s": border1s,
        "border2s": border2s,
        "n_test_samples": n_test_samples,
    }
    return trues, start_dates, meta


def load_patchtst_iv_pred(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected PatchTST pred shape: {arr.shape}")
    if arr.shape[-1] != 400:
        raise ValueError(f"Expected 400 IV features, got {arr.shape[-1]} in {path}")
    return arr.astype(np.float32, copy=False)


def load_hot_iv_pred(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 4:
        raise ValueError(f"Unexpected HOT pred shape: {arr.shape}")
    n, h, w, p = arr.shape
    arr = arr.transpose(0, 3, 1, 2)  # [N, pred, H, W]
    vec = arr.reshape(n, p, h * w, order="F")  # moneyness varies fastest, then tau
    return vec.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Compare PatchTST vs HOT vs Persistence on SPX IV surfaces")
    parser.add_argument("--csv_path", type=str, default="SPX_surfaces.csv")
    parser.add_argument("--seq_len", type=int, default=21)
    parser.add_argument("--pred_len", type=int, default=63)

    parser.add_argument(
        "--patchtst_pred",
        type=str,
        default="PatchTST-main/PatchTST_supervised/results/"
        "SPX_IV_21_63_PatchTST_custom_ftM_sl21_ll0_pl63_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0/pred.npy",
    )
    parser.add_argument(
        "--persist_pred",
        type=str,
        default="PatchTST-main/PatchTST_supervised/results/"
        "SPX_IV_21_63_Persistence_custom_ftM_sl21_ll0_pl63_Exp_0/pred.npy",
    )
    parser.add_argument(
        "--var_pred",
        type=str,
        default="PatchTST-main/PatchTST_supervised/results/"
        "SPX_IV_21_63_VAR1_ridge_intercept_custom_ftM_sl21_ll0_pl63_Exp_0/pred.npy",
    )
    parser.add_argument(
        "--hot_pred",
        type=str,
        default="HOT/results/SPX_IV_21_63_HOT_tensor_dh128_mlp512_b4_h8_p4_kronecker_product/pred.npy",
    )
    parser.add_argument(
        "--hot_dates",
        type=str,
        default="HOT/results/SPX_IV_21_63_HOT_tensor_dh128_mlp512_b4_h8_p4_kronecker_product/start_dates.npy",
    )
    args = parser.parse_args()

    trues_all, patch_start_dates_all, meta = build_patchtst_scaled_truth_iv(
        args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len
    )

    patch_pred = load_patchtst_iv_pred(args.patchtst_pred)
    persist_pred = load_patchtst_iv_pred(args.persist_pred)
    var_pred = load_patchtst_iv_pred(args.var_pred)
    hot_pred = load_hot_iv_pred(args.hot_pred)
    hot_dates = np.load(args.hot_dates).astype("datetime64[D]")

    n_patch = patch_pred.shape[0]
    trues = trues_all[:n_patch]
    patch_dates = patch_start_dates_all[:n_patch]

    if persist_pred.shape[0] != n_patch:
        persist_pred = persist_pred[:n_patch]
    if var_pred.shape[0] != n_patch:
        var_pred = var_pred[:n_patch]
    if patch_pred.shape[1:] != trues.shape[1:]:
        raise ValueError(f"PatchTST pred shape {patch_pred.shape} != truth {trues.shape}")
    if persist_pred.shape[1:] != trues.shape[1:]:
        raise ValueError(f"Persistence pred shape {persist_pred.shape} != truth {trues.shape}")
    if var_pred.shape[1:] != trues.shape[1:]:
        raise ValueError(f"VAR pred shape {var_pred.shape} != truth {trues.shape}")

    hot_map = {d: i for i, d in enumerate(hot_dates)}
    hot_idx = []
    missing = 0
    for d in patch_dates:
        i = hot_map.get(d)
        if i is None:
            missing += 1
            hot_idx.append(-1)
        else:
            hot_idx.append(i)
    if missing:
        raise ValueError(f"HOT is missing {missing} start_dates that exist in PatchTST test set.")
    hot_aligned = hot_pred[np.array(hot_idx, dtype=int)]

    preds = {
        "PatchTST": patch_pred,
        "HOT": hot_aligned,
        "VAR1": var_pred,
        "Persistence": persist_pred,
    }

    rows = []
    for name, arr in preds.items():
        ic_mean, ic_std, ic_h = compute_ic(arr, trues)
        rows.append(
            {
                "name": name,
                "mse": mse(arr, trues),
                "mae": mae(arr, trues),
                "rse": rse(arr, trues),
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_h": ic_h,
            }
        )

    by_name = {r["name"]: r for r in rows}
    print("=" * 78)
    print(f"{'Metric':<12} {'PatchTST':>18} {'HOT':>18} {'VAR1':>18} {'Persistence':>18}")
    print("=" * 78)
    for metric in ["mse", "mae", "rse", "ic_mean", "ic_std"]:
        print(
            f"{metric:<12} "
            f"{by_name['PatchTST'][metric]:>18.10f} "
            f"{by_name['HOT'][metric]:>18.10f} "
            f"{by_name['VAR1'][metric]:>18.10f} "
            f"{by_name['Persistence'][metric]:>18.10f}"
        )
    print("=" * 78)

    horizons_to_show = [1, 5, 10, 21, 42, 63]
    print(f"\nPer-horizon IC (averaged across {trues.shape[0]} test windows):")
    print(f"{'Horizon':<10} {'PatchTST':>14} {'HOT':>14} {'VAR1':>14} {'Persistence':>14}")
    print("-" * 56)
    for h in horizons_to_show:
        if 1 <= h <= args.pred_len:
            idx = h - 1
            print(
                f"t+{h:<7} "
                f"{by_name['PatchTST']['ic_h'][idx]:>14.6f} "
                f"{by_name['HOT']['ic_h'][idx]:>14.6f} "
                f"{by_name['VAR1']['ic_h'][idx]:>14.6f} "
                f"{by_name['Persistence']['ic_h'][idx]:>14.6f}"
            )
    print("-" * 56)


if __name__ == "__main__":
    main()
