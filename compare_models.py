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
- PatchTST: expects pred.npy with shape [N, pred_len, 400] (IV grid only)
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
    if h * w != 400:
        raise ValueError(f"Expected HOT surface grid to have 400 cells (got {h}x{w}={h*w}) in {path}")
    arr = arr.transpose(0, 3, 1, 2)  # [N, pred, H, W]
    vec = arr.reshape(n, p, h * w, order="F")  # moneyness varies fastest, then tau
    return vec.astype(np.float32, copy=False)

def _align_hot_to_patch_dates(
    *,
    hot_pred: np.ndarray,
    hot_dates: np.ndarray,
    patch_dates: np.ndarray,
    hot_name: str,
) -> np.ndarray:
    hot_dates = hot_dates.astype("datetime64[D]")
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
        raise ValueError(f"{hot_name} is missing {missing} start_dates that exist in PatchTST test set.")
    return hot_pred[np.array(hot_idx, dtype=int)]


def _align_pred_to_patch_dates(
    *,
    pred: np.ndarray,
    pred_dates: np.ndarray,
    patch_dates: np.ndarray,
    name: str,
) -> np.ndarray:
    pred_dates = pred_dates.astype("datetime64[D]")
    pred_map = {d: i for i, d in enumerate(pred_dates)}
    pred_idx = []
    missing = 0
    for d in patch_dates:
        i = pred_map.get(d)
        if i is None:
            missing += 1
            pred_idx.append(-1)
        else:
            pred_idx.append(i)
    if missing:
        raise ValueError(f"{name} is missing {missing} start_dates that exist in PatchTST test set.")
    return pred[np.array(pred_idx, dtype=int)]


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
        "--hot_pred_product",
        type=str,
        default="HOT/results/SPX_IV_21_63_HOT_tensor_dh128_mlp512_b4_h8_p4_kronecker_product/pred.npy",
    )
    parser.add_argument(
        "--hot_dates_product",
        type=str,
        default="HOT/results/SPX_IV_21_63_HOT_tensor_dh128_mlp512_b4_h8_p4_kronecker_product/start_dates.npy",
    )
    parser.add_argument(
        "--hot_pred_sum",
        type=str,
        default="HOT/results/SPX_IV_21_63_HOT_tensor_dh128_mlp512_b4_h8_p4_kronecker_sum/pred.npy",
    )
    parser.add_argument(
        "--hot_dates_sum",
        type=str,
        default="HOT/results/SPX_IV_21_63_HOT_tensor_dh128_mlp512_b4_h8_p4_kronecker_sum/start_dates.npy",
    )
    parser.add_argument(
        "--dyngwn_pred",
        type=str,
        default=None,
        help="Optional DynGWN pred.npy path (shape [N, pred_len, 400] in scaled space).",
    )
    parser.add_argument(
        "--dyngwn_dates",
        type=str,
        default=None,
        help="Optional DynGWN start_dates.npy path to align by date.",
    )
    args = parser.parse_args()

    trues_all, patch_start_dates_all, meta = build_patchtst_scaled_truth_iv(
        args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len
    )

    patch_pred = load_patchtst_iv_pred(args.patchtst_pred)
    persist_pred = load_patchtst_iv_pred(args.persist_pred)
    var_pred = load_patchtst_iv_pred(args.var_pred)
    hot_pred_product = load_hot_iv_pred(args.hot_pred_product)
    hot_dates_product = np.load(args.hot_dates_product).astype("datetime64[D]")
    hot_pred_sum = load_hot_iv_pred(args.hot_pred_sum)
    hot_dates_sum = np.load(args.hot_dates_sum).astype("datetime64[D]")

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

    hot_product_aligned = _align_hot_to_patch_dates(
        hot_pred=hot_pred_product,
        hot_dates=hot_dates_product,
        patch_dates=patch_dates,
        hot_name="HOT(product)",
    )
    hot_sum_aligned = _align_hot_to_patch_dates(
        hot_pred=hot_pred_sum,
        hot_dates=hot_dates_sum,
        patch_dates=patch_dates,
        hot_name="HOT(sum)",
    )

    preds = {
        "PatchTST": patch_pred,
        "HOT(product)": hot_product_aligned,
        "HOT(sum)": hot_sum_aligned,
        "VAR1": var_pred,
        "Persistence": persist_pred,
    }

    if args.dyngwn_pred:
        dyngwn_pred = load_patchtst_iv_pred(args.dyngwn_pred)
        if args.dyngwn_dates:
            dyngwn_dates = np.load(args.dyngwn_dates).astype("datetime64[D]")
            dyngwn_pred = _align_pred_to_patch_dates(
                pred=dyngwn_pred,
                pred_dates=dyngwn_dates,
                patch_dates=patch_dates,
                name="DynGWN",
            )
        else:
            dyngwn_pred = dyngwn_pred[:n_patch]
        preds["DynGWN"] = dyngwn_pred

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
    model_order = ["PatchTST", "HOT(product)", "HOT(sum)", "VAR1", "Persistence"]
    if "DynGWN" in by_name and "DynGWN" not in model_order:
        model_order.insert(3, "DynGWN")

    display = {"HOT(product)": "HOT(prod)", "HOT(sum)": "HOT(sum)", "Persistence": "Persistence"}
    colw = 14
    line_w = 12 + 1 + (colw + 1) * len(model_order)
    print("=" * line_w)
    print(f"{'Metric':<12} " + " ".join([f"{display.get(m, m):>{colw}}" for m in model_order]))
    print("=" * line_w)
    for metric in ["mse", "mae", "rse", "ic_mean", "ic_std"]:
        print(f"{metric:<12} " + " ".join([f"{by_name[m][metric]:>{colw}.10f}" for m in model_order]))
    print("=" * line_w)

    horizons_to_show = [1, 5, 10, 21, 42, 63]
    print(f"\nPer-horizon IC (averaged across {trues.shape[0]} test windows):")
    ic_colw = 12
    ic_line_w = 10 + 1 + (ic_colw + 1) * len(model_order)
    display_ic = {**display, "Persistence": "Persist"}
    print(f"{'Horizon':<10} " + " ".join([f"{display_ic.get(m, m):>{ic_colw}}" for m in model_order]))
    print("-" * ic_line_w)
    for h in horizons_to_show:
        if 1 <= h <= args.pred_len:
            idx = h - 1
            print(
                f"t+{h:<7} "
                + " ".join([f"{by_name[m]['ic_h'][idx]:>{ic_colw}.6f}" for m in model_order])
            )
    print("-" * 74)


if __name__ == "__main__":
    main()
