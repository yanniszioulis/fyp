#!/usr/bin/env python3
"""
Compare PatchTST, VAR(1), and Persistence predictions on the same test windows.

Loads model predictions from pred.npy (scaled space) and rebuilds the aligned test
ground truth using the same Dataset_Custom split + StandardScaler fitting.
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


def _load_and_scale_custom(
    root_path: str,
    data_path: str,
    features: str,
    target: str,
    seq_len: int,
):
    df_raw = pd.read_csv(os.path.join(root_path, data_path))

    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove("date")
    df_raw = df_raw[["date"] + cols + [target]]

    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test

    border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]

    if features in {"M", "MS"}:
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    elif features == "S":
        df_data = df_raw[[target]]
    else:
        raise ValueError(f"Unknown features={features!r} (expected 'M', 'MS', or 'S')")

    scaler = StandardScaler()
    train_data = df_data[border1s[0] : border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)

    test_border1 = border1s[2]
    test_border2 = border2s[2]
    test_data = data[test_border1:test_border2].astype(np.float32, copy=False)

    return test_data, df_data.shape[1]


def build_trues(test_data: np.ndarray, seq_len: int, pred_len: int, batch_size: int):
    n_test_samples = len(test_data) - seq_len - pred_len + 1
    if n_test_samples <= 0:
        raise ValueError("Not enough test data to build windows.")
    n_aligned = (n_test_samples // batch_size) * batch_size
    trues = np.zeros((n_aligned, pred_len, test_data.shape[1]), dtype=np.float32)
    for i in range(n_aligned):
        trues[i] = test_data[i + seq_len : i + seq_len + pred_len]
    return trues


def main():
    parser = argparse.ArgumentParser(description="Compare PatchTST vs VAR1 vs Persistence on SPX surfaces")
    parser.add_argument("--root_path", type=str, default="./dataset/")
    parser.add_argument("--data_path", type=str, default="SPX_surfaces.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="iv_1.1_1.0")
    parser.add_argument("--seq_len", type=int, default=21)
    parser.add_argument("--pred_len", type=int, default=63)
    parser.add_argument("--batch_size", type=int, default=24)

    parser.add_argument(
        "--patchtst_pred",
        type=str,
        default="./results/SPX_IV_21_63_PatchTST_custom_ftM_sl21_ll0_pl63_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0/pred.npy",
    )
    parser.add_argument(
        "--var1_pred",
        type=str,
        default="./results/SPX_IV_21_63_VAR1_ridge_intercept_custom_ftM_sl21_ll0_pl63_Exp_0/pred.npy",
    )
    parser.add_argument(
        "--persist_pred",
        type=str,
        default="./results/SPX_IV_21_63_Persistence_custom_ftM_sl21_ll0_pl63_Exp_0/pred.npy",
    )
    args = parser.parse_args()

    test_data, k = _load_and_scale_custom(
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        seq_len=args.seq_len,
    )
    trues = build_trues(test_data, args.seq_len, args.pred_len, args.batch_size)

    preds = {
        "PatchTST": np.load(args.patchtst_pred),
        "VAR1": np.load(args.var1_pred),
        "Persistence": np.load(args.persist_pred),
    }

    for name, arr in preds.items():
        if arr.shape != trues.shape:
            raise ValueError(f"{name} shape {arr.shape} != truth shape {trues.shape}")

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

    print("=" * 78)
    print(f"{'Metric':<12} {'PatchTST':>18} {'VAR1':>18} {'Persistence':>18}")
    print("=" * 78)
    by_name = {r["name"]: r for r in rows}
    for metric in ["mse", "mae", "rse", "ic_mean", "ic_std"]:
        print(
            f"{metric:<12} "
            f"{by_name['PatchTST'][metric]:>18.10f} "
            f"{by_name['VAR1'][metric]:>18.10f} "
            f"{by_name['Persistence'][metric]:>18.10f}"
        )
    print("=" * 78)

    horizons_to_show = [1, 5, 10, 21, 42, 63]
    print(f"\nPer-horizon IC (averaged across {trues.shape[0]} test windows):")
    print(f"{'Horizon':<10} {'PatchTST':>14} {'VAR1':>14} {'Persistence':>14}")
    print("-" * 56)
    for h in horizons_to_show:
        if 1 <= h <= args.pred_len:
            idx = h - 1
            print(
                f"t+{h:<7} "
                f"{by_name['PatchTST']['ic_h'][idx]:>14.6f} "
                f"{by_name['VAR1']['ic_h'][idx]:>14.6f} "
                f"{by_name['Persistence']['ic_h'][idx]:>14.6f}"
            )
    print("-" * 56)


if __name__ == "__main__":
    main()

