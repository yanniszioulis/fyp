#!/usr/bin/env python3
"""
VAR(1) baseline for SPX IV surface forecasting.

For each sliding test window:
- fit ridge VAR(1) with intercept on the context (seq_len)
- freeze coefficients and roll forward for pred_len steps (iterated forecast)

Saves predictions to: ./results/<setting>/pred.npy
Shape: [n_samples_aligned, pred_len, n_features] in the *scaled* space,
mirroring PatchTST's Dataset_Custom split + StandardScaler fitting.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _build_setting(args: argparse.Namespace) -> str:
    model_name = "VAR1_ridge_intercept"
    return (
        f"{args.model_id}_{args.seq_len}_{args.pred_len}_{model_name}_"
        f"{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"{args.des}_0"
    )


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
    test_data = data[test_border1:test_border2]

    meta = {
        "n_rows": int(len(df_raw)),
        "num_train": int(num_train),
        "num_vali": int(num_vali),
        "num_test": int(num_test),
        "border1s": [int(x) for x in border1s],
        "border2s": [int(x) for x in border2s],
        "test_border1": int(test_border1),
        "test_border2": int(test_border2),
        "n_features": int(df_data.shape[1]),
    }

    return test_data.astype(np.float64, copy=False), scaler, meta


def _var1_step_operator_with_intercept(context: np.ndarray, ridge_lambda: float):
    """
    Ridge VAR(1) with intercept, fitted on context window.

    Model: y = c + A x
    where x,y in R^k, c in R^k, A in R^{k x k}

    Uses dual formulation to avoid forming kxk A explicitly.
    Returns step(x_t) -> x_next, with coefficients frozen.
    """
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be >= 0")

    x = context[:-1]  # [n, k]
    y = context[1:]   # [n, k]
    n, k = x.shape

    # Augment regressors with intercept: X_aug = [1, x]
    ones = np.ones((n, 1), dtype=x.dtype)
    x_aug = np.concatenate([ones, x], axis=1)  # [n, 1+k]

    # Dual Gram matrix: G = X_aug X_aug^T + λ I
    g = x_aug @ x_aug.T  # [n, n]
    if ridge_lambda > 0:
        g = g + ridge_lambda * np.eye(n, dtype=g.dtype)

    y_t = y.T  # [k, n]

    def step(x_t: np.ndarray) -> np.ndarray:
        # x_t: [k]
        x_aug_t = np.concatenate([np.array([1.0], dtype=x_t.dtype), x_t.astype(x.dtype, copy=False)], axis=0)  # [1+k]
        v = x_aug @ x_aug_t  # [n]
        w = np.linalg.solve(g, v)  # [n]
        return y_t @ w  # [k]

    return step


def main():
    parser = argparse.ArgumentParser(description="VAR(1) baseline that writes PatchTST-style pred.npy")
    parser.add_argument("--root_path", type=str, default="./dataset/")
    parser.add_argument("--data_path", type=str, default="SPX_surfaces.csv")
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="iv_1.1_1.0")

    parser.add_argument("--seq_len", type=int, default=21)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=63)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument(
        "--ridge_lambda",
        type=float,
        default=500,
        help="L2 penalty strength for VAR(1) fit (applied in dual form).",
    )

    parser.add_argument("--model_id", type=str, default="SPX_IV")
    parser.add_argument("--des", type=str, default="Exp")
    args = parser.parse_args()

    if args.data != "custom":
        raise ValueError("This baseline currently mirrors Dataset_Custom only (use --data custom).")
    if args.label_len != 0:
        raise ValueError("This baseline assumes label_len=0 to match SPX_IV.sh.")
    if args.seq_len < 2:
        raise ValueError("seq_len must be >= 2 for VAR(1).")
    if args.pred_len < 1:
        raise ValueError("pred_len must be >= 1.")
    if args.ridge_lambda < 0:
        raise ValueError("ridge_lambda must be >= 0.")

    t0 = time.time()
    test_data, _scaler, meta = _load_and_scale_custom(
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        seq_len=args.seq_len,
    )

    n_test_samples = len(test_data) - args.seq_len - args.pred_len + 1
    if n_test_samples <= 0:
        raise ValueError(
            f"Not enough test rows for seq_len={args.seq_len}, pred_len={args.pred_len}: "
            f"len(test_data)={len(test_data)}"
        )

    n_aligned = (n_test_samples // args.batch_size) * args.batch_size
    if n_aligned <= 0:
        raise ValueError(
            f"After drop_last alignment (batch_size={args.batch_size}), no samples remain "
            f"(n_test_samples={n_test_samples})."
        )

    k = meta["n_features"]
    preds = np.zeros((n_aligned, args.pred_len, k), dtype=np.float32)

    for i in range(n_aligned):
        context = test_data[i : i + args.seq_len]  # [seq_len, k]
        step = _var1_step_operator_with_intercept(context, ridge_lambda=args.ridge_lambda)

        x_t = context[-1].astype(np.float64, copy=False)
        for h in range(args.pred_len):
            x_t = step(x_t)  # frozen coefficients, iterated forecast
            preds[i, h, :] = x_t.astype(np.float32, copy=False)

    setting = _build_setting(args)
    out_dir = os.path.join("./results", setting)
    os.makedirs(out_dir, exist_ok=True)

    pred_path = os.path.join(out_dir, "pred.npy")
    np.save(pred_path, preds)

    meta_out = {
        "setting": setting,
        "args": vars(args),
        "data_meta": meta,
        "n_test_samples": int(n_test_samples),
        "n_samples_aligned": int(n_aligned),
        "pred_shape": list(preds.shape),
        "seconds": float(time.time() - t0),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved: {pred_path}")
    print(f"preds shape: {preds.shape} (aligned from {n_test_samples} test windows)")
    print(f"meta: {os.path.join(out_dir, 'meta.json')}")


if __name__ == "__main__":
    main()

