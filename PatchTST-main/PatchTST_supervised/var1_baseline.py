#!/usr/bin/env python3
"""
VAR(1) baseline for SPX IV surface forecasting (ridge + intercept).

DEBUG MODE:
Adds diagnostics to catch/understand exploding forecasts:
- condition number / eigenvalues of the fitted transition A (estimated via probe)
- norm growth across horizons
- comparison vs persistence
- checks for NaN/Inf
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
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be >= 0")

    x = context[:-1]
    y = context[1:]
    n, k = x.shape

    ones = np.ones((n, 1), dtype=x.dtype)
    x_aug = np.concatenate([ones, x], axis=1)  # [n, 1+k]

    g = x_aug @ x_aug.T
    if ridge_lambda > 0:
        g = g + ridge_lambda * np.eye(n, dtype=g.dtype)

    y_t = y.T  # [k, n]

    def step(x_t: np.ndarray) -> np.ndarray:
        x_aug_t = np.concatenate(
            [np.array([1.0], dtype=x_t.dtype), x_t.astype(x.dtype, copy=False)], axis=0
        )
        v = x_aug @ x_aug_t
        w = np.linalg.solve(g, v)
        return y_t @ w

    # Expose internals for debugging
    return step, x_aug, y_t, g


def _debug_window(step, context, pred_len, x_aug, y_t, g, ridge_lambda, i, max_probe_dim=20):
    k = context.shape[1]
    last = context[-1]
    persist = np.repeat(last[None, :], pred_len, axis=0)

    x_t = last.copy()
    norms = []
    has_naninf = False
    for h in range(pred_len):
        x_t = step(x_t)
        nrm = float(np.linalg.norm(x_t))
        norms.append(nrm)
        if not np.isfinite(nrm) or not np.all(np.isfinite(x_t)):
            has_naninf = True
            break

    norms = np.array(norms, dtype=np.float64)
    print("\n" + "=" * 90)
    print(f"[DEBUG] window i={i}  ridge_lambda={ridge_lambda}")
    print(f"context last norm={np.linalg.norm(last):.6f}  context mean norm={np.linalg.norm(context.mean(axis=0)):.6f}")
    if len(norms) > 0:
        print(
            "forecast norms (h=1,5,10,21,42,63):",
            *(f"{norms[h-1]:.6f}" for h in [1, 5, 10, 21, 42, 63] if h <= len(norms)),
        )
        growth = norms[-1] / (norms[0] + 1e-12)
        print(f"norm growth (last/first) ~ {growth:.6f}")
    print(f"nan/inf encountered: {has_naninf}")

    # Gram matrix diagnostics
    try:
        cond_g = float(np.linalg.cond(g))
    except Exception:
        cond_g = float("nan")
    print(f"G shape={g.shape}, cond(G)={cond_g:.3e}")

    # Compare one-step vs persistence and context y_{last}
    one_step = step(last)
    mse_vs_persist_h1 = float(np.mean((one_step - last) ** 2))
    mse_vs_next_in_context = float(np.mean((one_step - context[-1]) ** 2))
    print(f"one-step MSE vs persistence(last): {mse_vs_persist_h1:.6e}")
    print(f"one-step norm={np.linalg.norm(one_step):.6f}")

    # Probe the effective linear map A and intercept c on a subset of dims (approx)
    # We estimate A[:,j] as step(e_j) - step(0) for selected j, and c as step(0).
    d = min(max_probe_dim, k)
    idx = np.linspace(0, k - 1, d, dtype=int)
    zero = np.zeros(k, dtype=last.dtype)
    c_full = step(zero)  # this is the intercept effect (since x=0 -> y=c)
    # build A_sub (d x d): rows/cols correspond to idx
    A_sub = np.zeros((d, d), dtype=np.float64)
    c_sub = c_full[idx].astype(np.float64)

    for jj, j in enumerate(idx):
        e = np.zeros(k, dtype=last.dtype)
        e[j] = 1.0
        col = step(e) - c_full
        A_sub[:, jj] = col[idx].astype(np.float64)

    # Spectral radius estimate on the submatrix
    try:
        eigvals = np.linalg.eigvals(A_sub)
        spec_rad = float(np.max(np.abs(eigvals)))
    except Exception:
        spec_rad = float("nan")
    print(f"probe dims d={d} (subset of features)")
    print(f"||c_sub||={np.linalg.norm(c_sub):.6f}")
    print(f"||A_sub||_F={np.linalg.norm(A_sub):.6f}")
    print(f"max |eig(A_sub)| (spectral radius approx)={spec_rad:.6f}")
    print("A_sub diag (first 10):", np.diag(A_sub)[: min(10, d)])

    # If spectral radius > 1, iterated forecasts can explode
    if np.isfinite(spec_rad) and spec_rad > 1.0:
        print("[DEBUG] spectral radius > 1 on probed subspace -> unstable recursion likely.")

    print("=" * 90)


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
    parser.add_argument("--ridge_lambda", type=float, default=100)

    parser.add_argument("--model_id", type=str, default="SPX_IV")
    parser.add_argument("--des", type=str, default="Exp")

    # Debug controls
    parser.add_argument("--debug", action="store_true", help="Print diagnostics for a few windows.")
    parser.add_argument("--debug_windows", type=int, default=3, help="How many initial windows to debug.")
    parser.add_argument("--debug_stride", type=int, default=200, help="Also debug every Nth window (0 disables).")
    parser.add_argument("--debug_probe_dim", type=int, default=20, help="Probe A on this many feature dims.")
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
        context = test_data[i : i + args.seq_len]
        step, x_aug, y_t, g = _var1_step_operator_with_intercept(context, ridge_lambda=args.ridge_lambda)

        if args.debug and (
            i < args.debug_windows or (args.debug_stride > 0 and i % args.debug_stride == 0)
        ):
            _debug_window(
                step=step,
                context=context,
                pred_len=args.pred_len,
                x_aug=x_aug,
                y_t=y_t,
                g=g,
                ridge_lambda=args.ridge_lambda,
                i=i,
                max_probe_dim=args.debug_probe_dim,
            )

        x_t = context[-1].astype(np.float64, copy=False)
        for h in range(args.pred_len):
            x_t = step(x_t)
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
