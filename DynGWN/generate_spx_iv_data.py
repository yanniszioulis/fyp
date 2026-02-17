#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Borders:
    border1: int
    border2: int


def _parse_iv_col(col: str) -> tuple[float, float]:
    # Expected: iv_<moneyness>_<tau>
    # Example: iv_0.9_0.04
    parts = col.split("_")
    if len(parts) != 3 or parts[0] != "iv":
        raise ValueError(f"Unexpected IV column name: {col!r}")
    return float(parts[1]), float(parts[2])


def _infer_grid_order(iv_cols: list[str]) -> dict:
    parsed = [(_parse_iv_col(c)[1], _parse_iv_col(c)[0], c) for c in iv_cols]  # (tau, moneyness, col)
    parsed_sorted = sorted(parsed)
    sorted_cols = [c for _tau, _m, c in parsed_sorted]
    if sorted_cols != iv_cols:
        raise ValueError(
            "IV columns are not sorted as (tau outer, moneyness inner). "
            "This breaks the 20x20 grid mapping assumed by DynGWN + HOT comparison."
        )
    taus = sorted({t for t, _m, _c in parsed_sorted})
    moneyness = sorted({m for _t, m, _c in parsed_sorted})
    if len(taus) != 20 or len(moneyness) != 20:
        raise ValueError(f"Expected a 20x20 grid, got {len(taus)}x{len(moneyness)}")
    return {
        "taus": taus,
        "moneyness": moneyness,
        "order": "tau_major_moneyness_minor",
    }


def _make_grid_adjacency(h: int = 20, w: int = 20, self_loops: bool = True) -> np.ndarray:
    n = h * w
    A = np.zeros((n, n), dtype=np.float32)

    def nid(r: int, c: int) -> int:
        return r * w + c

    for r in range(h):
        for c in range(w):
            i = nid(r, c)
            if self_loops:
                A[i, i] = 1.0
            if c - 1 >= 0:
                A[i, nid(r, c - 1)] = 1.0
            if c + 1 < w:
                A[i, nid(r, c + 1)] = 1.0
            if r - 1 >= 0:
                A[i, nid(r - 1, c)] = 1.0
            if r + 1 < h:
                A[i, nid(r + 1, c)] = 1.0
    return A


def _build_windows(series: np.ndarray, seq_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    # series: [T, 400]
    T, D = series.shape
    n = T - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError("Not enough data to build windows.")
    x = np.zeros((n, seq_len, D, 1), dtype=np.float32)
    y = np.zeros((n, pred_len, D, 1), dtype=np.float32)
    for i in range(n):
        x[i, :, :, 0] = series[i : i + seq_len]
        y[i, :, :, 0] = series[i + seq_len : i + seq_len + pred_len]
    return x, y


def main():
    p = argparse.ArgumentParser(description="Build DynGWN-ready SPX IV dataset (PatchTST-compatible splits/windowing).")
    p.add_argument("--csv_path", type=str, default="HOT/dataset/SPX_surfaces.csv")
    p.add_argument("--out_dir", type=str, default="DynGWN/data/SPX_IV_21_63")
    p.add_argument("--seq_len", type=int, default=21)
    p.add_argument("--pred_len", type=int, default=63)
    p.add_argument(
        "--graph_mode",
        type=str,
        default="grid_plus_adaptive",
        choices=["adaptive_only", "grid_plus_adaptive"],
        help="Which static adjacency to export. adaptive_only exports identity; grid_plus_adaptive exports 4-neighbor grid.",
    )
    p.add_argument("--self_loops", action="store_true", help="Add self loops in the grid adjacency.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path, low_memory=False)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    iv_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("iv_")]
    if len(iv_cols) != 400:
        raise ValueError(f"Expected 400 iv_* columns, got {len(iv_cols)}")

    grid_meta = _infer_grid_order(iv_cols)

    iv = df[iv_cols].to_numpy(dtype=np.float32, copy=False)  # [T, 400]
    dates = df["date"].to_numpy(dtype="datetime64[D]")
    T = iv.shape[0]

    num_train = int(T * 0.7)
    num_test = int(T * 0.2)
    num_val = T - num_train - num_test
    border1s = [0, num_train - args.seq_len, T - num_test - args.seq_len]
    border2s = [num_train, num_train + num_val, T]

    borders = {
        "train": Borders(border1s[0], border2s[0]),
        "val": Borders(border1s[1], border2s[1]),
        "test": Borders(border1s[2], border2s[2]),
    }

    # PatchTST-compatible scaling: fit on train interval by time (not on windows)
    train_time = iv[borders["train"].border1 : borders["train"].border2]
    mean = train_time.mean(axis=0).astype(np.float32, copy=False)
    std = train_time.std(axis=0).astype(np.float32, copy=False)
    std = np.where(std == 0.0, 1.0, std).astype(np.float32, copy=False)
    np.save(os.path.join(args.out_dir, "scaler_mean.npy"), mean)
    np.save(os.path.join(args.out_dir, "scaler_std.npy"), std)

    meta = {
        "csv_path": os.path.abspath(args.csv_path),
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "graph_mode": str(args.graph_mode),
        "T": int(T),
        "num_train": int(num_train),
        "num_val": int(num_val),
        "num_test": int(num_test),
        "border1s": border1s,
        "border2s": border2s,
        "iv_cols": iv_cols,
        "grid": grid_meta,
    }

    for split, b in borders.items():
        series = iv[b.border1 : b.border2]
        x, y = _build_windows(series, args.seq_len, args.pred_len)
        np.savez_compressed(os.path.join(args.out_dir, f"{split}.npz"), x=x, y=y)
        meta[f"{split}_n"] = int(x.shape[0])

    # test start_dates: first prediction day for each test sample
    tb1 = borders["test"].border1
    test_n = meta["test_n"]
    start_dates = np.zeros((test_n,), dtype="datetime64[D]")
    for i in range(test_n):
        start_dates[i] = dates[tb1 + i + args.seq_len]
    np.save(os.path.join(args.out_dir, "start_dates.npy"), start_dates)

    # adjacency
    if args.graph_mode == "adaptive_only":
        A = np.eye(400, dtype=np.float32)
    else:
        A = _make_grid_adjacency(20, 20, self_loops=args.self_loops)
    sensor_ids = list(range(400))
    sensor_id_to_ind = {i: i for i in sensor_ids}
    with open(os.path.join(args.out_dir, "adj_mx.pkl"), "wb") as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, A), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

