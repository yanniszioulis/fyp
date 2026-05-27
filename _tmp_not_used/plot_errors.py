#!/usr/bin/env python3
"""
plot_errors.py — 2×2 grid of per-horizon test errors over time.

For pred_len ∈ {5, 21, 63}, draw one subplot per horizon offset in
{t+1, t+5, t+10, t+21}. Each subplot has:
    x : target calendar date (the date being predicted at that horizon)
    y : mean squared error across the 150 IV cells of that
        single (window, horizon) forecast, in standardized log-IV space
    line: one per model loaded from `<ModelDir>/eval/63_<pred_len>/
          [<variant>/]preds.npy` (output of evaluate.py)

If `pred_len` is smaller than a requested horizon, that subplot is left
empty with an "unavailable" note.

Usage
-----
    python plot_errors.py --pred_len 21
    python plot_errors.py --pred_len 21 --model dlinear,hot,var --smooth 21
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from train import LOOKBACK, MODEL_DIR, ROOT, load_dataset


HORIZONS    = (1, 5, 10, 21)
DEEP_NAMES  = ("dlinear", "patchtst", "hot", "tucker_dlinear", "gwn")
ALL_NAMES   = (*DEEP_NAMES, "var")


# ─── Discovery ────────────────────────────────────────────────────────────

def find_preds_files(name: str, pred_len: int):
    """Return [(display_name, preds_path), ...]. One entry per variant."""
    base = os.path.join(ROOT, MODEL_DIR[name], "eval",
                        f"{LOOKBACK}_{pred_len}")
    if not os.path.isdir(base):
        return []
    found = []
    direct = os.path.join(base, "preds.npy")
    if os.path.isfile(direct):
        found.append((name, direct))
        return found
    # Variant subdirs.
    for entry in sorted(os.listdir(base)):
        sub = os.path.join(base, entry)
        p   = os.path.join(sub, "preds.npy")
        if os.path.isdir(sub) and os.path.isfile(p):
            found.append((f"{name}/{entry}", p))
    return found


# ─── Math ─────────────────────────────────────────────────────────────────

def per_window_horizon_mse(preds: np.ndarray, Yte: np.ndarray) -> np.ndarray:
    """`preds`/`Yte` shape (N_test, pred_len, n_channels). Returns
    (N_test, pred_len) — MSE averaged across the 150 cells."""
    if preds.shape != Yte.shape:
        raise ValueError(
            f"shape mismatch: preds {preds.shape} vs Yte {Yte.shape}")
    diff = preds - Yte
    return np.mean(diff * diff, axis=-1)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with `window` taps. Returns x unchanged
    when window <= 1. Edges use shrinking windows (min_periods)."""
    if window <= 1:
        return x
    s = pd.Series(x)
    return (s.rolling(window=window, center=True,
                      min_periods=max(1, window // 2))
             .mean()
             .to_numpy())


# ─── Entry point ──────────────────────────────────────────────────────────

def parse_models(arg: str) -> list[str]:
    if arg == "all":
        return list(ALL_NAMES)
    names = [s.strip() for s in arg.split(",") if s.strip()]
    bad = [n for n in names if n not in ALL_NAMES]
    if bad:
        raise SystemExit(
            f"Unknown model(s): {bad}. Pick from {ALL_NAMES} or 'all'.")
    return names


def main():
    ap = argparse.ArgumentParser(
        description="Plot per-horizon test errors of evaluated models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pred_len", required=True, type=int, choices=(5, 21, 63))
    ap.add_argument("--model",      default="all",
                    help="Comma list or 'all'. Defaults to 'all'.")
    ap.add_argument("--csv_path",   default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--data_end",   type=str,   default="2023-12-29")
    ap.add_argument("--ma",         type=int,   default=0,
                    help="Centered moving-average kernel (in days) applied "
                         "to each model's error series before plotting. "
                         "0 = raw, 21 ≈ one trading month, 63 ≈ one quarter. "
                         "Raw line is shown faintly underneath when MA > 1.")
    ap.add_argument("--out",        type=str,   default=None,
                    help="Output figure path. Defaults to "
                         "_test_results/63_<pred_len>/errors_by_horizon.png")
    args = ap.parse_args()

    # Load targets via the same pipeline used by evaluate.py.
    data_end = None if args.data_end.lower() == "none" else args.data_end
    data = load_dataset(args.csv_path, args.train_frac, args.val_frac,
                        LOOKBACK, args.pred_len, data_end=data_end)
    Xte, Yte = data["test"]
    r = data["rows"]
    N_test = Yte.shape[0]

    # Reconstruct test-window start indices to recover per-window target dates.
    N      = r["N"]
    n_win  = N - LOOKBACK - args.pred_len + 1
    starts = np.arange(n_win)
    target_end  = starts + LOOKBACK + args.pred_len
    test_starts = starts[target_end > r["val_end"]]
    assert test_starts.size == N_test, (test_starts.size, N_test)

    # Date series of the truncated dataframe.
    df = pd.read_csv(args.csv_path)
    if data_end is not None:
        df = df[df["date"] <= data_end].reset_index(drop=True)
    dates = pd.to_datetime(df["date"].to_numpy())

    # Discover candidate predictions.
    names = parse_models(args.model)
    series: list[tuple[str, np.ndarray]] = []
    missing: list[str] = []
    for n in names:
        files = find_preds_files(n, args.pred_len)
        if not files:
            missing.append(n)
            continue
        for display, path in files:
            preds = np.load(path)
            mse = per_window_horizon_mse(preds, Yte)   # (N_test, pred_len)
            series.append((display, mse))
            print(f"loaded {display}: preds={preds.shape}  "
                  f"per-window mse range=[{mse.min():.4f}, {mse.max():.4f}]")
    if missing:
        print(f"\nWARNING: no preds.npy found for: {missing}. "
              f"Run evaluate.py for them first.")
    if not series:
        raise SystemExit("No model predictions to plot.")

    ma_active = args.ma > 1
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    suptitle_extra = f"  [MA={args.ma}]" if ma_active else ""
    fig.suptitle(
        f"Test-set MSE by forecast horizon  "
        f"(lookback={LOOKBACK}, pred_len={args.pred_len}, "
        f"data_end={data.get('data_end')}, n_test={N_test}){suptitle_extra}",
        fontsize=12, y=0.995,
    )

    # Stable color per model across subplots.
    cmap = plt.get_cmap("tab10")
    color_by_model = {display: cmap(i % 10)
                      for i, (display, _) in enumerate(series)}

    for ax, h in zip(axes.flat, HORIZONS):
        if h > args.pred_len:
            ax.text(0.5, 0.5,
                    f"t+{h} unavailable (pred_len={args.pred_len})",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(f"t + {h}")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        h_idx = h - 1
        target_dates = dates[test_starts + LOOKBACK + h_idx]
        for display, mse in series:
            raw = mse[:, h_idx]
            col = color_by_model[display]
            if ma_active:
                ax.plot(target_dates, raw, color=col, linewidth=0.6, alpha=0.2)
                ax.plot(target_dates, smooth(raw, args.ma),
                        color=col, linewidth=1.6, label=display)
            else:
                ax.plot(target_dates, raw, color=col, linewidth=1.2,
                        label=display)
        ax.set_title(f"t + {h}")
        ax.set_ylabel("MSE (std log-IV)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
            ax.xaxis.get_major_locator()))

    # Single legend at the bottom.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.01),
               frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out = args.out or os.path.join(
        ROOT, "_test_results", f"{LOOKBACK}_{args.pred_len}",
        "errors_by_horizon.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved → {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
