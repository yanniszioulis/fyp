#!/usr/bin/env python3
"""
error_tables_santa.py — seed-averaged error tables for the SANTA family.

Focused variant of error_tables.py covering only:
    Persistence   (live-built, deterministic)
    SANTA         (10 seeds)
    SANTA-Flat    (10 seeds, joint-spatial ablation)
    SANTA-Temporal(10 seeds, temporal-only ablation)
    VAR           (deterministic, two-stage Gonçalves-Guidolin)

Same metric pack and conventions as error_tables.py:
  - All metrics in standardized log-IV space.
  - Per-seed metric first, then seed-mean ± seed-std (ddof=1); VAR and
    Persistence carry a single value (deterministic).
  - R2 vs Persist = 1 - MSE_model / MSE_pers   (per-seed then averaged).
  - R2 vs VAR     = 1 - MSE_model / MSE_VAR    (per-seed then averaged).
  - A test window is assigned to a calendar year by its last-horizon
    target date.
  - Output: five period tables (full + 2020/2021/2022/2023) and five
    horizon-breakdown tables. CSVs land under
    `_test_results/63_<P>/error_tables_santa/`.

Usage
-----
    python error_tables_santa.py --pred_len 21
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from train import LOOKBACK, ROOT, load_dataset
from error_tables import (
    HORIZONS,
    YEARS,
    _seed_preds,
    horizon_rows,
    load_var,
    render_horizon_table,
    render_table,
    table_rows,
    test_end_dates,
    write_csv,
    write_horizon_csv,
)


# (model_dir, display name)
SANTA_MODELS = [
    ("SANTA",          "SANTA"),
    ("SANTA_flat",     "SANTA-Flat"),
    ("SANTA_temporal", "SANTA-Temporal"),
]


def collect_santa(pred_len: int, y_shape: tuple,
                  budget: str = "50k") -> list[dict]:
    """Walk <SANTA*>/eval/63_<P>/<budget>/seed_<S>/ for each variant. The
    `budget` subfolder layer (added 2026-05 when the 50k matched-budget
    runs landed) lets multiple param-count regimes coexist under one
    `eval/` tree — pass e.g. "25k" or "100k" later without touching the
    eval scripts."""
    out = []
    for mdir, display in SANTA_MODELS:
        base = os.path.join(ROOT, mdir, "eval",
                            f"{LOOKBACK}_{pred_len}", budget)
        got = _seed_preds(base)
        if got is None:
            print(f"  skip {display}: no seed preds under "
                  f"{os.path.relpath(base, ROOT)}/")
            continue
        seeds, preds, n_params = got
        if preds.shape[1:] != y_shape:
            raise SystemExit(f"{display}: preds shape {preds.shape[1:]} "
                             f"per seed != Y_test {y_shape}")
        print(f"  {display:14s}  seeds={seeds}  n_params={n_params}")
        out.append({"display": display, "preds": preds, "seeds": seeds,
                    "n_params": n_params, "deterministic": False})
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_len", type=int, default=21,
                    choices=(5, 10, 21, 42, 63))
    ap.add_argument("--csv_path", default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--data_end", default="2023-12-29")
    ap.add_argument("--budget", default="50k",
                    help="Param-count subfolder under each SANTA/eval/63_<P>/. "
                         "Default '50k' matches the matched-budget runs.")
    args = ap.parse_args()

    data_end = None if args.data_end.lower() == "none" else args.data_end

    data = load_dataset(args.csv_path, 0.7, 0.1, LOOKBACK, args.pred_len,
                        data_end=data_end)
    Xte, Yte = data["test"]
    C = data["rows"]["n_channels"]

    persistence = np.broadcast_to(Xte[:, -1:, :], Yte.shape).copy()
    end_dates = test_end_dates(args.pred_len, args.csv_path, data_end,
                               data["rows"]["val_end"], data["rows"]["N"])
    if end_dates.shape[0] != Yte.shape[0]:
        raise SystemExit(f"date/window mismatch: {end_dates.shape[0]} dates "
                         f"vs {Yte.shape[0]} windows")

    print(f"\ndiscovering SANTA family (pred_len={args.pred_len}, "
          f"budget={args.budget}) ...")
    seeded = collect_santa(args.pred_len, Yte.shape, budget=args.budget)
    var    = load_var(args.pred_len, Yte.shape, C)
    if not seeded:
        raise SystemExit("no SANTA seed preds found — nothing to tabulate.")

    # Fixed row order across every period table: sort by full-period MSE
    # (seed-mean for the SANTA models, single value for VAR/Persistence).
    full_mask = np.ones(Yte.shape[0], dtype=bool)
    full_rows = table_rows(seeded, var, persistence, Yte, full_mask)
    order = sorted(full_rows, key=lambda d: full_rows[d]["mse"][0])

    years = pd.DatetimeIndex(end_dates).year
    tables = [("full", full_mask, full_rows)]
    for y in YEARS:
        m = (years == y)
        if not m.any():
            print(f"  (no windows for {y} — skipping that table)")
            continue
        tables.append((str(y), m,
                       table_rows(seeded, var, persistence, Yte, m)))

    out_dir = os.path.join(ROOT, "_test_results",
                           f"{LOOKBACK}_{args.pred_len}",
                           f"error_tables_santa_{args.budget}")
    os.makedirs(out_dir, exist_ok=True)

    horizons = [h for h in HORIZONS if h <= args.pred_len]

    def period_title(label: str, mask: np.ndarray) -> str:
        n = int(mask.sum())
        d0, d1 = end_dates[mask].min(), end_dates[mask].max()
        head = "FULL TEST PERIOD" if label == "full" else label
        return f"{head}  ({n} windows, {d0.date()} → {d1.date()})"

    print(f"\n{'='*72}\nSANTA-FAMILY ERROR TABLES — pred_len={args.pred_len}, "
          f"standardized log-IV space")
    print("± = sample std-dev across seeds (ddof=1); VAR and Persistence "
          "are deterministic.\n" + "=" * 72)

    for label, mask, rows in tables:
        print("\n" + render_table(period_title(label, mask), order, rows))
        write_csv(os.path.join(out_dir, f"error_table_{label}.csv"),
                  order, rows)

    print(f"\n\n{'='*72}\nHORIZON BREAKDOWN — error evaluated at "
          f"{', '.join('h+' + str(k) for k in horizons)}")
    print("each metric uses that horizon's slice only; baselines recomputed "
          "per horizon.\n" + "=" * 72)

    for label, mask, _ in tables:
        hrows = horizon_rows(seeded, var, persistence, Yte, mask, horizons)
        title = period_title(label, mask) + " — horizon breakdown"
        print("\n" + render_horizon_table(title, order, horizons, hrows))
        write_horizon_csv(os.path.join(out_dir, f"horizon_table_{label}.csv"),
                          order, horizons, hrows)

    print(f"\nCSVs written under {os.path.relpath(out_dir, ROOT)}/")


if __name__ == "__main__":
    main()
