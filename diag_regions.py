#!/usr/bin/env python3
"""
diag_regions.py — per-region test-error breakdown for finalised models.

Splits the 150-cell IV surface into 6 regions —
{short-tau, long-tau} x {put wing, ATM, call wing} — and reports, per
region, each model's MSE and its R^2 against the VAR baseline. The point
is to locate the corner(s) of the surface, if any, where a deep model
closes the gap to VAR or overtakes it: the per-region story behind a
flat overall result.

Reuses error_tables.py's seeded-preds discovery (the same
<ModelDir>/eval/63_<P>/seed_<S>/preds.npy layout), the deterministic VAR
baseline and a live-built persistence baseline. All metrics live in
standardized log-IV space — the space the models trained in.

Region pooling: a region's MSE is the mean squared error over every
(window, horizon, cell-in-region) entry. Because every cell carries the
same N*P observations, that equals the unweighted mean of per-cell MSE
over the region's cells — so per-cell MSE is computed once and every
region table is an average of it over a channel mask.

Channel layout: cell k = i_tau * n_money + i_money (tau outer, moneyness
inner), matching parse_grid() in train.py.

Outputs (under _test_results/63_<P>/diagnostics/):
  - region_breakdown.csv   tidy (model, region) rows: MSE and R^2-vs-VAR,
                           pooled over horizons and at horizon h+P
  - region_cell_gap.png    per-cell MSE heatmap: best deep model - VAR

Usage:
    python diag_regions.py --pred_len 21
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import LOOKBACK, ROOT, load_dataset
from error_tables import collect_seeded, load_var, test_end_dates


def region_masks(grid) -> dict[str, np.ndarray]:
    """6 boolean channel masks: {short-tau,long-tau} x {put,ATM,call}."""
    nt, nm = grid.n_tau, grid.n_money
    C = nt * nm
    it = np.arange(C) // nm
    im = np.arange(C) % nm
    t_mid = nt // 2
    m_a, m_b = nm // 3, 2 * (nm // 3)
    tau_bins = [("short-tau", it < t_mid), ("long-tau", it >= t_mid)]
    mon_bins = [("put wing",  im < m_a),
                ("ATM",       (im >= m_a) & (im < m_b)),
                ("call wing", im >= m_b)]
    regions = {}
    for tn, tmask in tau_bins:
        for mn, mmask in mon_bins:
            regions[f"{tn} / {mn}"] = tmask & mmask
    return regions


def per_cell_mse(pred: np.ndarray, Y: np.ndarray, hi: int | None = None
                 ) -> np.ndarray:
    """Per-channel MSE. `pred` may carry a leading seed axis; `Y` does
    not (it broadcasts). Returns (S, C) — S=1 for deterministic models."""
    if hi is None:
        d = pred - Y
        out = np.mean(d * d, axis=(-3, -2))
    else:
        d = pred[..., hi, :] - Y[..., hi, :]
        out = np.mean(d * d, axis=-2)
    return np.atleast_2d(out)


def _stat(values: np.ndarray) -> tuple[float, float]:
    """Seed-mean and sample std (ddof=1; 0 for a single value)."""
    return (float(values.mean()),
            float(values.std(ddof=1)) if values.size > 1 else 0.0)


def region_mse(cm: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return _stat(cm[:, mask].mean(axis=1))


def region_r2_vs_var(cm: np.ndarray, mask: np.ndarray,
                     var_region_mse: float) -> tuple[float, float]:
    """Per-seed R^2 = 1 - MSE_model / MSE_VAR, then seed-mean +/- std."""
    return _stat(1.0 - cm[:, mask].mean(axis=1) / var_region_mse)


def _fmt(stat: tuple[float, float], deep: bool, signed: bool) -> str:
    m, s = stat
    head = f"{m:+.4f}" if signed else f"{m:.4f}"
    return f"{head} ± {s:.4f}" if deep else head


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_len", type=int, default=21,
                    choices=(5, 10, 21, 42, 63))
    ap.add_argument("--csv_path",
                    default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--data_end", default="2023-12-29")
    ap.add_argument("--year", type=int, default=None,
                    help="restrict to one calendar year, e.g. 2023 "
                         "(window assigned by its last-horizon target date)")
    args = ap.parse_args()

    data_end = None if args.data_end.lower() == "none" else args.data_end
    data = load_dataset(args.csv_path, 0.7, 0.1, LOOKBACK, args.pred_len,
                        data_end=data_end)
    Xte, Yte = data["test"]
    grid = data["grid"]
    C = grid.n_tau * grid.n_money
    P = args.pred_len

    persistence = np.broadcast_to(Xte[:, -1:, :], Yte.shape).copy()

    print(f"\ndiscovering finalised models (pred_len={P}) ...")
    seeded = collect_seeded(P, Yte.shape)
    var = load_var(P, Yte.shape, C)
    if var is None:
        raise SystemExit("VAR baseline not found — region R^2 needs it.")
    if not seeded:
        raise SystemExit("no seeded models found — nothing to tabulate.")

    # Optional restriction to one calendar year. The full test period is
    # COVID-dominated (2020); the per-year view isolates a calm regime.
    end_dates = test_end_dates(P, args.csv_path, data_end,
                               data["rows"]["val_end"], data["rows"]["N"])
    if end_dates.shape[0] != Yte.shape[0]:
        raise SystemExit(f"date/window mismatch: {end_dates.shape[0]} dates "
                         f"vs {Yte.shape[0]} windows")
    tag, period = "", "full test period"
    if args.year is not None:
        wmask = pd.DatetimeIndex(end_dates).year == args.year
        if not wmask.any():
            raise SystemExit(f"no test windows land in {args.year}")
        Yte         = Yte[wmask]
        persistence = persistence[wmask]
        var["preds"] = var["preds"][wmask]
        for e in seeded:
            e["preds"] = e["preds"][:, wmask]
        tag, period = f"_{args.year}", str(args.year)
        d = pd.DatetimeIndex(end_dates[wmask])
        print(f"  restricted to {args.year}: {int(wmask.sum())} windows "
              f"({d.min().date()} → {d.max().date()})")

    # Per-cell MSE for every model: (S, C) pooled over horizons, and the
    # same at horizon h+P (the longest horizon — the corner most likely
    # to flip the ordering).
    models = [{"name": "Persistence", "deep": False,
               "cm":   per_cell_mse(persistence, Yte),
               "cm_h": per_cell_mse(persistence, Yte, P - 1)},
              {"name": "VAR", "deep": False,
               "cm":   per_cell_mse(var["preds"], Yte),
               "cm_h": per_cell_mse(var["preds"], Yte, P - 1)}]
    for e in seeded:
        models.append({"name": e["display"], "deep": True,
                       "cm":   per_cell_mse(e["preds"], Yte),
                       "cm_h": per_cell_mse(e["preds"], Yte, P - 1)})

    # Stable row order: overall (all-cell) seed-mean MSE.
    models.sort(key=lambda m: m["cm"].mean())

    regions = region_masks(grid)
    all_mask = np.ones(C, dtype=bool)
    var_cm   = next(m for m in models if m["name"] == "VAR")["cm"]
    var_cm_h = next(m for m in models if m["name"] == "VAR")["cm_h"]
    var_region = {r: float(var_cm[0, mask].mean())
                  for r, mask in regions.items()}
    var_region["ALL"] = float(var_cm[0].mean())
    var_region_h = {r: float(var_cm_h[0, mask].mean())
                    for r, mask in regions.items()}
    var_region_h["ALL"] = float(var_cm_h[0].mean())

    cols = list(regions) + ["ALL"]
    masks = {**regions, "ALL": all_mask}

    # ── Tidy CSV + three printed tables ──────────────────────────────
    recs = []
    mse_grid, r2_grid, r2h_grid = {}, {}, {}
    for m in models:
        mse_grid[m["name"]], r2_grid[m["name"]], r2h_grid[m["name"]] = {}, {}, {}
        for r in cols:
            mask = masks[r]
            mse = region_mse(m["cm"], mask)
            r2  = region_r2_vs_var(m["cm"], mask, var_region[r])
            r2h = region_r2_vs_var(m["cm_h"], mask, var_region_h[r])
            mse_grid[m["name"]][r] = _fmt(mse, m["deep"], signed=False)
            r2_grid[m["name"]][r]  = _fmt(r2,  m["deep"], signed=True)
            r2h_grid[m["name"]][r] = _fmt(r2h, m["deep"], signed=True)
            recs.append({"model": m["name"], "region": r,
                         "mse_mean": mse[0], "mse_std": mse[1],
                         "r2_var_mean": r2[0], "r2_var_std": r2[1],
                         "r2_var_h%d_mean" % P: r2h[0],
                         "r2_var_h%d_std" % P: r2h[1]})

    out_dir = os.path.join(ROOT, "_test_results", f"{LOOKBACK}_{P}",
                           "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(recs).to_csv(
        os.path.join(out_dir, f"region_breakdown{tag}.csv"), index=False)

    order = [m["name"] for m in models]
    bar = "=" * 78
    print(f"\n{bar}\nPER-REGION BREAKDOWN — pred_len={P}, {period}, "
          f"standardized log-IV space\n"
          f"± = sample std across seeds; VAR/Persistence deterministic.\n{bar}")
    for title, grid_d in [("MSE by region (pooled over horizons)", mse_grid),
                          ("R^2 vs VAR by region (pooled; + = beats VAR)",
                           r2_grid),
                          (f"R^2 vs VAR by region at horizon h+{P} "
                           f"(+ = beats VAR)", r2h_grid)]:
        df = pd.DataFrame(grid_d).T.reindex(order)[cols]
        print(f"\n{title}\n{df.to_string()}")

    # Winner per region (lowest pooled MSE seed-mean).
    print("\nwinner per region (lowest pooled MSE):")
    for r in cols:
        best = min(models, key=lambda m: m["cm"][:, masks[r]].mean())
        print(f"  {r:24s}  {best['name']}")

    # ── Per-cell MSE heatmap: best deep model - VAR ──────────────────
    best_deep = min((m for m in models if m["deep"]),
                    key=lambda m: m["cm"].mean())
    gap = (best_deep["cm"].mean(axis=0) - var_cm[0]).reshape(
        grid.n_tau, grid.n_money)
    lim = float(np.abs(gap).max())
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(gap, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   origin="lower")
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("tau (years)")
    ax.set_xticks(range(grid.n_money))
    ax.set_xticklabels([f"{m:+.3f}" for m in grid.money_vals],
                       rotation=90, fontsize=7)
    ax.set_yticks(range(grid.n_tau))
    ax.set_yticklabels([f"{t:.2f}" for t in grid.tau_vals], fontsize=7)
    ax.set_title(f"per-cell MSE gap: {best_deep['name']} - VAR  "
                 f"(blue = deep better, red = VAR better)")
    fig.colorbar(im, ax=ax, label="MSE difference")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"region_cell_gap{tag}.png"), dpi=150)
    plt.close(fig)

    print(f"\nwritten to {os.path.relpath(out_dir, ROOT)}/")
    print(f"  region_breakdown{tag}.csv, region_cell_gap{tag}.png")


if __name__ == "__main__":
    main()
