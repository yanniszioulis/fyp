#!/usr/bin/env python3
"""
dm_tests_santa.py — pairwise Diebold-Mariano significance tests for the
SANTA family vs persistence vs VAR, split by regime and forecast horizon.

For each (regime, horizon) cell we:
  1. Take per-window squared-error losses L_i = mean((pred_i - y_i)**2 over
     cells), averaged over seeds for the seeded models. The seed-averaging
     happens on PREDS first (an ensemble), then the squared error is taken;
     this matches the headline `metrics_test.json` convention.
  2. Run two-sided Diebold-Mariano with Newey-West HAC variance on
     d_t = L_a,t - L_b,t for every pair (a, b). Lag = horizon being tested
     (or 21 for the full-pred_len cell) — the standard for h-step forecasts
     with overlapping windows.
  3. Print a pairwise t-stat matrix and p-value matrix, then a one-line
     "best model + significance vs runner-up" verdict per cell.

Regimes are stress-vs-calm collapsed from eval_full.py's calendar regimes:
    stress = COVID (2019-12 → 2020-12) ∪ Bear 2022 (2022-01 → 2022-12)
    calm   = Reflation 2021 ∪ Normalisation 2023
    full   = all test windows
Each window is bucketed by its last-horizon target date (same convention
as error_tables.py / eval_full.py).

Horizons reported: full pred_len (loss averaged over h=1..P), plus
the per-horizon slices h+1, h+5, h+10, h+21 (only the ones <= pred_len).

Two extra sections appear after the full-surface MSE pass:

  SHORT-MATURITY ONLY — same regime × horizon grid, but per-window loss is
  restricted to the short-tau zone (tau idx 0:3 of the H=n_tau axis, the
  closest-to-expiry maturities). Surfaces these cells move on different
  dynamics (vol-of-vol, near-ATM convexity) so the winner often differs
  from the full-surface winner.

  DIRECTIONAL ACCURACY — per regime × horizon, score = per-window fraction
  of cells whose predicted change (vs today) has the same sign as the
  realised change. Persistence predicts zero so its sign is 0 → fails the
  exact-equality test on essentially every cell; it's included only as a
  floor. DM is run on per-window accuracy series; here HIGHER is better so
  the cell renderer flips sign conventions (row beats col when t>0).

Usage
-----
    python dm_tests_santa.py --pred_len 21
"""
from __future__ import annotations

import argparse
import os
from math import erf, sqrt

import numpy as np
import pandas as pd

from train import LOOKBACK, ROOT, load_dataset
from _tmp_not_used.error_tables import _seed_preds, load_var, test_end_dates


# Same list as error_tables_santa.py.
SANTA_MODELS = [
    ("SANTA",          "SANTA"),
    ("SANTA_flat",     "SANTA-Flat"),
    ("SANTA_temporal", "SANTA-Temporal"),
]

# Calendar regimes (same as eval_full.py.REGIMES), then collapsed.
REGIMES_CAL = [
    ("COVID",          "2019-12-02", "2020-12-31"),
    ("Reflation calm", "2021-01-01", "2021-12-31"),
    ("Bear 2022",      "2022-01-01", "2022-12-31"),
    ("Normalisation",  "2023-01-01", "2023-12-29"),
]
STRESS_NAMES = {"COVID", "Bear 2022"}
CALM_NAMES   = {"Reflation calm", "Normalisation"}

HORIZONS_REPORT = (1, 5, 10, 21)

# Tau-zone slicing (matches eval_full.py.TAU_ZONES). With the parsed grid
# laid out tau-outer / moneyness-inner, preds.reshape(N,P,H,W)[:,:,SHORT,:]
# selects the closest-to-expiry maturities (tau ≈ 30-60d at the current
# 10-tau grid). Only the short slot is exercised by main(); the others are
# left in the table so a future call can reuse the constant.
TAU_ZONES = [("short_tau", slice(0, 3)),
             ("mid_tau",   slice(3, 7)),
             ("long_tau",  slice(7, 10))]


# ─── Stats ────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def dm(L_a: np.ndarray, L_b: np.ndarray, lag: int) -> tuple[float, float, float]:
    """Diebold-Mariano with Newey-West HAC variance on d = L_a - L_b.
    Returns (mean_diff, t_stat, two_sided_p)."""
    d = L_a - L_b
    N = d.size
    mu = float(d.mean())
    e  = d - mu
    var = float((e * e).mean())
    for h in range(1, lag + 1):
        w = 1.0 - h / (lag + 1)
        var += 2.0 * w * float((e[h:] * e[:-h]).mean())
    var = max(var, 1e-12)
    se = sqrt(var / N)
    t  = mu / se
    p  = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return mu, t, p


# ─── Loss tensors ────────────────────────────────────────────────────────

def per_window_loss(preds: np.ndarray, Y: np.ndarray,
                    horizon: int | None) -> np.ndarray:
    """preds, Y: (N, P, C).  horizon=None -> mean over horizons; else int hi
    in [1, P] selects that single horizon slice. Returns (N,) MSE per window."""
    if horizon is None:
        e = preds - Y
        return (e * e).mean(axis=(1, 2))
    hi = horizon - 1
    e  = preds[:, hi, :] - Y[:, hi, :]
    return (e * e).mean(axis=1)


def per_window_loss_zone(preds: np.ndarray, Y: np.ndarray,
                         H: int, W: int, horizon: int | None,
                         tau_slice: slice) -> np.ndarray:
    """Same as per_window_loss but loss is restricted to a tau-zone.
    Reshape (N,P,C=H*W) → (N,P,H,W); slice rows of the tau axis; mean
    over (remaining) horizon, kept tau rows, and all moneyness."""
    N, P, C = preds.shape
    assert C == H * W
    p = preds.reshape(N, P, H, W)[:, :, tau_slice, :]
    y = Y    .reshape(N, P, H, W)[:, :, tau_slice, :]
    if horizon is None:
        e = p - y
        return (e * e).mean(axis=(1, 2, 3))
    hi = horizon - 1
    e  = p[:, hi, :, :] - y[:, hi, :, :]
    return (e * e).mean(axis=(1, 2))


def per_window_dir_acc(preds: np.ndarray, today: np.ndarray, Y: np.ndarray,
                       horizon: int | None) -> np.ndarray:
    """Per-window directional accuracy: fraction of cells where
    sign(pred_change) == sign(true_change), with the change measured
    against today's surface.

      preds, Y: (N, P, C)
      today:    (N, C)        — z at t (the input window's last day)
      horizon:  None -> mean over h=1..P; else single horizon slice

    Persistence predicts zero change (sign 0), so its accuracy is the
    fraction of cells whose true change is also exactly zero — typically
    ≈0 in practice. Included only as a floor.
    """
    todayP = today[:, None, :]                       # (N, 1, C)
    dpred  = np.sign(preds - todayP)                 # (N, P, C) in {-1,0,1}
    dtrue  = np.sign(Y     - todayP)
    correct = (dpred == dtrue).astype(np.float32)    # (N, P, C)
    if horizon is None:
        return correct.mean(axis=(1, 2))
    hi = horizon - 1
    return correct[:, hi, :].mean(axis=1)


def collect_model_preds(pred_len: int, y_shape: tuple,
                        n_channels: int,
                        budget: str = "50k") -> dict[str, dict]:
    """display -> {"preds": (N, P, C) ensemble (seed-mean), "n_seeds": int,
    "deterministic": bool}. Order is intentional: persistence, 3 SANTAs, VAR.

    The `budget` subfolder layer (added 2026-05 alongside the 50k matched-
    budget runs) lets multiple param-count regimes live under one eval/
    tree. Pass "25k", "100k" etc. to read a different budget without
    touching anything else.
    """
    out: dict[str, dict] = {}
    for mdir, display in SANTA_MODELS:
        base = os.path.join(ROOT, mdir, "eval",
                            f"{LOOKBACK}_{pred_len}", budget)
        got  = _seed_preds(base)
        if got is None:
            print(f"  skip {display}: no seed preds under "
                  f"{os.path.relpath(base, ROOT)}/")
            continue
        seeds, preds, _ = got
        if preds.shape[1:] != y_shape:
            raise SystemExit(f"{display}: shape mismatch")
        # Ensemble = seed-mean prediction. Standard "ensemble" reading.
        out[display] = {"preds": preds.mean(axis=0),
                        "n_seeds": preds.shape[0],
                        "deterministic": False}
    var = load_var(pred_len, y_shape, n_channels)
    if var is not None:
        out["VAR"] = {"preds": var["preds"],
                      "n_seeds": None, "deterministic": True}
    return out


# ─── Regime masks ─────────────────────────────────────────────────────────

def build_masks(end_dates: pd.DatetimeIndex, N: int):
    """Return {label: mask (N,)} for full/stress/calm and each calendar regime."""
    cal_label = np.full(N, "unassigned", dtype=object)
    for name, lo, hi in REGIMES_CAL:
        m = ((end_dates >= pd.Timestamp(lo))
             & (end_dates <= pd.Timestamp(hi)))
        cal_label[m] = name
    masks = {
        "full":   np.ones(N, dtype=bool),
        "stress": np.isin(cal_label, list(STRESS_NAMES)),
        "calm":   np.isin(cal_label, list(CALM_NAMES)),
    }
    for name in (n for n, _, _ in REGIMES_CAL):
        masks[name] = (cal_label == name)
    return masks


# ─── Rendering ────────────────────────────────────────────────────────────

def _stars(p: float) -> str:
    return "***" if p < 0.001 else ("**" if p < 0.01 else
                                    ("*"   if p < 0.05 else " "))


def render_pair_matrix(title: str, displays: list[str],
                       T: np.ndarray, P: np.ndarray,
                       mean_score: dict[str, float],
                       lower_better: bool = True,
                       score_label: str = "L") -> str:
    """Pretty-print a pairwise t-stat / p-value matrix.

    Cell (i, j) shows t and stars for d = score_i - score_j.
    - lower_better=True  (default, losses): t<0 means row beats column.
    - lower_better=False (e.g. accuracy):   t>0 means row beats column.
    Diagonal is the per-model mean score, labelled by `score_label`.
    """
    order = sorted(displays,
                   key=lambda d: mean_score[d] * (1 if lower_better else -1))
    idx   = {d: i for i, d in enumerate(displays)}

    name_w = max(len(d) for d in order)
    col_w  = 13

    def fmt_cell(i: int, j: int) -> str:
        if i == j:
            return f"{score_label}={mean_score[order[i]]:.4f}".rjust(col_w)
        t = T[idx[order[i]], idx[order[j]]]
        p = P[idx[order[i]], idx[order[j]]]
        return f"{t:+6.2f}{_stars(p)}".rjust(col_w)

    direction = ("rows beat cols when t<0" if lower_better
                 else "rows beat cols when t>0")
    lines = [title, f"  {direction}; *** p<.001, ** p<.01, * p<.05"]
    head = " " * name_w + " | " + " | ".join(d.rjust(col_w) for d in order)
    lines.append(head)
    lines.append("-" * len(head))
    for i, d in enumerate(order):
        row = d.ljust(name_w) + " | " + " | ".join(
            fmt_cell(i, j) for j in range(len(order)))
        lines.append(row)
    return "\n".join(lines)


def best_with_significance(displays: list[str], T: np.ndarray, P: np.ndarray,
                           mean_score: dict[str, float],
                           lower_better: bool = True,
                           score_label: str = "L") -> str:
    """One-line verdict: best model + the worst p-value over the
    pairwise tests of best vs each other. For lower_better=True we want
    t<0 (best smaller); for lower_better=False we want t>0 (best larger)."""
    order = sorted(displays,
                   key=lambda d: mean_score[d] * (1 if lower_better else -1))
    best  = order[0]
    rest  = order[1:]
    idx   = {d: i for i, d in enumerate(displays)}
    worst_p = 0.0
    worst_vs = None
    for d in rest:
        t = T[idx[best], idx[d]]
        p = P[idx[best], idx[d]]
        wrong_sign = (t > 0) if lower_better else (t < 0)
        if wrong_sign or p > worst_p:
            worst_p, worst_vs = p, d
    return (f"best: {best:14s}  {score_label}={mean_score[best]:.4f}   "
            f"vs runner-up '{worst_vs}': p={worst_p:.4f} {_stars(worst_p)}")


# ─── Main loop ────────────────────────────────────────────────────────────

def _run_dm_grid(scores: dict[str, np.ndarray], lag: int, lower_better: bool,
                 score_label: str, head: str) -> str:
    """Shared kernel: scores[display] = (N_sub,) series; compute pairwise
    DM, render matrix, append best-with-significance verdict."""
    displays = list(scores.keys())
    K = len(displays)
    T = np.zeros((K, K), dtype=np.float64)
    P = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            _, t, p = dm(scores[displays[i]], scores[displays[j]], lag)
            T[i, j], P[i, j] = t, p
    mean_score = {d: float(scores[d].mean()) for d in displays}
    body = render_pair_matrix("", displays, T, P, mean_score,
                              lower_better=lower_better,
                              score_label=score_label)
    verdict = best_with_significance(displays, T, P, mean_score,
                                     lower_better=lower_better,
                                     score_label=score_label)
    return f"{head}\n{body}\n\n{verdict}"


def run_cell(label: str, models: dict[str, dict], Y: np.ndarray,
             persistence: np.ndarray, mask: np.ndarray, horizon: int | None,
             lag: int) -> str:
    """Per-window MSE on the full surface for the (mask, horizon) subset."""
    sub_Y = Y[mask]
    scores: dict[str, np.ndarray] = {}
    scores["Persistence"] = per_window_loss(persistence[mask], sub_Y, horizon)
    for d, info in models.items():
        scores[d] = per_window_loss(info["preds"][mask], sub_Y, horizon)
    h_lbl = f"h+{horizon}" if horizon else "mean over h=1..P"
    head  = (f"\n{'='*78}\n{label.upper()}  ·  {h_lbl}  ·  "
             f"n_windows={int(mask.sum())}  ·  DM lag={lag}\n{'='*78}")
    return _run_dm_grid(scores, lag, lower_better=True, score_label="L",
                        head=head)


def run_cell_zone(label: str, models: dict[str, dict], Y: np.ndarray,
                  persistence: np.ndarray, mask: np.ndarray,
                  horizon: int | None, lag: int,
                  H: int, W: int, tau_slice: slice, zone_name: str) -> str:
    """Per-window MSE restricted to the given tau-zone."""
    sub_Y = Y[mask]
    scores: dict[str, np.ndarray] = {}
    scores["Persistence"] = per_window_loss_zone(
        persistence[mask], sub_Y, H, W, horizon, tau_slice)
    for d, info in models.items():
        scores[d] = per_window_loss_zone(
            info["preds"][mask], sub_Y, H, W, horizon, tau_slice)
    h_lbl = f"h+{horizon}" if horizon else "mean over h=1..P"
    head  = (f"\n{'='*78}\n{label.upper()}  ·  {zone_name}  ·  {h_lbl}  ·  "
             f"n_windows={int(mask.sum())}  ·  DM lag={lag}\n{'='*78}")
    return _run_dm_grid(scores, lag, lower_better=True, score_label="L",
                        head=head)


def run_cell_diracc(label: str, models: dict[str, dict], Y: np.ndarray,
                    persistence: np.ndarray, today: np.ndarray,
                    mask: np.ndarray, horizon: int | None, lag: int) -> str:
    """Per-window directional accuracy on the full surface."""
    sub_Y = Y[mask]
    sub_T = today[mask]
    scores: dict[str, np.ndarray] = {}
    scores["Persistence"] = per_window_dir_acc(persistence[mask], sub_T,
                                               sub_Y, horizon)
    for d, info in models.items():
        scores[d] = per_window_dir_acc(info["preds"][mask], sub_T,
                                       sub_Y, horizon)
    h_lbl = f"h+{horizon}" if horizon else "mean over h=1..P"
    head  = (f"\n{'='*78}\n{label.upper()}  ·  dir-accuracy  ·  {h_lbl}  ·  "
             f"n_windows={int(mask.sum())}  ·  DM lag={lag}\n{'='*78}")
    return _run_dm_grid(scores, lag, lower_better=False, score_label="acc",
                        head=head)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_len", type=int, default=21,
                    choices=(5, 10, 21, 42, 63))
    ap.add_argument("--csv_path", default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--data_end", default="2023-12-29")
    ap.add_argument("--budget", default="50k",
                    help="Param-count subfolder under each SANTA*/eval/63_<P>/. "
                         "Default '50k' matches the matched-budget runs.")
    args = ap.parse_args()

    data_end = None if args.data_end.lower() == "none" else args.data_end

    data = load_dataset(args.csv_path, 0.7, 0.1, LOOKBACK, args.pred_len,
                        data_end=data_end)
    Xte, Yte = data["test"]
    N_test = Yte.shape[0]
    C = data["rows"]["n_channels"]

    persistence = np.broadcast_to(Xte[:, -1:, :], Yte.shape).copy()
    end_dates = test_end_dates(args.pred_len, args.csv_path, data_end,
                               data["rows"]["val_end"], data["rows"]["N"])
    end_dates = pd.DatetimeIndex(end_dates)
    assert end_dates.shape[0] == N_test

    print(f"\nloading models (pred_len={args.pred_len}, "
          f"budget={args.budget}) ...")
    models = collect_model_preds(args.pred_len, Yte.shape, C,
                                 budget=args.budget)
    if not any(d.startswith("SANTA") for d in models):
        raise SystemExit("no SANTA seed preds found — nothing to test.")

    masks = build_masks(end_dates, N_test)
    # Print the regime sizes once so the reader knows the n_windows per cell.
    print("\nregime sizes (windows):")
    for label in ("full", "stress", "calm",
                  "COVID", "Bear 2022", "Reflation calm", "Normalisation"):
        print(f"  {label:18s}  n={int(masks[label].sum())}")

    horizons = [None] + [h for h in HORIZONS_REPORT if h <= args.pred_len]
    regime_keys = ("full", "stress", "calm",
                   "COVID", "Reflation calm", "Bear 2022", "Normalisation")

    H, W = data["grid"].n_tau, data["grid"].n_money
    short_slice = TAU_ZONES[0][1]   # ("short_tau", slice(0, 3))
    # `today` for the directional pass is the last input slice (same
    # tensor persistence is broadcast from), kept in (N, C) form.
    today = Xte[:, -1, :].copy()

    full_blocks: list[str] = []
    short_blocks: list[str] = []
    diracc_blocks: list[str] = []
    for label in regime_keys:
        m = masks[label]
        if not m.any():
            print(f"  skipping {label}: empty mask")
            continue
        for h in horizons:
            lag = h if h is not None else args.pred_len
            full_blocks.append(run_cell(label, models, Yte, persistence,
                                        m, h, lag))
            short_blocks.append(run_cell_zone(label, models, Yte, persistence,
                                              m, h, lag,
                                              H, W, short_slice, "short_tau"))
            diracc_blocks.append(run_cell_diracc(label, models, Yte,
                                                 persistence, today,
                                                 m, h, lag))

    out_dir = os.path.join(ROOT, "_test_results",
                           f"{LOOKBACK}_{args.pred_len}",
                           f"dm_tests_santa_{args.budget}")
    os.makedirs(out_dir, exist_ok=True)

    # Headline (full-surface MSE) → its own file, same content as before.
    out_full = os.path.join(out_dir, "dm_report.txt")
    with open(out_full, "w") as f:
        f.write("Diebold-Mariano pairwise tests, SANTA family vs VAR vs "
                "Persistence\n")
        f.write(f"pred_len={args.pred_len}  data_end={data_end}  "
                f"loss=MSE per window (seed-mean preds for SANTAs)\n")
        f.write("\n".join(full_blocks) + "\n")

    # Short-maturity MSE → its own file. Same cell layout, restricted loss.
    out_short = os.path.join(out_dir, "dm_report_short_tau.txt")
    with open(out_short, "w") as f:
        f.write("Diebold-Mariano pairwise tests, SHORT-MATURITY cells only "
                "(tau idx 0:3)\n")
        f.write(f"pred_len={args.pred_len}  data_end={data_end}  "
                f"H={H} W={W}  loss=MSE per window restricted to short_tau\n")
        f.write("\n".join(short_blocks) + "\n")

    # Directional accuracy → its own file. score=fraction sign-correct.
    out_dir_txt = os.path.join(out_dir, "dm_report_directional.txt")
    with open(out_dir_txt, "w") as f:
        f.write("Diebold-Mariano pairwise tests on DIRECTIONAL ACCURACY\n")
        f.write(f"pred_len={args.pred_len}  data_end={data_end}  "
                f"score=fraction of cells with sign(pred-today)=sign(true-today)\n")
        f.write("Persistence predicts zero so its sign is 0; included as a "
                "floor.\n")
        f.write("\n".join(diracc_blocks) + "\n")

    print("\n" + "#" * 78)
    print("# 1. FULL-SURFACE MSE")
    print("#" * 78)
    for b in full_blocks:
        print(b)
    print("\n" + "#" * 78)
    print("# 2. SHORT-MATURITY (tau idx 0:3) MSE")
    print("#" * 78)
    for b in short_blocks:
        print(b)
    print("\n" + "#" * 78)
    print("# 3. DIRECTIONAL ACCURACY")
    print("#" * 78)
    for b in diracc_blocks:
        print(b)
    print(f"\nreports written under {os.path.relpath(out_dir, ROOT)}/")
    print(f"  - dm_report.txt              (full surface MSE)")
    print(f"  - dm_report_short_tau.txt    (short-maturity MSE)")
    print(f"  - dm_report_directional.txt  (directional accuracy)")


if __name__ == "__main__":
    main()
