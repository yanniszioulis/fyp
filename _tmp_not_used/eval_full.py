#!/usr/bin/env python3
"""
eval_full.py — full evaluation report for the tuning-winner of every model.

Reads `<ModelDir>/eval/63_<pred_len>/[<variant>/]preds.npy` (produced by
evaluate.py from each tuning winner) and emits a stack of metrics and
figures under `_test_results/63_<pred_len>/full/`.

All metrics are computed in standardized log-IV space — the same space
the models were trained against. Persistence (= last input surface,
broadcast across all horizons) is the only baseline.

Usage
-----
    python eval_full.py --pred_len 21
"""
from __future__ import annotations

import argparse
import json
import os
from math import erf, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from train import LOOKBACK, MODEL_DIR, ROOT, load_dataset


DEEP_NAMES = ("dlinear", "patchtst", "hot", "tucker_dlinear", "gwn")
ALL_NAMES  = (*DEEP_NAMES, "var")

# Calendar regime buckets, anchored at each window's last-horizon target date.
REGIMES = [
    ("COVID",          "2019-12-02", "2020-12-31"),
    ("Reflation calm", "2021-01-01", "2021-12-31"),
    ("Bear 2022",      "2022-01-01", "2022-12-31"),
    ("Normalisation",  "2023-01-01", "2023-12-29"),
]

# Surface zones (indices into a (n_tau=10, n_money=15) grid).
# tau order:    [0.0822, 0.1085, 0.1432, 0.189, 0.2495, 0.3294, 0.4348, 0.5739, 0.7576, 1.0]
# money order:  [-0.1, -0.0857, ..., 0, ..., 0.0857, 0.1]
TAU_ZONES   = [("short_tau", slice(0, 3)),
               ("mid_tau",   slice(3, 7)),
               ("long_tau",  slice(7, 10))]
MONEY_ZONES = [("put_wing", slice(0, 5)),
               ("ATM",      slice(5, 10)),
               ("call_wing", slice(10, 15))]

# Example forecast-surface anchor dates (closest test target date is used).
EXAMPLE_DATES = [
    ("covid", "2020-03-23"),
    ("bear",  "2022-06-15"),
    ("calm",  "2023-08-15"),
]


# ─── Discovery ────────────────────────────────────────────────────────────

def discover_models(pred_len: int):
    """Return [(display, model_name, variant, [(seed_or_None, preds_path,
    source_path|None), ...]), ...]. Each (display) entry carries a list
    of one or more (seed, paths) tuples. Layouts supported:

      <ModelDir>/eval/63_<P>/preds.npy                 (single, seed=None)
      <ModelDir>/eval/63_<P>/seed_<S>/preds.npy        (multi-seed, no variant)
      <ModelDir>/eval/63_<P>/<variant>/preds.npy       (variant, single)
      <ModelDir>/eval/63_<P>/<variant>/seed_<S>/preds.npy  (variant, multi-seed)
    """
    out = []
    for name in ALL_NAMES:
        base = os.path.join(ROOT, MODEL_DIR[name], "eval",
                            f"{LOOKBACK}_{pred_len}")
        if not os.path.isdir(base):
            continue
        seed_dirs = _seed_dirs(base)
        direct    = os.path.join(base, "preds.npy")
        if seed_dirs:
            paths = [(s, os.path.join(d, "preds.npy"),
                      _maybe(os.path.join(d, "source.json")))
                     for s, d in seed_dirs]
            out.append((name, name, None, paths))
            continue
        if os.path.isfile(direct):
            out.append((name, name, None,
                        [(None, direct, _maybe(os.path.join(base, "source.json")))]))
            continue
        # No direct preds → scan variant subdirs.
        for entry in sorted(os.listdir(base)):
            sub = os.path.join(base, entry)
            if not os.path.isdir(sub) or entry.startswith("seed_"):
                continue
            seed_dirs_v = _seed_dirs(sub)
            direct_v    = os.path.join(sub, "preds.npy")
            if seed_dirs_v:
                paths = [(s, os.path.join(d, "preds.npy"),
                          _maybe(os.path.join(d, "source.json")))
                         for s, d in seed_dirs_v]
                out.append((f"{name}/{entry}", name, entry, paths))
            elif os.path.isfile(direct_v):
                out.append((f"{name}/{entry}", name, entry,
                            [(None, direct_v,
                              _maybe(os.path.join(sub, "source.json")))]))
    return out


def _seed_dirs(parent: str) -> list[tuple[int, str]]:
    """Return [(seed_int, dirpath), ...] for any seed_<N>/ subdirs that
    contain a preds.npy."""
    out = []
    if not os.path.isdir(parent):
        return out
    for entry in sorted(os.listdir(parent)):
        if not entry.startswith("seed_"):
            continue
        sub = os.path.join(parent, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "preds.npy")):
            try:
                s = int(entry[len("seed_"):])
            except ValueError:
                continue
            out.append((s, sub))
    return sorted(out, key=lambda t: t[0])


def _maybe(path: str) -> str | None:
    return path if os.path.isfile(path) else None


# ─── Dates / regimes ──────────────────────────────────────────────────────

def test_target_dates(data: dict, pred_len: int, csv_path: str, data_end):
    """Return (n_test,) datetime array — last-horizon target date per window."""
    L, P = LOOKBACK, pred_len
    r    = data["rows"]
    N    = r["N"]
    n_win = N - L - P + 1
    starts = np.arange(n_win)
    target_end = starts + L + P
    test_starts = starts[target_end > r["val_end"]]

    df = pd.read_csv(csv_path)
    if data_end is not None:
        df = df[df["date"] <= data_end].reset_index(drop=True)
    dates = pd.to_datetime(df["date"].to_numpy())
    last_target_idx = test_starts + L + P - 1
    return dates[last_target_idx], test_starts, dates


def assign_regime(end_dates: np.ndarray) -> np.ndarray:
    """Bucket each test window into one of REGIMES by its last target date.
    Windows outside every bucket get the label "unassigned" (shouldn't happen
    for the 21-day, 2019-12 → 2023-12 test window)."""
    labels = np.full(end_dates.shape, "unassigned", dtype=object)
    for name, lo, hi in REGIMES:
        m = (end_dates >= pd.Timestamp(lo)) & (end_dates <= pd.Timestamp(hi))
        labels[m] = name
    return labels


# ─── Stats helpers ────────────────────────────────────────────────────────

def block_bootstrap_idx(N: int, block: int, B: int, rng) -> np.ndarray:
    """Return (B, N) int64 indices via a moving-block bootstrap."""
    n_blocks = (N + block - 1) // block
    starts = rng.integers(0, N - block + 1, size=(B, n_blocks))
    grid = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)
    return grid[:, :N]


def ci(arr: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return (float(np.quantile(arr, alpha / 2)),
            float(np.quantile(arr, 1 - alpha / 2)))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def dm_test(L_a: np.ndarray, L_b: np.ndarray, lag: int) -> dict:
    """Diebold-Mariano with Newey-West HAC variance.
    L_a, L_b: per-window losses (N,). Returns mean diff (L_a - L_b),
    t-stat, two-sided p-value."""
    d = L_a - L_b
    N = d.size
    mu = float(d.mean())
    e = d - mu
    var = float((e * e).mean())
    for h in range(1, lag + 1):
        w = 1.0 - h / (lag + 1)
        var += 2.0 * w * float((e[h:] * e[:-h]).mean())
    var = max(var, 1e-12)
    se = sqrt(var / N)
    t  = mu / se
    p  = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return {"mean_diff": mu, "t": float(t), "p": float(p), "se": float(se)}


# ─── Metric pack ──────────────────────────────────────────────────────────

def per_window_loss(err: np.ndarray) -> np.ndarray:
    """Mean square error per window, averaged over (horizon, cells)."""
    return (err * err).mean(axis=(1, 2))


def headline(err: np.ndarray, err_pers: np.ndarray, Y: np.ndarray,
             preds: np.ndarray) -> dict:
    mse  = float((err * err).mean())
    rmse = float(np.sqrt(mse))
    mae  = float(np.abs(err).mean())
    mse_p = float((err_pers * err_pers).mean())
    r2_p  = float(1.0 - mse / mse_p)
    var_ratio = float(preds.std() / Y.std())
    return dict(MSE=mse, RMSE=rmse, MAE=mae, R2_pers=r2_p, var_ratio=var_ratio,
                MSE_pers=mse_p)


# ─── Load + cache the err tensors ────────────────────────────────────────

def load_all(pred_len: int, csv_path: str, data_end):
    print(f"loading dataset (data_end={data_end}) ...")
    data = load_dataset(csv_path, 0.7, 0.1, LOOKBACK, pred_len, data_end=data_end)
    Xte, Yte = data["test"]
    grid = data["grid"]
    N_test, P, C = Yte.shape
    H, W = grid.n_tau, grid.n_money
    assert C == H * W

    # Persistence: last input row, broadcast across all horizons.
    persistence = np.broadcast_to(Xte[:, -1:, :], (N_test, P, C)).copy()

    end_dates, test_starts, all_dates = test_target_dates(
        data, pred_len, csv_path, data_end)

    models = discover_models(pred_len)
    if not models:
        raise SystemExit("No preds.npy found under any <ModelDir>/eval/. "
                         "Run evaluate.py first.")

    sources       = {}
    preds_canon   = {}   # display -> (N, P, C) ensemble or single
    sq_err        = {}   # display -> (N, P, C) seed-mean squared err
    abs_err       = {}   # display -> (N, P, C) seed-mean absolute err
    L_seed        = {}   # display -> (S, N) per-seed per-window MSE
    per_seed      = {}   # display -> {"rmse": [...], "var_ratio": [...], "n_seeds", "seeds"}
    ens_sq_err    = {}   # display -> (N, P, C) ensemble squared err (only S>1)
    ens_abs_err   = {}   # display -> (N, P, C) ensemble absolute err
    ens_var_ratio = {}   # display -> float

    for display, name, variant, paths in models:
        seeds_list, preds_stack, src_first = [], [], None
        for s, ppath, spath in paths:
            p = np.load(ppath)
            if p.shape != Yte.shape:
                raise SystemExit(f"{display}: shape mismatch "
                                 f"{p.shape} vs Yte {Yte.shape}")
            preds_stack.append(p.astype(np.float32))
            seeds_list.append(s)
            if src_first is None and spath is not None:
                with open(spath) as f:
                    src_first = json.load(f)
        stack = np.stack(preds_stack, axis=0)              # (S, N, P, C)
        S = stack.shape[0]

        # Per-seed scalars.
        rmse_per_seed = [float(np.sqrt(((stack[i] - Yte) ** 2).mean()))
                         for i in range(S)]
        var_per_seed  = [float(stack[i].std() / Yte.std())
                         for i in range(S)]

        # Seed-mean squared / absolute err: collapse the seed axis up front.
        err = stack - Yte[None, ...]                       # (S, N, P, C)
        sq_err[display]  = (err ** 2).mean(axis=0)
        abs_err[display] = np.abs(err).mean(axis=0)
        L_seed[display]  = (err ** 2).mean(axis=(2, 3))    # (S, N)

        # Canonical preds = ensemble (or the single tensor if S=1).
        ens = stack.mean(axis=0)
        preds_canon[display] = ens

        if S > 1:
            err_e = ens - Yte
            ens_sq_err[display]    = err_e ** 2
            ens_abs_err[display]   = np.abs(err_e)
            ens_var_ratio[display] = float(ens.std() / Yte.std())

        per_seed[display] = {"rmse": rmse_per_seed,
                             "var_ratio": var_per_seed,
                             "n_seeds": S,
                             "seeds": seeds_list}
        sources[display] = src_first if src_first is not None else {}

    print(f"  Y_test shape:  {Yte.shape}   "
          f"test window: {end_dates[0].date()} → {end_dates[-1].date()}")
    print(f"  models found:")
    for display in preds_canon:
        ps = per_seed[display]
        print(f"    {display:30s}  n_seeds={ps['n_seeds']}  "
              f"seeds={ps['seeds']}")

    return {
        "data":              data,
        "Y":                 Yte,
        "X":                 Xte,
        "persistence":       persistence,
        "preds":             preds_canon,
        "sq_err":            sq_err,
        "abs_err":           abs_err,
        "L_seed":            L_seed,
        "per_seed":          per_seed,
        "ens_sq_err":        ens_sq_err,
        "ens_abs_err":       ens_abs_err,
        "ens_var_ratio":     ens_var_ratio,
        "sources":           sources,
        "displays_primary":  list(preds_canon.keys()),
        "end_dates":         end_dates,
        "test_starts":       test_starts,
        "all_dates":         all_dates,
        "grid":              grid,
        "H":                 H,
        "W":                 W,
        "N_test":            N_test,
        "P":                 P,
    }


# ─── Compute / write everything ──────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_len", type=int, default=21, choices=(5, 21, 63))
    ap.add_argument("--csv_path", default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--data_end", default="2023-12-29")
    ap.add_argument("--n_boot",   type=int, default=2000)
    ap.add_argument("--block",    type=int, default=None,
                    help="Block-bootstrap block length. Defaults to pred_len.")
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    data_end = None if args.data_end.lower() == "none" else args.data_end
    block = args.block or args.pred_len

    state = load_all(args.pred_len, args.csv_path, data_end)
    out_dir = os.path.join(ROOT, "_test_results",
                           f"{LOOKBACK}_{args.pred_len}", "full")
    os.makedirs(out_dir, exist_ok=True)

    state["out_dir"] = out_dir
    state["pred_len"] = args.pred_len
    state["block"]    = block
    state["n_boot"]   = args.n_boot
    state["seed"]     = args.seed

    # --- Stage 1: headline + bootstrap CIs + DM matrix ------------------
    compute_headline(state)
    # --- Stage 2: slices ------------------------------------------------
    compute_slices(state)
    # --- Stage 3: figures ----------------------------------------------
    figures(state)

    print(f"\nall outputs under {os.path.relpath(out_dir, ROOT)}/")


# ─── Stage 1 ──────────────────────────────────────────────────────────────

def _slug(s: str) -> str:
    return s.lower().replace(" ", "_")


def _headline_for_subset(state, mask: np.ndarray, *, do_dm: bool = False):
    """Build sorted headline rows for the test windows where mask is True.

    Rows include, in this order after sorting by RMSE:
      - one row per primary model (per-seed-averaged metrics)
      - one row per model with S>1, marked as ensemble (preds=mean over seeds)
      - one persistence baseline row

    Returns rows, cis, pw_loss, L_pers, dm. pw_loss keys cover primary
    *and* ensemble displays. DM matrix covers primary models only."""
    Y_m  = state["Y"][mask]
    P_m  = state["persistence"][mask]
    err_pers = P_m - Y_m
    L_pers   = per_window_loss(err_pers)
    N = int(mask.sum())
    rng = np.random.default_rng(state["seed"])
    boot_idx    = block_bootstrap_idx(N, state["block"], state["n_boot"], rng)
    L_pers_boot = L_pers[boot_idx].mean(axis=1)
    MSE_pers    = float(L_pers.mean())

    rows, pw_loss, cis = [], {}, {}

    # Per-seed-averaged rows.
    for d in state["displays_primary"]:
        sq    = state["sq_err"][d][mask]              # (N', P, C)
        abs_  = state["abs_err"][d][mask]
        L     = sq.mean(axis=(1, 2))                  # (N',)
        pw_loss[d] = L
        mse   = float(L.mean())
        rmse  = float(np.sqrt(mse))
        mae   = float(abs_.mean())
        var_ratio = float(state["preds"][d][mask].std() / Y_m.std())

        # Per-seed RMSE on this mask → seed-std diagnostic.
        seed_std_rmse = None
        n_seeds       = state["per_seed"][d]["n_seeds"]
        if n_seeds > 1:
            ls = state["L_seed"][d][:, mask]           # (S, N')
            per_seed_rmse = np.sqrt(ls.mean(axis=1))   # (S,)
            seed_std_rmse = float(np.std(per_seed_rmse))

        L_boot = L[boot_idx].mean(axis=1)
        cis[d] = {"RMSE_ci":    ci(np.sqrt(L_boot)),
                  "R2_pers_ci": ci(1.0 - L_boot / L_pers_boot)}

        rows.append({
            "display":       d,
            "MSE":           mse,
            "RMSE":          rmse,
            "MAE":           mae,
            "R2_pers":       float(1.0 - mse / MSE_pers),
            "var_ratio":     var_ratio,
            "n_params":      state["sources"].get(d, {}).get("n_params"),
            "val_loss":      state["sources"].get(d, {}).get("val_loss"),
            "n_seeds":       n_seeds,
            "seed_std_RMSE": seed_std_rmse,
            "is_ensemble":   False,
            "MSE_pers":      MSE_pers,
        })

    # Ensemble rows (only where S > 1).
    for d in state["displays_primary"]:
        if state["per_seed"][d]["n_seeds"] <= 1:
            continue
        sq_e  = state["ens_sq_err"][d][mask]
        abs_e = state["ens_abs_err"][d][mask]
        L_e   = sq_e.mean(axis=(1, 2))
        ens_d = f"{d}_ens"
        pw_loss[ens_d] = L_e
        mse_e = float(L_e.mean())
        L_e_boot = L_e[boot_idx].mean(axis=1)
        cis[ens_d] = {"RMSE_ci":    ci(np.sqrt(L_e_boot)),
                      "R2_pers_ci": ci(1.0 - L_e_boot / L_pers_boot)}
        rows.append({
            "display":       ens_d,
            "MSE":           mse_e,
            "RMSE":          float(np.sqrt(mse_e)),
            "MAE":           float(abs_e.mean()),
            "R2_pers":       float(1.0 - mse_e / MSE_pers),
            "var_ratio":     state["ens_var_ratio"][d],
            "n_params":      state["sources"].get(d, {}).get("n_params"),
            "val_loss":      None,
            "n_seeds":       state["per_seed"][d]["n_seeds"],
            "seed_std_RMSE": None,
            "is_ensemble":   True,
            "MSE_pers":      MSE_pers,
        })

    # Persistence row.
    rows.append({
        "display":       "persistence",
        "MSE":           MSE_pers,
        "RMSE":          float(np.sqrt(MSE_pers)),
        "MAE":           float(np.abs(err_pers).mean()),
        "R2_pers":       0.0,
        "var_ratio":     float(P_m.std() / Y_m.std()),
        "n_params":      None,
        "val_loss":      None,
        "n_seeds":       None,
        "seed_std_RMSE": None,
        "is_ensemble":   False,
        "MSE_pers":      MSE_pers,
    })
    cis["persistence"] = {"RMSE_ci":    ci(np.sqrt(L_pers_boot)),
                          "R2_pers_ci": (0.0, 0.0)}

    rows.sort(key=lambda r: r["RMSE"])

    dm = None
    if do_dm:
        primary = [r["display"] for r in rows
                   if not r["is_ensemble"] and r["display"] != "persistence"]
        K = len(primary)
        dm = {"models":    primary,
              "mean_diff": np.zeros((K, K)),
              "t":         np.zeros((K, K)),
              "p":         np.ones((K, K))}
        for i, a in enumerate(primary):
            for j, b in enumerate(primary):
                if i == j:
                    continue
                res = dm_test(pw_loss[a], pw_loss[b], lag=state["pred_len"])
                dm["mean_diff"][i, j] = res["mean_diff"]
                dm["t"][i, j]         = res["t"]
                dm["p"][i, j]         = res["p"]

    return rows, cis, pw_loss, L_pers, dm


def _write_headline_csv(rows, cis, path):
    csv_rows = []
    for r in rows:
        c = cis[r["display"]]
        csv_rows.append({
            "display":       r["display"],
            "is_ensemble":   r.get("is_ensemble", False),
            "n_seeds":       r.get("n_seeds"),
            "MSE":           r["MSE"],
            "RMSE":          r["RMSE"],
            "RMSE_ci_lo":    c["RMSE_ci"][0],
            "RMSE_ci_hi":    c["RMSE_ci"][1],
            "seed_std_RMSE": r.get("seed_std_RMSE"),
            "MAE":           r["MAE"],
            "R2_pers":       r["R2_pers"],
            "R2_pers_ci_lo": c["R2_pers_ci"][0],
            "R2_pers_ci_hi": c["R2_pers_ci"][1],
            "var_ratio":     r["var_ratio"],
            "n_params":      r["n_params"],
            "val_loss":      r["val_loss"],
            "MSE_pers":      r.get("MSE_pers"),
        })
    pd.DataFrame(csv_rows).to_csv(path, index=False)


def compute_headline(state):
    out_dir = state["out_dir"]
    N       = state["N_test"]

    # Overall (use every window).
    rows, cis, pw_loss, L_pers, dm = _headline_for_subset(
        state, np.ones(N, dtype=bool), do_dm=True)

    _write_headline_csv(rows, cis, os.path.join(out_dir, "headline.csv"))
    with open(os.path.join(out_dir, "headline.json"), "w") as f:
        json.dump({
            "rows":   rows,
            "cis":    cis,
            "dm": {"models":    dm["models"],
                   "mean_diff": dm["mean_diff"].tolist(),
                   "t":         dm["t"].tolist(),
                   "p":         dm["p"].tolist()},
            "settings": {
                "n_boot":  state["n_boot"],
                "block":   state["block"],
                "lag_HAC": state["pred_len"],
                "seed":    state["seed"],
            },
        }, f, indent=2, default=str)

    np.savez(os.path.join(out_dir, "per_window_loss.npz"),
             persistence=L_pers, **pw_loss)

    state["pw_loss"]  = pw_loss
    state["L_pers"]   = L_pers
    state["headline"] = rows
    state["cis"]      = cis
    state["dm"]       = dm
    state["displays"] = dm["models"]   # excludes persistence

    print("\n=== HEADLINE (sorted by RMSE; persistence + ensembles inline) ===")
    print(f"{'model':32s}  {'MSE':>8s}  {'RMSE':>8s}  {'±σ_seed':>9s}  "
          f"{'MAE':>8s}  {'R2_pers':>9s}  {'var_r':>6s}  {'params':>10s}")
    for r in rows:
        np_s = (f"{r['n_params']:,}" if r["n_params"] is not None else "—")
        sd = (f"{r['seed_std_RMSE']:.4f}"
              if r.get("seed_std_RMSE") is not None else "—")
        print(f"{r['display']:32s}  {r['MSE']:8.5f}  {r['RMSE']:8.5f}  "
              f"{sd:>9s}  {r['MAE']:8.5f}  {r['R2_pers']:>+9.4f}  "
              f"{r['var_ratio']:6.3f}  {np_s:>10s}")

    # Per-regime headlines.
    end_dates    = state["end_dates"]
    regimes_arr  = assign_regime(end_dates)
    state["regimes"] = regimes_arr
    state["headline_by_regime"] = {}
    for reg_name, _, _ in REGIMES:
        mask = (regimes_arr == reg_name)
        if not mask.any():
            continue
        rg_rows, rg_cis, _, _, _ = _headline_for_subset(state, mask)
        state["headline_by_regime"][reg_name] = {
            "rows": rg_rows, "cis": rg_cis, "n": int(mask.sum())}
        _write_headline_csv(
            rg_rows, rg_cis,
            os.path.join(out_dir, f"headline_{_slug(reg_name)}.csv"))
        print(f"\n  regime '{reg_name}' (n={int(mask.sum())}):")
        for r in rg_rows:
            sd = (f"±{r['seed_std_RMSE']:.4f}"
                  if r.get("seed_std_RMSE") is not None else "")
            print(f"    {r['display']:30s}  RMSE={r['RMSE']:.4f}{sd:9s}  "
                  f"R2_pers={r['R2_pers']:+.4f}")


# ─── Stage 2: slices ──────────────────────────────────────────────────────

def compute_slices(state):
    """Slice metrics use seed-mean squared/absolute err tensors. All
    aggregations are over primary displays (no ensemble rows in slices)."""
    Y, P_ = state["Y"], state["persistence"]
    sq_err  = state["sq_err"]
    abs_err = state["abs_err"]
    H, W  = state["H"], state["W"]
    P     = state["P"]
    regimes = state["regimes"]
    out_dir = state["out_dir"]

    err_pers       = P_ - Y                                   # (N, P, C)
    err_pers_sq    = err_pers ** 2
    L_pers_h       = err_pers_sq.mean(axis=(0, 2))            # (P,)
    err_pers_cell  = err_pers_sq.mean(axis=(0, 1)).reshape(H, W)  # (H, W)
    mae_pers_h     = np.abs(err_pers).mean(axis=(0, 2))

    # By horizon.
    rows = []
    for d in state["displays_primary"]:
        mse_h  = sq_err[d].mean(axis=(0, 2))
        mae_h  = abs_err[d].mean(axis=(0, 2))
        r2_h   = 1.0 - mse_h / L_pers_h
        for h in range(P):
            rows.append({"display": d, "h": h + 1,
                         "MSE":  float(mse_h[h]),
                         "RMSE": float(np.sqrt(mse_h[h])),
                         "MAE":  float(mae_h[h]),
                         "R2_pers": float(r2_h[h])})
    for h in range(P):
        rows.append({"display": "persistence", "h": h + 1,
                     "MSE":  float(L_pers_h[h]),
                     "RMSE": float(np.sqrt(L_pers_h[h])),
                     "MAE":  float(mae_pers_h[h]),
                     "R2_pers": 0.0})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "by_horizon.csv"),
                              index=False)

    # By cell.
    by_cell, by_cell_r2 = {}, {}
    for d in state["displays_primary"]:
        mse_cell  = sq_err[d].mean(axis=(0, 1)).reshape(H, W)
        by_cell[d]    = np.sqrt(mse_cell)
        by_cell_r2[d] = 1.0 - mse_cell / err_pers_cell
    np.savez(os.path.join(out_dir, "by_cell.npz"),
             persistence_rmse=np.sqrt(err_pers_cell),
             **{f"rmse__{k}":  v for k, v in by_cell.items()},
             **{f"r2pers__{k}": v for k, v in by_cell_r2.items()})

    # By zone.
    zone_rows = []
    for d in state["displays_primary"]:
        cell_mse = sq_err[d].mean(axis=(0, 1)).reshape(H, W)
        for t_label, t_slc in TAU_ZONES:
            for m_label, m_slc in MONEY_ZONES:
                mse_z = float(cell_mse[t_slc, m_slc].mean())
                mse_pz = float(err_pers_cell[t_slc, m_slc].mean())
                zone_rows.append({
                    "display": d,
                    "tau_zone": t_label,
                    "money_zone": m_label,
                    "MSE": mse_z,
                    "RMSE": float(np.sqrt(mse_z)),
                    "R2_pers": float(1.0 - mse_z / mse_pz),
                })
    pd.DataFrame(zone_rows).to_csv(os.path.join(out_dir, "by_zone.csv"),
                                   index=False)

    # By regime.
    reg_rows = []
    for d in state["displays_primary"]:
        L = sq_err[d].mean(axis=(1, 2))      # (N,)
        for reg_name, _, _ in REGIMES:
            m = (regimes == reg_name)
            if not m.any():
                continue
            mse_r  = float(L[m].mean())
            mse_pr = float(state["L_pers"][m].mean())
            reg_rows.append({
                "display": d,
                "regime":  reg_name,
                "n":       int(m.sum()),
                "MSE":     mse_r,
                "RMSE":    float(np.sqrt(mse_r)),
                "MAE":     float(abs_err[d][m].mean()),
                "R2_pers": float(1.0 - mse_r / mse_pr),
            })
    pd.DataFrame(reg_rows).to_csv(os.path.join(out_dir, "by_regime.csv"),
                                  index=False)

    # By regime × horizon.
    reg_h_rows = []
    for reg_name, _, _ in REGIMES:
        m = (regimes == reg_name)
        if not m.any():
            continue
        L_pers_h_reg = err_pers_sq[m].mean(axis=(0, 2))    # (P,)
        for d in state["displays_primary"]:
            mse_h = sq_err[d][m].mean(axis=(0, 2))
            r2_h  = 1.0 - mse_h / L_pers_h_reg
            for h in range(P):
                reg_h_rows.append({
                    "display": d, "regime": reg_name, "h": h + 1,
                    "MSE":     float(mse_h[h]),
                    "RMSE":    float(np.sqrt(mse_h[h])),
                    "R2_pers": float(r2_h[h]),
                })
    pd.DataFrame(reg_h_rows).to_csv(
        os.path.join(out_dir, "by_regime_horizon.csv"), index=False)

    state["by_cell"]    = by_cell
    state["by_cell_r2"] = by_cell_r2
    state["err_pers_cell_rmse"] = np.sqrt(err_pers_cell)


# ─── Stage 3: figures ────────────────────────────────────────────────────

def figures(state):
    out_dir = state["out_dir"]
    figs_dir = os.path.join(out_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    figs = []
    # Overall headline table.
    figs.append((
        "01_headline_table",
        _render_headline_table(
            state["headline"], state["cis"],
            f"Headline metrics — std log-IV space  "
            f"(overall, n_test={state['N_test']})",
            state["pred_len"]),
    ))
    # Per-regime headline tables (same layout, just filtered).
    for reg_name, _, _ in REGIMES:
        info = state["headline_by_regime"].get(reg_name)
        if info is None:
            continue
        figs.append((
            f"01_headline_{_slug(reg_name)}",
            _render_headline_table(
                info["rows"], info["cis"],
                f"Headline metrics — {reg_name}  "
                f"(n_test={info['n']})",
                state["pred_len"]),
        ))
    figs.append(("02_horizon_curves",      fig_horizon_curves(state)))
    figs.append(("03_per_cell_rmse",       fig_per_cell_rmse(state)))
    figs.append(("04_r2_pers_heatmap",     fig_r2_pers_heatmap(state)))
    figs.append(("05_winner_per_cell",     fig_winner_per_cell(state)))
    figs.append(("06_zone_r2_table",       fig_zone_r2(state)))
    figs.append(("07_regime_bars",         fig_regime_bars(state)))
    figs.append(("08_regime_horizon",      fig_regime_horizon(state)))
    figs.append(("09_dm_matrix",           fig_dm_matrix(state)))
    figs.append(("10_time_series",         fig_time_series(state)))

    # Example surfaces.
    for tag, anchor_date in EXAMPLE_DATES:
        fig = fig_example_surface(state, anchor_date, tag)
        if fig is not None:
            figs.append((f"11_example_{tag}", fig))

    # Save each PNG and bind into PDF.
    pdf_path = os.path.join(out_dir, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        for name, fig in figs:
            png = os.path.join(figs_dir, f"fig_{name}.png")
            fig.savefig(png, dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"\nsaved {len(figs)} figures → {os.path.relpath(figs_dir, ROOT)}/")
    print(f"saved report → {os.path.relpath(pdf_path, ROOT)}")


def _color_map(displays):
    cmap = plt.get_cmap("tab10")
    return {d: cmap(i % 10) for i, d in enumerate(displays)}


def _render_headline_table(rows, cis, title: str, pred_len: int):
    """Render a headline-style table as a matplotlib Figure.

    - RMSE cell shows point estimate + seed-std (when S>1) + bootstrap CI.
    - R² vs persistence cell shows point estimate + bootstrap CI.
    - Persistence row is yellow-tinted; ensemble rows are blue-tinted.
    """
    cols = ["display", "n_seeds", "RMSE\n(± σ_seed)\n[95% CI]",
            "MSE", "MAE", "R² vs pers\n[95% CI]",
            "var ratio", "params", "val_loss"]
    col_widths = [0.14, 0.05, 0.20, 0.07, 0.07, 0.20, 0.08, 0.09, 0.08]
    s = sum(col_widths)
    col_widths = [w / s for w in col_widths]

    body = []
    pers_idx = None
    ens_idx  = []
    for i, r in enumerate(rows):
        d = r["display"]
        c = cis[d]
        is_pers = (d == "persistence")
        is_ens  = bool(r.get("is_ensemble"))
        if is_pers:
            pers_idx = i
        if is_ens:
            ens_idx.append(i)
        ns = r.get("n_seeds")
        n_seeds_cell = ("—" if ns in (None, 0) else
                        (f"{ns} (ens)" if is_ens else str(ns)))
        # RMSE cell with optional seed-std.
        rmse_line = f"{r['RMSE']:.4f}"
        if r.get("seed_std_RMSE") is not None:
            rmse_line += f"  ± {r['seed_std_RMSE']:.4f}"
        rmse_cell = (f"{rmse_line}\n"
                     f"[{c['RMSE_ci'][0]:.4f}, {c['RMSE_ci'][1]:.4f}]")
        r2_cell = ("— (reference)" if is_pers else
                   f"{r['R2_pers']:+.4f}\n"
                   f"[{c['R2_pers_ci'][0]:+.4f}, "
                   f"{c['R2_pers_ci'][1]:+.4f}]")
        body.append([
            d,
            n_seeds_cell,
            rmse_cell,
            f"{r['MSE']:.4f}",
            f"{r['MAE']:.4f}",
            r2_cell,
            f"{r['var_ratio']:.3f}",
            (f"{r['n_params']:,}" if r["n_params"] is not None else "—"),
            (f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "—"),
        ])

    nrow = len(body)
    fig, ax = plt.subplots(figsize=(17, 1.0 + 0.7 * (nrow + 1)))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=cols, loc="center",
                     cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.3)
    # Header bold + light grey.
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#e8eef5")
        cell.set_text_props(weight="bold")
    if pers_idx is not None:
        for j in range(len(cols)):
            table[pers_idx + 1, j].set_facecolor("#fff2cc")
    for ei in ens_idx:
        for j in range(len(cols)):
            table[ei + 1, j].set_facecolor("#e3eef8")
    ax.set_title(f"{title}   (pred_len={pred_len})", fontsize=12, pad=14)
    return fig


def fig_horizon_curves(state):
    df = pd.read_csv(os.path.join(state["out_dir"], "by_horizon.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = _color_map(state["displays"])
    for d in state["displays"]:
        sub = df[df["display"] == d]
        ax.plot(sub["h"], sub["RMSE"], color=colors[d], linewidth=1.6, label=d)
    pers = df[df["display"] == "persistence"]
    ax.plot(pers["h"], pers["RMSE"], color="black", linestyle="--",
            linewidth=1.4, label="persistence", alpha=0.7)
    ax.set_xlabel("forecast horizon h (days)")
    ax.set_ylabel("RMSE (std log-IV)")
    ax.set_title("RMSE vs forecast horizon")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    return fig


def fig_per_cell_rmse(state):
    cells = state["by_cell"]
    displays = state["displays"]
    n = len(displays)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.8 * nrow),
                             squeeze=False)
    vmax = max(np.max(c) for c in cells.values())
    vmin = min(np.min(c) for c in cells.values())
    for i, d in enumerate(displays):
        ax = axes[i // ncol][i % ncol]
        im = ax.imshow(cells[d], aspect="auto", origin="lower",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        _label_grid_axes(ax, state["grid"])
        ax.set_title(d, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.045)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Per-cell RMSE (std log-IV) — each panel is one model",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    return fig


def fig_r2_pers_heatmap(state):
    r2 = state["by_cell_r2"]
    displays = state["displays"]
    n = len(displays)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.8 * nrow),
                             squeeze=False)
    vmax = max(np.max(v) for v in r2.values())
    vmin = min(np.min(v) for v in r2.values())
    bound = max(abs(vmin), abs(vmax))
    for i, d in enumerate(displays):
        ax = axes[i // ncol][i % ncol]
        im = ax.imshow(r2[d], aspect="auto", origin="lower",
                       cmap="RdBu", vmin=-bound, vmax=bound)
        _label_grid_axes(ax, state["grid"])
        ax.set_title(d, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.045)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("R² vs persistence per cell (red = worse than persistence)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    return fig


def fig_winner_per_cell(state):
    cells = state["by_cell"]
    displays = state["displays"]
    H, W = state["H"], state["W"]
    stack = np.stack([cells[d] for d in displays], axis=0)  # (K, H, W)
    winner = stack.argmin(axis=0)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(displays))]
    from matplotlib.colors import ListedColormap
    lc = ListedColormap(colors[:len(displays)])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(winner, aspect="auto", origin="lower",
                   cmap=lc, vmin=-0.5, vmax=len(displays) - 0.5)
    _label_grid_axes(ax, state["grid"])
    ax.set_title("Winner per cell (lowest RMSE)")
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(displays)))
    cbar.ax.set_yticklabels(displays)
    return fig


def fig_zone_r2(state):
    df = pd.read_csv(os.path.join(state["out_dir"], "by_zone.csv"))
    displays = state["displays"]
    # Build a (n_models, n_tau_zones, n_money_zones) array for R²-pers.
    tau_labels = [t for t, _ in TAU_ZONES]
    m_labels   = [m for m, _ in MONEY_ZONES]
    arr = np.zeros((len(displays), len(tau_labels), len(m_labels)))
    for i, d in enumerate(displays):
        for ti, t in enumerate(tau_labels):
            for mi, m in enumerate(m_labels):
                row = df[(df["display"] == d) &
                         (df["tau_zone"] == t) &
                         (df["money_zone"] == m)]
                arr[i, ti, mi] = row["R2_pers"].iloc[0]
    bound = float(np.abs(arr).max())
    fig, axes = plt.subplots(1, len(displays),
                             figsize=(3 * len(displays), 3.5),
                             squeeze=False, sharey=True)
    for i, d in enumerate(displays):
        ax = axes[0][i]
        im = ax.imshow(arr[i], cmap="RdBu", vmin=-bound, vmax=bound,
                       origin="lower")
        ax.set_xticks(range(len(m_labels)))
        ax.set_xticklabels(m_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(tau_labels)))
        ax.set_yticklabels(tau_labels, fontsize=8)
        ax.set_title(d, fontsize=9)
        for ti in range(len(tau_labels)):
            for mi in range(len(m_labels)):
                ax.text(mi, ti, f"{arr[i, ti, mi]:+.2f}",
                        ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("R² vs persistence by surface zone",
                 fontsize=11, y=1.02)
    return fig


def fig_regime_bars(state):
    df = pd.read_csv(os.path.join(state["out_dir"], "by_regime.csv"))
    displays = state["displays"]
    regs = [r for r, _, _ in REGIMES]
    x = np.arange(len(regs))
    width = 0.8 / len(displays)
    colors = _color_map(displays)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, d in enumerate(displays):
        sub = df[df["display"] == d].set_index("regime").reindex(regs)
        ax.bar(x + i * width, sub["R2_pers"].values,
               width=width, color=colors[d], label=d)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(displays) - 1) / 2)
    ax.set_xticklabels(regs)
    ax.set_ylabel("R² vs persistence")
    ax.set_title("R² vs persistence by calendar regime")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    # Window counts on top.
    counts = df[df["display"] == displays[0]].set_index("regime").reindex(regs)
    for xi, n in zip(x, counts["n"].values):
        ax.text(xi + width * (len(displays) - 1) / 2,
                ax.get_ylim()[1] * 0.95, f"n={int(n)}",
                ha="center", fontsize=8, alpha=0.7)
    return fig


def fig_regime_horizon(state):
    df = pd.read_csv(os.path.join(state["out_dir"], "by_regime_horizon.csv"))
    displays = state["displays"]
    regs = [r for r, _, _ in REGIMES]
    colors = _color_map(displays)
    fig, axes = plt.subplots(1, len(regs), figsize=(4 * len(regs), 4),
                             squeeze=False, sharey=True)
    for ri, reg in enumerate(regs):
        ax = axes[0][ri]
        sub_reg = df[df["regime"] == reg]
        if sub_reg.empty:
            ax.axis("off")
            continue
        for d in displays:
            s = sub_reg[sub_reg["display"] == d]
            ax.plot(s["h"], s["R2_pers"], color=colors[d], label=d, linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.set_title(reg, fontsize=10)
        ax.set_xlabel("h")
        if ri == 0:
            ax.set_ylabel("R² vs persistence")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(loc="lower left", fontsize=7)
    fig.suptitle("R² vs persistence by horizon, faceted by regime",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def fig_dm_matrix(state):
    dm = state["dm"]
    displays = dm["models"]
    K = len(displays)
    p = np.array(dm["p"])
    mean_diff = np.array(dm["mean_diff"])
    # Color by -log10(p), capped at 4.
    neglog = -np.log10(np.maximum(p, 1e-8))
    neglog = np.where(np.eye(K, dtype=bool), np.nan, neglog)

    fig, ax = plt.subplots(figsize=(0.9 * K + 4, 0.9 * K + 2))
    im = ax.imshow(neglog, cmap="viridis", vmin=0, vmax=4)
    ax.set_xticks(range(K))
    ax.set_xticklabels(displays, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels(displays, fontsize=8)
    ax.set_title("Diebold-Mariano pairwise — color: −log₁₀(p), sign: row vs col")
    for i in range(K):
        for j in range(K):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="white")
                continue
            stars = ""
            if p[i, j] < 0.001: stars = "***"
            elif p[i, j] < 0.01: stars = "**"
            elif p[i, j] < 0.05: stars = "*"
            # "+" if row wins (lower loss), "-" if row loses.
            sign = "+" if mean_diff[i, j] < 0 else "-"
            ax.text(j, i, f"{sign}{stars}\np={p[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if neglog[i, j] > 2 else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("−log₁₀(p)")
    return fig


def fig_time_series(state):
    out_dir = state["out_dir"]
    npz = np.load(os.path.join(out_dir, "per_window_loss.npz"))
    displays = state["displays"]
    end_dates = state["end_dates"]
    colors = _color_map(displays)
    fig, ax = plt.subplots(figsize=(13, 5))
    win = 21
    for d in displays:
        L = npz[d]
        s = pd.Series(L).rolling(win, min_periods=max(1, win // 2),
                                 center=True).mean()
        ax.plot(end_dates, s.values, color=colors[d], linewidth=1.4, label=d)
    Lp = npz["persistence"]
    s = pd.Series(Lp).rolling(win, min_periods=max(1, win // 2),
                              center=True).mean()
    ax.plot(end_dates, s.values, color="black", linestyle="--",
            linewidth=1.2, label="persistence", alpha=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("target end date")
    ax.set_ylabel("per-window MSE (std log-IV, 21d-smoothed)")
    ax.set_title("Per-window MSE over time, regime-shaded")
    ax.grid(which="both", alpha=0.2)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    # Regime shading.
    reg_colors = ["#fde0dd", "#e0f3db", "#fde9c0", "#dcdcfd"]
    for (rname, lo, hi), c in zip(REGIMES, reg_colors):
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=c, alpha=0.6,
                   zorder=0)
        ax.text(pd.Timestamp(lo) + (pd.Timestamp(hi) - pd.Timestamp(lo)) / 2,
                ax.get_ylim()[0] * 1.05, rname, ha="center", fontsize=8,
                alpha=0.7)
    return fig


def fig_example_surface(state, anchor_date_s, tag):
    """For a target date close to `anchor_date`, show truth + each model's
    prediction at h=pred_len and the (pred − true) error surface."""
    anchor = pd.Timestamp(anchor_date_s)
    end_dates = state["end_dates"]
    # Index of test window whose last target date is closest to anchor.
    if end_dates.min() > anchor or end_dates.max() < anchor:
        print(f"  example {tag}: anchor {anchor.date()} outside test range; skip")
        return None
    idx = int(np.argmin(np.abs(end_dates - anchor)))
    actual_date = end_dates[idx]

    Y    = state["Y"]
    preds = state["preds"]
    H, W = state["H"], state["W"]
    displays = state["displays"]

    last_h = state["P"] - 1  # show the last forecast horizon
    truth = Y[idx, last_h].reshape(H, W)
    err_stack = {d: (preds[d][idx, last_h] - Y[idx, last_h]).reshape(H, W)
                 for d in displays}
    pred_stack = {d: preds[d][idx, last_h].reshape(H, W) for d in displays}

    # Color scale: shared truth/pred range; symmetric for error.
    vmin = min(truth.min(), *(p.min() for p in pred_stack.values()))
    vmax = max(truth.max(), *(p.max() for p in pred_stack.values()))
    ebound = max(abs(e.min()) for e in err_stack.values())
    ebound = max(ebound, max(abs(e.max()) for e in err_stack.values()))

    ncol = len(displays) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 5.6))
    # Row 0: truth (col 0) + each model's pred.
    ax0 = axes[0, 0]
    im_t = ax0.imshow(truth, cmap="viridis", vmin=vmin, vmax=vmax,
                      aspect="auto", origin="lower")
    _label_grid_axes(ax0, state["grid"], small=True)
    ax0.set_title("truth", fontsize=9)
    for i, d in enumerate(displays):
        ax = axes[0, i + 1]
        ax.imshow(pred_stack[d], cmap="viridis", vmin=vmin, vmax=vmax,
                  aspect="auto", origin="lower")
        _label_grid_axes(ax, state["grid"], small=True)
        ax.set_title(d, fontsize=9)
    # Row 1: error surfaces. Col 0 is blank.
    axes[1, 0].axis("off")
    for i, d in enumerate(displays):
        ax = axes[1, i + 1]
        im_e = ax.imshow(err_stack[d], cmap="RdBu", vmin=-ebound, vmax=ebound,
                         aspect="auto", origin="lower")
        _label_grid_axes(ax, state["grid"], small=True)
        ax.set_title(f"err  rmse={np.sqrt((err_stack[d]**2).mean()):.3f}",
                     fontsize=8)
    fig.colorbar(im_t, ax=axes[0, :].tolist(), fraction=0.012, pad=0.01,
                 label="std log-IV")
    fig.colorbar(im_e, ax=axes[1, :].tolist(), fraction=0.012, pad=0.01,
                 label="pred − true")
    fig.suptitle(f"Example forecast — {tag}: target {actual_date.date()} "
                 f"(h={state['P']})", fontsize=11)
    return fig


# ─── Helpers ──────────────────────────────────────────────────────────────

def _label_grid_axes(ax, grid, small=False):
    fs = 7 if small else 8
    ax.set_xticks(range(grid.n_money))
    ax.set_xticklabels([f"{m:+.2f}" for m in grid.money_vals],
                       rotation=45, ha="right", fontsize=fs)
    ax.set_yticks(range(grid.n_tau))
    ax.set_yticklabels([f"{t:.2f}" for t in grid.tau_vals], fontsize=fs)
    ax.set_xlabel("log-moneyness", fontsize=fs + 1)
    ax.set_ylabel("τ (yrs)", fontsize=fs + 1)


if __name__ == "__main__":
    main()
