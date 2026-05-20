#!/usr/bin/env python3
"""
error_tables.py — seed-averaged error tables for finalised models.

Once a model is finalised for a pred_len, <ModelDir>/eval/63_<P>/ holds
its 6 seeded runs (one preds.npy per seed_<S>/ slot). This script reads
every finalised model's seeded preds — plus the deterministic VAR baseline
and a live-built persistence baseline — and emits 5 error tables:

  - the full test period
  - one each for calendar years 2020, 2021, 2022, 2023

A test window is assigned to a year by its last-horizon target date (the
same convention as eval_full.py's REGIMES).

Each table has columns:
  model | MSE | MAE | R2 vs persistence | R2 vs VAR | param count

For seeded (deep) models every error metric is reported as
seed-mean ± seed-std (sample std, ddof=1). VAR and Persistence are
deterministic, so they carry a single value with no ±.

  R2 vs persistence = 1 - MSE_model / MSE_persistence
  R2 vs VAR         = 1 - MSE_model / MSE_VAR

computed per-seed first, then averaged across seeds. Both baselines are
recomputed on each table's window subset. All metrics live in
standardized log-IV space — the space the models were trained against.

Param count: deep models read `n_params` from hyperparams.json;
VAR(p) uses p·C² + C (coefficient matrices + intercept); Persistence = 0.

It also emits a second stack of 5 horizon-breakdown tables (one per
period) showing how MSE, R2 vs persistence and R2 vs VAR evolve as the
forecast reaches further out — rows are each model repeated at horizons
h+1, h+5, h+10, h+21. Each metric is evaluated on that horizon's slice
alone (the error at exactly day k), with the baselines recomputed at the
same horizon.

Usage
-----
    python error_tables.py --pred_len 21
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from train import LOOKBACK, MODEL_DIR, ROOT, load_dataset


# (model_key, display name, variant subdir or None)
SEEDED_MODELS = [
    ("dlinear",        "DLinear",      None),
    ("tucker_dlinear", "Tucker",       None),
    ("gwn",            "GWN",          None),
    ("itransformer",   "iTransformer", None),
    ("patchtst",       "PatchTST",     None),
    ("hot",            "HOT (k-prod)", "kronecker_product"),
    ("hot",            "HOT (k-sum)",  "kronecker_sum"),
]

YEARS = (2020, 2021, 2022, 2023)

# Forecast horizons (days ahead) shown in the per-horizon breakdown.
HORIZONS = (1, 5, 10, 21)


# ─── Discovery ────────────────────────────────────────────────────────────

def _seed_preds(base: str):
    """Scan a dir of seed_<S>/ slots. Return (seeds, preds (S,N,P,C),
    n_params) or None when no seeded preds.npy is present."""
    if not os.path.isdir(base):
        return None
    found = []
    for entry in sorted(os.listdir(base)):
        if not entry.startswith("seed_"):
            continue
        d  = os.path.join(base, entry)
        pp = os.path.join(d, "preds.npy")
        if not os.path.isfile(pp):
            continue
        try:
            s = int(entry[len("seed_"):])
        except ValueError:
            continue
        found.append((s, d, pp))
    if not found:
        return None
    found.sort(key=lambda t: t[0])
    preds = np.stack([np.load(pp).astype(np.float32) for _, _, pp in found],
                     axis=0)
    n_params = None
    hp = os.path.join(found[0][1], "hyperparams.json")
    if os.path.isfile(hp):
        with open(hp) as f:
            n_params = json.load(f).get("n_params")
    return [s for s, _, _ in found], preds, n_params


def collect_seeded(pred_len: int, y_shape: tuple) -> list[dict]:
    out = []
    for key, display, variant in SEEDED_MODELS:
        base = os.path.join(ROOT, MODEL_DIR[key], "eval",
                            f"{LOOKBACK}_{pred_len}")
        if variant:
            base = os.path.join(base, variant)
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


def load_var(pred_len: int, y_shape: tuple, n_channels: int):
    base  = os.path.join(ROOT, MODEL_DIR["var"], "eval",
                         f"{LOOKBACK}_{pred_len}")
    ppath = os.path.join(base, "preds.npy")
    if not os.path.isfile(ppath):
        return None
    preds = np.load(ppath).astype(np.float32)
    if preds.shape != y_shape:
        raise SystemExit(f"VAR: preds shape {preds.shape} != Y_test {y_shape}")
    lag   = 1
    spath = os.path.join(base, "source.json")
    if os.path.isfile(spath):
        with open(spath) as f:
            lag = int(json.load(f).get("lag", 1))
    n_params = lag * n_channels * n_channels + n_channels
    print(f"  {'VAR':14s}  lag={lag}  n_params={n_params}")
    return {"preds": preds, "n_params": n_params, "lag": lag}


# ─── Window → calendar year ───────────────────────────────────────────────

def test_end_dates(pred_len: int, csv_path: str, data_end, val_end: int,
                   N: int) -> np.ndarray:
    """Last-horizon target date for every test window, in window order."""
    L, P  = LOOKBACK, pred_len
    starts = np.arange(N - L - P + 1)
    test_starts = starts[(starts + L + P) > val_end]
    df = pd.read_csv(csv_path)
    if data_end is not None:
        df = df[df["date"] <= data_end].reset_index(drop=True)
    dates = pd.to_datetime(df["date"].to_numpy())
    return dates[test_starts + L + P - 1]


# ─── Metrics ──────────────────────────────────────────────────────────────

def _mse_mae(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    e = pred - y
    return float((e * e).mean()), float(np.abs(e).mean())


def _mse_h(pred: np.ndarray, y: np.ndarray, hi: int) -> float:
    """MSE on a single horizon slice (axis-1 index hi) of (N, P, C) tensors."""
    e = pred[:, hi, :] - y[:, hi, :]
    return float((e * e).mean())


def _stat(values: list[float]) -> tuple[float, float]:
    """seed-mean and sample std (ddof=1; 0 when fewer than 2 values)."""
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    std  = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return mean, std


def table_rows(seeded: list[dict], var: dict | None, pers: np.ndarray,
               Y: np.ndarray, mask: np.ndarray) -> dict[str, dict]:
    """Return display -> stats dict for the windows selected by `mask`."""
    Ym  = Y[mask]
    mse_pers, mae_pers = _mse_mae(pers[mask], Ym)
    mse_var = mae_var = None
    if var is not None:
        mse_var, mae_var = _mse_mae(var["preds"][mask], Ym)

    rows: dict[str, dict] = {}

    for e in seeded:
        mses, maes, r2p, r2v = [], [], [], []
        for s in range(e["preds"].shape[0]):
            mse, mae = _mse_mae(e["preds"][s][mask], Ym)
            mses.append(mse)
            maes.append(mae)
            r2p.append(1.0 - mse / mse_pers)
            if mse_var is not None:
                r2v.append(1.0 - mse / mse_var)
        rows[e["display"]] = {
            "deterministic": False,
            "n_seeds":  e["preds"].shape[0],
            "n_params": e["n_params"],
            "mse":      _stat(mses),
            "mae":      _stat(maes),
            "r2_pers":  _stat(r2p),
            "r2_var":   _stat(r2v) if r2v else None,
        }

    if var is not None:
        rows["VAR"] = {
            "deterministic": True,
            "n_seeds":  None,
            "n_params": var["n_params"],
            "mse":      (mse_var, 0.0),
            "mae":      (mae_var, 0.0),
            "r2_pers":  (1.0 - mse_var / mse_pers, 0.0),
            "r2_var":   (0.0, 0.0),
        }

    rows["Persistence"] = {
        "deterministic": True,
        "n_seeds":  None,
        "n_params": 0,
        "mse":      (mse_pers, 0.0),
        "mae":      (mae_pers, 0.0),
        "r2_pers":  (0.0, 0.0),
        "r2_var":   ((1.0 - mse_pers / mse_var, 0.0)
                     if mse_var is not None else None),
    }
    return rows


def horizon_rows(seeded: list[dict], var: dict | None, pers: np.ndarray,
                 Y: np.ndarray, mask: np.ndarray,
                 horizons: list[int]) -> dict[str, dict]:
    """Return display -> {horizon -> stats} for the windows in `mask`.
    Each metric is evaluated on that horizon's slice alone."""
    Ym     = Y[mask]
    pers_m = pers[mask]
    var_m  = var["preds"][mask] if var is not None else None

    base = {}                       # horizon -> (MSE_pers, MSE_var | None)
    for k in horizons:
        hi = k - 1
        mp = _mse_h(pers_m, Ym, hi)
        mv = _mse_h(var_m, Ym, hi) if var_m is not None else None
        base[k] = (mp, mv)

    rows: dict[str, dict] = {}

    for e in seeded:
        per_k = {}
        for k in horizons:
            hi = k - 1
            mp, mv = base[k]
            mses, r2p, r2v = [], [], []
            for s in range(e["preds"].shape[0]):
                mse = _mse_h(e["preds"][s][mask], Ym, hi)
                mses.append(mse)
                r2p.append(1.0 - mse / mp)
                if mv is not None:
                    r2v.append(1.0 - mse / mv)
            per_k[k] = {
                "deterministic": False,
                "n_seeds": e["preds"].shape[0],
                "mse":     _stat(mses),
                "r2_pers": _stat(r2p),
                "r2_var":  _stat(r2v) if r2v else None,
            }
        rows[e["display"]] = per_k

    if var is not None:
        per_k = {}
        for k in horizons:
            mp, mv = base[k]
            per_k[k] = {
                "deterministic": True, "n_seeds": None,
                "mse":     (mv, 0.0),
                "r2_pers": (1.0 - mv / mp, 0.0),
                "r2_var":  (0.0, 0.0),
            }
        rows["VAR"] = per_k

    per_k = {}
    for k in horizons:
        mp, mv = base[k]
        per_k[k] = {
            "deterministic": True, "n_seeds": None,
            "mse":     (mp, 0.0),
            "r2_pers": (0.0, 0.0),
            "r2_var":  ((1.0 - mp / mv, 0.0) if mv is not None else None),
        }
    rows["Persistence"] = per_k
    return rows


# ─── Rendering ────────────────────────────────────────────────────────────

def _cell(stat, det: bool, signed: bool) -> str:
    if stat is None:
        return "—"
    mean, std = stat
    m = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    return m if det else f"{m} ± {std:.4f}"


def _params_cell(n) -> str:
    return "—" if n is None else f"{n:,}"


def render_table(title: str, order: list[str], rows: dict[str, dict]) -> str:
    headers = ["Model", "MSE", "MAE", "R2 vs Persist", "R2 vs VAR", "Params"]
    body = []
    for d in order:
        r = rows[d]
        det = r["deterministic"]
        body.append([
            d,
            _cell(r["mse"],     det, signed=False),
            _cell(r["mae"],     det, signed=False),
            _cell(r["r2_pers"], det, signed=True),
            _cell(r["r2_var"],  det, signed=True),
            _params_cell(r["n_params"]),
        ])
    widths = [max(len(headers[c]), *(len(row[c]) for row in body))
              for c in range(len(headers))]

    def fmt(cells):
        out = [cells[0].ljust(widths[0])]
        out += [cells[c].rjust(widths[c]) for c in range(1, len(cells))]
        return "  ".join(out)

    sep = "  ".join("-" * w for w in widths)
    lines = [title, fmt(headers), sep]
    lines += [fmt(row) for row in body]
    return "\n".join(lines)


def write_csv(path: str, order: list[str], rows: dict[str, dict]):
    recs = []
    for d in order:
        r = rows[d]
        rec = {"model": d, "n_seeds": r["n_seeds"], "n_params": r["n_params"]}
        for key in ("mse", "mae", "r2_pers", "r2_var"):
            stat = r[key]
            if stat is None:
                rec[f"{key}_mean"] = np.nan
                rec[f"{key}_std"]  = np.nan
            else:
                rec[f"{key}_mean"] = stat[0]
                rec[f"{key}_std"]  = np.nan if r["deterministic"] else stat[1]
        recs.append(rec)
    cols = ["model", "n_seeds", "mse_mean", "mse_std", "mae_mean", "mae_std",
            "r2_pers_mean", "r2_pers_std", "r2_var_mean", "r2_var_std",
            "n_params"]
    pd.DataFrame(recs)[cols].to_csv(path, index=False)


def render_horizon_table(title: str, order: list[str], horizons: list[int],
                         rows: dict[str, dict]) -> str:
    headers = ["Model", "Horizon", "MSE", "R2 vs Persist", "R2 vs VAR"]
    body = []
    for d in order:
        for k in horizons:
            r = rows[d][k]
            det = r["deterministic"]
            body.append([
                d,
                f"h+{k}",
                _cell(r["mse"],     det, signed=False),
                _cell(r["r2_pers"], det, signed=True),
                _cell(r["r2_var"],  det, signed=True),
            ])
    widths = [max(len(headers[c]), *(len(row[c]) for row in body))
              for c in range(len(headers))]

    def fmt(cells):
        out = [cells[0].ljust(widths[0]), cells[1].ljust(widths[1])]
        out += [cells[c].rjust(widths[c]) for c in range(2, len(cells))]
        return "  ".join(out)

    sep = "  ".join("-" * w for w in widths)
    lines = [title, fmt(headers), sep]
    lines += [fmt(row) for row in body]
    return "\n".join(lines)


def write_horizon_csv(path: str, order: list[str], horizons: list[int],
                      rows: dict[str, dict]):
    recs = []
    for d in order:
        for k in horizons:
            r = rows[d][k]
            rec = {"model": d, "horizon": k, "n_seeds": r["n_seeds"]}
            for key in ("mse", "r2_pers", "r2_var"):
                stat = r[key]
                if stat is None:
                    rec[f"{key}_mean"] = np.nan
                    rec[f"{key}_std"]  = np.nan
                else:
                    rec[f"{key}_mean"] = stat[0]
                    rec[f"{key}_std"]  = np.nan if r["deterministic"] else stat[1]
            recs.append(rec)
    cols = ["model", "horizon", "n_seeds", "mse_mean", "mse_std",
            "r2_pers_mean", "r2_pers_std", "r2_var_mean", "r2_var_std"]
    pd.DataFrame(recs)[cols].to_csv(path, index=False)


# ─── Entry ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_len", type=int, default=21, choices=(5, 21, 63))
    ap.add_argument("--csv_path", default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--data_end", default="2023-12-29")
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

    print(f"\ndiscovering finalised models (pred_len={args.pred_len}) ...")
    seeded = collect_seeded(args.pred_len, Yte.shape)
    var    = load_var(args.pred_len, Yte.shape, C)
    if not seeded:
        raise SystemExit("no seeded models found — nothing to tabulate.")

    # Fixed row order across all 5 tables: sort by full-period mean MSE.
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
        tables.append((str(y), m, table_rows(seeded, var, persistence,
                                              Yte, m)))

    out_dir = os.path.join(ROOT, "_test_results",
                           f"{LOOKBACK}_{args.pred_len}", "error_tables")
    os.makedirs(out_dir, exist_ok=True)

    horizons = [h for h in HORIZONS if h <= args.pred_len]

    def period_title(label: str, mask: np.ndarray) -> str:
        n = int(mask.sum())
        d0, d1 = end_dates[mask].min(), end_dates[mask].max()
        head = "FULL TEST PERIOD" if label == "full" else label
        return f"{head}  ({n} windows, {d0.date()} → {d1.date()})"

    print(f"\n{'='*72}\nERROR TABLES — pred_len={args.pred_len}, "
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
