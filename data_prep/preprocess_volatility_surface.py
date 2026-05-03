#!/usr/bin/env python3
"""
Preprocess OptionMetrics IvyDB Volatility_Surface data for SPX into the
flattened CSV format the project's models consume.

Replaces the older preprocess_optionmetrics.py (which built surfaces from
raw Option_Price via a homemade kernel smoother on a moneyness×tau grid).

Output grid (locked):
  • 10 maturities (days):      30, 60, 91, 122, 152, 182, 273, 365, 547, 730
  • 17 call-equivalent deltas Δ (×100):
                                 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                 55, 60, 65, 70, 75, 80, 85, 90
  • 170 IV cells per day, plus a leading 'date' column → 171 columns total.

Axis convention — call-equivalent delta Δ.
  Δ is a single dimension running 0.10 → 0.90 in 0.05 steps, 17 columns.
  Semantics:
    • Δ ≈ 0.10 — deep OTM call (high strike, K ≫ S; low IV expected)
    • Δ = 0.50 — ATM
    • Δ ≈ 0.90 — deep OTM put  (low  strike, K ≪ S; high IV expected, equity smirk)
  This matches OptionMetrics's own internal convention for kernel smoothing
  (manual page 39).

Selection rule (per axis position Δ):
  • Δ < 0.50:  cp_flag='C', delta = +Δ                 (e.g. Δ=0.10 ← C, +10)
  • Δ = 0.50:  mean of (C, +50) and (P, -50). Either alone is acceptable;
               cell is NaN (and the day is dropped) only if neither is present.
  • Δ > 0.50:  cp_flag='P', delta = -(1.0 - Δ)         (e.g. Δ=0.90 ← P, -10;
                                                        Δ=0.55 ← P, -45)

Column-ordering convention (LOCKED — do not re-sort):
  outer loop = maturities ascending,
  inner loop = Δ ascending.
  Names: iv_T{days}_D{int(round(Δ*100))}.
  Examples: iv_T30_D10, iv_T30_D15, …, iv_T30_D50, …, iv_T30_D90,
            iv_T60_D10, …, iv_T730_D90.
  Integer encoding avoids the alphabetical-sort bug that bit the legacy
  moneyness-grid CSV.

Quality control:
  • Drop a day if any required cell is missing/non-positive/non-finite.
  • Drop if any cell ≥ 3.0.
  • Drop if mean short-tenor (30/60/91 day) or mean long-tenor (365/547/730 day)
    IV is outside [0.05, 1.5].
  • Skew-shape rule (mean put-wing IV ≥ mean call-wing IV, with
    put-wing  = columns Δ > 0.50 and
    call-wing = columns Δ < 0.50): logged as warning per-day; halts
    overall if violated on > 1% of retained days.
  • Stricter shape check: for each (date, maturity), iv@Δ=0.90 ≥ iv@Δ=0.10.
    Halts if more than 1% of (day, maturity) pairs violate.
  • Date column strictly monotonic ascending in every output file.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Locked grid spec ─────────────────────────────────────────────────────────

DAYS_AXIS = [30, 60, 91, 122, 152, 182, 273, 365, 547, 730]   # 10 maturities

# Call-equivalent delta Δ as integer hundredths. 17 values.
DELTA_AXIS_X100 = [
    10, 15, 20, 25, 30, 35, 40, 45,
    50,
    55, 60, 65, 70, 75, 80, 85, 90,
]
ATM_X100 = 50                                                  # Δ = 0.50

# Sentinel for "missing IV" in OptionMetrics.
SENTINEL_IV = -99.99

N_CELLS = len(DAYS_AXIS) * len(DELTA_AXIS_X100)                # 170


def column_layout() -> list[str]:
    """Return the locked column ordering: ['date', iv_T..., iv_T..., ...]."""
    cols = ["date"]
    for days in DAYS_AXIS:
        for dx in DELTA_AXIS_X100:
            cols.append(f"iv_T{days}_D{dx}")
    return cols


def disp_column_layout() -> list[str]:
    """Same as column_layout but with disp_T... prefix."""
    cols = ["date"]
    for days in DAYS_AXIS:
        for dx in DELTA_AXIS_X100:
            cols.append(f"disp_T{days}_D{dx}")
    return cols


# ── Logging setup ────────────────────────────────────────────────────────────

def make_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("preprocess_volatility_surface")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                             datefmt="%H:%M:%S")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


# ── Loading and filtering input ─────────────────────────────────────────────

def load_filter_input(surface_file: Path, start_date: str, end_date: str,
                       log: logging.Logger) -> pd.DataFrame:
    """Stream the input CSV in chunks, retain only the (cp_flag, delta) rows
    we need to populate the call-equivalent-Δ axis."""
    log.info(f"Reading {surface_file} (this can take a minute)…")

    keep_cp_delta = set()
    for dx in DELTA_AXIS_X100:
        if dx <= ATM_X100:
            keep_cp_delta.add(("C", +dx))                       # Δ=0.10..0.50 ← call rows
        if dx >= ATM_X100:
            keep_cp_delta.add(("P", -(100 - dx)))               # Δ=0.50..0.90 ← put rows
    log.info(f"  retaining {len(keep_cp_delta)} (cp_flag, delta) combinations")

    keep_days = set(DAYS_AXIS)
    log.info(f"  retaining maturities (days) = {sorted(keep_days)}")

    n_rows_seen = 0
    n_rows_kept = 0
    chunks = []
    chunk_size = 1_000_000

    for chunk in pd.read_csv(
        surface_file,
        chunksize=chunk_size,
        usecols=["date", "days", "delta", "impl_volatility", "dispersion", "cp_flag"],
        dtype={"days": "int32", "delta": "int32",
                "impl_volatility": "float32", "dispersion": "float32",
                "cp_flag": "category", "date": "string"},
    ):
        n_rows_seen += len(chunk)

        # Date filter (lex compare works because YYYY-MM-DD)
        chunk = chunk[(chunk["date"] >= start_date) & (chunk["date"] <= end_date)]
        if len(chunk) == 0:
            continue

        chunk = chunk[chunk["days"].isin(keep_days)]
        if len(chunk) == 0:
            continue

        # Keep only the (cp_flag, delta) combos we need.
        idx = list(zip(chunk["cp_flag"].astype(str), chunk["delta"]))
        mask = np.array([t in keep_cp_delta for t in idx])
        chunk = chunk[mask]
        if len(chunk) == 0:
            continue

        # Drop sentinel and non-positive IVs (they will fail QC anyway).
        iv = chunk["impl_volatility"]
        chunk = chunk[(iv != SENTINEL_IV) & np.isfinite(iv) & (iv > 0)]
        if len(chunk) == 0:
            continue

        n_rows_kept += len(chunk)
        chunks.append(chunk)

    if not chunks:
        raise SystemExit("No rows survived filtering — check input file and date range.")

    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    log.info(f"  read {n_rows_seen:,} rows, kept {n_rows_kept:,} after filters")
    log.info(f"  unique input dates kept: {df['date'].nunique():,}")
    return df


# ── Build the per-day surface (one big pivot) ───────────────────────────────

def assign_call_equiv_delta_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Map each retained row to its call-equivalent Δ axis position (×100).

    Call rows (cp_flag='C', delta=+10..+50): delta_x100 = delta.
    Put  rows (cp_flag='P', delta=-50..-10): delta_x100 = 100 + delta
                                             (puts land at Δ=0.50..0.90;
                                              put -10 → Δ=0.90, put -45 → Δ=0.55).

    The ATM column (Δ=0.50) is fed by both (C, +50) and (P, -50); the
    downstream groupby-mean averages them into a single ATM cell.
    """
    out = df.copy()
    cp = out["cp_flag"].astype(str)
    delta = out["delta"].astype("int32")
    out["delta_x100"] = np.where(cp.eq("C"), delta, 100 + delta).astype("int32")
    return out


def build_wide(df: pd.DataFrame, log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorised construction of [n_dates, 170] wide IV and dispersion frames.

    Returns (iv_wide, disp_wide). Index = date (sorted ascending).
    Columns = MultiIndex (days, delta_x100) in the locked order.
    Cells with no input row will be NaN.
    """
    log.info("Building wide surfaces via groupby+pivot…")
    # Reduce ATM duplicates (call_50 and put_50 average to ATM cell).
    grouped = (
        df.groupby(["date", "days", "delta_x100"], observed=True)
          .agg(impl_volatility=("impl_volatility", "mean"),
                dispersion     =("dispersion",      "mean"))
          .reset_index()
    )

    iv_wide = grouped.pivot(index="date", columns=["days", "delta_x100"],
                              values="impl_volatility")
    disp_wide = grouped.pivot(index="date", columns=["days", "delta_x100"],
                                values="dispersion")

    # Reindex columns to the locked layout (and create columns missing from the
    # data, which will become all-NaN and force the day to be dropped).
    full_cols = pd.MultiIndex.from_tuples(
        [(d, dx) for d in DAYS_AXIS for dx in DELTA_AXIS_X100],
        names=["days", "delta_x100"],
    )
    iv_wide   = iv_wide.reindex(columns=full_cols)
    disp_wide = disp_wide.reindex(columns=full_cols)
    iv_wide   = iv_wide.sort_index()
    disp_wide = disp_wide.sort_index()
    log.info(f"  wide frame: dates={len(iv_wide)}, columns={iv_wide.shape[1]} (expect {N_CELLS})")
    return iv_wide, disp_wide


# ── Quality control ─────────────────────────────────────────────────────────

def qc_filter(iv_wide: pd.DataFrame, disp_wide: pd.DataFrame,
              log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply quality-control rules; return filtered frames + a reasons dict.

    Reasons tracked per dropped date: incomplete_grid, out_of_range, tenor_mean_oob.
    Skew-shape and per-maturity OTM-direction violations are logged as warnings;
    halt-tests (rate > 1%) live in main().
    """
    n0 = len(iv_wide)
    reasons = {"incomplete_grid": [], "out_of_range": [], "tenor_mean_oob": [],
               "skew_violations": [], "shape_violations": []}

    # 1) any NaN in the 170 IV columns → drop.
    nan_mask = iv_wide.isna().any(axis=1)
    drop_dates = list(iv_wide.index[nan_mask])
    reasons["incomplete_grid"] = drop_dates
    iv_wide = iv_wide[~nan_mask]
    disp_wide = disp_wide[~nan_mask]
    log.info(f"  after incomplete-grid drop: {len(iv_wide)}/{n0} dates "
             f"({len(drop_dates)} dropped)")

    # 2) cell range and finiteness.
    arr = iv_wide.to_numpy()
    bad_range = ~(np.isfinite(arr) & (arr > 0) & (arr < 3.0))
    bad_rows = bad_range.any(axis=1)
    drop_dates = list(iv_wide.index[bad_rows])
    reasons["out_of_range"] = drop_dates
    iv_wide = iv_wide[~bad_rows]
    disp_wide = disp_wide[~bad_rows]
    log.info(f"  after out-of-range drop:    {len(iv_wide)} ({len(drop_dates)} dropped)")

    # 3) tenor-mean-of-bounds.
    if len(iv_wide) == 0:
        return iv_wide, disp_wide, reasons

    short_cols = [c for c in iv_wide.columns if c[0] in (30, 60, 91)]
    long_cols  = [c for c in iv_wide.columns if c[0] in (365, 547, 730)]
    short_mean = iv_wide[short_cols].mean(axis=1)
    long_mean  = iv_wide[long_cols].mean(axis=1)
    oob = ((short_mean < 0.05) | (short_mean > 1.5) |
            (long_mean  < 0.05) | (long_mean  > 1.5))
    drop_dates = list(iv_wide.index[oob])
    reasons["tenor_mean_oob"] = drop_dates
    iv_wide = iv_wide[~oob]
    disp_wide = disp_wide[~oob]
    log.info(f"  after tenor-mean-oob drop:  {len(iv_wide)} ({len(drop_dates)} dropped)")

    # 4) skew-shape (mean put-wing IV ≥ mean call-wing IV).
    call_cols = [c for c in iv_wide.columns if c[1] < ATM_X100]
    put_cols  = [c for c in iv_wide.columns if c[1] > ATM_X100]
    put_wing  = iv_wide[put_cols].mean(axis=1)
    call_wing = iv_wide[call_cols].mean(axis=1)
    skew_bad = put_wing < call_wing
    reasons["skew_violations"] = list(iv_wide.index[skew_bad])
    log.info(f"  skew-shape violations: {int(skew_bad.sum())}/{len(iv_wide)} "
             f"({100*skew_bad.mean():.2f}%)")

    # 5) per-maturity OTM-direction shape check (Δ=0.90 ≥ Δ=0.10).
    shape_violations: list[tuple] = []
    for d in DAYS_AXIS:
        iv10 = iv_wide[(d, 10)]
        iv90 = iv_wide[(d, 90)]
        bad = iv90 < iv10
        if bad.any():
            for dt in iv_wide.index[bad]:
                shape_violations.append((dt, d))
    n_pairs = len(iv_wide) * len(DAYS_AXIS)
    shape_rate = len(shape_violations) / max(n_pairs, 1)
    reasons["shape_violations"] = shape_violations
    log.info(f"  shape-check (Δ=0.90 < Δ=0.10) violations: "
             f"{len(shape_violations)}/{n_pairs} ({100*shape_rate:.2f}%)")
    return iv_wide, disp_wide, reasons


# ── Underlying price ─────────────────────────────────────────────────────────

def load_underlying(price_file: Path, log: logging.Logger) -> pd.DataFrame | None:
    if price_file is None or not price_file.exists():
        log.warning(f"price_file not found at {price_file} — skipping SPX_underlying.csv")
        return None
    log.info(f"Loading underlying price from {price_file}")
    df = pd.read_csv(price_file)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "high" in df.columns and "low" in df.columns:
        df["underlying_price"] = (df["high"] + df["low"]) / 2.0
    elif "close" in df.columns:
        df["underlying_price"] = df["close"]
    else:
        raise ValueError(f"price_file lacks high/low or close columns: {df.columns}")
    return df[["date", "underlying_price"]].sort_values("date").reset_index(drop=True)


# ── Save outputs ─────────────────────────────────────────────────────────────

def to_flat_csv(wide: pd.DataFrame, layout: list[str], out_path: Path):
    """Flatten a (date, MultiIndex columns) wide frame into the locked CSV layout."""
    flat_names = layout[1:]                                    # skip 'date'
    arr = wide.to_numpy()                                       # already in locked order
    df = pd.DataFrame(arr, columns=flat_names, index=wide.index)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].astype(str)
    df.to_csv(out_path, index=False, float_format="%.6f")


# ── Sanity plots and reports ─────────────────────────────────────────────────

def make_sanity_plots(iv_wide: pd.DataFrame, out_dir: Path, log: logging.Logger,
                       n: int = 5, seed: int = 42):
    sp = out_dir / "sanity_plots"
    sp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if len(iv_wide) == 0:
        log.warning("no surfaces to plot")
        return
    pick = rng.choice(len(iv_wide), size=min(n, len(iv_wide)), replace=False)
    for i in pick:
        date = iv_wide.index[i]
        # Reshape [170] → [10 days, 17 deltas] in the locked column order.
        flat = iv_wide.iloc[i].to_numpy()
        grid = flat.reshape(len(DAYS_AXIS), len(DELTA_AXIS_X100))
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        im = ax.imshow(grid, aspect="auto", origin="upper", cmap="viridis")
        ax.set_xticks(range(len(DELTA_AXIS_X100)))
        ax.set_xticklabels([f"{dx/100:.2f}" for dx in DELTA_AXIS_X100],
                            rotation=45, fontsize=8)
        ax.set_yticks(range(len(DAYS_AXIS)))
        ax.set_yticklabels([f"{d}d" for d in DAYS_AXIS], fontsize=8)
        ax.set_xlabel("call-equivalent delta Δ")
        ax.set_ylabel("days to maturity")
        ax.set_title(f"SPX surface — {date}")
        plt.colorbar(im, ax=ax, label="implied vol")
        fig.tight_layout()
        fig.savefig(sp / f"random_surface_{date}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    log.info(f"saved {min(n, len(iv_wide))} sanity heatmaps to {sp}")


def make_delta_slice_plot(iv_wide: pd.DataFrame, out_dir: Path,
                            log: logging.Logger, target_date_str: str = "2019-11-22"):
    """Δ-axis slice diagnostic: 91d and 365d lines of IV vs Δ for one date.

    Picks `target_date_str` if it's in the retained set, otherwise the first
    retained date (e.g. on a smoke run that doesn't include it).
    """
    sp = out_dir / "sanity_plots"
    sp.mkdir(parents=True, exist_ok=True)
    if len(iv_wide) == 0:
        log.warning("no surfaces to plot")
        return
    target = pd.to_datetime(target_date_str).date()
    retained = set(iv_wide.index)
    date = target if target in retained else iv_wide.index[0]
    if date != target:
        log.info(f"  Δ-slice: target {target} not in retained set; using {date}")

    flat = iv_wide.loc[date].to_numpy()
    grid = flat.reshape(len(DAYS_AXIS), len(DELTA_AXIS_X100))
    xs = [dx / 100.0 for dx in DELTA_AXIS_X100]
    atm_idx = DELTA_AXIS_X100.index(ATM_X100)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, mat_days in [("91d", 91), ("365d", 365)]:
        i = DAYS_AXIS.index(mat_days)
        ax.plot(xs, grid[i], "-o", label=label, lw=1.4, ms=4)
        ax.plot(xs[atm_idx], grid[i][atm_idx],
                "o", mfc="none", mec="C0" if label == "91d" else "C1",
                ms=11, mew=1.6)
    ax.axvline(0.50, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("call-equivalent delta Δ")
    ax.set_ylabel("implied vol")
    ax.set_title(f"Δ-axis IV slice — {date}")
    ax.legend(title="maturity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = sp / f"delta_slice_{date}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved Δ-slice plot to {out}")


def write_report(report_path: Path, n_input_rows: int, n_input_dates: int,
                  n_after_qc: int, reasons: dict, disp_wide: pd.DataFrame,
                  date_range: tuple[str, str]):
    skew_pct = 100 * len(reasons["skew_violations"]) / max(n_after_qc, 1)
    n_pairs = max(n_after_qc * len(DAYS_AXIS), 1)
    shape_pct = 100 * len(reasons["shape_violations"]) / n_pairs
    md = []
    md += [f"# Preprocessing report"]
    md += [""]
    md += [f"- Date range requested: {date_range[0]} → {date_range[1]}"]
    md += [f"- Input rows after column/delta/maturity filter: {n_input_rows:,}"]
    md += [f"- Unique input dates kept (post-filter): {n_input_dates}"]
    md += [f"- Final retained dates: **{n_after_qc}**"]
    md += [""]
    md += [f"## Drop reasons"]
    md += [f"- `incomplete_grid` (any required cell missing): {len(reasons['incomplete_grid'])}"]
    md += [f"- `out_of_range`     (any cell ≤0, ≥3.0, or non-finite): {len(reasons['out_of_range'])}"]
    md += [f"- `tenor_mean_oob`   (short or long tenor mean IV outside [0.05, 1.5]): {len(reasons['tenor_mean_oob'])}"]
    md += [""]
    md += [f"## Skew-shape compliance"]
    md += [f"- Days where mean put-wing IV (Δ>0.50) < mean call-wing IV (Δ<0.50): "
            f"{len(reasons['skew_violations'])} (**{skew_pct:.2f}%** of retained days)"]
    md += [f"- (day, maturity) pairs where iv@Δ=0.90 < iv@Δ=0.10: "
            f"{len(reasons['shape_violations'])} (**{shape_pct:.2f}%** of {n_pairs} pairs)"]
    md += [""]
    md += [f"## Dispersion distribution (all retained cells)"]
    if disp_wide is not None and len(disp_wide):
        v = disp_wide.to_numpy().ravel()
        v = v[np.isfinite(v)]
        if len(v):
            md += [f"- median: {np.median(v):.6f}"]
            md += [f"- 90th percentile: {np.percentile(v, 90):.6f}"]
            md += [f"- 99th percentile: {np.percentile(v, 99):.6f}"]
            md += [f"- max: {v.max():.6f}"]
        else:
            md += ["- (no finite values)"]
    md += [""]
    md += [f"## Notes"]
    md += [f"- Output grid: 10 maturities × 17 call-equivalent Δ = **170 cells** per day "
            f"({1 + N_CELLS} CSV columns including `date`)."]
    md += [f"- Axis convention: Δ ∈ [0.10, 0.90] in 0.05 steps; Δ<0.50 ← call rows, "
            f"Δ>0.50 ← put rows (via 1−|δ_put|), Δ=0.50 = mean of call+put both at |δ|=0.50."]
    md += [f"- The HOT model script in this repo (`HOT/hot_spx_iv.py`) hardcodes a 20×20 "
            f"reshape and will not be compatible with the 170-cell layout until its "
            f"H_MONO/W_TAU constants are updated to (10, 17). All other models are agnostic "
            f"to N_IV and only require `N_IV = 170` in their config."]
    md += [""]
    md += [f"_Generated {datetime.now().isoformat(timespec='seconds')}_"]
    report_path.write_text("\n".join(md))


# ── Verification (must run before declaring done) ───────────────────────────

def verify_outputs(surfaces_csv: Path, log: logging.Logger):
    df = pd.read_csv(surfaces_csv, nrows=2)
    n_cols = df.shape[1]
    log.info(f"VERIFY: {surfaces_csv} cols={n_cols}, expect {1 + N_CELLS}")
    assert n_cols == 1 + N_CELLS, f"unexpected column count: got {n_cols}"

    cols = list(df.columns)
    log.info(f"VERIFY: first 5 cols after 'date': {cols[1:6]}")
    log.info(f"VERIFY: last 5 cols:                {cols[-5:]}")

    expected_first = ["iv_T30_D10", "iv_T30_D15", "iv_T30_D20", "iv_T30_D25", "iv_T30_D30"]
    expected_last  = ["iv_T730_D70", "iv_T730_D75", "iv_T730_D80", "iv_T730_D85", "iv_T730_D90"]
    assert cols[1:6]  == expected_first, f"first 5 mismatch: {cols[1:6]}"
    assert cols[-5:]  == expected_last,  f"last 5 mismatch: {cols[-5:]}"
    log.info("VERIFY: column ordering OK")

    # Pairwise correlation diagnostic on first day.
    full = pd.read_csv(surfaces_csv, nrows=300)
    if len(full) >= 30:
        sub = full.iloc[:30, 1:].to_numpy()                     # (30, 170)
        corr_full = np.corrcoef(sub.T)                          # (170, 170)
        adj_corrs, far_corrs = [], []
        for i in range(170 - 1):
            mat_i = i // 17
            mat_j = (i + 1) // 17
            if mat_i == mat_j:
                adj_corrs.append(corr_full[i, i + 1])
        for i in range(170):
            for j in range(i + 17 * 5, 170, 7):                 # far cells
                far_corrs.append(corr_full[i, j])
        log.info(f"VERIFY: same-maturity adjacent-cell mean corr = {np.mean(adj_corrs):.4f} "
                 f"(expect close to 1)")
        log.info(f"VERIFY: far-cell mean corr                   = {np.mean(far_corrs):.4f} "
                 f"(expect lower)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface_file", default="data_prep/volatility_surface.csv")
    ap.add_argument("--price_file",   default="data_prep/SPX_price.csv")
    ap.add_argument("--output_dir",   default=".")
    ap.add_argument("--start_date",   default="2009-01-01")
    ap.add_argument("--end_date",     default="2025-08-31")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "preprocessing_run.log"
    log = make_logger(log_path)

    # Step 0 — preserve any existing SPX_surfaces.csv as the broken-axis backup.
    legacy = out_dir / "SPX_surfaces.csv"
    if legacy.exists():
        rename_to = out_dir / "SPX_surfaces_broken_signed_delta.csv"
        if rename_to.exists():
            log.error(f"backup file {rename_to.name} already exists; refusing to "
                       f"overwrite. Move it aside first.")
            sys.exit(2)
        shutil.move(str(legacy), str(rename_to))
        log.info(f"renamed existing SPX_surfaces.csv → {rename_to.name}")

    # Step 1 — load + filter input.
    df = load_filter_input(Path(args.surface_file), args.start_date, args.end_date, log)
    n_input_rows  = len(df)
    n_input_dates = df["date"].nunique()

    # Step 2 — assign Δ grid and pivot to wide.
    df = assign_call_equiv_delta_grid(df)
    iv_wide, disp_wide = build_wide(df, log)

    # Step 3 — quality control.
    iv_wide, disp_wide, reasons = qc_filter(iv_wide, disp_wide, log)

    # Step 4 — halt if too many skew or shape violations.
    if len(iv_wide):
        skew_rate = len(reasons["skew_violations"]) / len(iv_wide)
        if skew_rate > 0.01:
            log.error(f"skew-shape violation rate {skew_rate:.2%} > 1%; halting")
            sys.exit(2)
        n_pairs = len(iv_wide) * len(DAYS_AXIS)
        shape_rate = len(reasons["shape_violations"]) / max(n_pairs, 1)
        if shape_rate > 0.01:
            log.error(f"shape-check violation rate {shape_rate:.2%} > 1%; halting")
            sys.exit(2)

    # Step 5 — date axis monotonic ascending check.
    iv_wide = iv_wide.sort_index()
    disp_wide = disp_wide.loc[iv_wide.index]
    assert iv_wide.index.is_monotonic_increasing
    assert disp_wide.index.is_monotonic_increasing

    # Step 6 — load underlying and inner-join (warn on missing dates).
    und = load_underlying(Path(args.price_file), log)
    if und is not None:
        keep = pd.Index(und["date"])
        missing = iv_wide.index.difference(keep)
        if len(missing):
            log.warning(f"{len(missing)} dates not in price_file — dropping")
        iv_wide = iv_wide.loc[iv_wide.index.intersection(keep)]
        disp_wide = disp_wide.loc[iv_wide.index]
        und = und[und["date"].isin(iv_wide.index)].reset_index(drop=True)

    # Step 7 — write outputs.
    surfaces_csv  = out_dir / "SPX_surfaces.csv"
    dispersion_csv = out_dir / "SPX_dispersion.csv"
    underlying_csv = out_dir / "SPX_underlying.csv"

    to_flat_csv(iv_wide,   column_layout(),       surfaces_csv)
    to_flat_csv(disp_wide, disp_column_layout(),  dispersion_csv)
    if und is not None and len(und):
        und["date"] = und["date"].astype(str)
        und.to_csv(underlying_csv, index=False, float_format="%.4f")
        log.info(f"saved SPX_underlying.csv  ({len(und)} dates)")

    log.info(f"saved SPX_surfaces.csv    ({len(iv_wide)} dates × {iv_wide.shape[1]} cells)")
    log.info(f"saved SPX_dispersion.csv  ({len(disp_wide)} dates × {disp_wide.shape[1]} cells)")

    # Step 8 — sanity plots and verification.
    make_sanity_plots(iv_wide, out_dir, log)
    make_delta_slice_plot(iv_wide, out_dir, log, target_date_str="2019-11-22")
    verify_outputs(surfaces_csv, log)

    # Step 9 — preprocessing report.
    write_report(out_dir / "preprocessing_report.md",
                  n_input_rows=n_input_rows,
                  n_input_dates=n_input_dates,
                  n_after_qc=len(iv_wide),
                  reasons=reasons,
                  disp_wide=disp_wide,
                  date_range=(args.start_date, args.end_date))
    log.info("DONE")


if __name__ == "__main__":
    main()
