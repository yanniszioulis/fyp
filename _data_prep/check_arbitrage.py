"""
Static no-arbitrage diagnostic for processed IV surfaces (SPX_surfaces.csv).

Checks all three static no-arbitrage conditions on the gridded surface:

  1. Butterfly  - the call price must be convex in strike at fixed maturity
                  (equivalently the risk-neutral density is non-negative).
  2. Calendar   - total implied variance w = sigma^2 * T must be non-decreasing
                  in maturity at fixed forward log-moneyness.
  3. Vertical   - the call price must be monotone decreasing in strike with
                  forward-undiscounted slope dC/dK in [-1, 0] (call spread).

Prices are undiscounted Black-76, computed in normalised form c = C/F on the
date-independent grid kappa = K/F = exp(m), so the tests are independent of the
forward level.

Usage:
    python _data_prep/check_arbitrage.py SURF1.csv [SURF2.csv ...]
    python _data_prep/check_arbitrage.py --plot baseline.csv new.csv

The first CSV is treated as the baseline for the side-by-side comparison;
--plot also writes a <stem>_worst_arb.png 3-panel butterfly figure per file.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# Noise floor: real violations are >~1e-6, round-off is ~1e-15, so anything
# below this magnitude is numerical dust rather than arbitrage.
TOL = 1e-8


def load_surfaces(path):
    """Return (dates, forward_ref, m_grid, t_grid, iv[n_days, n_tau, n_m])."""
    df = pd.read_csv(path)
    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    if not iv_cols:
        raise ValueError(f"{path}: no iv_* columns found")

    parsed = [c.split("_") for c in iv_cols]            # ['iv', m, tau]
    m_grid = np.unique([float(p[1]) for p in parsed])
    t_grid = np.unique([float(p[2]) for p in parsed])
    mi = {v: i for i, v in enumerate(m_grid)}
    ti = {v: i for i, v in enumerate(t_grid)}

    vals = df[iv_cols].to_numpy(dtype=float)
    iv = np.full((len(df), len(t_grid), len(m_grid)), np.nan)
    for k, (_, m, tau) in enumerate(parsed):
        iv[:, ti[float(tau)], mi[float(m)]] = vals[:, k]
    return df["date"].to_numpy(), df["forward_ref"].to_numpy(), m_grid, t_grid, iv


def call_prices(iv, m_grid, t_grid):
    """Normalised undiscounted Black-76 call price c = C/F on the kappa grid.
    c(kappa) = Phi(d1) - kappa * Phi(d2),  kappa = exp(m),  ln(F/K) = -m."""
    kappa = np.exp(m_grid)
    T = t_grid[None, :, None]
    sqrtT = np.sqrt(T)
    d1 = (-m_grid[None, None, :] + 0.5 * iv ** 2 * T) / (iv * sqrtT)
    d2 = d1 - iv * sqrtT
    return norm.cdf(d1) - kappa[None, None, :] * norm.cdf(d2), kappa


def _summary(viol, depth, n_days, axis_for_day):
    """Pack count / rate / day-rate / depth stats for one arbitrage type."""
    per_day = viol.reshape(n_days, -1).sum(axis=1)
    total_cells = int(np.prod(viol.shape))
    return {
        "viol":        viol,
        "n_viol":      int(viol.sum()),
        "total_cells": total_cells,
        "cell_rate":   100.0 * viol.sum() / max(total_cells, 1),
        "days_with":   int((per_day > 0).sum()),
        "day_rate":    100.0 * (per_day > 0).sum() / max(n_days, 1),
        "max_depth":   float(depth[viol].max()) if viol.any() else 0.0,
        "worst_day":   int(np.argmax(per_day)),
        "worst_n":     int(per_day.max()) if len(per_day) else 0,
    }


def analyse(path, tol=TOL):
    dates, fwd, m_grid, t_grid, iv = load_surfaces(path)
    n_days = len(dates)
    c, kappa = call_prices(iv, m_grid, t_grid)

    # 1. Butterfly: no-arb butterfly cost B >= 0 (depth = -B when B < 0).
    Kl, K0, Kr = kappa[:-2], kappa[1:-1], kappa[2:]
    B = ((Kr - K0) * c[..., :-2] - (Kr - Kl) * c[..., 1:-1]
         + (K0 - Kl) * c[..., 2:])
    bfly = _summary(B < -tol, -B, n_days, axis_for_day=1)
    bfly["per_tau"] = (B < -tol).sum(axis=(0, 2)) / (n_days * B.shape[2])
    bfly["B"], bfly["iv"], bfly["dates"] = B, iv, dates
    bfly["m_grid"], bfly["t_grid"] = m_grid, t_grid

    # 2. Calendar: total variance w = sigma^2 * T must not fall as T rises.
    w = iv ** 2 * t_grid[None, :, None]
    dw = np.diff(w, axis=1)                              # (days, n_tau-1, n_m)
    cal = _summary(dw < -tol, -dw, n_days, axis_for_day=1)
    cal["per_tau"] = (dw < -tol).sum(axis=(0, 2)) / (n_days * dw.shape[2])
    cal["t_pairs"] = list(zip(t_grid[:-1], t_grid[1:]))

    # 3. Vertical (call spread): slope dc/dkappa must lie in [-1, 0].
    slope = np.diff(c, axis=2) / np.diff(kappa)[None, None, :]
    vert_viol = (slope > tol) | (slope < -1.0 - tol)
    depth = np.maximum(slope, -1.0 - slope)              # +ve magnitude of breach
    vert = _summary(vert_viol, depth, n_days, axis_for_day=1)

    return {"path": path, "n_days": n_days,
            "butterfly": bfly, "calendar": cal, "vertical": vert}


def print_report(results):
    line = "=" * 80
    for r in results:
        print("\n" + line)
        print(f"ARBITRAGE REPORT  -  {r['path']}   ({r['n_days']:,} days)")
        print(line)
        for key, label, unit in [
            ("butterfly", "Butterfly (convexity / density >= 0)", "C/F"),
            ("calendar",  "Calendar  (total variance non-decr.)", "var"),
            ("vertical",  "Vertical  (call-spread slope in -1..0)", "slope"),
        ]:
            s = r[key]
            print(f"  {label}")
            print(f"      {s['n_viol']:>9,} / {s['total_cells']:,} cells "
                  f"({s['cell_rate']:.3f}%)   "
                  f"{s['days_with']:,} days ({s['day_rate']:.1f}%)   "
                  f"deepest breach = {s['max_depth']:.2e} {unit}")
        bt = r["butterfly"]
        print("  butterfly violation rate by tau:")
        for it, tau in enumerate(bt["t_grid"]):
            bar = "#" * int(round(60 * bt["per_tau"][it]))
            print(f"      tau={tau:6.3f}y  {100*bt['per_tau'][it]:6.2f}%  {bar}")
        cal = r["calendar"]
        if cal["n_viol"]:
            print("  calendar violation rate by tau step:")
            for ip, (ta, tb) in enumerate(cal["t_pairs"]):
                rate = 100 * cal["per_tau"][ip]
                if rate > 0:
                    print(f"      {ta:.3f}y -> {tb:.3f}y  {rate:6.2f}%")

    if len(results) >= 2:
        print("\n" + line)
        print(f"{'CHANGE vs baseline':<26}{'baseline':>14}{'new':>14}{'delta':>16}")
        print("-" * 80)
        for key, label in [("butterfly", "butterfly cells"),
                            ("calendar", "calendar cells"),
                            ("vertical", "vertical cells")]:
            b, n = results[0][key]["n_viol"], results[1][key]["n_viol"]
            d = n - b
            pct = f"{100.0*d/b:+.1f}%" if b else "  n/a"
            print(f"{label:<26}{b:>14,}{n:>14,}{d:>+10,} ({pct})")
        print(line)


def plot_worst_butterfly(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bt = r["butterfly"]
    d = bt["worst_day"]
    iv, B = bt["iv"][d], bt["B"][d]
    viol = B < -TOL
    m_grid, t_grid = bt["m_grid"], bt["t_grid"]
    it = int(np.argmax(viol.sum(axis=1)))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Worst butterfly-arbitrage surface - "
                 f"{str(bt['dates'][d])[:10]}  ({Path(r['path']).name})", fontsize=13)

    im = ax[0].imshow(iv, aspect="auto", origin="lower", cmap="viridis")
    ax[0].set_title("(a) implied-vol surface")
    ax[0].set_xlabel("log-moneyness k"); ax[0].set_ylabel("tau (yr)")
    ax[0].set_xticks(range(len(m_grid)))
    ax[0].set_xticklabels([f"{m:+.2f}" for m in m_grid], rotation=90, fontsize=7)
    ax[0].set_yticks(range(len(t_grid)))
    ax[0].set_yticklabels([f"{t:.3f}" for t in t_grid], fontsize=7)
    ax[0].axhline(it, color="red", ls="--", lw=1.5)
    fig.colorbar(im, ax=ax[0], label="IV")

    smile = iv[it]
    ax[1].plot(m_grid, smile, "o-", color="tab:blue")
    bad = np.where(viol[it])[0] + 1
    if len(bad):
        ax[1].plot(m_grid[bad], smile[bad], "o", color="red", ms=11,
                   label="non-convex middle")
        ax[1].legend()
    ax[1].set_title(f"(b) smile at tau={t_grid[it]:.3f}yr")
    ax[1].set_xlabel("log-moneyness k"); ax[1].set_ylabel("IV"); ax[1].grid(alpha=0.3)

    Brow = B[it]
    colours = ["red" if b < -TOL else "tab:blue" for b in Brow]
    ax[2].bar(m_grid[1:-1], Brow, width=0.011, color=colours)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("(c) butterfly cost B - red = negative = arbitrage")
    ax[2].set_xlabel("log-moneyness k"); ax[2].set_ylabel("B  (~ d2C/dK2)")

    fig.tight_layout()
    out = Path(r["path"]).with_name(Path(r["path"]).stem + "_worst_arb.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    p = argparse.ArgumentParser(description="Static no-arbitrage diagnostic")
    p.add_argument("csvs", nargs="+", help="SPX_surfaces.csv file(s); the first "
                   "is the baseline for the comparison")
    p.add_argument("--tol", type=float, default=TOL,
                   help=f"breach magnitude below which a cell is ignored as "
                        f"numerical noise (default {TOL:g})")
    p.add_argument("--plot", action="store_true",
                   help="write a <stem>_worst_arb.png butterfly figure per file")
    args = p.parse_args()

    results = []
    for path in args.csvs:
        print(f"Analysing {path} ...")
        results.append(analyse(path, tol=args.tol))

    print_report(results)
    if args.plot:
        print("\nWriting worst-day figures...")
        for r in results:
            plot_worst_butterfly(r)


if __name__ == "__main__":
    main()
