"""
Plot raw OTM call/put quotes against the kernel-smoothed IV surface for
two trading days side-by-side, chosen to illustrate the two regimes
the smoother has to handle:

  LEFT  — 2004-03-22: ~488 raw quotes, ATM 30d IV ~19%   (sparse early)
  RIGHT — 2020-03-16: ~20k  raw quotes, ATM 30d IV ~79%  (dense COVID)

Each panel shows the smoothed surface as a viridis-shaded mesh and the
raw quotes as a 3D scatter (calls blue triangles, puts red dots). A
single shared legend sits along the bottom and a shared colourbar on
the left.

Usage:
    python _data_prep/plot_quotes_vs_surface.py \
        --surfaces SPX_surfaces.csv \
        --options  _data_prep/SPX_options.csv \
        --forward  _data_prep/SPX_forward.csv \
        --out      _data_prep/quotes_vs_surface.pdf
    # override the baked-in pair:
    #   --dates 2008-09-15 2023-10-19
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')
import numpy as np
import pandas as pd

CALL_COLOR = "#1f77b4"   # blue
PUT_COLOR  = "#d62728"   # red

# LaTeX rendering for thesis figures. Requires TeX on PATH.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
})

# Same filter window as the surface preprocessor.
M_LO, M_HI = -0.25, 0.25
TTM_LO, TTM_HI = 1.0 / 365.0, 1.75
IV_MIN, IV_MAX = 0.01, 3.0
MIN_VEGA = 0.5
ATM_THR = 0.02


def load_forward(path):
    fwd = pd.read_csv(path).rename(columns={
        "expiration": "exdate", "AMSettlement": "am_settlement",
        "ForwardPrice": "forward_price",
    })[["date", "exdate", "am_settlement", "forward_price"]]
    fwd["date"]   = pd.to_datetime(fwd["date"])
    fwd["exdate"] = pd.to_datetime(fwd["exdate"])
    fwd = (fwd.sort_values(["date", "exdate", "am_settlement"])
              .drop_duplicates(subset=["date", "exdate"], keep="first")
              .drop(columns=["am_settlement"]))
    return fwd


def quotes_for_dates(options_path, fwd, target_dates):
    """Stream the raw options file, keep only rows on target_dates that pass
    the same OTM + quality filters used in preprocess_optionmetrics.py."""
    target = pd.to_datetime(list(target_dates))
    usecols = ["date", "exdate", "cp_flag", "strike_price",
               "impl_volatility", "volume", "vega"]
    dtypes  = {"cp_flag": "category", "strike_price": "float64",
               "impl_volatility": "float32", "volume": "int64",
               "vega": "float32"}

    kept = []
    for chunk in pd.read_csv(options_path, chunksize=1_000_000,
                              usecols=usecols, dtype=dtypes):
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk = chunk[chunk["date"].isin(target)]
        if not len(chunk):
            continue
        chunk["exdate"] = pd.to_datetime(chunk["exdate"])

        chunk = chunk.merge(fwd, on=["date", "exdate"], how="left")
        chunk = chunk[chunk["forward_price"] > 0]

        chunk["ttm"] = (chunk["exdate"] - chunk["date"]).dt.days / 365.0
        chunk = chunk[(chunk["ttm"] >= TTM_LO) & (chunk["ttm"] <= TTM_HI)]

        strike = chunk["strike_price"] / 1000.0
        chunk["moneyness"] = np.log(strike / chunk["forward_price"])
        chunk = chunk[(chunk["moneyness"] >= M_LO) & (chunk["moneyness"] <= M_HI)]

        iv = chunk["impl_volatility"]
        base_ok = (np.isfinite(iv) & (iv > IV_MIN) & (iv < IV_MAX)
                   & np.isfinite(chunk["vega"]))
        is_wing = chunk["moneyness"].abs() > 0.07
        vega_ok = np.where(is_wing, chunk["vega"] >= 0.1,
                           chunk["vega"] >= MIN_VEGA)
        vol_ok  = (chunk["volume"] > 0) | is_wing
        chunk = chunk[base_ok & vega_ok & vol_ok]

        calls = chunk[chunk["cp_flag"] == "C"]
        puts  = chunk[chunk["cp_flag"] == "P"]
        calls = calls[(calls["moneyness"] > 0.0)
                      | (calls["moneyness"].abs() < ATM_THR)]
        puts  = puts[(puts["moneyness"] < 0.0)
                     | (puts["moneyness"].abs() < ATM_THR)]
        kept.append(pd.concat([calls, puts]))

    if not kept:
        raise RuntimeError(f"no quotes found for {list(target_dates)}")
    return pd.concat(kept, ignore_index=True)


def parse_surface_grid(columns):
    """Recover the (n_tau, n_m) IV grid from iv_{m}_{tau} column names."""
    iv_cols = [c for c in columns if c.startswith("iv_")]
    parts = [c[3:].rsplit("_", 1) for c in iv_cols]
    m_vals   = sorted({float(m) for m, _ in parts})
    tau_vals = sorted({float(t) for _, t in parts})
    return m_vals, tau_vals, iv_cols


def surface_grid_for_row(row, m_vals, tau_vals):
    """Build a (n_tau, n_m) IV grid from a single row of SPX_surfaces.csv."""
    grid = np.full((len(tau_vals), len(m_vals)), np.nan)
    m_idx   = {m: i for i, m in enumerate(m_vals)}
    tau_idx = {t: i for i, t in enumerate(tau_vals)}
    for col, val in row.items():
        if not col.startswith("iv_"):
            continue
        m_str, t_str = col[3:].rsplit("_", 1)
        i = tau_idx[float(t_str)]
        j = m_idx[float(m_str)]
        grid[i, j] = val
    return grid


def render_panel(ax, m_vals, tau_vals, surface, quotes, date_str, n_quotes,
                 vmin, vmax, cmap):
    # Restrict everything to the surface support: m and T axes only span
    # the smoothed grid (m in [-0.10, +0.10], T in [tau_min, tau_max]).
    m_min, m_max     = float(min(m_vals)), float(max(m_vals))
    tau_min, tau_max = float(min(tau_vals)), float(max(tau_vals))

    # log10(T) puts the short-dated quotes on a readable scale.
    log_tau = np.log10(np.asarray(tau_vals))
    M, LT = np.meshgrid(m_vals, log_tau)
    surf_mappable = ax.plot_surface(
        M, LT, surface, cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=0.42, linewidth=0.2, edgecolor="0.3",
        rstride=1, cstride=1, antialiased=True,
    )

    in_box = (
        (quotes["moneyness"] >= m_min) & (quotes["moneyness"] <= m_max)
        & (quotes["ttm"] >= tau_min)   & (quotes["ttm"]       <= tau_max)
    )
    q = quotes[in_box]
    calls = q[q["cp_flag"] == "C"]
    puts  = q[q["cp_flag"] == "P"]
    lt_c = np.log10(calls["ttm"].to_numpy())
    lt_p = np.log10(puts["ttm"].to_numpy())

    ax.scatter(puts["moneyness"], lt_p, puts["impl_volatility"],
               c=PUT_COLOR, marker="o", s=14, alpha=0.85,
               depthshade=False, edgecolors="none")
    ax.scatter(calls["moneyness"], lt_c, calls["impl_volatility"],
               c=CALL_COLOR, marker="^", s=16, alpha=0.85,
               depthshade=False, edgecolors="none")

    # Snap each round-number maturity to its nearest kernel grid value so the
    # wall gridlines coincide with the surface mesh stripes. The labels are
    # nominal: e.g. "6m" actually marks the closest grid tau (~5.2m here).
    tau_arr = np.asarray(tau_vals)

    def snap(target):
        return float(tau_arr[int(np.argmin(np.abs(tau_arr - target)))])

    tick_t = [(snap(0.0822), "1m"), (snap(0.25), "3m"),
              (snap(0.5), "6m"), (snap(1.0), "1y")]
    # de-duplicate in case two targets snap to the same grid value
    seen = set(); tick_t = [tl for tl in tick_t
                            if not (tl[0] in seen or seen.add(tl[0]))]
    ax.set_yticks([np.log10(t) for t, _ in tick_t])
    ax.set_yticklabels([lbl for _, lbl in tick_t])

    y_pad = 0.04 * (np.log10(tau_max) - np.log10(tau_min))
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(np.log10(tau_min) - y_pad, np.log10(tau_max) + y_pad)
    ax.set_zlim(vmin, vmax)
    ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])
    ax.set_xticklabels([r"$-0.10$", r"$-0.05$", r"$0$",
                        r"$0.05$", r"$0.10$"])
    ax.set_xlabel(r"Log moneyness $m$", labelpad=4)
    ax.set_ylabel(r"Maturity $\tau$", labelpad=4)
    ax.set_zlabel("")                          # \sigma is on the colourbar
    # Two-line title: date on top, raw-quote count underneath. The smaller
    # second line is what makes the sparse-vs-dense contrast legible at a
    # glance without needing the caption to spell it out.
    ax.set_title(f"{date_str}\n" rf"{{\small ${n_quotes:,}$ quotes}}",
                 pad=6, fontsize=14)
    ax.set_proj_type("ortho")                  # no perspective skew
    # Camera at (+x, +y): the (m=+0.10, tau=1y) corner faces the viewer,
    # so the long-maturity edge runs along the front of the box and the
    # surface unfolds front-to-back along tau without folding back on
    # itself.
    ax.view_init(elev=20, azim=55)
    return surf_mappable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surfaces", default="SPX_surfaces.csv")
    p.add_argument("--options",  default="_data_prep/SPX_options.csv")
    p.add_argument("--forward",  default="_data_prep/SPX_forward.csv")
    p.add_argument("--out",      default="_data_prep/quotes_vs_surface.pdf")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--dates",    nargs="*",
                   default=["2004-03-22", "2020-03-16"],
                   help="two YYYY-MM-DD dates; default is the report's "
                        "sparse-early vs dense-COVID contrast pair.")
    args = p.parse_args()

    surf = pd.read_csv(args.surfaces)
    surf["date"] = pd.to_datetime(surf["date"])
    m_vals, tau_vals, iv_cols = parse_surface_grid(surf.columns)

    target = pd.to_datetime(args.dates)
    missing = [d for d in target if d not in set(surf["date"])]
    if missing:
        raise SystemExit(f"dates not in surfaces: {missing}")
    target = pd.DatetimeIndex(sorted(target))
    print(f"plotting dates: {[d.date().isoformat() for d in target]}")

    fwd = load_forward(args.forward)
    quotes = quotes_for_dates(args.options, fwd, target)

    # Shared colour range AND z-axis range across panels. Derived from the
    # SMOOTHED SURFACE IVs only — not from the raw quotes — because deep-OTM
    # wing quotes on crisis days can sit at >150% IV and would otherwise
    # blow vmax up and squash both surfaces flat against the bottom. The
    # surface itself is what we want to give the full colour bandwidth to;
    # the handful of outlier wing dots saturate in colour space but still
    # plot at their true z-position.
    surf_ivs = []
    for d in target:
        row = surf.loc[surf["date"] == d].iloc[0]
        surf_ivs.append(row[iv_cols].to_numpy(dtype=float))
    pool = np.concatenate([a[np.isfinite(a)] for a in surf_ivs])
    vmin = float(np.floor(pool.min()      * 10) / 10)
    vmax = float(np.ceil (pool.max() * 1.05 * 10) / 10)  # +5% headroom

    cmap = "viridis"
    fig = plt.figure(figsize=(8.4, 3.8))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    last_surf = None
    for ax, d in zip(axes, target):
        row   = surf.loc[surf["date"] == d].iloc[0]
        grid  = surface_grid_for_row(row, m_vals, tau_vals)
        day_q = quotes[quotes["date"] == d]
        last_surf = render_panel(
            ax, m_vals, tau_vals, grid, day_q,
            d.strftime("%Y-%m-%d"), int(len(day_q)),
            vmin, vmax, cmap,
        )

    # shared legend along the top (calls / puts), colourbar on the right
    legend_handles = [
        plt.Line2D([0], [0], marker="^", color="none",
                   markerfacecolor=CALL_COLOR, markeredgecolor="none",
                   markersize=7, label="Calls"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=PUT_COLOR, markeredgecolor="none",
                   markersize=7, label="Puts"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.55, 0.01), fontsize=11)

    # Layout, left to right:  "Implied volatility" label | colourbar | panels.
    fig.text(0.005, 0.5, r"Implied Volatility $\sigma$",
             rotation=90, va="center", ha="left")

    fig.subplots_adjust(left=0.11, right=0.99, top=0.97, bottom=0.10,
                        wspace=0.05)
    cbar_ax = fig.add_axes([0.085, 0.22, 0.018, 0.60])
    cbar = fig.colorbar(last_surf, cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position("left")    # ticks on the left of the bar
    cbar.outline.set_visible(True)              # keep the surrounding box

    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
