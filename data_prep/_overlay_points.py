"""
For each of the three sample dates, plot the wide_8_sharp surface and overlay
the raw (vega-weighted, OTM+ATM-filtered) option observations the kernel
actually saw. Lets us eyeball whether the smoother is faithful to the cloud
of observations or sliding off them.
"""
import os, sys
from os.path import dirname, abspath, join

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "_wide_8_sharp_full"
N = 8
M_LO, M_HI = -0.20, 0.20
FILTER_M_LO, FILTER_M_HI = -0.40, 0.40
ATM_TH = 0.01
MIN_VEGA = 0.5
IV_MIN, IV_MAX = 0.01, 3.0
TTM_MIN, TTM_MAX = 0.04, 1.0
FILTER_TTM_MAX = 1.5

DATES = [
    pd.Timestamp("2020-03-16"),
    pd.Timestamp("2018-09-04"),
    pd.Timestamp("2017-08-08"),
]


def load_raw_for_dates(dates):
    """Stream the options csv and keep rows whose date is in `dates`."""
    target = {d.strftime("%Y-%m-%d") for d in dates}
    print(f"Streaming options for {sorted(target)} ...")
    keep = []
    for chunk in pd.read_csv("SPX_options.csv", chunksize=1_000_000):
        m = chunk["date"].isin(target)
        if m.any():
            keep.append(chunk.loc[m].copy())
    opts = pd.concat(keep, ignore_index=True)
    print(f"  raw options rows for these 3 dates: {len(opts):,}")
    return opts


def attach_forward(opts):
    fwd = pd.read_csv("SPX_forward.csv")
    rename_map = {"expiration":"exdate","AMSettlement":"am_settlement","ForwardPrice":"forward_price"}
    fwd = fwd.rename(columns={k:v for k,v in rename_map.items() if k in fwd.columns})
    fwd = fwd[["date","exdate","am_settlement","forward_price"]].copy()
    fwd["date"]   = pd.to_datetime(fwd["date"])
    fwd["exdate"] = pd.to_datetime(fwd["exdate"])
    fwd = fwd.drop_duplicates(subset=["date","exdate","am_settlement"])
    fwd = (fwd.sort_values(["date","exdate","am_settlement"])
              .drop_duplicates(subset=["date","exdate"], keep="first")
              .drop(columns=["am_settlement"]))
    opts["date"]   = pd.to_datetime(opts["date"])
    opts["exdate"] = pd.to_datetime(opts["exdate"])
    if "forward_price" in opts.columns:
        opts = opts.drop(columns=["forward_price"])
    if "vega" not in opts.columns:
        opts["vega"] = 1.0
    opts = opts.merge(fwd, on=["date","exdate"], how="left").dropna(subset=["forward_price"])
    opts = opts[opts["forward_price"] > 0].copy()
    return opts


def filter_day(do):
    """Reproduce the filtering inside process_daily_surfaces()."""
    do = do.copy()
    do["ttm"] = (do["exdate"] - do["date"]).dt.days / 365.0
    eps_t = 1.0 / 365.0
    do = do[(do["ttm"] >= eps_t) & (do["ttm"] <= FILTER_TTM_MAX)]
    do["strike"] = do["strike_price"] / 1000.0
    do["m"] = np.log(do["strike"] / do["forward_price"])
    do = do[(do["m"] >= FILTER_M_LO) & (do["m"] <= FILTER_M_HI)]
    do = do[
        np.isfinite(do["impl_volatility"])
        & (do["impl_volatility"] > IV_MIN) & (do["impl_volatility"] < IV_MAX)
    ]
    do = do[np.isfinite(do["vega"]) & (do["vega"] >= MIN_VEGA)]
    do = do[do["volume"] > 0]
    calls = do[do["cp_flag"] == "C"]
    puts  = do[do["cp_flag"] == "P"]
    calls_keep = calls[(calls["m"] > 0.0) | (np.abs(calls["m"]) < ATM_TH)]
    puts_keep  = puts [(puts ["m"] < 0.0) | (np.abs(puts ["m"]) < ATM_TH)]
    return pd.concat([calls_keep, puts_keep]).reset_index(drop=True)


def main():
    df = pd.read_csv(join(OUT, "SPX_surfaces.csv"))
    df["date"] = pd.to_datetime(df["date"])
    iv = df.iloc[:, 2:].values.reshape(-1, N, N)

    raw = attach_forward(load_raw_for_dates(DATES))
    by_date = {d: g for d, g in raw.groupby("date")}

    m_grid = np.linspace(M_LO, M_HI, N)
    t_grid = np.exp(np.linspace(np.log(TTM_MIN), np.log(TTM_MAX), N))
    Mg, Tg = np.meshgrid(m_grid, t_grid)

    fig = plt.figure(figsize=(18, 6))
    for i, d in enumerate(DATES):
        diff = (df["date"] - d).abs()
        idx = int(diff.idxmin())
        actual = df.iloc[idx]["date"]
        surf = iv[idx]

        day_raw = filter_day(by_date.get(d, pd.DataFrame()))
        # restrict scatter to the saved grid window and tau range, just for
        # readability against the surface
        m_in = (day_raw["m"] >= M_LO) & (day_raw["m"] <= M_HI)
        t_in = (day_raw["ttm"] >= TTM_MIN) & (day_raw["ttm"] <= TTM_MAX)
        pts = day_raw[m_in & t_in]

        ax = fig.add_subplot(1, 3, i+1, projection="3d")
        ax.plot_surface(Mg, Tg, surf, cmap="viridis", alpha=0.55,
                        edgecolor="k", linewidth=0.25)
        ax.scatter(pts["m"], pts["ttm"], pts["impl_volatility"],
                   c="red", s=6, alpha=0.45, label=f"{len(pts)} obs")
        ax.set_title(f"{actual.date()}  ({len(pts)} obs in window, "
                     f"{len(day_raw)} total used by kernel)")
        ax.set_xlabel("log-moneyness")
        ax.set_ylabel("tau (yrs)")
        ax.set_zlabel("IV")
        ax.legend(loc="upper left", fontsize=8)
        # consistent z range for honesty
        z_lo = float(min(surf.min(), pts["impl_volatility"].min())) if len(pts) else surf.min()
        z_hi = float(max(surf.max(), pts["impl_volatility"].max())) if len(pts) else surf.max()
        ax.set_zlim(z_lo*0.95, z_hi*1.05)

    out_path = join(OUT, "sample_surfaces_with_obs.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
