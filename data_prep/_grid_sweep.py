"""
Joint sweep over (n_axis_points, h_m, h_tau): does coarsening the grid +
matching the kernel bandwidth to ~1 grid step produce surfaces that look
less synthetically smooth (lower PCA-top-1 on daily diffs, lower adjacent-
diff correlation)?

Loads ~1 year of options (2018) and runs the existing OptionMetricsPreprocess.
"""
import os, sys
from os.path import dirname, join, abspath

import numpy as np
import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))
from preprocess_optionmetrics import OptionMetricsPreprocess

START = "2018-01-01"
END   = "2019-01-01"
OPTIONS_FILE = "SPX_options.csv"
FORWARD_FILE = "SPX_forward.csv"
OUT_ROOT = "_grid_out"

CONFIGS = [
    # label, n, h_m, h_tau, m_low, m_high, filter_m_low, filter_m_high, uniform_weights
    ("baseline_20",       20, 0.001,  0.05, -0.10, 0.10, -0.30, 0.30, False),
    ("coarse_8_sharp",     8, 0.0004, 0.10, -0.10, 0.10, -0.30, 0.30, False),
    ("wide_8_sharp",       8, 0.0010, 0.10, -0.20, 0.20, -0.40, 0.40, False),
    ("wide_10",           10, 0.0008, 0.13, -0.20, 0.20, -0.40, 0.40, False),
    ("wide_8_uniform",     8, 0.0010, 0.10, -0.20, 0.20, -0.40, 0.40, True),
]


def load_subset():
    print(f"Streaming options for {START} <= date < {END} ...")
    keep = []
    for chunk in pd.read_csv(OPTIONS_FILE, chunksize=1_000_000):
        m = (chunk["date"] >= START) & (chunk["date"] < END)
        if m.any():
            keep.append(chunk.loc[m].copy())
    opts = pd.concat(keep, ignore_index=True)
    print(f"  options rows: {len(opts):,}")
    return opts


def attach_forward(pre, opts_df):
    pre.options_df = opts_df.copy()
    fwd = pd.read_csv(FORWARD_FILE)
    rename_map = {"expiration":"exdate","AMSettlement":"am_settlement","ForwardPrice":"forward_price"}
    fwd = fwd.rename(columns={k:v for k,v in rename_map.items() if k in fwd.columns})
    fwd = fwd[["date","exdate","am_settlement","forward_price"]].copy()
    fwd["date"] = pd.to_datetime(fwd["date"])
    fwd["exdate"] = pd.to_datetime(fwd["exdate"])
    fwd = fwd.drop_duplicates(subset=["date","exdate","am_settlement"])
    fwd = (fwd.sort_values(["date","exdate","am_settlement"])
              .drop_duplicates(subset=["date","exdate"], keep="first")
              .drop(columns=["am_settlement"]))
    pre.options_df["date"] = pd.to_datetime(pre.options_df["date"])
    pre.options_df["exdate"] = pd.to_datetime(pre.options_df["exdate"])
    if "forward_price" in pre.options_df.columns:
        pre.options_df = pre.options_df.drop(columns=["forward_price"])
    if "vega" not in pre.options_df.columns:
        pre.options_df["vega"] = 1.0
    pre.options_df = pre.options_df.merge(fwd, on=["date","exdate"], how="left")
    pre.options_df = pre.options_df.dropna(subset=["forward_price"])
    pre.options_df = pre.options_df[pre.options_df["forward_price"] > 0].copy()


def run_one(label, n, h_m, h_tau, m_low, m_high, fm_low, fm_high, uniform, opts):
    out_dir = join(OUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    pre = OptionMetricsPreprocess(
        options_file=OPTIONS_FILE,
        forward_file=FORWARD_FILE,
        output_dir=out_dir,
        n_axis_points=n,
        h_m=h_m,
        h_tau=h_tau,
        moneyness_min=m_low, moneyness_max=m_high,
        filter_moneyness_min=fm_low, filter_moneyness_max=fm_high,
    )
    attach_forward(pre, opts)
    if uniform:
        pre.options_df["vega"] = 1.0
    pre.process_daily_surfaces()
    return join(out_dir, "SPX_surfaces.csv"), n


def diagnose(csv_path, n, label):
    df = pd.read_csv(csv_path)
    iv = df.iloc[:, 2:].values.reshape(-1, n, n)
    T = iv.shape[0]

    def adj_corr(S, axis):
        a = np.take(S, np.arange(S.shape[axis]-1), axis=axis)
        b = np.take(S, np.arange(1, S.shape[axis]), axis=axis)
        a = a.reshape(a.shape[0], -1); b = b.reshape(b.shape[0], -1)
        am = a - a.mean(0); bm = b - b.mean(0)
        num = (am*bm).mean(0); den = a.std(0)*b.std(0) + 1e-12
        return (num/den).mean()

    d = np.diff(iv, axis=0)
    Xd = d.reshape(d.shape[0], n*n); Xd = Xd - Xd.mean(0)
    Sv = np.linalg.svd(Xd, compute_uv=False)
    cum = np.cumsum(Sv**2) / np.sum(Sv**2)

    print(f"\n=== {label} (T={T}, grid={n}x{n}={n*n} cells) ===")
    print(f"  LEVEL adj corr (m / tau)        : {adj_corr(iv,2):.4f} / {adj_corr(iv,1):.4f}")
    print(f"  DIFFS adj corr (m / tau)        : {adj_corr(d,2):.4f} / {adj_corr(d,1):.4f}")
    top3 = cum[2]*100 if len(cum) >= 3 else float('nan')
    top5 = cum[4]*100 if len(cum) >= 5 else float('nan')
    print(f"  PCA on diffs top-1 / 3 / 5 (%)  : "
          f"{cum[0]*100:.2f} / {top3:.2f} / {top5:.2f}")
    print(f"  mean |neighbor diff| (m / tau)  : "
          f"{np.abs(np.diff(iv,axis=2)).mean():.5f} / {np.abs(np.diff(iv,axis=1)).mean():.5f}")
    print(f"  mean cross-surface std (per day): {iv.reshape(T,n*n).std(axis=1).mean():.5f}")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    opts = load_subset()
    runs = []
    for cfg in CONFIGS:
        label, n, h_m, h_tau, ml, mh, fml, fmh, uniform = cfg
        print(f"\n----- {label}: n={n}, h_m={h_m}, h_tau={h_tau}, "
              f"m=[{ml},{mh}], filter_m=[{fml},{fmh}], uniform={uniform} -----")
        path, _ = run_one(label, n, h_m, h_tau, ml, mh, fml, fmh, uniform, opts)
        runs.append((label, n, path))
    for label, n, path in runs:
        diagnose(path, n, label)


if __name__ == "__main__":
    main()
