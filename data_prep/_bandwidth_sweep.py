"""
Sweep h_m values on a 1-year subset to see whether reducing the moneyness
bandwidth recovers cross-sectional structure that the current h_m=0.001 erases.

Loads ~1 year of SPX_options rows by streaming the 4.2 GB CSV in chunks and
filtering on date. Runs the existing OptionMetricsPreprocess on each candidate
h_m, then reports the same smoothness diagnostics we used to flag the issue:
  - adjacent-diff correlation along moneyness/tau axes
  - PCA-on-daily-diffs explained variance for top-{1,3,5}
  - mean cross-surface std and mean |neighbor diff|
  - share of grid cells filled by ffill/bfill (reported by process_daily_surfaces)

Run from data_prep/. Writes outputs to ./_sweep_out/{h_m}/SPX_surfaces.csv.
"""
import os
import sys
from os.path import dirname, join, abspath

import numpy as np
import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))
from preprocess_optionmetrics import OptionMetricsPreprocess

START = "2018-01-01"
END   = "2019-01-01"
OPTIONS_FILE = "SPX_options.csv"
FORWARD_FILE = "SPX_forward.csv"
OUT_ROOT = "_sweep_out"

H_M_VALUES = [0.001, 0.0004, 0.0002, 0.0001]   # current is 0.001


def load_subset():
    print(f"Streaming options for {START} <= date < {END} ...")
    keep = []
    chunk_size = 1_000_000
    for chunk in pd.read_csv(OPTIONS_FILE, chunksize=chunk_size):
        m = (chunk["date"] >= START) & (chunk["date"] < END)
        if m.any():
            keep.append(chunk.loc[m].copy())
    if not keep:
        raise RuntimeError("no rows in window")
    opts = pd.concat(keep, ignore_index=True)
    print(f"  options rows in window: {len(opts):,}")
    return opts


def run_one(h_m, opts_df):
    out_dir = join(OUT_ROOT, f"hm_{h_m}")
    os.makedirs(out_dir, exist_ok=True)
    pre = OptionMetricsPreprocess(
        options_file=OPTIONS_FILE,
        forward_file=FORWARD_FILE,
        output_dir=out_dir,
        h_m=h_m,
    )
    # bypass load_data's full read; reuse its forward-merge logic by calling
    # load_data after stuffing options_df. cleanest: copy the relevant bits.
    pre.options_df = opts_df.copy()

    # forward load + merge -- mirror load_data but skip the chunked options read
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

    pre.process_daily_surfaces()
    return join(out_dir, "SPX_surfaces.csv")


def diagnose(csv_path, label):
    df = pd.read_csv(csv_path)
    iv = df.iloc[:, 2:].values.reshape(-1, 20, 20)
    T = iv.shape[0]

    def adj_corr(S, axis):
        a = np.take(S, np.arange(S.shape[axis]-1), axis=axis)
        b = np.take(S, np.arange(1, S.shape[axis]), axis=axis)
        a = a.reshape(a.shape[0], -1); b = b.reshape(b.shape[0], -1)
        am = a - a.mean(0); bm = b - b.mean(0)
        num = (am*bm).mean(0); den = a.std(0)*b.std(0) + 1e-12
        return (num/den).mean()

    d = np.diff(iv, axis=0)
    Xd = d.reshape(d.shape[0], 400); Xd = Xd - Xd.mean(0)
    Sv = np.linalg.svd(Xd, compute_uv=False)
    cum = np.cumsum(Sv**2) / np.sum(Sv**2)

    print(f"\n=== {label} (T={T}) ===")
    print(f"  LEVEL adj corr (m / tau)        : {adj_corr(iv,2):.4f} / {adj_corr(iv,1):.4f}")
    print(f"  DIFFS adj corr (m / tau)        : {adj_corr(d,2):.4f} / {adj_corr(d,1):.4f}")
    print(f"  PCA on diffs top-1 / 3 / 5 (%)  : "
          f"{cum[0]*100:.2f} / {cum[2]*100:.2f} / {cum[4]*100:.2f}")
    print(f"  mean |neighbor diff| (m / tau)  : "
          f"{np.abs(np.diff(iv,axis=2)).mean():.5f} / {np.abs(np.diff(iv,axis=1)).mean():.5f}")
    print(f"  mean cross-surface std (per day): {iv.reshape(T,400).std(axis=1).mean():.5f}")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    opts = load_subset()
    paths = {}
    for h in H_M_VALUES:
        print(f"\n----- running h_m = {h} -----")
        paths[h] = run_one(h, opts)
    for h, p in paths.items():
        diagnose(p, f"h_m = {h}")


if __name__ == "__main__":
    main()
