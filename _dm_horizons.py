#!/usr/bin/env python3
"""DM significance per forecast horizon within the h=21 runs.

For a single horizon slice k, the loss is the per-window MSE over the 150
cells at exactly day k. k-step forecast errors are MA(k-1), so the DM test
uses Newey-West truncation M = k-1 (and a block bootstrap with block = k)."""
import os
import numpy as np
from scipy import stats
import pandas as pd

from train import LOOKBACK, MODEL_DIR, ROOT, load_dataset
from error_tables import test_end_dates

PRED_LEN, DATA_END, CSV = 21, "2023-12-29", os.path.join(ROOT, "SPX_surfaces.csv")

data = load_dataset(CSV, 0.7, 0.1, LOOKBACK, PRED_LEN, data_end=DATA_END)
Xte, Yte = data["test"]
pers = np.broadcast_to(Xte[:, -1:, :], Yte.shape).copy()
end_dates = test_end_dates(PRED_LEN, CSV, DATA_END, data["rows"]["val_end"],
                           data["rows"]["N"])
yr = pd.DatetimeIndex(end_dates).year
var = np.load(f"{MODEL_DIR['var']}/results/63_21/preds.npy").astype(np.float32)


def seed_preds(base):
    arrs = [np.load(os.path.join(base, e, "preds.npy")).astype(np.float32)
            for e in sorted(os.listdir(base))
            if e.startswith("seed_") and os.path.isfile(os.path.join(base, e, "preds.npy"))]
    return np.stack(arrs, 0)


def dm(d, h):
    """HLN-corrected DM; truncation/MA order h (=k-1+1). h>=1."""
    n = d.size
    dbar = d.mean()
    dc = d - dbar
    M = max(h - 1, 0)
    g0 = (dc * dc).mean()
    lrv = g0
    for k in range(1, M + 1):
        lrv += 2.0 * (1.0 - k / (M + 1.0)) * (dc[k:] * dc[:-k]).mean()
    lrv = max(lrv, 1e-18)
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    dm_hln = (dbar / np.sqrt(lrv / n)) * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return dm_hln, p


def block_boot(d, blk, B=15000, seed=0):
    rng = np.random.default_rng(seed)
    n = d.size
    blk = max(blk, 1)
    dext = np.concatenate([d, d])
    nb = int(np.ceil(n / blk))
    obs = d.mean()
    cnt = sum(abs(dext[(rng.integers(0, n, nb)[:, None] + np.arange(blk)).ravel()[:n] % n].mean()
                  - obs) >= abs(obs) for _ in range(B))
    return (cnt + 1) / (B + 1)


def slice_loss(pred, mask, k=None, lo=None):
    """Per-window MSE. k: single horizon (1-indexed). lo: average over
    horizons lo..21 (1-indexed, inclusive)."""
    e = (pred - Yte)[mask]
    if k is not None:
        e = e[:, k - 1, :]
        return (e * e).mean(axis=1)
    e = e[:, lo - 1:, :]
    return (e * e).mean(axis=(1, 2))


def analyse(name, sp, year, slices):
    m = (yr == year)
    n = int(m.sum())
    print(f"\n{'='*78}\n{name}  —  {year}   n={n} windows  seeds={sp.shape[0]}")
    print(f"  {'slice':<14}{'MSE m/V/P':<26}{'vs Persistence':<26}vs VAR")
    for tag, kw, hh in slices:
        pl = slice_loss(pers, m, **kw)
        vl = slice_loss(var, m, **kw)
        ens = slice_loss(sp.mean(0), m, **kw)
        msestr = f"{ens.mean():.3f}/{vl.mean():.3f}/{pl.mean():.3f}"
        cells = []
        for bl in (pl, vl):
            d = bl - ens
            _, p = dm(d, hh)
            pb = block_boot(d, hh)
            cells.append(f"d={d.mean():+.3f} p={p:.3f} bb={pb:.3f}")
        print(f"  {tag:<14}{msestr:<26}{cells[0]:<26}{cells[1]}")


SL = [
    ("h+1",        dict(k=1),  1),
    ("h+5",        dict(k=5),  5),
    ("h+10",       dict(k=10), 10),
    ("h+15",       dict(k=15), 15),
    ("h+21",       dict(k=21), 21),
    ("h+11..21avg", dict(lo=11), 21),
]

analyse("PatchTST",     seed_preds(f"{MODEL_DIR['patchtst']}/eval/63_21"), 2022, SL)
analyse("HOT (k-prod)", seed_preds(f"{MODEL_DIR['hot']}/eval/63_21/kronecker_product"), 2022, SL)
analyse("HOT (k-sum)",  seed_preds(f"{MODEL_DIR['hot']}/eval/63_21/kronecker_sum"), 2022, SL)
analyse("PCAFormer",    seed_preds(f"{MODEL_DIR['pcaformer']}/eval/63_21"), 2022, SL)
analyse("DLinear",      seed_preds(f"{MODEL_DIR['dlinear']}/eval/63_21"), 2020, SL)
print("\nd = baseline_MSE - model_MSE (d>0 => model better) | p = DM(HLN) | bb = block-boot")
