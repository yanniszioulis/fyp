#!/usr/bin/env python3
"""Diebold-Mariano significance check for the 2022 h+21 'outperformance'."""
import os
import numpy as np
from scipy import stats

from train import LOOKBACK, MODEL_DIR, ROOT, load_dataset
from error_tables import test_end_dates
import pandas as pd

PRED_LEN, DATA_END, CSV = 21, "2023-12-29", os.path.join(ROOT, "SPX_surfaces.csv")
H = PRED_LEN  # forecast horizon -> overlap, DM truncation lag = H-1

data = load_dataset(CSV, 0.7, 0.1, LOOKBACK, PRED_LEN, data_end=DATA_END)
Xte, Yte = data["test"]
pers = np.broadcast_to(Xte[:, -1:, :], Yte.shape).copy()
end_dates = test_end_dates(PRED_LEN, CSV, DATA_END, data["rows"]["val_end"],
                           data["rows"]["N"])
yr = pd.DatetimeIndex(end_dates).year
m22 = (yr == 2022)
n = int(m22.sum())
print(f"2022 windows: n = {n}  ({end_dates[m22][0].date()} .. {end_dates[m22][-1].date()})")


def seed_preds(base):
    arrs = []
    for e in sorted(os.listdir(base)):
        p = os.path.join(base, e, "preds.npy")
        if e.startswith("seed_") and os.path.isfile(p):
            arrs.append(np.load(p).astype(np.float32))
    return np.stack(arrs, 0)  # (S,N,P,C)


models = {
    "PatchTST":     seed_preds(f"{MODEL_DIR['patchtst']}/eval/63_21"),
    "PCAFormer":    seed_preds(f"{MODEL_DIR['pcaformer']}/eval/63_21"),
    "HOT (k-prod)": seed_preds(f"{MODEL_DIR['hot']}/eval/63_21/kronecker_product"),
    "HOT (k-sum)":  seed_preds(f"{MODEL_DIR['hot']}/eval/63_21/kronecker_sum"),
}
var = np.load(f"{MODEL_DIR['var']}/results/63_21/preds.npy").astype(np.float32)


def win_loss(pred):
    """Per-window MSE over the 2022 subset: mean over (horizon, cell)."""
    e = (pred - Yte)[m22]
    return (e * e).mean(axis=(1, 2))   # (n,)


def dm(d, h):
    """DM test on loss differential d (baseline - model); d>0 => model better.
    Newey-West (Bartlett) LRV with truncation M=h-1, plus HLN small-sample fix."""
    n = d.size
    dbar = d.mean()
    dc = d - dbar
    M = h - 1
    g0 = (dc * dc).mean()
    lrv = g0
    for k in range(1, M + 1):
        gk = (dc[k:] * dc[:-k]).mean()
        lrv += 2.0 * (1.0 - k / (M + 1.0)) * gk
    lrv = max(lrv, 1e-18)
    dm_stat = dbar / np.sqrt(lrv / n)
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    dm_hln = dm_stat * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    # naive (no autocorrelation correction) for contrast
    se0 = np.sqrt(g0 / n)
    p0 = 2 * (1 - stats.norm.cdf(abs(dbar / se0)))
    return dm_hln, p, dbar / se0, p0


def block_boot(d, h, B=20000, seed=0):
    """Circular block bootstrap p-value, block length = h."""
    rng = np.random.default_rng(seed)
    n = d.size
    L = h
    nb = int(np.ceil(n / L))
    dext = np.concatenate([d, d])
    cnt = 0
    obs = d.mean()
    for _ in range(B):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + np.arange(L)).ravel()[:n]
        bs = dext[idx % n].mean()
        if abs(bs - obs) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (B + 1)


pers_l = win_loss(pers)
var_l = win_loss(var)

for name, sp in models.items():
    seed_l = np.stack([win_loss(sp[s]) for s in range(sp.shape[0])], 0)  # (S,n)
    ens_l = win_loss(sp.mean(0))     # ensemble-mean forecast loss
    print(f"\n{'='*70}\n{name}   seeds={sp.shape[0]}")
    print(f"  mean window-MSE: model(ens)={ens_l.mean():.4f}  "
          f"seed-mean={seed_l.mean():.4f}  VAR={var_l.mean():.4f}  "
          f"Pers={pers_l.mean():.4f}")
    for bname, bl in [("Persistence", pers_l), ("VAR", var_l)]:
        d = bl - ens_l                       # ensemble forecast vs baseline
        dm_s, p, dm0, p0 = dm(d, H)
        pb = block_boot(d, H)
        print(f"  vs {bname:11s}: dbar={d.mean():+.4f}  "
              f"DM(HLN,M={H-1})={dm_s:+.2f} p={p:.3f}  | "
              f"naive p={p0:.4f}  | block-boot p={pb:.3f}")
        # per-seed sign check
        sig = 0
        for s in range(sp.shape[0]):
            ds = bl - seed_l[s]
            if dm(ds, H)[1] < 0.05:
                sig += 1
        print(f"               per-seed: {sig}/{sp.shape[0]} seeds individually p<0.05")
