#!/usr/bin/env python3
"""DM significance check: DLinear vs persistence/VAR in 2020 (covid) at h+21."""
import os
import numpy as np
from scipy import stats
import pandas as pd

from train import LOOKBACK, MODEL_DIR, ROOT, load_dataset
from error_tables import test_end_dates

PRED_LEN, DATA_END, CSV = 21, "2023-12-29", os.path.join(ROOT, "SPX_surfaces.csv")
H = PRED_LEN

data = load_dataset(CSV, 0.7, 0.1, LOOKBACK, PRED_LEN, data_end=DATA_END)
Xte, Yte = data["test"]
pers = np.broadcast_to(Xte[:, -1:, :], Yte.shape).copy()
end_dates = test_end_dates(PRED_LEN, CSV, DATA_END, data["rows"]["val_end"],
                           data["rows"]["N"])
yr = pd.DatetimeIndex(end_dates).year
m20 = (yr == 2020)
n = int(m20.sum())
print(f"2020 windows: n = {n}  ({end_dates[m20][0].date()} .. {end_dates[m20][-1].date()})")


def seed_preds(base):
    arrs = []
    for e in sorted(os.listdir(base)):
        p = os.path.join(base, e, "preds.npy")
        if e.startswith("seed_") and os.path.isfile(p):
            arrs.append(np.load(p).astype(np.float32))
    return np.stack(arrs, 0)


dl = seed_preds(f"{MODEL_DIR['dlinear']}/eval/63_21")
var = np.load(f"{MODEL_DIR['var']}/results/63_21/preds.npy").astype(np.float32)


def win_loss(pred):
    e = (pred - Yte)[m20]
    return (e * e).mean(axis=(1, 2))


def dm(d, h):
    n = d.size
    dbar = d.mean()
    dc = d - dbar
    M = h - 1
    g0 = (dc * dc).mean()
    lrv = g0
    for k in range(1, M + 1):
        lrv += 2.0 * (1.0 - k / (M + 1.0)) * (dc[k:] * dc[:-k]).mean()
    lrv = max(lrv, 1e-18)
    dm_stat = dbar / np.sqrt(lrv / n)
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    dm_hln = dm_stat * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    se0 = np.sqrt(g0 / n)
    p0 = 2 * (1 - stats.norm.cdf(abs(dbar / se0)))
    return dm_hln, p, p0


def block_boot(d, h, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = d.size
    dext = np.concatenate([d, d])
    nb = int(np.ceil(n / h))
    obs = d.mean()
    cnt = 0
    for _ in range(B):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + np.arange(h)).ravel()[:n]
        if abs(dext[idx % n].mean() - obs) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (B + 1)


pers_l = win_loss(pers)
var_l = win_loss(var)
seed_l = np.stack([win_loss(dl[s]) for s in range(dl.shape[0])], 0)
ens_l = win_loss(dl.mean(0))

print(f"\nDLinear  seeds={dl.shape[0]}")
print(f"  mean window-MSE: DLinear(ens)={ens_l.mean():.4f}  "
      f"seed-mean={seed_l.mean():.4f}  VAR={var_l.mean():.4f}  Pers={pers_l.mean():.4f}")

# how concentrated is the loss? share from the worst 21-day cluster
order = np.argsort(pers_l)[::-1]
print(f"  loss concentration: top-21 windows hold "
      f"{pers_l[order[:21]].sum() / pers_l.sum() * 100:.0f}% of persistence's 2020 SSE; "
      f"worst window ends {end_dates[m20][order[0]].date()}")

for bname, bl in [("Persistence", pers_l), ("VAR", var_l)]:
    d = bl - ens_l
    dm_s, p, p0 = dm(d, H)
    pb = block_boot(d, H)
    sig = sum(dm(bl - seed_l[s], H)[1] < 0.05 for s in range(dl.shape[0]))
    print(f"  vs {bname:11s}: dbar={d.mean():+.4f}  "
          f"DM(HLN,M={H-1})={dm_s:+.2f} p={p:.3f}  | naive p={p0:.4f}  | "
          f"block-boot p={pb:.3f}  | per-seed {sig}/{dl.shape[0]} p<0.05")
