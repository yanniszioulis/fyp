"""_run_gwn_graph_diag.py — GWN graph-init diagnostic.

3 variants × 5 seeds at pred_len=21, data_end=2023-12-29 (matches the
existing baselines), same training harness as the published GWN config
in train.py (Adam, LR_GWN=1e-3, WD_GWN=1e-3, no grad_clip).

Variant A — current default (random adaptive rank-3, no static support)
Variant B — static Gaussian kernel only (no adaptive)
Variant C — static + adaptive initialized to the Gaussian kernel
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("DLinear", "PatchTST", "HOT", "Tucker_DLinear", "GWN", "VAR"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import train as T
from gwn import GWN, build_gaussian_adjacency


PRED_LEN   = 21
SEEDS      = [0, 1, 2]
DATA_END   = "2023-12-29"
EPOCHS     = 100
PATIENCE   = 15
MIN_EPOCHS = 15
OUT_DIR    = os.path.join(ROOT, "_test_results", f"{T.LOOKBACK}_{PRED_LEN}",
                          "gwn_graph_diag")


# ─── Helpers ─────────────────────────────────────────────────────────

def log(msg=""):
    print(msg, flush=True)


def test_target_last_dates(data, csv_path, data_end):
    L, P = T.LOOKBACK, PRED_LEN
    r = data["rows"]
    starts = np.arange(r["N"] - L - P + 1)
    target_end = starts + L + P
    test_starts = starts[target_end > r["val_end"]]
    df = pd.read_csv(csv_path)
    if data_end is not None:
        df = df[df["date"] <= data_end].reset_index(drop=True)
    dates = pd.to_datetime(df["date"].to_numpy())
    return dates[test_starts + L + P - 1]


def per_year_mse(pred, target, dates):
    years = pd.DatetimeIndex(dates).year.to_numpy()
    out = {}
    for y in sorted(set(int(yy) for yy in years)):
        mask = years == y
        out[int(y)] = float(((pred[mask] - target[mask]) ** 2).mean())
    return out


def build_variant(name, C, gaussian_adj):
    """Build a GWN variant using the tuning winner architecture
    (8/8/8/16 channels, blocks=4, layers=1 → 11,369 params) — same as
    train.py's current default and the published baseline."""
    L, P = T.LOOKBACK, PRED_LEN
    common = dict(
        num_nodes=C, seq_len=L, pred_len=P,
        in_dim=1, gcn_bool=True,
        residual_channels=8, dilation_channels=8,
        skip_channels=8, end_channels=16,
        kernel_size=2, blocks=4, layers=1,
        dropout=0.3,
    )
    if name == "A":
        return GWN(**common, supports=None,            addaptadj=True,  aptinit=None)
    if name == "B":
        return GWN(**common, supports=[gaussian_adj],  addaptadj=False, aptinit=None)
    if name == "C":
        return GWN(**common, supports=[gaussian_adj],  addaptadj=True,
                   aptinit=gaussian_adj)
    raise ValueError(name)


def train_seed(variant_name, seed, data, gaussian_adj, device):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    C = data["rows"]["n_channels"]

    model = build_variant(variant_name, C, gaussian_adj)
    adapter = T._GWNAdapter(model)
    adapter.to(device)
    n_params = sum(p.numel() for p in adapter.parameters())

    # Same as existing harness GWN config: Adam + WD, no grad_clip.
    optimizer = torch.optim.Adam(adapter.parameters(),
                                 lr=T.LR_GWN, weight_decay=T.WD_GWN)

    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]
    Xte, Yte = data["test"]

    best_val   = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone()
                  for k, v in adapter.state_dict().items()}
    stop_epoch = None

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr = T._epoch(adapter, Xtr, Ytr, T.BATCH_SIZE, device,
                      optimizer=optimizer, generator=gen)
        va = T._epoch(adapter, Xva, Yva, T.BATCH_SIZE, device, optimizer=None)
        if va < best_val:
            best_val   = va
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in adapter.state_dict().items()}
        if epoch >= MIN_EPOCHS and (epoch - best_epoch) >= PATIENCE:
            stop_epoch = epoch
            break
    if stop_epoch is None:
        stop_epoch = epoch
    dt = time.time() - t_start

    adapter.load_state_dict(best_state)
    adapter.eval()
    preds_te = T._predict(adapter, Xte, T.BATCH_SIZE, device)
    test_mse = float(((preds_te - Yte) ** 2).mean())

    log(f"  variant {variant_name} seed {seed}: "
        f"best_val={best_val:.4f} @ep{best_epoch}/{stop_epoch}  "
        f"test_mse={test_mse:.4f}  ({dt:.0f}s, n_params={n_params:,})")

    return {
        "variant":    variant_name,
        "seed":       seed,
        "best_val":   best_val,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "test_mse":   test_mse,
        "n_params":   n_params,
        "preds":      preds_te,
        "adapter":    adapter,
    }


def adj_from_adapter(adapter, gaussian_adj):
    """Return the effective spatial adjacency the model used during
    forward — softmax(ReLU(E1 @ E2)) when adaptive is on, otherwise the
    fixed Gaussian kernel."""
    m = adapter.model
    if getattr(m, "addaptadj", False) and hasattr(m, "nodevec1"):
        with torch.no_grad():
            E1 = m.nodevec1.detach().cpu()
            E2 = m.nodevec2.detach().cpu()
            A_ad = torch.softmax(torch.relu(E1 @ E2), dim=1)
        return A_ad
    return gaussian_adj.cpu()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    device = T.pick_device()
    log(f"[gwn-diag] {datetime.now(timezone.utc).isoformat()}Z")
    log(f"[gwn-diag] device = {device}")
    log(f"[gwn-diag] data_end = {DATA_END}, seeds = {SEEDS}")
    log(f"[gwn-diag] epochs<={EPOCHS}, patience={PATIENCE}, "
        f"min_epochs={MIN_EPOCHS}")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(ROOT, "SPX_surfaces.csv")
    data = T.load_dataset(csv_path=csv_path, train_frac=0.7, val_frac=0.1,
                          lookback=T.LOOKBACK, pred_len=PRED_LEN,
                          data_end=DATA_END)
    test_dates = test_target_last_dates(data, csv_path, DATA_END)
    log(f"[gwn-diag] test windows: {len(data['test'][0])}  "
        f"date range: {test_dates.min().date()} → {test_dates.max().date()}")

    # Build Gaussian kernel once (the same instance is reused for all
    # variants and seeds).
    gaussian_adj = build_gaussian_adjacency(data["grid"])
    inv_h = 1.0 / (gaussian_adj ** 2).sum(dim=1)
    log(f"[gwn-diag] Gaussian kernel Herfindahl: "
        f"min={inv_h.min().item():.1f}, median={inv_h.median().item():.1f}, "
        f"max={inv_h.max().item():.1f}")

    Yte = data["test"][1]
    pers = np.broadcast_to(data["test"][0][:, -1:, :], Yte.shape).copy()
    pers_mse = float(((pers - Yte) ** 2).mean())
    pers_py  = per_year_mse(pers, Yte, test_dates)
    log(f"[gwn-diag] persistence test MSE = {pers_mse:.4f}")

    results: dict = {"A": [], "B": [], "C": []}
    for variant in ("A", "B", "C"):
        log(f"\n=== Variant {variant} ===")
        for seed in SEEDS:
            try:
                r = train_seed(variant, seed, data, gaussian_adj, device)
                results[variant].append(r)
            except Exception:
                log(f"  variant {variant} seed {seed} FAILED:")
                traceback.print_exc()

    # ── Table 1: per-variant seed variance ─────────────────────────
    log(f"\n{'=' * 78}")
    log(f"Table 1 — per-variant seed variance summary")
    log(f"{'=' * 78}")
    log(f"  {'variant':<10s} {'n_params':>9s}  {'test_mses':<40s}  "
        f"{'mean':>7s}  {'std':>7s}  {'mean/pers':>9s}")
    for v in ("A", "B", "C"):
        if not results[v]:
            log(f"  {v}: NO RESULTS")
            continue
        mses = [r["test_mse"] for r in results[v]]
        params = results[v][0]["n_params"]
        mean = float(np.mean(mses)); std = float(np.std(mses))
        mses_str = "[" + ", ".join(f"{m:.4f}" for m in mses) + "]"
        log(f"  {v:<10s} {params:>9,}  {mses_str:<40s}  "
            f"{mean:>7.4f}  {std:>7.4f}  {mean/pers_mse:>9.4f}")
    log(f"  {'persistence':<10s} {0:>9,}  {'':<40s}  "
        f"{pers_mse:>7.4f}  {'':>7s}  {1.0:>9.4f}")

    # ── Table 2: per-year ─────────────────────────────────────────
    log(f"\n{'=' * 78}")
    log(f"Table 2 — per-year test MSE (mean across seeds per variant)")
    log(f"{'=' * 78}")
    years = (2020, 2021, 2022, 2023)
    hdr = "  " + f"{'variant':<12s} " + "  ".join(f"{y:>8d}" for y in years) + f"  {'overall':>8s}"
    log(hdr)
    for v in ("A", "B", "C"):
        if not results[v]:
            continue
        per_year_means = {y: [] for y in years}
        for r in results[v]:
            pys = per_year_mse(r["preds"], Yte, test_dates)
            for y in years:
                if y in pys:
                    per_year_means[y].append(pys[y])
        row = "  " + f"{v:<12s} " + "  ".join(
            f"{(np.mean(per_year_means[y]) if per_year_means[y] else np.nan):>8.4f}"
            for y in years)
        overall = float(np.mean([r["test_mse"] for r in results[v]]))
        row += f"  {overall:>8.4f}"
        log(row)
    pers_row = "  " + f"{'persistence':<12s} " + "  ".join(
        f"{pers_py.get(y, float('nan')):>8.4f}" for y in years) + f"  {pers_mse:>8.4f}"
    log(pers_row)

    # Look up Tucker / DLinear per-year from the regenerated report.
    by_regime_path = os.path.join(ROOT, "_test_results",
                                  f"{T.LOOKBACK}_{PRED_LEN}", "full",
                                  "by_regime.csv")
    head_path = os.path.join(ROOT, "_test_results",
                             f"{T.LOOKBACK}_{PRED_LEN}", "full",
                             "headline.csv")
    if os.path.isfile(by_regime_path) and os.path.isfile(head_path):
        log(f"\n  Reference (from {os.path.relpath(by_regime_path, ROOT)}):")
        df_r = pd.read_csv(by_regime_path)
        head = pd.read_csv(head_path)
        regime_to_year = {"COVID": 2020, "Reflation calm": 2021,
                          "Bear 2022": 2022, "Normalisation": 2023}
        for model_name in ("dlinear", "tucker_dlinear"):
            sub = df_r[df_r["display"] == model_name].set_index("regime")
            if sub.empty:
                continue
            mses = [sub.loc[k, "MSE"] if k in sub.index else float("nan")
                    for k in regime_to_year.keys()]
            o = head[head["display"] == model_name]
            overall = float(o["MSE"].iloc[0]) if not o.empty else float("nan")
            row = "  " + f"{model_name:<12s} " + "  ".join(
                f"{m:>8.4f}" for m in mses) + f"  {overall:>8.4f}"
            log(row)

    # ── Table 3: adjacency examination ────────────────────────────
    log(f"\n{'=' * 78}")
    log(f"Table 3 — effective adjacency examination (seed 0)")
    log(f"{'=' * 78}")
    for v in ("A", "B", "C"):
        if not results[v]:
            continue
        r0 = results[v][0]
        A = adj_from_adapter(r0["adapter"], gaussian_adj)
        off = A - torch.diag(torch.diag(A))
        inv_h_v = 1.0 / (A ** 2).sum(dim=1)
        in_deg = A.sum(dim=0)
        top5 = in_deg.topk(5)
        log(f"\n  Variant {v}:")
        log(f"    off-diag mean    = {off.mean().item():.4f}")
        log(f"    off-diag max     = {off.max().item():.4f}")
        log(f"    Herfindahl (eff. neighbors): "
            f"median={inv_h_v.median().item():.1f}, "
            f"min={inv_h_v.min().item():.1f}, "
            f"max={inv_h_v.max().item():.1f}")
        log(f"    top-5 in-degree nodes: idx={top5.indices.tolist()}  "
            f"vals={[round(v_, 3) for v_ in top5.values.tolist()]}")
        if v == "C":
            init = gaussian_adj.cpu()
            diff = (A - init).norm().item()
            init_norm = init.norm().item()
            log(f"    Frobenius distance from Gaussian init: {diff:.4f}  "
                f"(init norm = {init_norm:.4f}, ratio = {diff/init_norm:.3f})")

    # ── Persist artefacts ─────────────────────────────────────────
    for v in ("A", "B", "C"):
        if results[v]:
            preds = np.stack([r["preds"] for r in results[v]], axis=0)
            np.save(os.path.join(OUT_DIR, f"variant_{v}_preds.npy"), preds)
    summary = {}
    for v in ("A", "B", "C"):
        summary[v] = [{k: vv for k, vv in r.items()
                       if k not in ("preds", "adapter")}
                      for r in results[v]]
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {"pred_len": PRED_LEN, "seeds": SEEDS,
                       "data_end": DATA_END, "epochs": EPOCHS,
                       "patience": PATIENCE, "min_epochs": MIN_EPOCHS},
            "persistence_test_mse":      pers_mse,
            "persistence_test_per_year": pers_py,
            "results":                   summary,
        }, f, indent=2, default=str)
    log(f"\n[gwn-diag] saved → {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
