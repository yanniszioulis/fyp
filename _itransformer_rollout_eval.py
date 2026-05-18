#!/usr/bin/env python3
"""
Recursive 21-step rollout of a trained 1-step iTransformer, evaluated on
the same test windows as the direct pred_len=21 setup so the comparison
is apples-to-apples.

Pipeline
--------
1. Load 1-step iTransformer checkpoint (run_dir at pred_len=1).
2. Load the dataset at pred_len=21 to get the standard test windows
   (1007 windows starting at last-target-date 2019-12-02).
3. For each window, recursively predict 21 steps in standardised log-IV
   space. At iteration t the model's input is the rolling lookback
   buffer of length L; we slide the buffer left by 1 and append the
   latest prediction each step.
4. Save the rollout preds and compare to:
     - v2 direct (pred_len=21 multi-step head)
     - DLinear (seed=0)
     - Persistence

Outputs
-------
<run_dir>/diagnostics/rollout_preds.npy        # [N, 21, C] in std log-IV
<run_dir>/diagnostics/rollout_metrics.json     # pooled MSE / RMSE / MAE
<run_dir>/diagnostics/comparison_rollout_vs_direct.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from train import (                                  # noqa: E402
    LOOKBACK, ITransformer, _ITransformerAdapter,
    load_dataset, pick_device,
)
from eval_full import (                              # noqa: E402
    REGIMES, assign_regime, test_target_dates,
)


# ─── Loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: str):
    with open(os.path.join(run_dir, "hyperparams.json")) as f:
        hp = json.load(f)
    model = ITransformer(**hp["model_kwargs"])
    grid = hp["grid"]
    adapter = _ITransformerAdapter(model, grid["n_tau"], grid["n_money"])
    state = torch.load(os.path.join(run_dir, "best_model.pt"),
                       map_location="cpu", weights_only=True)
    adapter.load_state_dict(state)
    adapter.eval()
    return hp, model, adapter


# ─── Rollout ──────────────────────────────────────────────────────────────

def rollout_21(adapter, model, Xte, device, target_horizon=21, batch=64):
    """Recursive 21-step rollout from each test window's lookback.

    Xte: [N, L, C] numpy. Returns [N, target_horizon, C] numpy in the
    same canonical [B, T, C] space the adapter expects.
    """
    N, L, C = Xte.shape
    rollout = np.empty((N, target_horizon, C), dtype=np.float32)

    adapter.to(device)
    with torch.no_grad():
        for s in range(0, N, batch):
            xb = torch.from_numpy(Xte[s:s+batch]).to(device)         # [B, L, C]
            buf = xb.clone()
            for t in range(target_horizon):
                # The adapter returns [B, model.pred_len, C]; we trained
                # with pred_len=1 so the second dim is 1.
                step_pred = adapter(buf)                             # [B, 1, C]
                rollout[s:s+batch, t : t+1] = step_pred.cpu().numpy()
                # Slide the buffer: drop the oldest, append step_pred.
                buf = torch.cat([buf[:, 1:, :], step_pred], dim=1)
    adapter.cpu()
    return rollout


# ─── Comparison helpers ───────────────────────────────────────────────────

def metric_block(preds: np.ndarray, Y: np.ndarray) -> dict:
    err2 = (preds - Y) ** 2
    return {
        "mse":  float(err2.mean()),
        "rmse": float(np.sqrt(err2.mean())),
        "mae":  float(np.abs(preds - Y).mean()),
    }


def per_horizon(preds, Y):
    return [(h + 1, float(((preds[:, h] - Y[:, h]) ** 2).mean()))
            for h in range(preds.shape[1])]


def per_regime(preds, Y, regimes, regime_names):
    out = {}
    for r in regime_names:
        mask = regimes == r
        if mask.sum() == 0:
            continue
        out[r] = float(((preds[mask] - Y[mask]) ** 2).mean())
    return out


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Trained pred_len=1 iTransformer run directory.")
    ap.add_argument("--target_pred_len", type=int, default=21)
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)

    out_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")
    hp, model, adapter = load_run(run_dir)
    print(f"  model: pred_len={model.pred_len}, params="
          f"{sum(p.numel() for p in adapter.parameters()):,}")
    if model.pred_len != 1:
        raise SystemExit(
            f"This script expects a pred_len=1 model, got "
            f"pred_len={model.pred_len}"
        )

    # Load the standard pred_len=21 test windows (same split as v2 eval).
    csv_path = os.path.join(ROOT, "SPX_surfaces.csv")
    data21 = load_dataset(
        csv_path, train_frac=0.7, val_frac=0.1,
        lookback=LOOKBACK, pred_len=args.target_pred_len,
        data_end=hp.get("data_end"),
    )
    Xte, Yte = data21["test"]
    N, L, C = Xte.shape
    print(f"  test windows (pred_len={args.target_pred_len}): "
          f"{N} (expect 1007)")
    assert N == 1007, f"unexpected test-window count {N}"

    # 21-step recursive rollout.
    print(f"  running {args.target_pred_len}-step recursive rollout …")
    rollout = rollout_21(adapter, model, Xte, device,
                         target_horizon=args.target_pred_len)
    np.save(os.path.join(out_dir, "rollout_preds.npy"),
            rollout.astype(np.float32))

    # Regime labels.
    end_dates, _, _ = test_target_dates(
        data21, args.target_pred_len, csv_path, hp.get("data_end")
    )
    regimes = assign_regime(end_dates)
    regime_names = [r[0] for r in REGIMES]

    # Reference preds.
    persistence = np.broadcast_to(
        Xte[:, -1:, :], (N, args.target_pred_len, C)
    ).copy()
    ref_paths = {
        "DLinear (seed=0)":          "DLinear/eval/63_21/seed_0/preds.npy",
        "iT v2 direct (per-cell head)": "iTransformer/63_21/"
                                         "2026-05-17T18-18-25Z/preds.npy",
        "iT v1 direct (shared head)":   "iTransformer/63_21/"
                                         "2026-05-17T17-59-05Z/preds.npy",
    }
    refs = {"persistence": persistence}
    for name, path in ref_paths.items():
        if os.path.isfile(path):
            refs[name] = np.load(path)

    # Pooled metrics.
    all_preds = {**refs, "iT 1-step → 21-roll": rollout}
    pooled = {n: metric_block(p, Yte) for n, p in all_preds.items()}
    with open(os.path.join(out_dir, "rollout_metrics.json"), "w") as fp:
        json.dump({"rollout_pooled": pooled[ "iT 1-step → 21-roll"],
                   "all_pooled": pooled}, fp, indent=2)

    # Build comparison markdown.
    md = [
        "# iTransformer: 1-step rollout vs 21-step direct",
        "",
        f"- 1-step model: `{os.path.relpath(run_dir, ROOT)}/best_model.pt` "
        f"(pred_len=1, trained as a single-step forecaster).",
        f"- Rollout: recursive 21-step prediction in standardised log-IV "
        f"space; at iteration t, the lookback buffer slides one step left "
        f"and the latest prediction is appended.",
        f"- All test predictions are evaluated against the same Yte from "
        f"the pred_len=21 split ({N} windows).",
        "",
        "## Pooled (test MSE / RMSE / MAE)",
        "",
        "| Model | Test MSE | RMSE | MAE | vs DLinear |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    dl_mse = pooled["DLinear (seed=0)"]["mse"]
    order = ["DLinear (seed=0)", "iT 1-step → 21-roll",
             "iT v2 direct (per-cell head)", "iT v1 direct (shared head)",
             "persistence"]
    for name in order:
        if name not in pooled:
            continue
        b = pooled[name]
        md.append(
            f"| {name} | **{b['mse']:.4f}**" if name == "iT 1-step → 21-roll"
            else f"| {name} | {b['mse']:.4f}"
        )
        md[-1] += f" | {b['rmse']:.4f} | {b['mae']:.4f} | {b['mse']/dl_mse:.3f}× |"
    md += [""]

    # Per-horizon.
    md += ["## Per-horizon MSE ratio vs DLinear",
           "",
           "| h | DLinear MSE | persist | v2 direct | **rollout** |",
           "| --: | ---: | ---: | ---: | ---: |"]
    for h in (0, 1, 2, 3, 4, 6, 9, 14, 20):
        vdl = ((refs["DLinear (seed=0)"][:, h] - Yte[:, h]) ** 2).mean()
        vpe = ((persistence[:, h] - Yte[:, h]) ** 2).mean()
        vv2 = ((refs["iT v2 direct (per-cell head)"][:, h]
                - Yte[:, h]) ** 2).mean() \
            if "iT v2 direct (per-cell head)" in refs else float("nan")
        vrl = ((rollout[:, h] - Yte[:, h]) ** 2).mean()
        md.append(f"| {h+1} | {vdl:.4f} | {vpe/vdl:.3f} | "
                  f"{vv2/vdl:.3f} | **{vrl/vdl:.3f}** |")
    md += [""]

    # Per-regime.
    md += ["## Per-regime pooled MSE (and ratios vs DLinear)",
           "",
           "| Regime | n | DLinear | persistence | v2 direct | **rollout** | "
           "rollout/DL | rollout vs v2 |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in regime_names:
        mask = regimes == r
        n_r = int(mask.sum())
        v_dl = ((refs["DLinear (seed=0)"][mask] - Yte[mask]) ** 2).mean()
        v_pe = ((persistence[mask] - Yte[mask]) ** 2).mean()
        v_v2 = ((refs["iT v2 direct (per-cell head)"][mask]
                 - Yte[mask]) ** 2).mean()
        v_rl = ((rollout[mask] - Yte[mask]) ** 2).mean()
        md.append(
            f"| {r} | {n_r} | {v_dl:.4f} | {v_pe:.4f} | {v_v2:.4f} | "
            f"**{v_rl:.4f}** | {v_rl/v_dl:.3f}× | {v_rl/v_v2:.3f}× |"
        )
    md += [""]

    # Interpretation hooks.
    rl_pool = pooled["iT 1-step → 21-roll"]["mse"]
    v2_pool = pooled.get("iT v2 direct (per-cell head)", {}).get("mse")
    md += ["## Interpretation",
           "",
           f"- Rollout pooled MSE: **{rl_pool:.4f}** vs v2-direct "
           f"**{v2_pool:.4f}** → rollout is "
           f"{'better' if v2_pool and rl_pool < v2_pool else 'worse'} by "
           f"{abs(rl_pool - (v2_pool or rl_pool))*1000:.1f} milliMSE.",
           "- Compare h=1 and h=21 ratios specifically: at h=1 the rollout "
           "uses its native target, at h=21 it has compounded 20 steps of "
           "error.",
           "- If the rollout is uniformly worse than direct, error "
           "compounding ('exposure bias') dominates — the model never saw "
           "its own predictions during training.",
           "- If the rollout beats direct at short horizons (h=1–3) but "
           "loses at long horizons, the 1-step model fits short-horizon "
           "dynamics better but compounds them too fast.",
           "- If the rollout beats direct *everywhere*, the multi-step head "
           "was wasting capacity that the 1-step head spends on getting the "
           "near-term dynamics right.",
           ""]

    out_md = os.path.join(out_dir, "comparison_rollout_vs_direct.md")
    with open(out_md, "w") as fp:
        fp.write("\n".join(md) + "\n")
    print(f"  wrote {os.path.relpath(out_md, ROOT)}")
    print(f"  rollout pooled mse: {rl_pool:.4f}  "
          f"(v2 direct: {v2_pool:.4f}, DLinear: {dl_mse:.4f})")


if __name__ == "__main__":
    main()
