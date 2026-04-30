#!/usr/bin/env python3
"""
Post-hoc per-channel bias correction for DCISM.

Loads a trained DCISM result dir, computes per-channel additive bias on the
validation set (in scaled space), applies it to the test predictions, and
writes a sibling `<src>_bc/` directory with bias-corrected outputs.

Motivation: DCISM(MAE) optimises for the conditional median; IV is right-skewed,
so its predictions sit below the conditional mean → systematic under-prediction
(bias ≈ -0.077 in scaled space in our benchmark). A constant per-channel offset
fit on validation closes the MSE gap to DCISM(MSE) without retraining and
without disturbing rank correlations (IC unchanged by additive shifts).

Usage:
    python DCISM/apply_bias_correction.py \\
        --src_dir DCISM/results/SPX_IV_21_63_DCISMv0_k13_ck3_ep100_lossmae

Produces (alongside <src_dir>):
    <src_dir>_bc/
      best_model.pt        (copy of original)
      config.json          (copy + bias_correction=true)
      bias.npy             (per-channel additive bias, [400], scaled space)
      pred.npy             (test predictions, BC-applied, scaled space)
      start_dates.npy      (copy)
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make the DCISM module importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcism_spx_iv import DCISM, load_splits, N_IV   # noqa: E402


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():           return torch.device("cuda")
    if torch.backends.mps.is_available():   return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description="Apply per-channel bias correction to a DCISM result dir.")
    ap.add_argument("--src_dir", required=True,
                    help="Existing DCISM result dir (must contain best_model.pt and config.json).")
    ap.add_argument("--device",  default="auto")
    args = ap.parse_args()

    src = args.src_dir.rstrip("/")
    cfg_path  = os.path.join(src, "config.json")
    ckpt_path = os.path.join(src, "best_model.pt")
    if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
        raise SystemExit(f"src_dir missing config.json or best_model.pt: {src}")

    if src.endswith("_bc"):
        raise SystemExit(f"refusing to bias-correct an already-corrected dir: {src}")
    dst = src + "_bc"
    os.makedirs(dst, exist_ok=True)

    with open(cfg_path) as f:
        cfg = json.load(f)

    device = _device(args.device)
    print(f"src      : {src}")
    print(f"dst      : {dst}")
    print(f"device   : {device}")

    # ── Rebuild model and load checkpoint ─────────────────────────────────
    model = DCISM(
        seq_len      = cfg["seq_len"],
        pred_len     = cfg["pred_len"],
        n_channels   = N_IV,
        kernel_size  = cfg["kernel_size"],
        conv_kernel  = cfg["conv_kernel"],
        conv_dropout = cfg.get("conv_dropout", 0.0),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"loaded   : {ckpt_path}  ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Load data ─────────────────────────────────────────────────────────
    X_tr, y_tr, X_va, y_va, X_te, test_dates, info, _scaler = load_splits(
        cfg["csv_path"], cfg["seq_len"], cfg["pred_len"],
    )
    print(f"data     : T={info['T']}  val={info['val_windows']}  test={info['test_windows']}")

    bs = cfg.get("batch_size", 64)

    # ── Compute per-channel bias on validation set (scaled space) ─────────
    # bias[c] = mean over (window, horizon) of (y_val[c] - pred_val[c])
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=bs, num_workers=0,
    )
    sum_resid = np.zeros(N_IV, dtype=np.float64)
    n_resid   = 0
    with torch.no_grad():
        for xb, yb in va_loader:
            pred = model(xb.to(device)).cpu().numpy()      # [B, P, 400]
            true = yb.numpy()                               # [B, P, 400]
            sum_resid += (true - pred).reshape(-1, N_IV).sum(axis=0).astype(np.float64)
            n_resid   += pred.shape[0] * pred.shape[1]
    bias = (sum_resid / n_resid).astype(np.float32)         # [400], scaled space
    print(f"bias     : mean={bias.mean():+.6f}  min={bias.min():+.6f}  max={bias.max():+.6f}")

    # ── Predict on test, apply bias ───────────────────────────────────────
    te_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te)),
        batch_size=bs, num_workers=0,
    )
    preds = []
    with torch.no_grad():
        for (xb,) in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0).astype(np.float32)   # [N_test, P, 400]
    preds_bc = preds + bias[None, None, :]                       # broadcast over [N, P]

    # ── Persist outputs to <src>_bc/ ──────────────────────────────────────
    np.save(os.path.join(dst, "bias.npy"),         bias)
    np.save(os.path.join(dst, "pred.npy"),         preds_bc)
    np.save(os.path.join(dst, "start_dates.npy"),  test_dates)

    # Copy ckpt and config (mark BC = true).
    shutil.copy2(ckpt_path, os.path.join(dst, "best_model.pt"))
    cfg_bc = {**cfg, "bias_correction": True, "bc_source_dir": src}
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg_bc, f, indent=2)

    # train_log.csv if it exists in src.
    src_log = os.path.join(src, "train_log.csv")
    if os.path.exists(src_log):
        shutil.copy2(src_log, os.path.join(dst, "train_log.csv"))

    print(f"\n✓ pred.npy        shape={preds_bc.shape}")
    print(f"✓ start_dates.npy range={test_dates[0]} → {test_dates[-1]}")
    print(f"✓ Done. Bias-corrected results in {dst}/")


if __name__ == "__main__":
    main()
