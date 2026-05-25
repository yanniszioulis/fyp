#!/usr/bin/env python3
"""
train.py — train one (or all) IV-surface forecasters on SPX_surfaces.csv.

Models
------
Deep:  dlinear, patchtst, hot, tucker_dlinear, gwn
       Each uses its own __init__ defaults; lr is read from the LR_*
       constants below. Shared trainer: Adam, MSE on standardized log-IV,
       max EPOCHS, early stop with PATIENCE (suppressed until MIN_EPOCHS).
Stat:  var (Gonçalves–Guidolin two-stage: daily 5-param cross-sectional
       OLS on the surface basis [1, M, M², τ, Mτ] with M = k/√τ, then a
       BIC-selected VAR on the 5-dim β series — train-only fit, frozen
       on val/test. No training loop.

`--model all` runs the deep models sequentially. VAR must be invoked
explicitly with `--model var`.

Data
----
Columns of SPX_surfaces.csv shaped iv_{moneyness}_{tau} are parsed into a
[H=tau × W=moneyness] grid (current default: 10 × 11 = 110 cells, from
the preprocessed file under _data_prep/data/optionmetrics_processed/). All
IV values are taken in log space. A per-channel StandardScaler is fit on
the training rows; the same transform is applied to val/test. Train/val/
test split is chronological by window-end position (default 80/10/10).

Output
------
Deep models:  <ModelDir>/63_<pred_len>/<UTC-timestamp>/
                  hyperparams.json
                  metrics_test.json
                  train_log.csv
                  preds.npy            (N_test, pred_len, n_channels)
VAR:          VAR/results/63_<pred_len>/
                  hyperparams.json
                  metrics_test.json
                  preds.npy

Preds and stats are in standardized log-IV space (the training space).
hyperparams.json stores the scaler mean/scale so preds can be inverted
back to IV later.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Make the per-model packages importable as top-level modules.
ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("DLinear", "PatchTST", "HOT", "Tucker_DLinear", "GWN",
            "PCAFormer", "iTransformer", "ConvLSTM", "SANTA", "SANTA_flat",
            "SANTA_temporal", "VAR"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from dlinear import DLinear                # noqa: E402
from patchtst import PatchTST              # noqa: E402
from hot import HOT                        # noqa: E402
from tucker_dlinear import TuckerDLinear   # noqa: E402
from gwn import GWN                        # noqa: E402
from pcaformer import PCAFormer            # noqa: E402
from itransformer import ITransformer      # noqa: E402
from convlstm import ConvLSTM              # noqa: E402
from santa import (                        # noqa: E402
    SANTA, Config as SANTAConfig,
)
from santa_flat import SANTAFlat           # noqa: E402
from santa_temporal import SANTATemporal   # noqa: E402
from var import run_var_baseline           # noqa: E402


# ─── User-editable per-model optimiser settings ───────────────────────────
# Edit these to set the lr / weight_decay used at training time. Adam
# optimiser; all other training knobs (epochs, patience, min_epochs) are
# shared, see below.
LR_DLINEAR        = 4e-3
LR_PATCHTST       = 1e-3
LR_HOT            = 5e-4
LR_TUCKER_DLINEAR = 1e-3
LR_GWN            = 1e-3
LR_PCAFORMER      = 1e-3
LR_ITRANSFORMER   = 1e-3
LR_CONVLSTM       = 1e-3
# SANTA (Surface-Aware Neural Tensor Attention): transformer of comparable
# complexity to HOT/PatchTST. Same starting point — small lr, light
# decoupled WD with AdamW + grad_clip below.
LR_SANTA          = 5e-4
# SANTA-Flat: joint-spatial ablation of SANTA. Same per-block hyperparams
# as SANTA so the comparison isolates the factoring choice — start at the
# same lr / wd / optimiser.
LR_SANTA_FLAT     = 5e-4
# SANTA-Temporal: temporal-only ablation. Same per-block hyperparams as
# SANTA so the comparison isolates the spatial-block question — start at
# the same lr / wd / optimiser.
LR_SANTA_TEMPORAL = 5e-4

WD_DLINEAR        = 0.0
WD_PATCHTST       = 1e-4
WD_HOT            = 0.3
WD_TUCKER_DLINEAR = 1e-2
WD_GWN            = 1e-3
# PCAFormer: frozen PCA basis is the main regulariser; the transformer
# itself is unconstrained. PatchTST-style defaults (light WD, AdamW).
WD_PCAFORMER      = 1e-3
# iTransformer: same shape of starting point — light WD, AdamW. The
# attention-on-cells design is structurally similar to PatchTST's
# attention-on-patches; matching its WD is the natural default.
WD_ITRANSFORMER   = 3e-1
# ConvLSTM: Medvedev & Wang (2022) train with plain Adam and no weight
# decay — WD stays 0 and convlstm is kept off the AdamW list below.
WD_CONVLSTM       = 0.0
WD_SANTA          = 1e-3
WD_SANTA_FLAT     = 1e-3
WD_SANTA_TEMPORAL = 1e-3

# Tucker-only (with AdamW): the G core gets its own multipliers on top of
# LR_TUCKER_DLINEAR / WD_TUCKER_DLINEAR. The factor matrices stay at the
# base values. Lower G LR dampens gauge-direction noise; higher G WD
# regularises the dominant (overfit-prone) parameter group.
# Set both to 1.0 to apply the base values uniformly.
LR_TUCKER_DLINEAR_G_MULT = 0.25
WD_TUCKER_DLINEAR_G_MULT = 30

# Shared trainer settings (same for every deep model).
EPOCHS     = 100
PATIENCE   = 15
MIN_EPOCHS = 15
BATCH_SIZE = 32

LOOKBACK   = 63   # fixed across the project
VALID_PRED_LEN = (1, 5, 10, 21, 42, 63)
DEEP_MODELS    = ("dlinear", "patchtst", "hot", "tucker_dlinear", "gwn",
                  "pcaformer", "itransformer", "convlstm", "santa",
                  "santa_flat", "santa_temporal")

# Folder names per model (where outputs land relative to repo root).
MODEL_DIR = {
    "dlinear":        "DLinear",
    "patchtst":       "PatchTST",
    "hot":            "HOT",
    "tucker_dlinear": "Tucker_DLinear",
    "gwn":            "GWN",
    "pcaformer":      "PCAFormer",
    "itransformer":   "iTransformer",
    "convlstm":       "ConvLSTM",
    "santa":          "SANTA",
    "santa_flat":     "SANTA_flat",
    "santa_temporal": "SANTA_temporal",
    "var":            "VAR",
}


# ─── Device pick: cuda → mps → cpu ────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Data loading ─────────────────────────────────────────────────────────

@dataclass
class GridSpec:
    n_tau: int          # H
    n_money: int        # W
    iv_cols: list       # ordered column names matching reshape [H, W]
    tau_vals: list
    money_vals: list


def parse_grid(columns: list[str]) -> GridSpec:
    """Parse `iv_{moneyness}_{tau}` columns into a (tau × moneyness) grid.

    Column order in the reshape is set to (tau outer, moneyness inner) so
    that `data[:, 150].reshape(N, H=n_tau, W=n_money)` matches the CSV
    column ordering when iv_cols is laid out in that order.
    """
    parsed = []
    for c in columns:
        if not c.startswith("iv_"):
            continue
        _, m, t = c.split("_", 2)
        parsed.append((c, float(m), float(t)))
    money_vals = sorted({m for _, m, _ in parsed})
    tau_vals   = sorted({t for _, _, t in parsed})
    # Layout iv_cols as tau-outer, moneyness-inner.
    lookup = {(m, t): name for name, m, t in parsed}
    iv_cols = [lookup[(m, t)] for t in tau_vals for m in money_vals]
    return GridSpec(
        n_tau=len(tau_vals), n_money=len(money_vals),
        iv_cols=iv_cols, tau_vals=tau_vals, money_vals=money_vals,
    )


def load_dataset(csv_path: str, train_frac: float, val_frac: float,
                 lookback: int, pred_len: int, data_end: str | None = None):
    """Return windowed train/val/test tensors and the per-channel scaler.

    Steps
    -----
    1. Read CSV, optionally truncate to rows with date <= data_end,
       parse the (tau × moneyness) grid, take log of IV values.
    2. Row-level split: first train_frac rows define the scaler-fit domain.
    3. StandardScaler per channel, fit on train rows, applied globally.
    4. Build sliding windows (input=L, target=P) and split each window
       into train/val/test by the position of its *target end*. Train
       windows therefore have every value within the scaler's fit domain
       (no leakage).
    """
    df = pd.read_csv(csv_path)
    if data_end is not None:
        n_before = len(df)
        df = df[df["date"] <= data_end].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows with date <= {data_end}.")
        print(f"  truncated to date <= {data_end}: "
              f"{n_before} → {len(df)} rows  "
              f"(last kept: {df['date'].iloc[-1]})")
    grid = parse_grid(df.columns.tolist())
    raw = df[grid.iv_cols].to_numpy(dtype=np.float64)  # [N, C]
    if not np.all(raw > 0):
        raise ValueError("Non-positive IV values; log is undefined.")
    log_iv = np.log(raw).astype(np.float32)

    N, C = log_iv.shape
    train_end = int(N * train_frac)
    val_end   = int(N * (train_frac + val_frac))

    mean  = log_iv[:train_end].mean(axis=0)            # [C]
    std   = log_iv[:train_end].std(axis=0) + 1e-12     # [C]
    scaled = (log_iv - mean) / std                     # [N, C]

    L, P = lookback, pred_len
    n_win = N - L - P + 1
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({N}) for L={L}, P={P}.")
    starts = np.arange(n_win)
    target_end = starts + L + P    # exclusive

    train_mask = target_end <= train_end
    val_mask   = (target_end > train_end) & (target_end <= val_end)
    test_mask  = target_end > val_end

    def stack(idx):
        X = np.stack([scaled[s : s + L]         for s in idx], axis=0)
        Y = np.stack([scaled[s + L : s + L + P] for s in idx], axis=0)
        return X, Y

    Xtr, Ytr = stack(starts[train_mask])
    Xva, Yva = stack(starts[val_mask])
    Xte, Yte = stack(starts[test_mask])

    dates = df["date"].tolist()
    test_starts = starts[test_mask]
    test_first_target_date = (dates[int(test_starts[0]) + L]
                              if test_starts.size else None)

    return {
        "grid":   grid,
        "scaler": {"mean": mean.astype(np.float32),
                   "std":  std.astype(np.float32)},
        "rows":   {"N": N, "train_end": train_end, "val_end": val_end,
                   "n_channels": C},
        "data_end": data_end,
        "first_date": str(df["date"].iloc[0]),
        "last_date":  str(df["date"].iloc[-1]),
        "test_first_target_date": test_first_target_date,
        "train":  (Xtr, Ytr),
        "val":    (Xva, Yva),
        "test":   (Xte, Yte),
        # Per-row dates (after data_end filtering), length N. Used by
        # the post-training per-regime breakdown so we don't need to
        # re-read the CSV; also lets eval scripts share the same source.
        "dates_iso": [str(d) for d in dates],
        "scaled_log_iv": scaled,  # full series for VAR
    }


# ─── Model adapters ───────────────────────────────────────────────────────
# Each adapter wraps a model so that its forward sees its native input
# shape, and outputs are reshaped back to [B, P, C] (the trainer's space).

class _Adapter(nn.Module):
    """Shared interface: input/output in [B, T, C] (the canonical space)."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _DLinearAdapter(_Adapter):
    def forward(self, x):           # [B, L, C] → [B, P, C]
        return self.model(x)


class _PatchTSTAdapter(_Adapter):
    def forward(self, x):           # [B, L, C] → [B, P, C]
        return self.model(x)


class _HOTAdapter(_Adapter):
    def __init__(self, model, n_tau, n_money):
        super().__init__(model)
        self.H, self.W = n_tau, n_money

    def forward(self, x):
        # x: [B, L, C=H*W] → [B, H, W, L] → model → [B, H, W, P] → [B, P, C]
        B, L, C = x.shape
        z = x.reshape(B, L, self.H, self.W).permute(0, 2, 3, 1).contiguous()
        z = self.model(z)
        return z.permute(0, 3, 1, 2).reshape(B, -1, C)


class _TuckerAdapter(_Adapter):
    def __init__(self, model, n_tau, n_money):
        super().__init__(model)
        # Tucker takes [B, L, W, H]; we have [B, L, H, W] from the reshape.
        self.H, self.W = n_tau, n_money

    def forward(self, x):
        # x: [B, L, C=H*W] → [B, L, H, W] → permute to [B, L, W, H]
        B, L, C = x.shape
        z = x.reshape(B, L, self.H, self.W).permute(0, 1, 3, 2).contiguous()
        z = self.model(z)        # [B, P, W, H]
        return z.permute(0, 1, 3, 2).reshape(B, -1, C)


class _GWNAdapter(_Adapter):
    def forward(self, x):
        # x: [B, L, C] → [B, 1, C, L] → model → [B, P, C, 1] → [B, P, C]
        B, L, C = x.shape
        z = x.permute(0, 2, 1).unsqueeze(1)
        z = self.model(z)
        return z.squeeze(-1)


class _PCAFormerAdapter(_Adapter):
    def __init__(self, model, n_tau, n_money):
        super().__init__(model)
        # PCAFormer takes [B, L, W, H]; same convention as TuckerDLinear.
        self.H, self.W = n_tau, n_money

    def forward(self, x):
        # x: [B, L, C=H*W] → [B, L, H, W] → permute to [B, L, W, H]
        B, L, C = x.shape
        z = x.reshape(B, L, self.H, self.W).permute(0, 1, 3, 2).contiguous()
        z = self.model(z)        # [B, P, W, H]
        return z.permute(0, 1, 3, 2).reshape(B, -1, C)


class _ITransformerAdapter(_Adapter):
    def __init__(self, model, n_tau, n_money):
        super().__init__(model)
        # ITransformer takes [B, L, W, H]; same shape convention as
        # PCAFormer / TuckerDLinear, so the reshape is identical.
        self.H, self.W = n_tau, n_money

    def forward(self, x):
        # x: [B, L, C=H*W] → [B, L, H, W] → permute to [B, L, W, H]
        B, L, C = x.shape
        z = x.reshape(B, L, self.H, self.W).permute(0, 1, 3, 2).contiguous()
        z = self.model(z)        # [B, P, W, H]
        return z.permute(0, 1, 3, 2).reshape(B, -1, C)


class _ConvLSTMAdapter(_Adapter):
    def __init__(self, model, n_tau, n_money):
        super().__init__(model)
        # ConvLSTM takes [B, L, W, H]; same shape convention as
        # iTransformer / PCAFormer / TuckerDLinear.
        self.H, self.W = n_tau, n_money

    def forward(self, x):
        # x: [B, L, C=H*W] → [B, L, H, W] → permute to [B, L, W, H]
        B, L, C = x.shape
        z = x.reshape(B, L, self.H, self.W).permute(0, 1, 3, 2).contiguous()
        z = self.model(z)        # [B, P, W, H]
        return z.permute(0, 1, 3, 2).reshape(B, -1, C)


class _SANTAAdapter(_Adapter):
    """Adapter for SANTA-family models (SANTA, SANTA-Flat ablation, …).

    Wraps any model whose forward signature is the SANTA contract:
    `(B, L, M, T)` standardised log-IV in, `netDelta (B, Hh, M, T)` out.

    The native model takes `(B, L, M, T)` standardised log-IV with
    M=moneyness, T=maturity, and emits `netDelta (B, Hh, M, T)` — the
    residual added to today's surface. The trainer here works in
    `(B, L, C=H*W)` with C laid out tau-outer / moneyness-inner
    (parse_grid convention). Forward path:
        x [B,L,C] -> [B,L,H=n_tau,W=n_money] -> permute -> [B,L,M=W,T=H]
        model -> netDelta [B, Hh, M, T]
        zhat = z_today + netDelta            (level in standardised log-IV)
        zhat [B,Hh,M,T] -> permute -> [B,Hh,T,M] -> reshape -> [B,Hh,C]
    Hh equals pred_len because SANTA is configured with
    horizons=(1, …, pred_len) in build_model, so the output time axis
    matches the trainer's Y axis exactly and MSE on (zhat, y) reduces
    to surface_loss(netDelta, z_today, z_future) with uniform gamma.
    """
    def __init__(self, model, n_tau, n_money):
        super().__init__(model)
        self.H, self.W = n_tau, n_money

    def forward(self, x):
        B, L, C = x.shape
        z = (x.reshape(B, L, self.H, self.W)
              .permute(0, 1, 3, 2)
              .contiguous())                            # [B, L, M, T]
        netDelta = self.model(z)                        # [B, Hh=P, M, T]
        z_today  = z[:, -1, :, :]                       # [B, M, T]
        zhat     = z_today.unsqueeze(1) + netDelta      # [B, P, M, T]
        return (zhat.permute(0, 1, 3, 2)
                    .contiguous()
                    .reshape(B, -1, C))                 # [B, P, C]


def build_adapter_from_kwargs(name: str, model_kwargs: dict,
                              n_tau: int, n_money: int) -> nn.Module:
    """Construct an adapter-wrapped model from explicit model_kwargs
    (no defaults applied). Used by --from_winner to reproduce a tuning
    combo exactly. Mirrors evaluate.rebuild_adapter."""
    if name == "dlinear":
        return _DLinearAdapter(DLinear(**model_kwargs))
    if name == "patchtst":
        return _PatchTSTAdapter(PatchTST(**model_kwargs))
    if name == "hot":
        return _HOTAdapter(HOT(**model_kwargs), n_tau, n_money)
    if name == "tucker_dlinear":
        return _TuckerAdapter(TuckerDLinear(**model_kwargs), n_tau, n_money)
    if name == "gwn":
        return _GWNAdapter(GWN(**model_kwargs))
    if name == "pcaformer":
        return _PCAFormerAdapter(PCAFormer(**model_kwargs), n_tau, n_money)
    if name == "itransformer":
        return _ITransformerAdapter(ITransformer(**model_kwargs), n_tau, n_money)
    if name == "convlstm":
        return _ConvLSTMAdapter(ConvLSTM(**model_kwargs), n_tau, n_money)
    if name == "santa":
        # model_kwargs holds the saved SANTAConfig as a plain dict
        # (tuples become lists in JSON; Config is fine with either).
        cfg = SANTAConfig(**model_kwargs)
        return _SANTAAdapter(SANTA(cfg), n_tau, n_money)
    if name == "santa_flat":
        # SANTA-Flat shares Config + adapter with SANTA — only the inner
        # backbone differs (joint M·T spatial block vs SANTA's factored
        # A+B). Same I/O shape so _SANTAAdapter wraps it unchanged.
        cfg = SANTAConfig(**model_kwargs)
        return _SANTAAdapter(SANTAFlat(cfg), n_tau, n_money)
    if name == "santa_temporal":
        # SANTA-Temporal: same Config + adapter as SANTA, only the inner
        # backbone differs (no spatial blocks — temporal-only per-cell).
        cfg = SANTAConfig(**model_kwargs)
        return _SANTAAdapter(SANTATemporal(cfg), n_tau, n_money)
    raise ValueError(f"Unknown model: {name}")


def load_winner_config(name: str, pred_len: int) -> dict:
    """Read <ModelDir>/tuning_results/<lookback>_<pred_len>/summary.json,
    locate the winner combo, and return its config.json dict augmented
    with `_winner_combo` and `_winner_source` (path relative to ROOT)."""
    summary_path = os.path.join(
        ROOT, MODEL_DIR[name], "tuning_results",
        f"{LOOKBACK}_{pred_len}", "summary.json",
    )
    if not os.path.isfile(summary_path):
        raise SystemExit(
            f"--from_winner: no tuning summary at "
            f"{os.path.relpath(summary_path, ROOT)} "
            f"(run hyperparameter_tuning.py first).")
    with open(summary_path) as f:
        summary = json.load(f)
    winner = summary.get("winner")
    if not winner:
        raise SystemExit(
            f"--from_winner: no winner in {os.path.relpath(summary_path, ROOT)}")
    cfg_path = os.path.join(ROOT, winner["config_path"])
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["_winner_combo"]  = winner["combo_id"]
    cfg["_winner_source"] = os.path.relpath(cfg_path, ROOT)
    return cfg


def build_model(name: str, pred_len: int, n_channels: int,
                n_tau: int, n_money: int,
                tau_vals: list | None = None,
                money_vals: list | None = None) -> tuple[nn.Module, dict]:
    """Construct a model using its own __init__ defaults. Returns
    (adapter, resolved_kwargs).

    `tau_vals` / `money_vals` are the actual grid coordinates (length
    n_tau / n_money respectively) read from the CSV by parse_grid. They
    are only needed by models whose embeddings/positional encodings live
    in coordinate space (currently only santa, which feeds them to
    its CoordinateEmbedding for moneyness and sqrt(tau)). All other
    models work off axis sizes alone and ignore these args."""
    L, P, C = LOOKBACK, pred_len, n_channels
    if name == "dlinear":
        # Matches the prior tuning winner (combo_0044): kernel_size=31.
        kw = dict(seq_len=L, pred_len=P, n_channels=C, kernel_size=31)
        m = DLinear(**kw)
        return _DLinearAdapter(m), {**kw, "revin": True,
                                    "revin_affine": True, "revin_eps": 1e-5}
    if name == "patchtst":
        # Matches the prior tuning winner (combo_0001): patch_len=stride=7,
        # n_heads=2, d_ff=64, dropout=0.1, head_dropout=0.01, revin=False.
        # `decomposition` enables the DLinear-style trend/residual split
        # (two independent backbones + heads, summed); `kernel_size` is
        # the moving-average kernel used only when decomposition=True
        # (must be odd).
        kw = dict(
            c_in=C, seq_len=L, pred_len=P,
            patch_len=7, stride=7, d_model=32, n_heads=8,
            n_layers=2, d_ff=64, attn_dropout=0.0, dropout=0.2,
            head_dropout=0.05, revin=True, padding_patch="end",
            decomposition=False, kernel_size=31,
        )
        m = PatchTST(**kw)
        return _PatchTSTAdapter(m), kw
    if name == "hot":
        # Typical small HOT that lives in the current tuning grid (one
        # representative point per axis: middle-of-grid). Lets us smoke-
        # test HOT locally on MPS at a fast size before kicking off the
        # full sweep.
        kw = dict(
            context_length=L, prediction_length=P,
            d_hidden=64, n_blocks=1, n_head=8, patch_size=7,
            attention_type="kronecker_sum",
            dropout=0.2, attn_dropout=0.0, head_dropout=0.0,
            pe="rope", norm=True, head_type="flatten",
        )
        m = HOT(**kw)
        return _HOTAdapter(m, n_tau, n_money), kw
    if name == "tucker_dlinear":
        # Two-branch (trend + seasonal) DLinear with Tucker-decomposed
        # weights. Trend uses full spatial rank (W × H, default in
        # TuckerDLinear) so the spatial pathway is identity at init.
        # Seasonal uses a low-rank spatial bias (rank_W=6, rank_H=4)
        # since the high-frequency residual lives mostly in the dominant
        # surface modes (level/slope/skew/butterfly).
        # Mean init: at step 0 each branch outputs the per-cell lookback
        # mean of its band, broadcast across the horizon — same starting
        # point as DLinear's trend init.
        kw = dict(
            seq_len=L, pred_len=P, W=n_money, H=n_tau,
            rank_L_trend=1,    rank_P_trend=min(2, P),
            rank_W_trend=10, rank_H_trend=7,
            rank_L_seasonal=4, rank_P_seasonal=min(1, P),
            rank_W_seasonal=3, rank_H_seasonal=3,
            kernel_trend=41,
            g_init_noise=1e-3,
        )
        m = TuckerDLinear(**kw)
        return _TuckerAdapter(m, n_tau, n_money), kw
    if name == "gwn":
        # Tuning winner at h=21 (GWN/tuning_results/63_21, combo_0034):
        # 8/8/8/16 channels, blocks=4, layers=1 → 11,369 params,
        # val_loss ≈ 0.155. This is the config the published baseline
        # in headline.csv was trained with. Was previously 16/16/16/32
        # with blocks=2 (~37k params) — tuning preferred a deeper,
        # narrower model.
        kw = dict(
            num_nodes=C, seq_len=L, pred_len=P,
            in_dim=1, supports=None,
            gcn_bool=True, addaptadj=False, aptinit=None,
            residual_channels=8, dilation_channels=8,
            skip_channels=8, end_channels=16,
            kernel_size=2, blocks=4, layers=1,
            dropout=0.3,
        )
        m = GWN(**kw)
        return _GWNAdapter(m), kw
    if name == "pcaformer":
        # PCAFormer hand-tuned to a small footprint: 3 PCs (top-3
        # explain >95% of IV-surface variance), d_model=8, 1 encoder
        # layer, 4 heads, FFN=32, dropout=0.3, RevIN no-affine. Frozen
        # PCA basis is fit by train_deep_model via inner.fit_pca(Xtr)
        # before the first epoch.
        kw = dict(
            seq_len=L, pred_len=P, W=n_money, H=n_tau,
            n_factors=3, d_model=4, n_heads=4,
            n_layers=1, d_ff=16, dropout=0.3,
            revin_affine=False,
        )
        m = PCAFormer(**kw)
        return _PCAFormerAdapter(m, n_tau, n_money), kw
    if name == "itransformer":
        # iTransformer at the conservative defaults from the model file:
        # d_model=64, n_blocks=2, n_heads=4, ffn_ratio=2, dropout=0.1,
        # head_init_scale=0.001. No normalisation layer, no factor
        # bottleneck — full cross-cell self-attention with per-cell
        # time handling. Sized for ~1.2k training windows.
        kw = dict(
            seq_len=L, pred_len=P, W=n_money, H=n_tau,
            d_model=64, n_blocks=1, n_heads=8,
            ffn_ratio=4, dropout=0.3, head_init_scale=0.001,
        )
        m = ITransformer(**kw)
        return _ITransformerAdapter(m, n_tau, n_money), kw
    if name == "convlstm":
        # ConvLSTM of Medvedev & Wang (2022) at the paper's recipe:
        # 2 stacked ConvLSTM layers (16 then 8 kernels, 4×4 then 3×3),
        # average-pool 2×2 after each, 0.25 dropout, flatten+dense head.
        # W=n_money, H=n_tau; with the 15×10 grid the two pools take it
        # 15×10 → 7×5 → 3×2 before the dense head.
        # revin: optional per-cell RevIN (off by default to match the
        # paper's outside-the-model min-max scaling). Flip to True for
        # the RevIN-on ablation; matches the per-cell norm used by HOT /
        # PatchTST / iTransformer in this project.
        kw = dict(
            seq_len=L, pred_len=P, W=n_money, H=n_tau,
            hidden_channels=(16, 8), kernel_sizes=(4, 3),
            pool=2, dropout=0.25,
            revin=True,
        )
        m = ConvLSTM(**kw)
        return _ConvLSTMAdapter(m, n_tau, n_money), kw
    if name == "santa":
        # SANTA — Surface-Aware Neural Tensor Attention (SANTA/santa.py).
        # The model is grid-aware: it embeds the CONTINUOUS moneyness
        # coordinate (k) and √τ via an MLP, so the actual CSV grid values
        # must be passed in. M is the moneyness axis (15) and T the
        # maturity axis (10) — matching the (B, L, M, T) layout the
        # adapter feeds in. horizons is set to (1, …, P) so the head
        # emits one prediction per trainer-side target step. Hyperparams
        # below (d=32, n_heads=4, n_layers=2, d_ff_mult=1,
        # d_head_hidden=24, dropout=0.1) put the model at ~44k params at
        # L=63 — n_layers=2 is fixed across SANTA / SANTAFlat /
        # SANTATemporal so the budget is spent on depth, with `d` the
        # only knob that varies between variants to compensate for
        # 3/2/1 SubBlocks per layer (so all three land within ~44–51k).
        # The santa.py file's own Config defaults are different
        # (d=48, n_layers=2, ~120k params).
        if tau_vals is None or money_vals is None:
            raise ValueError("santa needs tau_vals and money_vals "
                             "from the parsed grid.")
        cfg = SANTAConfig(
            M=n_money, T=n_tau, L=L,
            horizons=tuple(range(1, P + 1)),
            d=32, n_heads=4, n_layers=2, d_ff_mult=1,
            d_head_hidden=24, dropout=0.1,
            k_grid=tuple(float(v) for v in money_vals),
            tau_grid_years=tuple(float(v) for v in tau_vals),
            centre_on="last",
        )
        m = SANTA(cfg)
        # Persist the resolved config as a plain dict so
        # build_adapter_from_kwargs can rebuild this exact model.
        kw = {
            "M": cfg.M, "T": cfg.T, "L": cfg.L,
            "horizons": list(cfg.horizons),
            "d": cfg.d, "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers, "d_ff_mult": cfg.d_ff_mult,
            "d_head_hidden": cfg.d_head_hidden, "dropout": cfg.dropout,
            "k_grid": list(cfg.k_grid),
            "tau_grid_years": list(cfg.tau_grid_years),
            "centre_on": cfg.centre_on,
        }
        return _SANTAAdapter(m, n_tau, n_money), kw
    if name == "santa_flat":
        # SANTA-Flat ablation: same Config shape as the santa branch
        # above (n_heads=4, n_layers=2, d_ff_mult=1, d_head_hidden=24,
        # dropout=0.1) — only `d` differs to compensate for having 2
        # SubBlocks per layer (vs SANTA's 3) so the parameter budget
        # stays near 50k. The inner backbone is the only architectural
        # diff (joint M·T spatial block vs SANTA's factored A+B).
        # d=40 lands at ~47k params; divisible by 4 for the heads.
        if tau_vals is None or money_vals is None:
            raise ValueError("santa_flat needs tau_vals and money_vals "
                             "from the parsed grid.")
        cfg = SANTAConfig(
            M=n_money, T=n_tau, L=L,
            horizons=tuple(range(1, P + 1)),
            d=40, n_heads=4, n_layers=2, d_ff_mult=1,
            d_head_hidden=24, dropout=0.1,
            k_grid=tuple(float(v) for v in money_vals),
            tau_grid_years=tuple(float(v) for v in tau_vals),
            centre_on="last",
        )
        m = SANTAFlat(cfg)
        kw = {
            "M": cfg.M, "T": cfg.T, "L": cfg.L,
            "horizons": list(cfg.horizons),
            "d": cfg.d, "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers, "d_ff_mult": cfg.d_ff_mult,
            "d_head_hidden": cfg.d_head_hidden, "dropout": cfg.dropout,
            "k_grid": list(cfg.k_grid),
            "tau_grid_years": list(cfg.tau_grid_years),
            "centre_on": cfg.centre_on,
        }
        return _SANTAAdapter(m, n_tau, n_money), kw
    if name == "santa_temporal":
        # SANTA-Temporal ablation: same Config shape as santa /
        # santa_flat above (n_heads=4, n_layers=2, d_ff_mult=1,
        # d_head_hidden=24, dropout=0.1) — only `d` differs to
        # compensate for having 1 SubBlock per layer (vs SANTA's 3,
        # SANTA-Flat's 2) so the parameter budget stays near 50k. The
        # inner backbone is the only architectural diff (temporal-only,
        # no spatial blocks).
        # d=56 lands at ~50.6k params; divisible by 4 for the heads.
        if tau_vals is None or money_vals is None:
            raise ValueError("santa_temporal needs tau_vals and money_vals "
                             "from the parsed grid.")
        cfg = SANTAConfig(
            M=n_money, T=n_tau, L=L,
            horizons=tuple(range(1, P + 1)),
            d=56, n_heads=4, n_layers=2, d_ff_mult=1,
            d_head_hidden=24, dropout=0.1,
            k_grid=tuple(float(v) for v in money_vals),
            tau_grid_years=tuple(float(v) for v in tau_vals),
            centre_on="last",
        )
        m = SANTATemporal(cfg)
        kw = {
            "M": cfg.M, "T": cfg.T, "L": cfg.L,
            "horizons": list(cfg.horizons),
            "d": cfg.d, "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers, "d_ff_mult": cfg.d_ff_mult,
            "d_head_hidden": cfg.d_head_hidden, "dropout": cfg.dropout,
            "k_grid": list(cfg.k_grid),
            "tau_grid_years": list(cfg.tau_grid_years),
            "centre_on": cfg.centre_on,
        }
        return _SANTAAdapter(m, n_tau, n_money), kw
    raise ValueError(f"Unknown model: {name}")


LR_BY_MODEL = {
    "dlinear":        LR_DLINEAR,
    "patchtst":       LR_PATCHTST,
    "hot":            LR_HOT,
    "tucker_dlinear": LR_TUCKER_DLINEAR,
    "gwn":            LR_GWN,
    "pcaformer":      LR_PCAFORMER,
    "itransformer":   LR_ITRANSFORMER,
    "convlstm":       LR_CONVLSTM,
    "santa":          LR_SANTA,
    "santa_flat":     LR_SANTA_FLAT,
    "santa_temporal": LR_SANTA_TEMPORAL,
}

WD_BY_MODEL = {
    "dlinear":        WD_DLINEAR,
    "patchtst":       WD_PATCHTST,
    "hot":            WD_HOT,
    "tucker_dlinear": WD_TUCKER_DLINEAR,
    "gwn":            WD_GWN,
    "pcaformer":      WD_PCAFORMER,
    "itransformer":   WD_ITRANSFORMER,
    "convlstm":       WD_CONVLSTM,
    "santa":          WD_SANTA,
    "santa_flat":     WD_SANTA_FLAT,
    "santa_temporal": WD_SANTA_TEMPORAL,
}


# ─── Post-training per-regime breakdown ──────────────────────────────────
# Calendar regime buckets, anchored at each window's last-horizon target
# date. Matches eval_full.py's REGIMES exactly so the breakdown printed
# here lines up with the cross-model comparison tables. Kept inline (not
# imported) to avoid a circular import — eval_full.py already imports
# from train.py.
_REGIMES = (
    ("COVID",          "2019-12-02", "2020-12-31"),
    ("Reflation calm", "2021-01-01", "2021-12-31"),
    ("Bear 2022",      "2022-01-01", "2022-12-31"),
    ("Normalisation",  "2023-01-01", "2023-12-29"),
)


def _per_regime_breakdown(preds: np.ndarray, Yte: np.ndarray,
                          Xte: np.ndarray, data: dict, pred_len: int) -> None:
    """Print per-regime test MSE for the trained model alongside two
    baselines (persistence + VAR), grouped by the calendar regime of
    each test window's last target date.

    `preds`/`Yte`: [N_test, pred_len, n_channels] in standardised log-IV.
    `Xte`:        [N_test, lookback, n_channels] (for the persistence baseline).
    `data`:       the dict returned by load_dataset (used for `dates_iso`,
                  `rows`, `data_end`).
    """
    L, P = LOOKBACK, pred_len
    rows = data["rows"]
    N = rows["N"]
    val_end = rows["val_end"]
    n_win = N - L - P + 1
    starts = np.arange(n_win)
    target_end = starts + L + P
    test_starts = starts[target_end > val_end]
    if test_starts.size == 0:
        return
    last_target_idx = test_starts + L + P - 1

    dates_iso = data.get("dates_iso")
    if dates_iso is None:
        print("  per-regime breakdown skipped (data has no dates_iso).")
        return
    dates = pd.to_datetime(np.asarray(dates_iso))
    last_target = dates[last_target_idx]

    # Regime labels per test window.
    labels = np.full(last_target.shape, "unassigned", dtype=object)
    for name, lo, hi in _REGIMES:
        m = (last_target >= pd.Timestamp(lo)) & (last_target <= pd.Timestamp(hi))
        labels[m] = name

    # Persistence: predict the last lookback day for every horizon.
    persistence = np.broadcast_to(Xte[:, -1:, :], (Xte.shape[0], P, Xte.shape[2]))

    # VAR baseline if available — preferred path is the canonical eval
    # output; fall back to the most recent train_var() result.
    var_paths = [
        os.path.join(ROOT, "VAR", "eval",    f"{L}_{P}", "preds.npy"),
        os.path.join(ROOT, "VAR", "results", f"{L}_{P}", "preds.npy"),
    ]
    var_preds = None
    var_src = None
    for p in var_paths:
        if os.path.isfile(p):
            cand = np.load(p)
            if cand.shape == Yte.shape:
                var_preds = cand
                var_src = p
                break

    print("  per-regime test MSE:")
    print(f"    {'regime':<18}{'n':>5}    {'model':>9}  {'persist':>9}  "
          f"{'VAR':>9}    {'m/persist':>9}  {'m/VAR':>6}")
    for name, _, _ in _REGIMES:
        mask = labels == name
        n_r = int(mask.sum())
        if n_r == 0:
            continue
        m_mse = float(((preds[mask] - Yte[mask]) ** 2).mean())
        p_mse = float(((persistence[mask] - Yte[mask]) ** 2).mean())
        m_p   = m_mse / p_mse if p_mse > 0 else float("nan")
        if var_preds is not None:
            v_mse = float(((var_preds[mask] - Yte[mask]) ** 2).mean())
            v_str = f"{v_mse:>9.4f}"
            m_v   = m_mse / v_mse if v_mse > 0 else float("nan")
            m_v_str = f"{m_v:>6.2f}"
        else:
            v_str = f"{'n/a':>9s}"
            m_v_str = f"{'n/a':>6s}"
        print(f"    {name:<18}{n_r:>5}    "
              f"{m_mse:>9.4f}  {p_mse:>9.4f}  {v_str}    "
              f"{m_p:>9.2f}  {m_v_str}")
    if var_preds is None:
        print(f"    (VAR baseline not found at "
              f"VAR/eval/{L}_{P}/preds.npy or VAR/results/{L}_{P}/preds.npy)")
    else:
        print(f"    (VAR baseline from {os.path.relpath(var_src, ROOT)})")


# ─── Training loop ────────────────────────────────────────────────────────

def _iter_batches(X: np.ndarray, Y: np.ndarray, batch: int,
                  shuffle: bool, device: torch.device, generator=None):
    n = X.shape[0]
    idx = (torch.randperm(n, generator=generator).numpy()
           if shuffle else np.arange(n))
    for s in range(0, n, batch):
        sel = idx[s : s + batch]
        xb = torch.from_numpy(X[sel]).to(device)
        yb = torch.from_numpy(Y[sel]).to(device)
        yield xb, yb


def _epoch(model, X, Y, batch, device, optimizer=None, generator=None,
           grad_clip=None,
           ortho_q_fn=None, ortho_q_weight=0.0,
           ortho_L_fn=None, ortho_L_weight=0.0):
    """Run one epoch. Returns (mean_data_loss, mean_ortho_q, mean_ortho_L).

    Optional penalties (no-arg callables returning scalar tensors)
    fold into the training optimisation target:
        total = data_loss
              + ortho_q_weight * ortho_q_fn()
              + ortho_L_weight * ortho_L_fn()
    No model currently registers a penalty — the hooks are kept so the
    signature stays uniform with hyperparameter_tuning.py. The reported
    `mean_data_loss` is the MSE only (never the combined objective) so
    train/val numbers stay comparable across models with and without
    the penalties.
    """
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.MSELoss()
    total, ortho_q_total, ortho_L_total, n = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in _iter_batches(X, Y, batch, shuffle=train,
                                    device=device, generator=generator):
            pred = model(xb)
            data_loss = loss_fn(pred, yb)
            total_loss = data_loss
            if ortho_q_fn is not None:
                ortho_q = ortho_q_fn()
                total_loss = total_loss + ortho_q_weight * ortho_q
                ortho_q_val = float(ortho_q.detach())
            else:
                ortho_q_val = 0.0
            if ortho_L_fn is not None:
                ortho_L = ortho_L_fn()
                total_loss = total_loss + ortho_L_weight * ortho_L
                ortho_L_val = float(ortho_L.detach())
            else:
                ortho_L_val = 0.0
            if train:
                optimizer.zero_grad()
                total_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = xb.shape[0]
            total       += data_loss.item() * bs
            ortho_q_total += ortho_q_val   * bs
            ortho_L_total += ortho_L_val   * bs
            n += bs
    return (total       / max(n, 1),
            ortho_q_total / max(n, 1),
            ortho_L_total / max(n, 1))


def _predict(model, X, batch, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[s : s + batch]).to(device)
            out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0,))


def train_deep_model(name: str, data: dict, pred_len: int,
                     device: torch.device, seed: int,
                     batch_size: int = BATCH_SIZE,
                     winner_cfg: dict | None = None,
                     out_dir: str | None = None):
    """Train one deep model with the shared trainer; save artefacts.

    If `winner_cfg` is provided (the saved config.json from a tuning
    winner), its `model_kwargs`, optimizer choice, `lr`, `weight_decay`,
    `lr_g`, `wd_g`, `grad_clip`, `min_epochs`, and `batch_size` override
    the train.py defaults. Used by multi_seed.py --from_winner.

    `out_dir` overrides the default timestamped output path. eval_seeds.py
    uses this to route outputs directly to the canonical seed slot.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    grid = data["grid"]
    C    = data["rows"]["n_channels"]
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]
    Xte, Yte = data["test"]

    # Resolve model + trainer knobs. winner_cfg, if given, fully drives
    # them so we reproduce the tuning combo exactly (same optimizer,
    # same grad_clip policy, same min_epochs).
    if winner_cfg is None:
        adapter, resolved = build_model(
            name, pred_len, C, grid.n_tau, grid.n_money,
            tau_vals=grid.tau_vals, money_vals=grid.money_vals,
        )
        lr = LR_BY_MODEL[name]
        wd = WD_BY_MODEL[name]
        min_epochs = MIN_EPOCHS
        # AdamW + grad_clip=1.0 for every deep model. AdamW's decoupled
        # weight decay matters for transformers (HOT/PatchTST) and for
        # GWN's gated dilated convs; for DLinear/Tucker it's a no-op
        # while WD is small but lets us add decoupled WD without re-
        # tuning. grad_clip stabilises GWN's noisy-val updates and HOT/
        # PatchTST's attention init. santa is another transformer
        # so it also lands on AdamW.
        use_adamw = name in ("tucker_dlinear", "hot", "patchtst", "dlinear",
                             "pcaformer", "itransformer", "santa",
                             "santa_flat", "santa_temporal")
        grad_clip = 1.0 if use_adamw else None
        if name == "tucker_dlinear":
            # G core gets its own multipliers; factor matrices stay at
            # base lr / wd.
            lr_g = lr * LR_TUCKER_DLINEAR_G_MULT
            wd_g = wd * WD_TUCKER_DLINEAR_G_MULT
        else:
            lr_g = wd_g = None
    else:
        resolved = winner_cfg["model_kwargs"]
        adapter  = build_adapter_from_kwargs(
            name, resolved, grid.n_tau, grid.n_money,
        )
        lr  = float(winner_cfg["lr"])
        wd  = float(winner_cfg["weight_decay"])
        min_epochs = int(winner_cfg["min_epochs"])
        batch_size = int(winner_cfg["batch_size"])
        use_adamw  = (winner_cfg.get("optimizer") == "AdamW")
        gc = winner_cfg.get("grad_clip")
        grad_clip = float(gc) if gc is not None else None
        lr_g = winner_cfg.get("lr_g")
        wd_g = winner_cfg.get("wd_g")
        if lr_g is not None:
            lr_g = float(lr_g)
        if wd_g is not None:
            wd_g = float(wd_g)

    adapter.to(device)
    n_params = sum(p.numel() for p in adapter.parameters())

    # Shared forecast-head bias warm-start (iTransformer).
    #
    # If the underlying model exposes a shared `head` (nn.Linear with
    # bias of shape [P]), set the bias to the per-horizon training
    # mean (averaged over cells). This makes the model's step-0
    # forecast "predict the cross-cell historical mean per horizon" —
    # a coarse but reasonable level baseline that the model then
    # refines, rather than spending early-training capacity learning
    # absolute levels from scratch.
    #
    # Ytr is [N_train, P, C=H*W] with C laid out tau-outer-moneyness-
    # inner (parse_grid convention). Per-horizon mean is just
    # Ytr.mean(axis=0).mean(axis=1) → shape [P].
    inner_model = getattr(adapter, "model", adapter)
    if name == "itransformer":
        # Skip the warm-start when per-cell RevIN is active: under RevIN
        # the model's output is denormalised by adding the input's
        # per-cell lookback mean back, so head.bias=0 already yields a
        # "predict per-cell lookback mean" warm start. Leaving the bias
        # at zero also keeps it interpretable as a learned deviation
        # from RevIN's baseline.
        revin_on = bool(getattr(inner_model, "revin", False))
        head_module = getattr(inner_model, "head", None)
        if (isinstance(head_module, nn.Linear)
                and head_module.bias is not None
                and head_module.out_features == pred_len
                and not revin_on):
            # Ytr: numpy [N, P, C]; mean over N and over cells gives [P].
            per_horizon_mean = Ytr.mean(axis=0).mean(axis=1)         # [P]
            with torch.no_grad():
                head_module.bias.copy_(
                    torch.from_numpy(per_horizon_mean.astype(np.float32)).to(device)
                )
            print(f"  itransformer head.bias warm-started: "
                  f"mean={head_module.bias.mean().item():+.4f}  "
                  f"std={head_module.bias.std().item():.4f}  "
                  f"shape={tuple(head_module.bias.shape)}")
        elif revin_on and isinstance(head_module, nn.Linear):
            print(f"  itransformer revin=True: head.bias kept at zero; "
                  f"per-cell RevIN supplies the level warm-start.")

    # PCAFormer one-shot PCA fit. The model's PCA basis must be set
    # before the first forward pass; we fit on the same training
    # windows the trainer is about to use, in the same RevIN-normalised
    # space the forward will see. Mirrors the adapter's [B,L,C] →
    # [B,L,W,H] reshape so the model fits on the canonical surface
    # layout.
    inner = getattr(adapter, "model", adapter)
    if name == "pcaformer":
        Xtr_t = torch.from_numpy(Xtr).to(device)
        N_w, L_w, C_w = Xtr_t.shape
        surfaces = (Xtr_t
                    .reshape(N_w, L_w, grid.n_tau, grid.n_money)
                    .permute(0, 1, 3, 2)
                    .contiguous())                            # [N, L, W, H]
        inner.fit_pca(surfaces)
        ev = inner.explained_variance_ratio(surfaces).item()
        print(f"  pcaformer PCA basis fit on {N_w} train windows × "
              f"{L_w} timesteps; top-{inner.n_factors} "
              f"explained variance ratio = {ev:.4f}")
        del Xtr_t, surfaces

    # Optional per-model regularisers added to the training loss. No
    # model currently registers a penalty (the old AxialFactor
    # orthogonality hooks were removed with that model); the
    # `ortho_q_fn` / `ortho_L_fn` plumbing is kept null so the
    # `_epoch` signature stays uniform for hyperparameter_tuning.py,
    # which calls the same helper.
    ortho_q_fn = None
    ortho_q_weight = 0.0
    ortho_L_fn = None
    ortho_L_weight = 0.0
    has_ortho = False

    if name == "tucker_dlinear" and lr_g is not None:
        g_params, other_params = [], []
        for pname, p in adapter.named_parameters():
            (g_params if pname.endswith(".G") else other_params).append(p)
        optimizer = torch.optim.AdamW(
            [{"params": other_params, "lr": lr,   "weight_decay": wd},
             {"params": g_params,     "lr": lr_g, "weight_decay": wd_g}],
        )
        opt_name = "AdamW"
    else:
        opt_cls  = torch.optim.AdamW if use_adamw else torch.optim.Adam
        optimizer = opt_cls(adapter.parameters(), lr=lr, weight_decay=wd)
        opt_name = "AdamW" if use_adamw else "Adam"

    if out_dir is None:
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        out_dir = os.path.join(ROOT, MODEL_DIR[name],
                               f"{LOOKBACK}_{pred_len}", ts)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{name}  pred_len={pred_len}  device={device}]")
    print(f"  windows: train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")
    print(f"  params:  {n_params:,}")
    lr_str = f"lr={lr}" + (f" (lr_G={lr_g:.4g})" if lr_g is not None else "")
    wd_str = f"wd={wd}" + (f" (wd_G={wd_g:.4g})" if wd_g is not None else "")
    print(f"  {opt_name}  {lr_str}  {wd_str}  batch={batch_size}  "
          f"epochs<={EPOCHS}  patience={PATIENCE} (min_epochs={min_epochs})"
          f"{f'  grad_clip={grad_clip}' if grad_clip is not None else ''}")
    if winner_cfg is not None:
        print(f"  from_winner: {winner_cfg['_winner_source']} "
              f"(combo={winner_cfg['_winner_combo']})")
    print(f"  out:     {os.path.relpath(out_dir, ROOT)}")

    log_path = os.path.join(out_dir, "train_log.csv")
    log_f    = open(log_path, "w", newline="")
    log_w    = csv.writer(log_f)
    # `ortho_q` and `ortho_L` columns are the mean per-batch penalty
    # values reported by `_epoch` (0 when no penalty is registered —
    # currently the case for every model). Kept for CSV schema
    # stability across the per-model train logs.
    log_w.writerow(["epoch", "train_loss", "val_loss", "lr",
                    "epoch_time_s", "ortho_q", "ortho_L"])

    best_val   = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone()
                  for k, v in adapter.state_dict().items()}
    stop_epoch = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_ortho_q, tr_ortho_L = _epoch(
            adapter, Xtr, Ytr, batch_size, device,
            optimizer=optimizer, generator=gen, grad_clip=grad_clip,
            ortho_q_fn=ortho_q_fn, ortho_q_weight=ortho_q_weight,
            ortho_L_fn=ortho_L_fn, ortho_L_weight=ortho_L_weight,
        )
        va_loss, _, _ = _epoch(adapter, Xva, Yva, batch_size, device,
                               optimizer=None)
        dt = time.time() - t0
        improved = va_loss < best_val
        if improved:
            best_val   = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in adapter.state_dict().items()}

        marker = "  [best]" if improved else ""
        ortho_str = (f"  ortho_q={tr_ortho_q:.6f}  ortho_L={tr_ortho_L:.6f}"
                     if has_ortho else "")
        print(f"  epoch {epoch:3d}/{EPOCHS}  "
              f"train={tr_loss:.6f}  val={va_loss:.6f}{ortho_str}  "
              f"({dt:.1f}s){marker}")
        log_w.writerow([epoch, f"{tr_loss:.8f}", f"{va_loss:.8f}",
                        f"{lr:.8g}", f"{dt:.3f}",
                        f"{tr_ortho_q:.8f}", f"{tr_ortho_L:.8f}"])
        log_f.flush()

        # Early stop only after min_epochs.
        if epoch >= min_epochs and (epoch - best_epoch) >= PATIENCE:
            stop_epoch = epoch
            print(f"  early stop at epoch {epoch} "
                  f"(best val={best_val:.6f} @ epoch {best_epoch})")
            break

    log_f.close()
    if stop_epoch is None:
        stop_epoch = EPOCHS
        print(f"  finished {EPOCHS} epochs (best val={best_val:.6f} "
              f"@ epoch {best_epoch})")

    # Restore best weights for the test pass.
    adapter.load_state_dict(best_state)
    preds_te = _predict(adapter, Xte, batch_size, device)   # [N, P, C]
    np.save(os.path.join(out_dir, "preds.npy"), preds_te.astype(np.float32))
    # Persist the best weights alongside the predictions so downstream
    # analysis (probes, ablations) can load the trained model without
    # retraining. best_state is already a CPU clone (see the val loop).
    torch.save(best_state, os.path.join(out_dir, "best_model.pt"))

    mse = float(np.mean((preds_te - Yte) ** 2))
    mae = float(np.mean(np.abs(preds_te - Yte)))
    rmse = float(np.sqrt(mse))
    stats = {
        "model":        name,
        "pred_len":     pred_len,
        "lookback":     LOOKBACK,
        "n_test":       int(Xte.shape[0]),
        "best_val_mse": float(best_val),
        "best_epoch":   best_epoch,
        "stop_epoch":   stop_epoch,
        "test_mse":     mse,
        "test_rmse":    rmse,
        "test_mae":     mae,
        "space":        "standardized_log_iv",
    }
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(stats, f, indent=2)

    hyper = {
        "model":          name,
        "model_kwargs":   resolved,
        "optimizer":      opt_name,
        "lr":             lr,
        "weight_decay":   wd,
        "lr_g":           lr_g,
        "wd_g":           wd_g,
        "grad_clip":      grad_clip,
        "epochs":         EPOCHS,
        "patience":       PATIENCE,
        "min_epochs":     min_epochs,
        "batch_size":     batch_size,
        "lookback":       LOOKBACK,
        "pred_len":       pred_len,
        "seed":           seed,
        "device":         str(device),
        "n_params":       n_params,
        "winner_source":  (winner_cfg["_winner_source"]
                           if winner_cfg is not None else None),
        "winner_combo":   (winner_cfg["_winner_combo"]
                           if winner_cfg is not None else None),
        "data_end":       data.get("data_end"),
        "first_date":     data.get("first_date"),
        "last_date":      data.get("last_date"),
        "test_first_target_date": data.get("test_first_target_date"),
        "grid": {
            "n_tau":      grid.n_tau,
            "n_money":    grid.n_money,
            "tau_vals":   grid.tau_vals,
            "money_vals": grid.money_vals,
        },
        "scaler": {
            "space":      "log_iv",
            "mean":       data["scaler"]["mean"].tolist(),
            "std":        data["scaler"]["std"].tolist(),
            "channel_order": grid.iv_cols,
        },
    }
    with open(os.path.join(out_dir, "hyperparams.json"), "w") as f:
        json.dump(hyper, f, indent=2)

    print(f"  test: mse={mse:.6f}  rmse={rmse:.6f}  mae={mae:.6f}")
    _per_regime_breakdown(preds_te, Yte, Xte, data, pred_len)
    print(f"  saved → {os.path.relpath(out_dir, ROOT)}\n")


# ─── VAR (Gonçalves–Guidolin two-stage on the 5-coefficient basis) ──────

def train_var(data: dict, pred_len: int, seed: int):
    """Thin wrapper around var.run_var_baseline.

    Stage 1 fits ℓ = β₀ + β₁M + β₂M² + β₃τ + β₄Mτ across the 150 cells
    daily (M = k/√τ, intra-day OLS, no temporal leakage). Stage 2 fits a
    VAR(p) on the 5-dim β series over the TRAIN slice with p chosen by
    BIC; parameters are frozen and used unchanged on val/test. The frozen
    VAR is iterated forward at each test base date, β̂_{t+h} is plugged
    back into the Stage-1 formula to reconstruct ℓ̂ on every cell, and
    the result is restandardised so it can be scored against the same Yte
    the neural models use. Run the per-regime breakdown afterwards on the
    standardised preds for parity with the deep-model rows."""
    Xte, Yte = data["test"]
    out_dir = os.path.join(ROOT, MODEL_DIR["var"], "results",
                           f"{LOOKBACK}_{pred_len}")
    run_var_baseline(data, pred_len, seed=seed,
                     out_dir=out_dir, verbose=True)
    preds = np.load(os.path.join(out_dir, "preds.npy"))
    _per_regime_breakdown(preds, Yte, Xte, data, pred_len)
    print(f"  saved → {os.path.relpath(out_dir, ROOT)}\n")


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train an IV-surface forecaster on SPX_surfaces.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    choices=(*DEEP_MODELS, "var", "all"))
    ap.add_argument("--pred_len", required=True, type=int,
                    choices=VALID_PRED_LEN)
    ap.add_argument("--csv_path",
                    default=os.path.join(ROOT, "SPX_surfaces.csv"),
                    help="Path to the SPX surfaces CSV (iv_{m}_{tau} columns). "
                         "Default is the repo-root SPX_surfaces.csv (15x10 "
                         "grid) — same file eval_seeds.py uses, and the only "
                         "copy that survives a fresh clone since "
                         "_data_prep/data/* is gitignored.")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--data_end",   type=str,   default="2023-12-29",
                    help="Drop CSV rows with date > this (YYYY-MM-DD). "
                         "Pass 'none' to keep all rows.")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--batch_size", type=int,   default=BATCH_SIZE,
                    help=f"Mini-batch size. Default: {BATCH_SIZE} "
                         "(the BATCH_SIZE constant in train.py). "
                         "Ignored when --from_winner is set.")
    ap.add_argument("--from_winner", action="store_true",
                    help="Override train.py defaults with the tuning "
                         "winner's config.json for (--model, --pred_len). "
                         "Reads <ModelDir>/tuning_results/63_<pred_len>/"
                         "summary.json. Not valid with --model var/all.")
    args = ap.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must be < 1 to leave a test split.")

    device = pick_device()
    print(f"device: {device}")
    print(f"loading {os.path.relpath(args.csv_path, ROOT)} ...")
    data_end = None if args.data_end.lower() == "none" else args.data_end
    data = load_dataset(args.csv_path, args.train_frac, args.val_frac,
                        LOOKBACK, args.pred_len, data_end=data_end)
    g, r = data["grid"], data["rows"]
    print(f"  rows: N={r['N']}  train_end={r['train_end']}  "
          f"val_end={r['val_end']}  C={r['n_channels']}  "
          f"grid={g.n_tau}×{g.n_money} (tau×moneyness)")
    print(f"  windows: train={data['train'][0].shape[0]}  "
          f"val={data['val'][0].shape[0]}  test={data['test'][0].shape[0]}")
    print(f"  test starts predicting at: {data['test_first_target_date']}\n")

    if args.from_winner and args.model in ("var", "all"):
        raise SystemExit("--from_winner requires a single deep model "
                         "(not 'var' or 'all').")

    if args.model == "var":
        train_var(data, args.pred_len, args.seed)
        return
    if args.model == "all":
        for name in DEEP_MODELS:
            train_deep_model(name, data, args.pred_len, device, args.seed,
                             batch_size=args.batch_size)
        return

    winner_cfg = (load_winner_config(args.model, args.pred_len)
                  if args.from_winner else None)
    train_deep_model(args.model, data, args.pred_len, device, args.seed,
                     batch_size=args.batch_size, winner_cfg=winner_cfg)


if __name__ == "__main__":
    main()
