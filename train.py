#!/usr/bin/env python3
"""
train.py — train one (or all) IV-surface forecasters on SPX_surfaces.csv.

Models
------
Deep:  santa, santa_flat, santa_temporal, transformer, per_cell_transformer,
       dlinear. All share the surface contract — (B,L,M,T) standardised log-IV
       in, netDelta out (the per-cell change from today) — and the same trainer:
       AdamW, grad-clip 1.0, MSE on standardised log-IV, max EPOCHS, early stop
       with PATIENCE (suppressed until MIN_EPOCHS). Per-model lr / weight_decay
       are the LR_* / WD_* constants below; the architecture and embedding width
       are set in build_model.
Stat:  var (Gonçalves–Guidolin two-stage: daily 5-param cross-sectional OLS on
       the surface basis [1, M, M², τ, Mτ] with M = k/√τ, then a BIC-selected
       VAR on the 5-dim β series — fit on TRAIN only, frozen on val/test). No
       training loop.

`--model all` runs the deep models sequentially. VAR must be invoked explicitly
with `--model var`.

Data
----
Columns of SPX_surfaces.csv shaped iv_{moneyness}_{tau} are parsed into a
[H=tau × W=moneyness] grid (current file: 10 × 11 = 110 cells). IV is taken in
log space; a per-channel StandardScaler is fit on the training rows and applied
to val/test. The train/val/test split is chronological by window-end position
(default 80/10/10).

Output
------
Deep models:  <ModelDir>/63_<pred_len>/<UTC-timestamp>/
                  hyperparams.json, metrics_test.json, train_log.csv,
                  preds.npy  (N_test, pred_len, n_channels)
VAR:          VAR/results/63_<pred_len>/
                  hyperparams.json, metrics_test.json, preds.npy

Preds and stats are in standardised log-IV space (the training space);
hyperparams.json stores the scaler mean/scale so preds can be inverted to IV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Repo root holds the shared framework (surface_core, embeddings); each model
# lives in its own folder. Put the root and the model folders on sys.path so the
# flat `from <model> import ...` imports resolve.
ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in (".", "SANTA", "SANTA_flat", "SANTA_temporal", "transformer",
            "per_cell_transformer", "DLinear", "Linear", "VAR"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from surface_core import Config            # noqa: E402
from santa import SANTA                    # noqa: E402
from santa_flat import SANTAFlat           # noqa: E402
from santa_temporal import SANTATemporal   # noqa: E402
from transformer import VanillaTransformer # noqa: E402
from per_cell_transformer import PerCellTransformer  # noqa: E402
from dlinear import DLinear                # noqa: E402
from linear import LinearForecaster        # noqa: E402
from var import run_var_baseline           # noqa: E402


# ─── User-editable per-model optimiser settings ───────────────────────────
# lr / weight_decay used at training time. Every model trains with AdamW +
# grad_clip=1.0 (decoupled WD; clip stabilises the attention init); all other
# knobs (epochs, patience, min_epochs, batch) are shared, see below.
#
# The SANTA family (SANTA, its two spatial ablations, and the two transformer
# floors) share one recipe — small lr, light decoupled WD — so cross-model
# comparisons isolate architecture, not tuning. DLinear keeps its own faster lr
# and zero WD (a pure linear map needs neither warm-up nor decoupled decay).
LR_DLINEAR              = 4e-3
LR_LINEAR               = 4e-3
LR_SANTA                = 5e-4
LR_SANTA_FLAT           = 5e-4
LR_SANTA_TEMPORAL       = 5e-4
LR_TRANSFORMER          = 5e-4
LR_PER_CELL_TRANSFORMER = 5e-4

WD_DLINEAR              = 0.0
WD_LINEAR               = 0.0

# DLinear moving-average kernel for the trend/seasonal split (odd, ≤ L). Larger =
# smoother trend / higher-frequency seasonal residual. Linear (no decomposition)
# ignores it.
DLINEAR_KERNEL_SIZE = 31
WD_SANTA                = 1e-3
WD_SANTA_FLAT           = 1e-3
WD_SANTA_TEMPORAL       = 1e-3
WD_TRANSFORMER          = 1e-3
WD_PER_CELL_TRANSFORMER = 1e-3

# Shared trainer settings (same for every deep model).
EPOCHS     = 100
PATIENCE   = 15
MIN_EPOCHS = 15
BATCH_SIZE = 64

LOOKBACK   = 63   # fixed across the project
VALID_PRED_LEN = (1, 5, 10, 21, 42, 63)
# Every deep model shares the SANTA forecasting contract (netDelta on the centred
# surface) and the _SANTAAdapter.
DEEP_MODELS    = ("santa", "santa_flat", "santa_temporal",
                  "transformer", "per_cell_transformer", "dlinear", "linear")

# Folder names per model (where outputs land relative to repo root).
MODEL_DIR = {
    "santa":                "SANTA",
    "santa_flat":           "SANTA_flat",
    "santa_temporal":       "SANTA_temporal",
    "transformer":          "transformer",
    "per_cell_transformer": "per_cell_transformer",
    "dlinear":              "DLinear",
    "linear":               "Linear",
    "var":                  "VAR",
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


def split_starts(N: int, L: int, P: int, train_end: int, val_end: int):
    """Window start indices per split under the whole-horizon-within-split rule.

    A window occupies lookback rows [s, s+L) and forecasts target rows
    [s+L, s+L+P). It is assigned to the split whose row range fully contains its
    *target horizon*, so no window ever forecasts a day belonging to another
    split; windows whose horizon straddles a boundary are dropped (a natural
    P-day embargo at each edge). Concretely:

        train: target horizon entirely in [0, train_end)      -> target_end <= train_end
        val:   target horizon entirely in [train_end, val_end) -> first_target >= train_end
                                                                 and target_end <= val_end
        test:  target horizon entirely in [val_end, N)         -> first_target >= val_end

    Train is identical to the old target-end rule (its whole horizon is already
    < train_end, so the model never trains on future). Val/test additionally
    require the FIRST forecast day to be on/after their boundary, which removes
    any backward reach-back into the prior split's days. Lookback (the model
    input) may still span a boundary — that is observed history, not a label,
    and is the standard, leakage-free way to condition a forecast.
    """
    starts = np.arange(N - L - P + 1)
    first_target = starts + L          # first forecast row (today + 1)
    target_end   = starts + L + P      # one past the last forecast row
    train = starts[target_end <= train_end]
    val   = starts[(first_target >= train_end) & (target_end <= val_end)]
    test  = starts[first_target >= val_end]
    return train, val, test


def load_dataset(csv_path: str, train_frac: float, val_frac: float,
                 lookback: int, pred_len: int, data_end: str | None = None):
    """Return windowed train/val/test tensors and the per-channel scaler.

    Steps
    -----
    1. Read CSV, optionally truncate to rows with date <= data_end,
       parse the (tau × moneyness) grid, take log of IV values.
    2. Row-level split: first train_frac rows define the scaler-fit domain.
    3. StandardScaler per channel, fit on train rows, applied globally.
    4. Build sliding windows (input=L, target=P) and assign each to a split via
       `split_starts` (whole forecast horizon within the split; straddling
       windows dropped). No window forecasts a day from another split, and no
       training target lies in the scaler's held-out region — no leakage.
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
    train_s, val_s, test_s = split_starts(N, L, P, train_end, val_end)

    def stack(idx):
        X = np.stack([scaled[s : s + L]         for s in idx], axis=0)
        Y = np.stack([scaled[s + L : s + L + P] for s in idx], axis=0)
        return X, Y

    Xtr, Ytr = stack(train_s)
    Xva, Yva = stack(val_s)
    Xte, Yte = stack(test_s)

    dates = df["date"].tolist()
    test_starts = test_s
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
        # Window start indices of the test split (single source of truth for the
        # split, so the regime breakdown and VAR align with the trained models
        # instead of recomputing the masks and risking drift).
        "test_starts": test_starts,
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


class _SANTAAdapter(_Adapter):
    """Adapter for every model in the family (SANTA, its ablations, the two
    transformer floors, and DLinear) — they all share the same contract:
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


def build_model(name: str, pred_len: int,
                n_tau: int, n_money: int,
                tau_vals: list | None = None,
                money_vals: list | None = None) -> tuple[nn.Module, dict]:
    """Construct an adapter-wrapped model; return (adapter, resolved_kwargs).

    Every model shares the surface contract — (B,L,M,T) in, netDelta out — so all
    are wrapped by _SANTAAdapter and built from one shared Config. The per-model
    differences are the backbone class and the embedding width d, chosen so each
    attention model lands in the same ~44-51k parameter envelope; DLinear is the
    exception (a channel-independent linear map, naturally ~296k, with a
    moving-average kernel_size as its only extra knob). horizons is set to (1,…,P)
    so the output time axis matches the trainer's targets.

    tau_vals / money_vals are the live grid coordinates; only the
    coordinate-embedding models (SANTA and its spatial ablations) consume them, but
    every model records them in its saved Config. The returned kwargs are that
    resolved Config, written to hyperparams.json for the record only.
    """
    L, P = LOOKBACK, pred_len
    if tau_vals is None or money_vals is None:
        raise ValueError("build_model needs tau_vals and money_vals from the "
                         "parsed grid.")

    def make_cfg(d: int) -> Config:
        return Config(
            M=n_money, T=n_tau, L=L,
            horizons=tuple(range(1, P + 1)),
            d=d, n_heads=4, n_layers=2, d_ff_mult=1,
            d_head_hidden=24, dropout=0.1,
            k_grid=tuple(float(v) for v in money_vals),
            tau_grid_years=tuple(float(v) for v in tau_vals),
        )

    def resolved_kw(cfg: Config, **extra) -> dict:
        kw = {
            "M": cfg.M, "T": cfg.T, "L": cfg.L,
            "horizons": list(cfg.horizons),
            "d": cfg.d, "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers, "d_ff_mult": cfg.d_ff_mult,
            "d_head_hidden": cfg.d_head_hidden, "dropout": cfg.dropout,
            "k_grid": list(cfg.k_grid),
            "tau_grid_years": list(cfg.tau_grid_years),
        }
        kw.update(extra)
        return kw

    # Embedding width d per model, picked so each lands in the ~44-51k envelope by
    # compensating for the number of SubBlocks per layer:
    #   SANTA              (3 SubBlocks/layer)  d=32 → 43.8k
    #   SANTA-Flat         (2 SubBlocks/layer)  d=40 → 47.0k
    #   SANTA-Temporal     (1 SubBlock /layer)  d=56 → 50.6k
    #   PerCellTransformer (S-Temporal backbone, no coords) d=56 → 44.0k
    if name == "santa":
        # Factored spatial (A: moneyness, B: maturity) + temporal (C).
        cfg = make_cfg(d=32)
        return _SANTAAdapter(SANTA(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "santa_flat":
        # Joint-spatial ablation: A+B replaced by one block over all M·T cells.
        cfg = make_cfg(d=40)
        return _SANTAAdapter(SANTAFlat(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "santa_temporal":
        # Temporal-only ablation: both spatial blocks removed (widest d to
        # compensate for the missing spatial mixing).
        cfg = make_cfg(d=56)
        return _SANTAAdapter(SANTATemporal(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "per_cell_transformer":
        # SANTA-Temporal's backbone with the coordinate embeddings removed; matched
        # d=56 so the param delta vs SANTA-Temporal is exactly the coord-embed cost.
        cfg = make_cfg(d=56)
        return _SANTAAdapter(PerCellTransformer(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "transformer":
        # Day-token floor. The surface-wide head scales linearly with n_horizons,
        # so d is chosen per pred_len to keep total params in the family envelope;
        # the closed-form fallback solves the ~50k budget for unlisted horizons.
        d_by_P = {1: 48, 10: 24, 21: 16, 42: 8, 63: 8}
        d_t = d_by_P.get(P)
        if d_t is None:
            disc = (410 + 110 * P) ** 2 + 48 * (50_000 - 110 * P)
            d_raw = max(8.0, (-(410 + 110 * P) + math.sqrt(max(disc, 0))) / 24.0)
            d_t = int(round(d_raw / 4.0) * 4)
        cfg = make_cfg(d=d_t)
        return _SANTAAdapter(VanillaTransformer(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "dlinear":
        # Linear floor with decomposition: channel-independent per-cell DLinear
        # (trend+seasonal split → one affine map of the lookback per cell). No
        # attention, no coordinate embeddings — it ignores the attention-only
        # Config fields. Its only extra knob is the moving-average kernel_size.
        cfg = make_cfg(d=16)
        kernel_size = DLINEAR_KERNEL_SIZE
        return (_SANTAAdapter(DLinear(cfg, kernel_size=kernel_size), n_tau, n_money),
                resolved_kw(cfg, kernel_size=kernel_size))
    if name == "linear":
        # The simplest floor: DLinear without the decomposition — one linear map of
        # the centred lookback per cell (NLinear-style, since centring subtracts
        # today). DLinear vs linear isolates the value of the trend/seasonal split.
        cfg = make_cfg(d=16)
        return _SANTAAdapter(LinearForecaster(cfg), n_tau, n_money), resolved_kw(cfg)
    raise ValueError(f"Unknown model: {name}")


LR_BY_MODEL = {
    "santa":                LR_SANTA,
    "santa_flat":           LR_SANTA_FLAT,
    "santa_temporal":       LR_SANTA_TEMPORAL,
    "transformer":          LR_TRANSFORMER,
    "per_cell_transformer": LR_PER_CELL_TRANSFORMER,
    "dlinear":              LR_DLINEAR,
    "linear":               LR_LINEAR,
}

WD_BY_MODEL = {
    "santa":                WD_SANTA,
    "santa_flat":           WD_SANTA_FLAT,
    "santa_temporal":       WD_SANTA_TEMPORAL,
    "transformer":          WD_TRANSFORMER,
    "per_cell_transformer": WD_PER_CELL_TRANSFORMER,
    "dlinear":              WD_DLINEAR,
    "linear":               WD_LINEAR,
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
    test_starts = data["test_starts"]            # same split the models were trained/scored on
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
           grad_clip=None):
    """Run one epoch and return the mean MSE over samples. Trains when an
    optimizer is given (with optional grad-norm clipping); otherwise evaluates
    under no_grad. The MSE on (ẑ, y) equals surface_loss(netDelta, …) with uniform
    horizon weights, because the adapter reconstructs ẑ = z_today + netDelta."""
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in _iter_batches(X, Y, batch, shuffle=train,
                                    device=device, generator=generator):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = xb.shape[0]
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)


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
                     out_dir: str | None = None):
    """Train one deep model with the shared trainer and save artefacts.

    Every model uses the same recipe — AdamW with decoupled weight decay,
    grad-norm clipping at 1.0, and early stopping on val MSE after MIN_EPOCHS —
    so cross-model comparisons isolate architecture, not the training procedure.
    Per-model lr / weight_decay come from LR_BY_MODEL / WD_BY_MODEL. `out_dir`
    overrides the default timestamped path (e.g. to route to a fixed seed slot).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    grid = data["grid"]
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]
    Xte, Yte = data["test"]

    adapter, resolved = build_model(
        name, pred_len, grid.n_tau, grid.n_money,
        tau_vals=grid.tau_vals, money_vals=grid.money_vals,
    )
    lr = LR_BY_MODEL[name]
    wd = WD_BY_MODEL[name]
    grad_clip = 1.0
    adapter.to(device)
    n_params = sum(p.numel() for p in adapter.parameters())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=wd)

    if out_dir is None:
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        out_dir = os.path.join(ROOT, MODEL_DIR[name],
                               f"{LOOKBACK}_{pred_len}", ts)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{name}  pred_len={pred_len}  device={device}]")
    print(f"  windows: train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")
    print(f"  params:  {n_params:,}")
    print(f"  AdamW  lr={lr}  wd={wd}  batch={batch_size}  "
          f"epochs<={EPOCHS}  patience={PATIENCE} (min_epochs={MIN_EPOCHS})  "
          f"grad_clip={grad_clip}")
    print(f"  out:     {os.path.relpath(out_dir, ROOT)}")

    log_path = os.path.join(out_dir, "train_log.csv")
    log_f    = open(log_path, "w", newline="")
    log_w    = csv.writer(log_f)
    log_w.writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_time_s"])

    best_val   = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone()
                  for k, v in adapter.state_dict().items()}
    stop_epoch = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = _epoch(adapter, Xtr, Ytr, batch_size, device,
                         optimizer=optimizer, generator=gen, grad_clip=grad_clip)
        va_loss = _epoch(adapter, Xva, Yva, batch_size, device)
        dt = time.time() - t0
        improved = va_loss < best_val
        if improved:
            best_val   = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in adapter.state_dict().items()}

        marker = "  [best]" if improved else ""
        print(f"  epoch {epoch:3d}/{EPOCHS}  "
              f"train={tr_loss:.6f}  val={va_loss:.6f}  ({dt:.1f}s){marker}")
        log_w.writerow([epoch, f"{tr_loss:.8f}", f"{va_loss:.8f}",
                        f"{lr:.8g}", f"{dt:.3f}"])
        log_f.flush()

        # Early stop only after MIN_EPOCHS.
        if epoch >= MIN_EPOCHS and (epoch - best_epoch) >= PATIENCE:
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
        "optimizer":      "AdamW",
        "lr":             lr,
        "weight_decay":   wd,
        "grad_clip":      grad_clip,
        "epochs":         EPOCHS,
        "patience":       PATIENCE,
        "min_epochs":     MIN_EPOCHS,
        "batch_size":     batch_size,
        "lookback":       LOOKBACK,
        "pred_len":       pred_len,
        "seed":           seed,
        "device":         str(device),
        "n_params":       n_params,
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
                         "Default is the repo-root SPX_surfaces.csv (11×10 grid).")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--data_end",   type=str,   default="2023-12-29",
                    help="Drop CSV rows with date > this (YYYY-MM-DD). "
                         "Pass 'none' to keep all rows.")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--batch_size", type=int,   default=BATCH_SIZE,
                    help=f"Mini-batch size. Default: {BATCH_SIZE}.")
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

    if args.model == "var":
        train_var(data, args.pred_len, args.seed)
        return
    if args.model == "all":
        for name in DEEP_MODELS:
            train_deep_model(name, data, args.pred_len, device, args.seed,
                             batch_size=args.batch_size)
        return

    train_deep_model(args.model, data, args.pred_len, device, args.seed,
                     batch_size=args.batch_size)


if __name__ == "__main__":
    main()
