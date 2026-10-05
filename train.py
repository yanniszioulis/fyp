#!/usr/bin/env python3
"""train.py — train one (or all) IV-surface forecasters on SPX_surfaces.csv.

Models
------
Deep:  santa, santa_flat, santa_temporal, transformer, per_cell_transformer, nlinear.
       All share the surface contract ((B,L,M,T) standardised log-IV in, netDelta out)
       and the same trainer (AdamW, grad-clip 1.0, MSE, early stop on val). Per-model
       lr / weight_decay are the LR_* / WD_* constants below; architecture and width
       are set in build_model.
Stat:  var (Gonçalves–Guidolin two-stage: daily cross-sectional OLS on the basis
       [1, M, M², τ, Mτ] with M = k/√τ, then a BIC-selected VAR on the β series,
       fit on TRAIN only). No training loop.

Usage:  python train.py --model <name|all|var> --pred_len <1|5|10|21|42|63>
        (--model all runs the deep models; VAR is run explicitly with --model var.)

Data:   iv_{moneyness}_{tau} columns → a (tau × moneyness) grid (10 × 11 = 110 cells),
        log-IV, per-channel StandardScaler fit on train rows. Chronological
        train/val/test split by window-end position (70/10/20 by default).

Output: <ModelDir>/eval/63_<pred_len>/seed_<seed>/ for deep models,
        VAR/eval/63_<pred_len>/ for VAR — each with hyperparams.json, metrics_test.json
        (+ best_model.pt, train_log.csv for deep; preds.npy for VAR). Preds/stats are in
        standardised log-IV; hyperparams.json stores the scaler so preds invert to IV.
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# put the repo root + model folders on sys.path so the flat model imports resolve
ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in (".", "SANTA", "SANTA_flat", "SANTA_temporal", "transformer",
            "per_cell_transformer", "NLinear", "VAR"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from surface_core import Config            # noqa: E402
from santa import SANTA                    # noqa: E402
from santa_flat import SANTAFlat           # noqa: E402
from santa_temporal import SANTATemporal   # noqa: E402
from transformer import VanillaTransformer # noqa: E402
from per_cell_transformer import PerCellTransformer  # noqa: E402
from nlinear import NLinearForecaster      # noqa: E402
from var import run_var_baseline           # noqa: E402


# user-editable per-model lr / weight_decay (all AdamW + grad_clip=1.0; SANTA family shares one recipe, NLinear faster lr + zero WD)
LR_NLINEAR              = 4e-3
LR_SANTA                = 5e-4
LR_SANTA_FLAT           = 5e-4
LR_SANTA_TEMPORAL       = 5e-4
LR_TRANSFORMER          = 5e-4
LR_PER_CELL_TRANSFORMER = 5e-4

WD_NLINEAR              = 0.0

WD_SANTA                = 1e-3
WD_SANTA_FLAT           = 1e-3
WD_SANTA_TEMPORAL       = 1e-3
WD_TRANSFORMER          = 1e-3
WD_PER_CELL_TRANSFORMER = 1e-3

# shared trainer settings (same for every deep model)
EPOCHS     = 100
PATIENCE   = 15
MIN_EPOCHS = 15
BATCH_SIZE = 64

LOOKBACK   = 63   # fixed across the project
VALID_PRED_LEN = (1, 5, 10, 21, 42, 63)
# every deep model shares the SANTA forecasting contract (netDelta on the centred surface) and the _SANTAAdapter
DEEP_MODELS    = ("santa", "santa_flat", "santa_temporal",
                  "transformer", "per_cell_transformer", "nlinear")

# folder names per model (where outputs land relative to repo root)
MODEL_DIR = {
    "santa":                "SANTA",
    "santa_flat":           "SANTA_flat",
    "santa_temporal":       "SANTA_temporal",
    "transformer":          "transformer",
    "per_cell_transformer": "per_cell_transformer",
    "nlinear":              "NLinear",
    "var":                  "VAR",
}


# device pick: cuda → mps → cpu

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# data loading

@dataclass
class GridSpec:
    n_tau: int          # H
    n_money: int        # W
    iv_cols: list       # ordered column names matching reshape [H, W]
    tau_vals: list
    money_vals: list


def parse_grid(columns: list[str]) -> GridSpec:
    """Parse `iv_{moneyness}_{tau}` columns into a (tau × moneyness) grid,
    laying iv_cols out tau-outer / moneyness-inner.
    """
    parsed = []
    for c in columns:
        if not c.startswith("iv_"):
            continue
        _, m, t = c.split("_", 2)
        parsed.append((c, float(m), float(t)))
    money_vals = sorted({m for _, m, _ in parsed})
    tau_vals   = sorted({t for _, _, t in parsed})
    # layout iv_cols as tau-outer, moneyness-inner
    lookup = {(m, t): name for name, m, t in parsed}
    iv_cols = [lookup[(m, t)] for t in tau_vals for m in money_vals]
    return GridSpec(
        n_tau=len(tau_vals), n_money=len(money_vals),
        iv_cols=iv_cols, tau_vals=tau_vals, money_vals=money_vals,
    )


def split_starts(N: int, L: int, P: int, train_end: int, val_end: int):
    """Window start indices per split under the whole-horizon-within-split rule.

    A window occupies lookback rows [s, s+L) and forecasts [s+L, s+L+P). It is
    assigned to the split that fully contains its target horizon, so no window
    forecasts a day from another split; windows straddling a boundary are dropped
    (a natural P-day embargo). Lookback may still span a boundary — observed
    history, not a label, so leakage-free.
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

    Reads the CSV (optionally truncated to date <= data_end), parses the grid, takes
    log-IV, fits a per-channel StandardScaler on the first train_frac rows, builds
    sliding windows (input L, target P), and assigns each to a split via split_starts
    (whole horizon within the split; straddling windows dropped — no leakage).
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
        # test-split window starts (single source of truth so VAR aligns with the trained models)
        "test_starts": test_starts,
        # per-row dates (after data_end filtering), length N
        "dates_iso": [str(d) for d in dates],
        "scaled_log_iv": scaled,  # full series for VAR
    }


# model adapters: reshape between the trainer's [B, L, C] and each model's native [B, L, M, T]

class _Adapter(nn.Module):
    """Shared interface: input/output in [B, T, C] (the canonical space)."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _SANTAAdapter(_Adapter):
    """Adapter for every model in the family — all share the contract (B, L, M, T)
    standardised log-IV in, netDelta (B, Hh, M, T) out.

    The trainer works in (B, L, C) with C tau-outer / moneyness-inner (parse_grid
    order); this reshapes to (B, L, M, T), runs the model, adds today's level to get
    ẑ = z_today + netDelta, and reshapes back to (B, P, C). Hh == pred_len (horizons
    (1..P) set in build_model), so MSE on (ẑ, y) == surface_loss with uniform gamma.
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

    Every model shares the surface contract and is wrapped by _SANTAAdapter from one
    Config; the per-model differences are the backbone class and the embedding width d,
    chosen so each attention model lands in the same ~44-51k parameter envelope (NLinear
    is the exception, ~1.3k). horizons = (1..P) so the output axis matches the targets.
    tau_vals / money_vals are the live grid coords (only SANTA + spatial ablations use
    them); all are recorded in the saved Config written to hyperparams.json.
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

    # embedding width d per model, chosen so each attention model lands in the ~44-51k param envelope (NLinear ~1.3k)
    if name == "santa":
        # factored spatial (A: moneyness, B: maturity) + temporal (C)
        cfg = make_cfg(d=32)
        return _SANTAAdapter(SANTA(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "santa_flat":
        # joint-spatial ablation: A+B replaced by one block over all M·T cells
        cfg = make_cfg(d=40)
        return _SANTAAdapter(SANTAFlat(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "santa_temporal":
        # temporal-only ablation: both spatial blocks removed (widest d to compensate)
        cfg = make_cfg(d=56)
        return _SANTAAdapter(SANTATemporal(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "per_cell_transformer":
        # SANTA-Temporal backbone minus coordinate embeddings (d=56 → param delta == coord-embed cost)
        cfg = make_cfg(d=56)
        return _SANTAAdapter(PerCellTransformer(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "transformer":
        # day-token floor; d chosen per pred_len to keep params in the family envelope
        d_by_P = {1: 48, 10: 24, 21: 16, 42: 8, 63: 8}
        d_t = d_by_P.get(P)
        if d_t is None:
            disc = (410 + 110 * P) ** 2 + 48 * (50_000 - 110 * P)
            d_raw = max(8.0, (-(410 + 110 * P) + math.sqrt(max(disc, 0))) / 24.0)
            d_t = int(round(d_raw / 4.0) * 4)
        cfg = make_cfg(d=d_t)
        return _SANTAAdapter(VanillaTransformer(cfg), n_tau, n_money), resolved_kw(cfg)
    if name == "nlinear":
        # simplest floor: one linear map of the centred lookback, shared across cells (NLinear)
        cfg = make_cfg(d=16)
        return _SANTAAdapter(NLinearForecaster(cfg), n_tau, n_money), resolved_kw(cfg)
    raise ValueError(f"Unknown model: {name}")


LR_BY_MODEL = {
    "santa":                LR_SANTA,
    "santa_flat":           LR_SANTA_FLAT,
    "santa_temporal":       LR_SANTA_TEMPORAL,
    "transformer":          LR_TRANSFORMER,
    "per_cell_transformer": LR_PER_CELL_TRANSFORMER,
    "nlinear":              LR_NLINEAR,
}

WD_BY_MODEL = {
    "santa":                WD_SANTA,
    "santa_flat":           WD_SANTA_FLAT,
    "santa_temporal":       WD_SANTA_TEMPORAL,
    "transformer":          WD_TRANSFORMER,
    "per_cell_transformer": WD_PER_CELL_TRANSFORMER,
    "nlinear":              WD_NLINEAR,
}


# training loop

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
    """Run one epoch and return mean MSE. Trains when an optimizer is given (with
    optional grad-norm clipping); otherwise evaluates under no_grad. MSE on (ẑ, y)
    equals surface_loss(netDelta, …) since the adapter reconstructs ẑ = z_today + netDelta.
    """
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

    Same recipe for every model — AdamW with decoupled weight decay, grad-norm clip 1.0,
    early stop on val MSE after MIN_EPOCHS — so comparisons isolate architecture, not the
    trainer. Per-model lr / weight_decay from LR_BY_MODEL / WD_BY_MODEL; out_dir overrides
    the default eval/<L>_<P>/seed_<seed>/ path.
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
        out_dir = os.path.join(ROOT, MODEL_DIR[name], "eval",
                               f"{LOOKBACK}_{pred_len}", f"seed_{seed}")
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

        # early stop only after MIN_EPOCHS
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

    # restore best weights for the test pass
    adapter.load_state_dict(best_state)
    preds_te = _predict(adapter, Xte, batch_size, device)   # [N, P, C]
    np.save(os.path.join(out_dir, "preds.npy"), preds_te.astype(np.float32))
    # persist best weights so they can be reloaded without retraining (best_state is a cpu clone)
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
    print(f"  saved → {os.path.relpath(out_dir, ROOT)}\n")


# VAR (Gonçalves–Guidolin two-stage on the 5-coefficient basis)

def train_var(data: dict, pred_len: int, seed: int):
    """Thin wrapper around var.run_var_baseline.

    Stage 1 fits ℓ = β₀ + β₁M + β₂M² + β₃τ + β₄Mτ daily across the 110 cells
    (M = k/√τ, intra-day OLS); stage 2 fits a BIC-selected VAR(p) on the 5-dim β
    series over the TRAIN slice, frozen on val/test. The VAR is iterated forward at
    each test base date and plugged back into the stage-1 formula to reconstruct ℓ̂,
    then restandardised to score against the same Yte the neural models use.
    """
    out_dir = os.path.join(ROOT, MODEL_DIR["var"], "eval",
                           f"{LOOKBACK}_{pred_len}")
    run_var_baseline(data, pred_len, seed=seed,
                     out_dir=out_dir, verbose=True)
    print(f"  saved → {os.path.relpath(out_dir, ROOT)}\n")


# entry point

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
