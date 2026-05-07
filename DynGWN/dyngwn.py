#!/usr/bin/env python3
"""
DynGWN (Graph-WaveNet) training script for SPX IV surface forecasting.

All model code is inlined — no dependency on the legacy DynGWN/ pipeline files.

Architecture:
  - WaveNet-style dilated temporal convolutions (blocks=4, layers=2, kernel_size=2)
  - Graph convolution at each WaveNet step using one or two supports:
        graph_mode='grid_plus_adaptive'  (default):
            [4-neighbor (tau, delta) grid, learned adaptive (nodevec1 @ nodevec2)]
        graph_mode='adaptive_only':
            [learned adaptive only]

Grid layout (derived from CSV `iv_*` column names at load time):
    Columns are `iv_{moneyness}_{tau}`, ordered (tau OUTER, moneyness INNER):
    col k = i_t · W + i_m = (moneyness[i_m], tau[i_t]).
    H = h_tau        (OUTER axis: number of distinct maturities)
    W = w_moneyness  (INNER axis: number of distinct log-moneyness slices)
    nid(r, c) = r·W + c = i_t·W + i_m = csv_col(i_t, i_m)        ← matches CSV order
    Row-neighbour edge ↔ adjacent maturity at same moneyness;
    Column-neighbour edge ↔ adjacent moneyness at same maturity.
The current dataset is 8×8 = 64 cells (m in [-0.20, 0.20], tau in [0.04, 1.0]);
earlier 20×20 = 400 and 10×17 = 170-cell layouts also fit this convention
(same `nid = r·W + c`). For non-square grids the (outer, inner) ordering
matters — we verify it explicitly at load time.

Loss menu (`--loss`):
    mse              (default)  — MSE in scaled space.
    huber_scaled                — Huber in scaled space (δ in stdev units).
    mae_original    (legacy)    — masked-MAE on inverse-transformed pred vs y in
                                  ORIGINAL IV space. Up-weights high-vol cells.
                                  Reproduces legacy DynGWN/engine.py.
    huber_original  (legacy)    — Huber on inverse-transformed pred vs y in
                                  ORIGINAL IV space (δ in vol-points).

Inputs are standardised with a single global (mean, std) fitted on the training
portion of the target-space data — pooled across time and all IV cells, not
per-column. The model trains in scaled space; predictions are inverse-transformed
and saved in original target-space units.

Dataset selection:
    --dataset full      use the full CSV (default).
    --dataset precovid  slice to date <= 2019-12-31 before splitting.

Outputs (in --out_dir; default
`DynGWN/results/{dataset}_SPX_IV_{seq_len}_{pred_len}_DynGWN_{graph_mode}_nh{nh}_b{b}_l{l}_ep{ep}{loss_suffix}`):
    pred.npy          [N_test, pred_len, n_iv]   original-space predictions  (gitignored)
    start_dates.npy   [N_test]                   datetime64[D] start of each window
    train_log.csv     epoch, train_loss, val_loss
    best_model.pt     checkpoint of best validation weights
    config.json       full hyperparam + split + git record
"""

import argparse
import csv
import json
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FRAC           = 0.70
TEST_FRAC            = 0.20
PRECOVID_END         = "2019-12-31"
DATASET_CHOICES      = ["full", "precovid"]
TARGET_SPACE_CHOICES = ["level", "logdiff"]
# Grid dims (h_tau OUTER, w_moneyness INNER) are derived from CSV columns at
# load time — see _parse_grid_dims below.


# ─── Loss (legacy masked_mae from DynGWN/util.py) ─────────────────────────────

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val=float("nan")) -> torch.Tensor:
    """Mean absolute error with masking for null/NaN labels (legacy semantics)."""
    if null_val != null_val:  # NaN
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


# ─── Static grid adjacency ────────────────────────────────────────────────────

def _make_grid_adjacency(h: int, w: int, self_loops: bool = True) -> np.ndarray:
    """
    4-neighbor adjacency over a flattened H×W grid with `nid(r, c) = r·w + c`.

    With h = number of taus (OUTER) and w = number of moneyness values (INNER),
    node indices match CSV column ordering `csv_col(i_t, i_m) = i_t·w + i_m`.
    Row neighbours = adjacent maturities (same moneyness); column neighbours =
    adjacent moneyness (same maturity). Both are real surface-smoothness
    relationships.
    """
    n = h * w
    A = np.zeros((n, n), dtype=np.float32)
    nid = lambda r, c: r * w + c
    for r in range(h):
        for c in range(w):
            i = nid(r, c)
            if self_loops:
                A[i, i] = 1.0
            if c - 1 >= 0:    A[i, nid(r, c - 1)] = 1.0
            if c + 1 < w:     A[i, nid(r, c + 1)] = 1.0
            if r - 1 >= 0:    A[i, nid(r - 1, c)] = 1.0
            if r + 1 < h:     A[i, nid(r + 1, c)] = 1.0
    return A


def _row_normalize(A: np.ndarray) -> np.ndarray:
    """Row-stochastic (D^-1 A) — matches the einsum 'ncvl,vw->ncwl' message-passing form."""
    rowsum = A.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return (A / rowsum).astype(np.float32)


# ─── Graph WaveNet Model ──────────────────────────────────────────────────────

class _nconv(nn.Module):
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncvl,vw->ncwl", (x, A)).contiguous()


class _linear(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _gcn(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float,
                 support_len: int = 1, order: int = 2):
        super().__init__()
        self.nconv = _nconv()
        self.mlp   = _linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order   = order

    def forward(self, x: torch.Tensor, support: list) -> torch.Tensor:
        out = [x]
        for a in support:
            a  = a.to(x.device)
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class DynGWN(nn.Module):
    """
    WaveNet + adaptive graph convolution.

    Input:  [B, in_dim=1, num_nodes, seq_len]  (padded +1 inside forward)
    Output: [B, pred_len, num_nodes, 1]        (in scaled space)
    """
    def __init__(self, num_nodes: int, dropout: float = 0.3,
                 in_dim: int = 1, seq_len: int = 21, pred_len: int = 63,
                 nhid: int = 32, kernel_size: int = 2,
                 blocks: int = 4, layers: int = 2,
                 static_supports: list = None):
        super().__init__()
        self.blocks    = blocks
        self.layers    = layers
        self.num_nodes = num_nodes

        skip_channels = nhid * 2
        end_channels  = nhid * 4
        order         = 2

        self.static_supports = static_supports or []
        for i, sup in enumerate(self.static_supports):
            self.register_buffer(f"static_support_{i}", sup, persistent=False)
        support_len = len(self.static_supports) + 1   # +1 for adaptive

        self.start_conv = nn.Conv2d(in_dim, nhid, kernel_size=(1, 1))

        # Adaptive adjacency node vectors (rank-10 low-rank embedding)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 5))
        self.nodevec2 = nn.Parameter(torch.randn(5, num_nodes))

        self.filter_convs = nn.ModuleList()
        self.gate_convs   = nn.ModuleList()
        self.skip_convs   = nn.ModuleList()
        self.gconv        = nn.ModuleList()

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(nhid, nhid, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(
                    nn.Conv2d(nhid, nhid, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.skip_convs.append(
                    nn.Conv2d(nhid, skip_channels, kernel_size=(1, 1)))

                if (i + 1) * (b + 1) - 1 < blocks * layers - 1:
                    self.gconv.append(
                        _gcn(nhid, nhid, dropout, support_len=support_len, order=order))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.receptive_field = receptive_field
        end_kernel = seq_len + 1 - receptive_field + 1   # +1 for the input pad
        assert end_kernel >= 1, (
            f"Temporal kernel {end_kernel} < 1; reduce blocks/layers or increase seq_len")

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels,
                                    kernel_size=(1, end_kernel), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len,
                                    kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim, nodes, seq_len]
        x = F.pad(x, (1, 0, 0, 0))        # [B, in_dim, nodes, seq_len+1]
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))

        x    = self.start_conv(x)
        skip = 0
        adp  = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        statics = [getattr(self, f"static_support_{i}")
                   for i in range(len(self.static_supports))]
        new_supports = statics + [adp]

        gcn_idx = 0
        for i in range(self.blocks * self.layers):
            residual = x
            f = torch.tanh(self.filter_convs[i](residual))
            g = torch.sigmoid(self.gate_convs[i](residual))
            x = f * g

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            if i < self.blocks * self.layers - 1:
                x = self.gconv[gcn_idx](x, new_supports)
                gcn_idx += 1
                x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)             # [B, pred_len, nodes, 1]
        return x


# ─── Data ─────────────────────────────────────────────────────────────────────

def _slice_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "full":
        return df
    if dataset == "precovid":
        end = pd.Timestamp(PRECOVID_END)
        return df[df["date"] <= end].reset_index(drop=True)
    raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASET_CHOICES}")


def _apply_target_space(iv_raw: np.ndarray, dates_full: np.ndarray, target_space: str):
    """level → unchanged; logdiff → log(IV)[1:]-log(IV)[:-1], dates trimmed by 1."""
    if target_space == "level":
        return iv_raw, dates_full
    if target_space == "logdiff":
        if (iv_raw <= 0).any():
            raise ValueError("logdiff target_space requires all IV > 0")
        log_iv = np.log(iv_raw)
        return (log_iv[1:] - log_iv[:-1]).astype(iv_raw.dtype), dates_full[1:]
    raise ValueError(f"Unknown target_space {target_space!r}; choose from {TARGET_SPACE_CHOICES}")


def _parse_iv_col(col: str) -> tuple[str, str]:
    """`iv_{moneyness}_{tau}` → (moneyness_str, tau_str). String keys avoid
    float-precision pitfalls when verifying grid order."""
    rest = col[len("iv_"):]
    return tuple(rest.rsplit("_", 1))  # (moneyness_str, tau_str)


def _parse_grid_dims(iv_cols: list) -> tuple:
    """
    Derive (h_tau, w_moneyness) from `iv_*` column names and verify that the
    columns form a complete grid laid out as **(tau outer, moneyness inner)**:
    cols[i_t·w + i_m] = (moneyness[i_m], tau[i_t]).

    With h = number of taus and w = number of moneyness values, DynGWN's
    `nid(r, c) = r·w + c` then matches CSV column ordering with r = i_t,
    c = i_m. Returns (h_tau, w_moneyness, moneyness_strs, tau_strs).
    """
    pairs = [_parse_iv_col(c) for c in iv_cols]
    moneyness_seen, tau_seen = [], []
    for m, t in pairs:
        if m not in moneyness_seen:
            moneyness_seen.append(m)
        if t not in tau_seen:
            tau_seen.append(t)
    h, w = len(tau_seen), len(moneyness_seen)
    if h * w != len(iv_cols):
        raise ValueError(
            f"iv_ columns are not a complete grid: {len(iv_cols)} cols, "
            f"{h} unique tau × {w} unique moneyness = {h * w}"
        )
    for i_t in range(h):
        for i_m in range(w):
            k = i_t * w + i_m
            if pairs[k] != (moneyness_seen[i_m], tau_seen[i_t]):
                raise ValueError(
                    f"unexpected column order at index {k}: got {iv_cols[k]!r}, "
                    f"expected (moneyness={moneyness_seen[i_m]}, tau={tau_seen[i_t]}). "
                    f"DynGWN requires (tau outer, moneyness inner)."
                )
    return h, w, moneyness_seen, tau_seen


def load_splits(csv_path: str, dataset: str, target_space: str,
                seq_len: int, pred_len: int):
    """
    Returns BOTH x and y in SCALED space, plus the (mean, std) used to scale them.
    A single global mean/std is fit on the training portion of the target-space
    data (pooled across time and all IV cells, not per-column). The same scalars
    are used to transform the entire series.

    Original-space losses (mae_original, huber_original) denorm internally inside
    `_compute_loss`; main() denormalises pred.npy before saving.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = _slice_dataset(df, dataset)

    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    if not iv_cols:
        raise ValueError(f"No iv_* columns found in {csv_path}")
    n_iv = len(iv_cols)
    h_tau, w_moneyness, _moneyness, _taus = _parse_grid_dims(iv_cols)

    iv_raw     = df[iv_cols].to_numpy(dtype=np.float32)
    dates_full = df["date"].to_numpy(dtype="datetime64[D]")
    data, dates = _apply_target_space(iv_raw, dates_full, target_space)

    T = len(data)
    n_train = int(T * TRAIN_FRAC)
    n_test  = int(T * TEST_FRAC)
    n_val   = T - n_train - n_test

    b1 = [0,           n_train - seq_len,  T - n_test - seq_len]
    b2 = [n_train,     n_train + n_val,    T]

    train_data = data[b1[0]:b2[0]]
    mean = float(train_data.mean())
    std  = float(train_data.std())
    iv_sc = ((data - mean) / std).astype(np.float32)

    def _windows(start, end):
        sl = iv_sc[start:end]                                       # SCALED for both x and y
        n  = len(sl) - seq_len - pred_len + 1
        # X: [N, seq_len, nodes, 1]   y: [N, pred_len, nodes, 1]
        X = np.stack([sl[i        : i+seq_len,    :, None] for i in range(n)])
        y = np.stack([sl[i+seq_len : i+seq_len+pred_len, :, None] for i in range(n)])
        return X.astype(np.float32), y.astype(np.float32)

    X_tr, y_tr = _windows(b1[0], b2[0])
    X_va, y_va = _windows(b1[1], b2[1])
    X_te, _y   = _windows(b1[2], b2[2])

    test_slice_dates = dates[b1[2]:b2[2]]
    test_start_dates = np.array([test_slice_dates[i + seq_len] for i in range(len(X_te))])

    info = dict(
        T=T, n_iv=n_iv, h_tau=h_tau, w_moneyness=w_moneyness,
        n_train=n_train, n_val=n_val, n_test=n_test,
        train_end_date=str(dates[n_train - 1]),
        train_windows=len(X_tr), val_windows=len(X_va), test_windows=len(X_te),
    )
    return X_tr, y_tr, X_va, y_va, X_te, test_start_dates, info, mean, std


def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    # X: [N, seq, nodes, 1]  → store as [N, 1, nodes, seq] for model
    X_t = torch.from_numpy(X.transpose(0, 3, 2, 1))   # [N, 1, nodes, seq]
    y_t = torch.from_numpy(y.transpose(0, 3, 2, 1))   # [N, 1, nodes, pred]
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _denorm(x_scaled: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse-transform a scaled tensor to original target-space units. mean/std are scalars."""
    return x_scaled * std + mean


def _compute_loss(pred_scaled: torch.Tensor, real_scaled: torch.Tensor,
                  mean: torch.Tensor, std: torch.Tensor,
                  loss_kind: str, huber_delta: float) -> torch.Tensor:
    """
    Loss menu:
      mse / huber_scaled              — operate on (pred_scaled, real_scaled).
      mae_original / huber_original   — denorm both pred and real, then compare in
                                        original IV space (vol-points).
    """
    if loss_kind == "mse":
        return F.mse_loss(pred_scaled, real_scaled)
    if loss_kind == "huber_scaled":
        return F.smooth_l1_loss(pred_scaled, real_scaled, beta=huber_delta)
    pred_orig = _denorm(pred_scaled, mean, std)
    real_orig = _denorm(real_scaled, mean, std)
    if loss_kind == "mae_original":
        return masked_mae(pred_orig, real_orig, null_val=float("nan"))
    if loss_kind == "huber_original":
        return F.smooth_l1_loss(pred_orig, real_orig, beta=huber_delta)
    raise ValueError(f"unknown loss_kind: {loss_kind!r}")


# ─── Training ─────────────────────────────────────────────────────────────────

def _epoch(model, loader, opt, device, mean, std, loss_kind, huber_delta, clip: float = 5.0):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out  = model(xb)                                    # [B, pred, nodes, 1] scaled
        out  = out.squeeze(-1).permute(0, 2, 1)             # [B, nodes, pred] scaled
        real = yb[:, 0, :, :]                               # [B, nodes, pred] scaled
        loss = _compute_loss(out, real, mean, std, loss_kind, huber_delta)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


@torch.no_grad()
def _val_loss(model, loader, device, mean, std, loss_kind, huber_delta):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out  = model(xb).squeeze(-1).permute(0, 2, 1)
        real = yb[:, 0, :, :]
        total += _compute_loss(out, real, mean, std, loss_kind, huber_delta).item() * len(xb)
        n     += len(xb)
    return total / n


def train(model, X_tr, y_tr, X_va, y_va, mean, std, args, out_dir, device):
    tr_loader = _make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    va_loader = _make_loader(X_va, y_va, args.batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device)

    best_val, wait = float("inf"), 0
    ckpt = os.path.join(out_dir, "best_model.pt")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = _epoch(model, tr_loader, opt, device, mean_t, std_t, args.loss, args.huber_delta)
        va_loss = _val_loss(model, va_loader, device, mean_t, std_t, args.loss, args.huber_delta)
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val:
            best_val, wait = va_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:4}/{args.epochs}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  best={best_val:.6f}")

        if wait >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader(); w.writerows(log_rows)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"  best val loss ({args.loss}): {best_val:.6f}")
    return model


# ─── Misc ────────────────────────────────────────────────────────────────────

def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _build_config(args, info: dict) -> dict:
    return {
        "model":          "DynGWN",
        "dataset":        args.dataset,
        "target_space":   args.target_space,
        "csv_path":       args.csv_path,
        "seq_len":        args.seq_len,
        "pred_len":       args.pred_len,
        "graph_mode":     args.graph_mode,
        "nhid":           args.nhid,
        "blocks":         args.blocks,
        "layers":         args.layers,
        "kernel_size":    args.kernel_size,
        "dropout":        args.dropout,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "patience":       args.patience,
        "seed":           args.seed,
        "loss":           args.loss,
        "huber_delta":    args.huber_delta,
        "n_iv":           info["n_iv"],
        "h_tau":          info["h_tau"],
        "w_moneyness":    info["w_moneyness"],
        "n_train":        info["n_train"],
        "n_val":          info["n_val"],
        "n_test":         info["n_test"],
        "train_end_date": info["train_end_date"],
        "git_commit":     _git_commit(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train DynGWN on SPX IV surface")
    ap.add_argument("--csv_path",     default="SPX_surfaces.csv")
    ap.add_argument("--dataset",      default="full", choices=DATASET_CHOICES,
                    help="full = entire CSV; precovid = dates <= 2019-12-31")
    ap.add_argument("--target_space", default="level", choices=TARGET_SPACE_CHOICES,
                    help="level = train on raw IV (default); logdiff = train on "
                         "log(IV)[1:]-log(IV)[:-1]. logdiff loses one day at the front. "
                         "NOTE: with --loss mae_original/huber_original AND "
                         "target_space=logdiff, the 'original' space the loss denorms "
                         "to is raw log-diff (not vol-points).")
    ap.add_argument("--seq_len",      type=int,   default=21)
    ap.add_argument("--pred_len",     type=int,   default=63)
    ap.add_argument("--graph_mode",   default="grid_plus_adaptive",
                    choices=["grid_plus_adaptive", "adaptive_only"],
                    help="grid_plus_adaptive (default) adds a fixed 4-neighbor "
                         "(tau, moneyness) grid to the learned adaptive adjacency.")
    ap.add_argument("--nhid",         type=int,   default=32,
                    help="Residual and dilation channels (skip=nhid*8, end=nhid*16)")
    ap.add_argument("--blocks",       type=int,   default=4)
    ap.add_argument("--layers",       type=int,   default=2)
    ap.add_argument("--kernel_size",  type=int,   default=2)
    ap.add_argument("--dropout",      type=float, default=0.3)
    ap.add_argument("--epochs",       type=int,   default=500)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience",     type=int,   default=15)
    ap.add_argument("--device",       default="auto")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--out_dir",      default=None)
    ap.add_argument("--predict_only", action="store_true",
                    help="Skip training; load best_model.pt + config.json from --out_dir, "
                         "run inference, write pred.npy.")
    ap.add_argument("--loss",         default="mse",
                    choices=["mse", "huber_scaled", "mae_original", "huber_original"],
                    help="Default `mse` (scaled space, apples-to-apples with the "
                         "rest of the benchmark). `mae_original`/`huber_original` "
                         "reproduce the legacy DynGWN loss in vol-points.")
    ap.add_argument("--huber_delta",  type=float, default=0.02,
                    help="Threshold for huber_*: ~stdev units for huber_scaled, "
                         "vol-points for huber_original (default 0.02 = 2 vol pts).")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():           device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)

    if args.predict_only:
        if args.out_dir is None:
            raise SystemExit("--predict_only requires --out_dir <dir with config.json + best_model.pt>")
        cfg_path  = os.path.join(args.out_dir, "config.json")
        ckpt_path = os.path.join(args.out_dir, "best_model.pt")
        if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
            raise SystemExit(f"--predict_only: need config.json and best_model.pt in {args.out_dir}")
        with open(cfg_path) as f:
            cfg = json.load(f)
        for k in ("csv_path", "dataset", "target_space", "seq_len", "pred_len",
                  "graph_mode", "nhid", "blocks", "layers", "kernel_size", "dropout",
                  "loss", "huber_delta"):
            if k in cfg:
                setattr(args, k, cfg[k])
        print(f"[predict_only] {cfg_path}")

    if args.out_dir is None:
        loss_suffix = {
            "mse":            "",
            "huber_scaled":   f"_losshuberscaled_d{args.huber_delta:g}",
            "mae_original":   "_lossmaeoriginal",
            "huber_original": f"_losshuberoriginal_d{args.huber_delta:g}",
        }[args.loss]
        args.out_dir = (f"DynGWN/results/"
                        f"{args.dataset}_{args.target_space}_SPX_IV_"
                        f"{args.seq_len}_{args.pred_len}"
                        f"_DynGWN_{args.graph_mode}"
                        f"_nh{args.nhid}_b{args.blocks}_l{args.layers}"
                        f"_ep{args.epochs}{loss_suffix}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Dataset    : {args.dataset}")
    print(f"Target     : {args.target_space}")
    print(f"Device     : {device}")
    print(f"Output dir : {args.out_dir}")
    print(f"Loss       : {args.loss}"
          + (f"  (delta={args.huber_delta})" if "huber" in args.loss else ""))

    print("\nLoading data...")
    X_tr, y_tr, X_va, y_va, X_te, test_dates, info, mean, std = load_splits(
        args.csv_path, args.dataset, args.target_space, args.seq_len, args.pred_len)
    print(f"  T={info['T']}  n_iv={info['n_iv']}  "
          f"train={info['train_windows']}  "
          f"val={info['val_windows']}  test={info['test_windows']} windows")
    print(f"  Train ends {info['train_end_date']}")

    n_iv = info["n_iv"]
    static_supports = []
    if args.graph_mode == "grid_plus_adaptive":
        A_grid = _row_normalize(_make_grid_adjacency(
            h=info["h_tau"], w=info["w_moneyness"], self_loops=True))
        static_supports = [torch.from_numpy(A_grid)]

    model = DynGWN(num_nodes=n_iv, dropout=args.dropout,
                   in_dim=1, seq_len=args.seq_len, pred_len=args.pred_len,
                   nhid=args.nhid, kernel_size=args.kernel_size,
                   blocks=args.blocks, layers=args.layers,
                   static_supports=static_supports).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  graph_mode : {args.graph_mode}  ({len(static_supports)+1} supports)")
    print(f"  Parameters : {n_params:,}  receptive_field={model.receptive_field}")

    if args.predict_only:
        ckpt_path = os.path.join(args.out_dir, "best_model.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"  Loaded checkpoint from {ckpt_path}")
    else:
        config = _build_config(args, info)
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print("\nTraining...")
        model = train(model, X_tr, y_tr, X_va, y_va, mean, std, args, args.out_dir, device)

    print("\nPredicting on test set...")
    te_loader = _make_loader(X_te, np.zeros_like(X_te), 8, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in te_loader:
            out = model(xb.to(device))            # [B, pred_len, nodes, 1]  scaled
            out = out.squeeze(-1).cpu().numpy()   # [B, pred_len, nodes]  (compare_models flat)
            preds.append(out)
    preds = np.concatenate(preds, axis=0)[:len(test_dates)].astype(np.float32)
    preds = (preds * std + mean).astype(np.float32)

    np.save(os.path.join(args.out_dir, "pred.npy"),        preds)
    np.save(os.path.join(args.out_dir, "start_dates.npy"), test_dates)
    print(f"  pred.npy        shape={preds.shape}")
    print(f"  start_dates.npy range={test_dates[0]} → {test_dates[-1]}")
    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
