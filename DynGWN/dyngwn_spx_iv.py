#!/usr/bin/env python3
"""
DynGWN (Graph-WaveNet) training script for SPX IV surface forecasting.

All model code is inlined — no dependency on the legacy DynGWN/ pipeline files.

Architecture:
  - WaveNet-style dilated temporal convolutions (blocks=4, layers=2, kernel_size=2)
  - Graph convolution at each WaveNet step using one or two supports:
        graph_mode='grid_plus_adaptive'  (default, matches legacy run):
            [4-neighbor moneyness×tau grid, learned adaptive (nodevec1 @ nodevec2)]
        graph_mode='adaptive_only':
            [learned adaptive only]

Training loss (matches legacy engine.py): masked_mae on inverse-transformed
predictions vs ground truth in ORIGINAL IV space. Inputs (x) are scaled by
StandardScaler; targets (y) stay in original space; the model output is
inverse-transformed inside the loss. pred.npy is still saved in scaled space
(compare_models contract).

Outputs (in --out_dir):
    pred.npy          [N_test, pred_len, 400]  scaled-space predictions
    start_dates.npy   [N_test]                  datetime64[D] start of each window
    train_log.csv     epoch, train_loss, val_loss
    best_model.pt     checkpoint of best validation weights
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
N_IV       = 400
H_MONO     = 20   # moneyness axis
W_TAU      = 20   # tau axis


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

def _make_grid_adjacency(h: int = H_MONO, w: int = W_TAU,
                         self_loops: bool = True) -> np.ndarray:
    """4-neighbor adjacency over a flattened H×W grid (matches legacy generator)."""
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

    Input:  [B, in_dim=1, num_nodes=400, seq_len=21]  (padded +1 inside forward)
    Output: [B, pred_len=63, num_nodes=400, 1]
    """
    def __init__(self, num_nodes: int = 400, dropout: float = 0.3,
                 in_dim: int = 1, seq_len: int = 21, pred_len: int = 63,
                 nhid: int = 32, kernel_size: int = 2,
                 blocks: int = 4, layers: int = 2,
                 static_supports: list = None):
        super().__init__()
        self.blocks    = blocks
        self.layers    = layers
        self.num_nodes = num_nodes

        skip_channels = nhid * 8
        end_channels  = nhid * 16
        order         = 2

        # static_supports: list of fixed [N, N] adjacencies registered as buffers.
        # Adaptive adjacency adds one more support, computed at every forward.
        self.static_supports = static_supports or []
        for i, sup in enumerate(self.static_supports):
            self.register_buffer(f"static_support_{i}", sup, persistent=False)
        support_len = len(self.static_supports) + 1   # +1 for adaptive

        self.start_conv = nn.Conv2d(in_dim, nhid, kernel_size=(1, 1))

        # Adaptive adjacency node vectors
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes))

        self.filter_convs = nn.ModuleList()
        self.gate_convs   = nn.ModuleList()
        self.skip_convs   = nn.ModuleList()
        self.gconv        = nn.ModuleList()
        self.bn           = nn.ModuleList()

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
                    self.bn.append(nn.BatchNorm2d(nhid))
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
                x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)             # [B, pred_len, nodes, 1]
        return x


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_splits(csv_path: str, seq_len: int, pred_len: int):
    """
    Returns x in SCALED space (input to model), y in ORIGINAL IV space (loss target).
    The legacy loss inverse-transforms model output before computing masked_mae,
    so y must remain in the original (un-scaled) space.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    # CSV column order = (tau outer, moneyness inner). Do NOT sort — alphabetical
    # order scrambles the surface and breaks cross-sectional alignment with
    # compare_models.py (which uses CSV order).
    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    assert len(iv_cols) == N_IV

    T = len(df)
    n_train = int(T * TRAIN_FRAC)
    n_test  = int(T * TEST_FRAC)
    n_val   = T - n_train - n_test

    b1 = [0,           n_train - seq_len,  T - n_test - seq_len]
    b2 = [n_train,     n_train + n_val,    T]

    iv_raw = df[iv_cols].to_numpy(dtype=np.float32)               # original space
    scaler = StandardScaler().fit(iv_raw[b1[0]:b2[0]])
    iv_sc  = scaler.transform(iv_raw).astype(np.float32)           # scaled space
    dates  = df["date"].to_numpy(dtype="datetime64[D]")

    def _windows(start, end):
        sl_x = iv_sc[start:end]                                     # scaled  → x
        sl_y = iv_raw[start:end]                                    # original → y
        n  = len(sl_x) - seq_len - pred_len + 1
        # X: [N, seq_len, 400, 1]   y: [N, pred_len, 400, 1]
        X = np.stack([sl_x[i        : i+seq_len,    :, None] for i in range(n)])
        y = np.stack([sl_y[i+seq_len : i+seq_len+pred_len, :, None] for i in range(n)])
        return X.astype(np.float32), y.astype(np.float32)

    X_tr, y_tr = _windows(b1[0], b2[0])
    X_va, y_va = _windows(b1[1], b2[1])
    X_te, _y   = _windows(b1[2], b2[2])

    test_slice_dates = dates[b1[2]:b2[2]]
    test_start_dates = np.array([test_slice_dates[i + seq_len] for i in range(len(X_te))])

    info = dict(T=T, n_train=n_train, n_val=n_val, n_test=n_test,
                train_windows=len(X_tr), val_windows=len(X_va), test_windows=len(X_te))
    return X_tr, y_tr, X_va, y_va, X_te, test_start_dates, info, scaler


def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    # X: [N, seq, nodes, 1]  → store as [N, 1, nodes, seq] for model
    X_t = torch.from_numpy(X.transpose(0, 3, 2, 1))   # [N, 1, nodes, seq]
    y_t = torch.from_numpy(y.transpose(0, 3, 2, 1))   # [N, 1, nodes, pred]
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


# ─── Training ─────────────────────────────────────────────────────────────────

def _denorm(out_scaled: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse-transform [B, nodes, pred] scaled output to original space."""
    # mean/std are [nodes] → broadcast over [B, nodes, pred].
    return out_scaled * std.view(1, -1, 1) + mean.view(1, -1, 1)


def _epoch(model, loader, opt, device, mean, std, clip: float = 5.0):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out  = model(xb)                                    # [B, pred, nodes, 1] scaled
        out  = out.squeeze(-1).permute(0, 2, 1)             # [B, nodes, pred] scaled
        pred = _denorm(out, mean, std)                      # [B, nodes, pred] original
        real = yb[:, 0, :, :]                               # [B, nodes, pred] original
        loss = masked_mae(pred, real, null_val=float("nan"))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


@torch.no_grad()
def _val_loss(model, loader, device, mean, std):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out  = model(xb).squeeze(-1).permute(0, 2, 1)
        pred = _denorm(out, mean, std)
        real = yb[:, 0, :, :]
        total += masked_mae(pred, real, null_val=float("nan")).item() * len(xb)
        n     += len(xb)
    return total / n


def train(model, X_tr, y_tr, X_va, y_va, scaler, args, out_dir, device):
    tr_loader = _make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    va_loader = _make_loader(X_va, y_va, args.batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std  = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    best_val, wait = float("inf"), 0
    ckpt = os.path.join(out_dir, "best_model.pt")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = _epoch(model, tr_loader, opt, device, mean, std)
        va_loss = _val_loss(model, va_loader, device, mean, std)
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val:
            best_val, wait = va_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:4}/{args.epochs}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  best={best_val:.6f}")

        if wait >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader(); w.writerows(log_rows)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"  best val masked-MAE: {best_val:.6f}")
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train DynGWN on SPX IV surface")
    ap.add_argument("--csv_path",     default="SPX_surfaces.csv")
    ap.add_argument("--seq_len",      type=int,   default=21)
    ap.add_argument("--pred_len",     type=int,   default=63)
    ap.add_argument("--graph_mode",   default="grid_plus_adaptive",
                    choices=["grid_plus_adaptive", "adaptive_only"],
                    help="grid_plus_adaptive (legacy default) adds a fixed 4-neighbor "
                         "moneyness×tau grid to the learned adaptive adjacency.")
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
    ap.add_argument("--patience",     type=int,   default=30)
    ap.add_argument("--device",       default="auto")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--out_dir",      default=None)
    ap.add_argument("--predict_only", action="store_true",
                    help="Skip training; load best_model.pt + config.json from --out_dir, "
                         "run inference, write pred.npy.")
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
        for k in ("csv_path", "seq_len", "pred_len", "graph_mode", "nhid",
                  "blocks", "layers", "kernel_size", "dropout", "batch_size"):
            if k in cfg:
                setattr(args, k, cfg[k])
        print(f"[predict_only] {cfg_path}")

    if args.out_dir is None:
        args.out_dir = (f"DynGWN/results/SPX_IV_{args.seq_len}_{args.pred_len}"
                        f"_DynGWN_{args.graph_mode}"
                        f"_nh{args.nhid}_b{args.blocks}_l{args.layers}"
                        f"_ep{args.epochs}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Output dir : {args.out_dir}")

    print("\nLoading data...")
    X_tr, y_tr, X_va, y_va, X_te, test_dates, info, scaler = load_splits(
        args.csv_path, args.seq_len, args.pred_len)
    print(f"  T={info['T']}  train={info['train_windows']}  "
          f"val={info['val_windows']}  test={info['test_windows']} windows")

    static_supports = []
    if args.graph_mode == "grid_plus_adaptive":
        A_grid = _row_normalize(_make_grid_adjacency(self_loops=True))
        static_supports = [torch.from_numpy(A_grid)]

    model = DynGWN(num_nodes=N_IV, dropout=args.dropout,
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
        config = {
            "model":         "DynGWN",
            "csv_path":      args.csv_path,
            "seq_len":       args.seq_len,
            "pred_len":      args.pred_len,
            "graph_mode":    args.graph_mode,
            "nhid":          args.nhid,
            "blocks":        args.blocks,
            "layers":        args.layers,
            "kernel_size":   args.kernel_size,
            "dropout":       args.dropout,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "lr":            args.lr,
            "weight_decay":  args.weight_decay,
            "patience":      args.patience,
            "seed":          args.seed,
        }
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print("\nTraining...")
        model = train(model, X_tr, y_tr, X_va, y_va, scaler, args, args.out_dir, device)

    print("\nPredicting on test set...")
    te_loader = _make_loader(X_te, np.zeros_like(X_te), args.batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in te_loader:
            out = model(xb.to(device))            # [B, pred_len=63, nodes=400, 1]
            out = out.squeeze(-1).cpu().numpy()   # [B, 63, 400]  (compare_models flat format)
            preds.append(out)
    preds = np.concatenate(preds, axis=0)[:len(test_dates)].astype(np.float32)

    np.save(os.path.join(args.out_dir, "pred.npy"),        preds)
    np.save(os.path.join(args.out_dir, "start_dates.npy"), test_dates)
    print(f"  pred.npy        shape={preds.shape}")
    print(f"  start_dates.npy range={test_dates[0]} → {test_dates[-1]}")
    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
