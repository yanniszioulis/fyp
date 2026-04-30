#!/usr/bin/env python3
"""
PatchTST training script for SPX IV surface forecasting.

Channel-independent patch-based Transformer with RevIN normalisation.
All model code is inlined — no dependency on PatchTST-main/.

Outputs (in --out_dir):
    pred.npy          [N_test, pred_len, 400]  scaled-space predictions
    start_dates.npy   [N_test]                  datetime64[D] start of each window
    train_log.csv     epoch, train_loss, val_loss
    best_model.pt     checkpoint of best validation weights
"""

import argparse
import csv
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
N_IV       = 400


# ─── RevIN ────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(mode)

    def _get_statistics(self, x: Tensor):
        dims = tuple(range(1, x.ndim - 1))
        self.mean  = x.mean(dim=dims, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=dims, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        return x * self.stdev + self.mean


# ─── Helpers ──────────────────────────────────────────────────────────────────

class _Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.dims)


def _positional_encoding(q_len: int, d_model: int) -> nn.Parameter:
    """Learnable 'zeros'-initialised positional encoding."""
    W = torch.empty((q_len, d_model))
    nn.init.uniform_(W, -0.02, 0.02)
    return nn.Parameter(W, requires_grad=True)


# ─── Attention ────────────────────────────────────────────────────────────────

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.,
                 res_attention: bool = True):
        super().__init__()
        self.attn_dropout  = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        self.scale = nn.Parameter(torch.tensor((d_model // n_heads) ** -0.5))

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                prev: Optional[Tensor] = None) -> tuple:
        # q: [B, heads, Q, d_k]  k: [B, heads, d_k, S]  v: [B, heads, S, d_v]
        scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            scores = scores + prev
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        output  = torch.matmul(weights, v)
        if self.res_attention:
            return output, weights, scores
        return output, weights


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.,
                 proj_dropout: float = 0., res_attention: bool = True):
        super().__init__()
        d_k = d_model // n_heads
        self.n_heads, self.d_k = n_heads, d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        self.res_attention = res_attention
        self.sdp_attn  = _ScaledDotProductAttention(d_model, n_heads, attn_dropout, res_attention)
        self.to_out    = nn.Sequential(nn.Linear(n_heads * d_k, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, prev: Optional[Tensor] = None):
        bs = Q.size(0)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(Q).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        if self.res_attention:
            output, _, scores = self.sdp_attn(q_s, k_s, v_s, prev=prev)
        else:
            output, _ = self.sdp_attn(q_s, k_s, v_s)
            scores = None
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        output = self.to_out(output)
        if self.res_attention:
            return output, scores
        return output, None


# ─── Encoder ──────────────────────────────────────────────────────────────────

class _TSTEncoderLayer(nn.Module):
    def __init__(self, q_len: int, d_model: int, n_heads: int,
                 d_ff: int = 256, attn_dropout: float = 0.,
                 dropout: float = 0., res_attention: bool = True):
        super().__init__()
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, attn_dropout, dropout, res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(d_model), _Transpose(1, 2))
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(d_model), _Transpose(1, 2))

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):
        src2, scores = self.self_attn(src, prev=prev)
        src = self.norm_attn(src + self.dropout_attn(src2))
        src2 = self.ff(src)
        src = self.norm_ffn(src + self.dropout_ffn(src2))
        if self.res_attention:
            return src, scores
        return src


class _TSTEncoder(nn.Module):
    def __init__(self, q_len: int, d_model: int, n_heads: int, d_ff: int,
                 attn_dropout: float, dropout: float, n_layers: int, res_attention: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            _TSTEncoderLayer(q_len, d_model, n_heads, d_ff, attn_dropout, dropout, res_attention)
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src: Tensor) -> Tensor:
        output, scores = src, None
        for layer in self.layers:
            if self.res_attention:
                output, scores = layer(output, prev=scores)
            else:
                output = layer(output)
        return output


class _TSTiEncoder(nn.Module):
    """Channel-independent encoder: 400 channels processed in parallel."""
    def __init__(self, c_in: int, patch_num: int, patch_len: int, d_model: int,
                 n_heads: int, d_ff: int, attn_dropout: float, dropout: float,
                 n_layers: int, res_attention: bool):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.W_P   = nn.Linear(patch_len, d_model)
        self.W_pos = _positional_encoding(patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _TSTEncoder(patch_num, d_model, n_heads, d_ff,
                                   attn_dropout, dropout, n_layers, res_attention)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, patch_len, patch_num]
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)            # [B, C, patch_num, patch_len]
        x = self.W_P(x)                       # [B, C, patch_num, d_model]
        u = x.reshape(-1, x.shape[2], x.shape[3])   # [B*C, patch_num, d_model]
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)                   # [B*C, patch_num, d_model]
        z = z.reshape(-1, n_vars, z.shape[-2], z.shape[-1])  # [B, C, patch_num, d_model]
        return z.permute(0, 1, 3, 2)          # [B, C, d_model, patch_num]


class _FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear  = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, d_model, patch_num]
        return self.dropout(self.linear(self.flatten(x)))  # [B, C, target_window]


# ─── Model ────────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    Channel-independent PatchTST with RevIN.
    Input:  [B, seq_len, C]  (scaled)
    Output: [B, pred_len, C] (scaled)
    """
    def __init__(self, c_in: int, seq_len: int, pred_len: int,
                 patch_len: int = 7, stride: int = 7,
                 d_model: int = 128, n_heads: int = 16, n_layers: int = 3,
                 d_ff: int = 256, attn_dropout: float = 0., dropout: float = 0.,
                 head_dropout: float = 0., res_attention: bool = True,
                 revin: bool = True):
        super().__init__()
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(c_in)

        patch_num = int((seq_len - patch_len) / stride + 1)
        self.patch_len = patch_len
        self.stride    = stride

        self.backbone = _TSTiEncoder(c_in, patch_num, patch_len, d_model, n_heads,
                                     d_ff, attn_dropout, dropout, n_layers, res_attention)
        nf = d_model * patch_num
        self.head = _FlattenHead(c_in, nf, pred_len, head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, seq_len, C]
        if self.revin:
            x = self.revin_layer(x, "norm")
        z = x.permute(0, 2, 1)                                # [B, C, seq_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, num_patches, patch_len]
        z = z.permute(0, 1, 3, 2)                             # [B, C, patch_len, num_patches]
        z = self.backbone(z)                                   # [B, C, d_model, num_patches]
        z = self.head(z)                                       # [B, C, pred_len]
        z = z.permute(0, 2, 1)                                 # [B, pred_len, C]
        if self.revin:
            z = self.revin_layer(z, "denorm")
        return z


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_splits(csv_path: str, seq_len: int, pred_len: int):
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    iv_cols = sorted([c for c in df.columns if c.startswith("iv_")])
    assert len(iv_cols) == N_IV

    T = len(df)
    n_train = int(T * TRAIN_FRAC)
    n_test  = int(T * TEST_FRAC)
    n_val   = T - n_train - n_test

    b1 = [0,           n_train - seq_len,  T - n_test - seq_len]
    b2 = [n_train,     n_train + n_val,    T]

    iv = df[iv_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(iv[b1[0]:b2[0]])
    iv = scaler.transform(iv).astype(np.float32)

    dates = df["date"].to_numpy(dtype="datetime64[D]")

    def _windows(start, end):
        sl = iv[start:end]
        n  = len(sl) - seq_len - pred_len + 1
        X  = np.stack([sl[i        : i + seq_len]            for i in range(n)])
        y  = np.stack([sl[i+seq_len : i+seq_len+pred_len]    for i in range(n)])
        return X, y

    X_tr, y_tr = _windows(b1[0], b2[0])
    X_va, y_va = _windows(b1[1], b2[1])
    X_te, _y   = _windows(b1[2], b2[2])

    test_slice_dates = dates[b1[2]:b2[2]]
    test_start_dates = np.array([test_slice_dates[i + seq_len] for i in range(len(X_te))])

    return X_tr, y_tr, X_va, y_va, X_te, test_start_dates, dict(
        T=T, n_train=n_train, n_val=n_val, n_test=n_test,
        train_windows=len(X_tr), val_windows=len(X_va), test_windows=len(X_te)
    )


# ─── Training ─────────────────────────────────────────────────────────────────

def _epoch(model, loader, opt, device):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = F.mse_loss(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


@torch.no_grad()
def _val_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += F.mse_loss(model(xb), yb).item() * len(xb); n += len(xb)
    return total / n


def train(model, X_tr, y_tr, X_va, y_va, args, out_dir, device):
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                           batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                           batch_size=args.batch_size, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, wait = float("inf"), 0
    ckpt = os.path.join(out_dir, "best_model.pt")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = _epoch(model, tr_loader, opt, device)
        va_loss = _val_loss(model, va_loader, device)
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val:
            best_val, wait = va_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4}/{args.epochs}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  best={best_val:.6f}")

        if wait >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader(); w.writerows(log_rows)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"  best val MSE: {best_val:.6f}")
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train PatchTST on SPX IV surface")
    ap.add_argument("--csv_path",    default="SPX_surfaces.csv")
    ap.add_argument("--seq_len",     type=int,   default=21)
    ap.add_argument("--pred_len",    type=int,   default=63)
    ap.add_argument("--patch_len",   type=int,   default=7)
    ap.add_argument("--stride",      type=int,   default=7)
    ap.add_argument("--d_model",     type=int,   default=128)
    ap.add_argument("--n_heads",     type=int,   default=16)
    ap.add_argument("--n_layers",    type=int,   default=3)
    ap.add_argument("--d_ff",        type=int,   default=256)
    ap.add_argument("--dropout",     type=float, default=0.0)
    ap.add_argument("--attn_dropout",type=float, default=0.0)
    ap.add_argument("--head_dropout",type=float, default=0.0)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--batch_size",  type=int,   default=64)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--weight_decay",type=float, default=1e-4)
    ap.add_argument("--patience",    type=int,   default=15)
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--out_dir",     default=None)
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():      device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                              device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)

    if args.out_dir is None:
        args.out_dir = (f"PatchTST/results/SPX_IV_{args.seq_len}_{args.pred_len}"
                        f"_PatchTST_pl{args.patch_len}_s{args.stride}"
                        f"_dm{args.d_model}_nh{args.n_heads}_nl{args.n_layers}"
                        f"_ep{args.epochs}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Output dir : {args.out_dir}")

    print("\nLoading data...")
    X_tr, y_tr, X_va, y_va, X_te, test_dates, info = load_splits(
        args.csv_path, args.seq_len, args.pred_len)
    print(f"  T={info['T']}  train={info['train_windows']}  "
          f"val={info['val_windows']}  test={info['test_windows']} windows")

    model = PatchTST(N_IV, args.seq_len, args.pred_len,
                     patch_len=args.patch_len, stride=args.stride,
                     d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                     d_ff=args.d_ff, attn_dropout=args.attn_dropout,
                     dropout=args.dropout, head_dropout=args.head_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    print("\nTraining...")
    model = train(model, X_tr, y_tr, X_va, y_va, args, args.out_dir, device)

    print("\nPredicting on test set...")
    te_loader = DataLoader(TensorDataset(torch.from_numpy(X_te)),
                           batch_size=args.batch_size, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0).astype(np.float32)

    np.save(os.path.join(args.out_dir, "pred.npy"),        preds)
    np.save(os.path.join(args.out_dir, "start_dates.npy"), test_dates)
    print(f"  pred.npy        shape={preds.shape}")
    print(f"  start_dates.npy range={test_dates[0]} → {test_dates[-1]}")
    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
