#!/usr/bin/env python3
"""
PatchTST training script for SPX IV surface forecasting.

Channel-independent patch-based Transformer. Inputs are standardised with a
single global (mean, std) fitted on the training portion of the target-space
data — pooled across time and all 170 IV cells, not per-column. The model
trains in scaled space; predictions are inverse-transformed and saved in
original target-space units. RevIN (per-window per-channel norm/denorm inside
the model) is opt-in via --revin=on; default is off, apples-to-apples with
DLinear/DynGWN/HOT. All model code is inlined — no dependency on PatchTST-main/.

Dataset selection:
    --dataset full      use the full CSV (default).
    --dataset precovid  slice to date <= 2019-12-31 before splitting.

Outputs (in --out_dir; default
`PatchTST/results/{dataset}_SPX_IV_{seq_len}_{pred_len}_PatchTST_pl{pl}_s{s}_dm{dm}_nh{nh}_nl{nl}_ep{ep}{loss_suffix}`):
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
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FRAC           = 0.70
TEST_FRAC            = 0.20
PRECOVID_END         = "2019-12-31"
DATASET_CHOICES      = ["full", "precovid"]
TARGET_SPACE_CHOICES = ["level", "logdiff"]


# ─── RevIN ────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
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
    """Channel-independent encoder: all channels processed in parallel via reshape."""
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
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)
        u = x.reshape(-1, x.shape[2], x.shape[3])
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = z.reshape(-1, n_vars, z.shape[-2], z.shape[-1])
        return z.permute(0, 1, 3, 2)


class _FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear  = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear(self.flatten(x)))


# ─── Model ────────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    Channel-independent PatchTST with RevIN.
    Input:  [B, seq_len, C]  (scaled)
    Output: [B, pred_len, C] (scaled)

    padding_patch: 'end' replicates the last value `stride` times before unfolding,
    yielding patch_num+1 patches. Matches legacy run_longExp.py default.
    """
    def __init__(self, c_in: int, seq_len: int, pred_len: int,
                 patch_len: int = 7, stride: int = 7,
                 d_model: int = 128, n_heads: int = 16, n_layers: int = 3,
                 d_ff: int = 256, attn_dropout: float = 0., dropout: float = 0.,
                 head_dropout: float = 0., res_attention: bool = True,
                 revin: bool = True, affine: bool = False,
                 padding_patch: str = "end"):
        super().__init__()
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(c_in, affine=affine)

        self.patch_len     = patch_len
        self.stride        = stride
        self.padding_patch = padding_patch
        patch_num = int((seq_len - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = _TSTiEncoder(c_in, patch_num, patch_len, d_model, n_heads,
                                     d_ff, attn_dropout, dropout, n_layers, res_attention)
        nf = d_model * patch_num
        self.head = _FlattenHead(c_in, nf, pred_len, head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.revin:
            x = self.revin_layer(x, "norm")
        z = x.permute(0, 2, 1)
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)
        z = self.head(z)
        z = z.permute(0, 2, 1)
        if self.revin:
            z = self.revin_layer(z, "denorm")
        return z


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


def load_splits(csv_path: str, dataset: str, target_space: str,
                seq_len: int, pred_len: int):
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = _slice_dataset(df, dataset)

    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    if not iv_cols:
        raise ValueError(f"No iv_* columns found in {csv_path}")
    n_iv = len(iv_cols)

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
    iv   = ((data - mean) / std).astype(np.float32)

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

    info = dict(
        T=T, n_iv=n_iv, n_train=n_train, n_val=n_val, n_test=n_test,
        train_end_date=str(dates[n_train - 1]),
        train_windows=len(X_tr), val_windows=len(X_va), test_windows=len(X_te),
    )
    return X_tr, y_tr, X_va, y_va, X_te, test_start_dates, info, mean, std


# ─── Training ─────────────────────────────────────────────────────────────────

def _compute_loss(pred, y, loss_kind: str, huber_delta: float = 1.0):
    if loss_kind == "mse":
        return F.mse_loss(pred, y)
    if loss_kind == "huber_scaled":
        return F.smooth_l1_loss(pred, y, beta=huber_delta)
    raise ValueError(f"unknown loss_kind: {loss_kind!r}")


def _epoch(model, loader, opt, device, loss_kind, huber_delta):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = _compute_loss(model(xb), yb, loss_kind, huber_delta)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


@torch.no_grad()
def _val_loss(model, loader, device, loss_kind, huber_delta):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += _compute_loss(model(xb), yb, loss_kind, huber_delta).item() * len(xb)
        n += len(xb)
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
        tr_loss = _epoch(model, tr_loader, opt, device, args.loss, args.huber_delta)
        va_loss = _val_loss(model, va_loader, device, args.loss, args.huber_delta)
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
    print(f"  best val loss: {best_val:.6f}")
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
        "model":          "PatchTST",
        "dataset":        args.dataset,
        "target_space":   args.target_space,
        "csv_path":       args.csv_path,
        "seq_len":        args.seq_len,
        "pred_len":       args.pred_len,
        "patch_len":      args.patch_len,
        "stride":         args.stride,
        "padding_patch":  args.padding_patch,
        "revin":          args.revin,
        "revin_affine":   args.revin_affine,
        "d_model":        args.d_model,
        "n_heads":        args.n_heads,
        "n_layers":       args.n_layers,
        "d_ff":           args.d_ff,
        "dropout":        args.dropout,
        "attn_dropout":   args.attn_dropout,
        "head_dropout":   args.head_dropout,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "patience":       args.patience,
        "seed":           args.seed,
        "loss":           args.loss,
        "huber_delta":    args.huber_delta,
        "n_iv":           info["n_iv"],
        "n_train":        info["n_train"],
        "n_val":          info["n_val"],
        "n_test":         info["n_test"],
        "train_end_date": info["train_end_date"],
        "git_commit":     _git_commit(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train PatchTST on SPX IV surface")
    ap.add_argument("--csv_path",    default="SPX_surfaces.csv")
    ap.add_argument("--dataset",     default="full", choices=DATASET_CHOICES,
                    help="full = entire CSV; precovid = dates <= 2019-12-31")
    ap.add_argument("--target_space", default="level", choices=TARGET_SPACE_CHOICES,
                    help="level = train on raw IV (default); logdiff = train on "
                         "log(IV)[1:]-log(IV)[:-1]. logdiff loses one day at the front.")
    ap.add_argument("--seq_len",     type=int,   default=21)
    ap.add_argument("--pred_len",    type=int,   default=63)
    ap.add_argument("--patch_len",   type=int,   default=3)
    ap.add_argument("--stride",      type=int,   default=3)
    ap.add_argument("--d_model",     type=int,   default=64)
    ap.add_argument("--n_heads",     type=int,   default=4)
    ap.add_argument("--n_layers",    type=int,   default=2)
    ap.add_argument("--d_ff",        type=int,   default=128)
    ap.add_argument("--dropout",     type=float, default=0.2)
    ap.add_argument("--attn_dropout",type=float, default=0.0)
    ap.add_argument("--head_dropout",type=float, default=0.1)
    ap.add_argument("--padding_patch", default="end",
                    choices=["end", "none"],
                    help="'end' replicates last value stride-times before unfold "
                         "(adds +1 patch). Matches legacy default.")
    ap.add_argument("--revin", default="off", choices=["on", "off"],
                    help="Per-window RevIN normalisation. 'off' (default) = pass "
                         "scaled inputs through directly (apples-to-apples with "
                         "DLinear/DynGWN/HOT). 'on' = legacy RevIN (per-window "
                         "per-channel norm/denorm).")
    ap.add_argument("--revin_affine", type=int, default=0,
                    help="RevIN affine params: 1=on, 0=off (legacy default). "
                         "Only effective when --revin=on.")
    ap.add_argument("--epochs",      type=int,   default=150)
    ap.add_argument("--batch_size",  type=int,   default=64)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--weight_decay",type=float, default=1e-4)
    ap.add_argument("--patience",    type=int,   default=15)
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--out_dir",     default=None)
    ap.add_argument("--predict_only", action="store_true",
                    help="Skip training; load best_model.pt + config.json from --out_dir, "
                         "run inference, write pred.npy.")
    ap.add_argument("--loss",         default="mse",
                    choices=["mse", "huber_scaled"])
    ap.add_argument("--huber_delta",  type=float, default=1.0)
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
                  "patch_len", "stride",
                  "d_model", "n_heads", "n_layers", "d_ff",
                  "dropout", "attn_dropout", "head_dropout",
                  "padding_patch", "revin", "revin_affine", "loss", "huber_delta"):
            if k in cfg:
                setattr(args, k, cfg[k])
        print(f"[predict_only] {cfg_path}")

    if args.out_dir is None:
        loss_suffix = {
            "mse":          "",
            "huber_scaled": f"_losshuberscaled_d{args.huber_delta:g}",
        }[args.loss]
        revin_suffix = "" if args.revin == "off" else "_revin"
        args.out_dir = (f"PatchTST/results/"
                        f"{args.dataset}_{args.target_space}_SPX_IV_"
                        f"{args.seq_len}_{args.pred_len}"
                        f"_PatchTST_pl{args.patch_len}_s{args.stride}"
                        f"_dm{args.d_model}_nh{args.n_heads}_nl{args.n_layers}"
                        f"_ep{args.epochs}{loss_suffix}{revin_suffix}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Dataset    : {args.dataset}")
    print(f"Target     : {args.target_space}")
    print(f"Device     : {device}")
    print(f"Output dir : {args.out_dir}")

    print("\nLoading data...")
    X_tr, y_tr, X_va, y_va, X_te, test_dates, info, mean, std = load_splits(
        args.csv_path, args.dataset, args.target_space, args.seq_len, args.pred_len)
    print(f"  T={info['T']}  n_iv={info['n_iv']}  "
          f"train={info['train_windows']}  "
          f"val={info['val_windows']}  test={info['test_windows']} windows")
    print(f"  Train ends {info['train_end_date']}")

    n_iv = info["n_iv"]
    model = PatchTST(n_iv, args.seq_len, args.pred_len,
                     patch_len=args.patch_len, stride=args.stride,
                     d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                     d_ff=args.d_ff, attn_dropout=args.attn_dropout,
                     dropout=args.dropout, head_dropout=args.head_dropout,
                     padding_patch=args.padding_patch,
                     revin=(args.revin == "on"),
                     affine=bool(args.revin_affine)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if args.predict_only:
        ckpt_path = os.path.join(args.out_dir, "best_model.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"  Loaded checkpoint from {ckpt_path}")
    else:
        config = _build_config(args, info)
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

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
    preds = (preds * std + mean).astype(np.float32)

    np.save(os.path.join(args.out_dir, "pred.npy"),        preds)
    np.save(os.path.join(args.out_dir, "start_dates.npy"), test_dates)
    print(f"  pred.npy        shape={preds.shape}")
    print(f"  start_dates.npy range={test_dates[0]} → {test_dates[-1]}")
    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
