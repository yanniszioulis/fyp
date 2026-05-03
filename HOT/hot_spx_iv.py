#!/usr/bin/env python3
"""
HOT (Higher-Order Transformer) training script for SPX IV surface forecasting.

Treats the 170 IV cells as a 17×10 structured tensor (delta × tau) and applies
Kronecker attention across both spatial axes plus the temporal patches.

Layout:
    H = DELTA = 17  (call-equivalent delta axis, 0.10 → 0.90 in 0.05 steps)
    W = TAU   = 10  (maturities 30, 60, 91, 122, 152, 182, 273, 365, 547, 730 d)

CSV column order is (T outer, D inner): col k = i_T·17 + i_D. The F-order
reshape `iv.reshape(-1, 17, 10, order='F')` produces `result[i_D, i_T] = col k`,
giving H=delta on axis 0 and W=tau on axis 1. compare_models.load_hot
mirror-reshapes with the same F-order to recover the flat CSV column order.

Dataset selection:
    --dataset full      use the full CSV (default).
    --dataset precovid  slice to date <= 2019-12-31 before splitting.

Per-cell normalisation (--norm):
    --norm on           HOT.forward normalises each (H,W) cell by its context-window
                        mean/std and denormalises the prediction with the same stats
                        (legacy behaviour). Strips level information per cell.
    --norm off          Skip the per-cell normalisation/denormalisation. Model sees
                        StandardScaler-scaled inputs and produces predictions directly
                        in scaled space. Typically reduces long-horizon bias on this
                        dataset (cf. PatchTST --revin off).

Requires:  pip install einops

Outputs (in --out_dir; default
`HOT/results/{dataset}_SPX_IV_{seq_len}_{pred_len}_HOT_tensor_dh{dh}_nb{nb}_nh{nh}_ps{ps}_{attention_type}_pe{pe}_ep{ep}{loss_suffix}{norm_suffix}`):
    pred.npy          [N_test, H=17, W=10, pred_len]   HOT tensor format  (gitignored)
    start_dates.npy   [N_test]                          datetime64[D]
    train_log.csv     epoch, train_loss, val_loss
    best_model.pt     checkpoint of best validation weights
    config.json       full hyperparam + split + git record
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FRAC           = 0.70
TEST_FRAC            = 0.20
PRECOVID_END         = "2019-12-31"
DATASET_CHOICES      = ["full", "precovid"]
TARGET_SPACE_CHOICES = ["level", "logdiff"]
H_DELTA              = 17    # delta axis (HOT H dim)
W_TAU                = 10    # maturity axis (HOT W dim)
N_IV_EXPECTED        = H_DELTA * W_TAU  # 170; checked against CSV at load time


# ─── Positional Encoding / Embeddings ─────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 64, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self._set_cos_sin_cache(max_position_embeddings,
                                self.inv_freq.device, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.max_seq_len_cached < seq_len:
            self.max_seq_len_cached = seq_len
            t    = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb  = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1]//2], x[..., x.shape[-1]//2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, l, nh, dh = x.shape
        self._set_cos_sin_cache(l, x.device, x.dtype)
        cos = self.cos_cached[:l].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:l].unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self._rotate_half(x) * sin)


# ─── Kronecker Attention ──────────────────────────────────────────────────────

class KroneckerAttention(nn.Module):
    def __init__(self, num_modes: int, d_model: int, n_head: int,
                 dropout: float = 0., rotary_emb=None,
                 mode: str = "product", rope_dims: list = []):
        super().__init__()
        self.n_head   = n_head
        self.d_model  = d_model
        self.d_head   = d_model // n_head
        self.mode     = mode
        self.rotary_emb = rotary_emb
        self.rope_dims  = rope_dims
        self.query_proj = nn.Linear(d_model, d_model * num_modes)
        self.key_proj   = nn.Linear(d_model, d_model * num_modes)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj   = nn.Linear(d_model, d_model)
        self.att_dropout  = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)

    def compute_attention(self, query, key, value, dim, use_rope=True):
        def pool(x):
            return einsum(x, "bs ... l nh dh -> bs l nh dh")

        q = query.transpose(dim, -2)
        k = key.transpose(dim, -2)
        v = value.transpose(dim, -3)
        q = q.unflatten(dim=-1, sizes=(self.n_head, self.d_head))
        k = k.unflatten(dim=-1, sizes=(self.n_head, self.d_head))
        q = self.q_norm(pool(q))
        k = self.k_norm(pool(k))
        if use_rope and self.rotary_emb is not None:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        att = einsum(q, k, "bs l1 nh d, bs l2 nh d -> bs l1 l2 nh") / math.sqrt(q.shape[3])
        att = self.att_dropout(F.softmax(att, dim=2))
        h   = einsum(att, v, "bs l1 l2 nh, bs ... l2 nh d -> bs ... l1 nh d")
        return h.transpose(dim, -3), att

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(X).split(self.d_model, dim=-1)
        key   = self.key_proj(X).split(self.d_model, dim=-1)
        value = self.value_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head))

        if self.mode == "product":
            for idx, dim in enumerate(range(1, len(X.shape) - 1)):
                use_rope = (self.rotary_emb is not None) and (dim in self.rope_dims)
                value, _ = self.compute_attention(query[idx], key[idx], value, dim, use_rope)
            value = value.flatten(start_dim=-2)
        elif self.mode == "sum":
            res = 0
            for idx, dim in enumerate(range(1, len(X.shape) - 1)):
                use_rope = (self.rotary_emb is not None) and (dim in self.rope_dims)
                v, _ = self.compute_attention(query[idx], key[idx], value, dim, use_rope)
                res += v
            value = res.flatten(start_dim=-2)

        return self.proj_dropout(self.out_proj(value))


# ─── Transformer Block ────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_hidden: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_hidden: int, d_mlp: int):
        super().__init__()
        self.w1 = nn.Linear(d_hidden, d_mlp, bias=False)
        self.w2 = nn.Linear(d_mlp, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_mlp, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_hidden: int, d_mlp: int, n_head: int, dropout: float = 0.,
                 attention_type: str = "kronecker_product", num_modes: int = 2,
                 rope_dims: list = [], input_size: int = 6):
        super().__init__()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)

        rotary_emb = None
        if len(rope_dims) > 0:
            rotary_emb = RotaryEmbedding(d_hidden // n_head, max_position_embeddings=input_size)

        assert "kronecker" in attention_type, f"Only kronecker attention supported; got {attention_type}"
        mode = attention_type.split("_")[1]
        self.attention   = KroneckerAttention(num_modes, d_hidden, n_head, dropout,
                                              rotary_emb, mode, rope_dims)
        self.feedforward = SwiGLUFeedForward(d_hidden, d_mlp)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.attention(self.norm1(X))
        h = X + self.drop1(h)
        return h + self.drop2(self.feedforward(self.norm2(h)))


# ─── HOT Model ────────────────────────────────────────────────────────────────

class HOT(nn.Module):
    """
    Higher-Order Transformer for structured IV surface forecasting.

    Input:  [B, H=17, W=10, context_length=21]
    Output: [B, H=17, W=10, prediction_length=63]

    If `norm=True` (legacy), normalises each (H,W) point over the context window
    inside forward() and denormalises the prediction with the same stats. This
    strips per-cell level information in the same way RevIN does for PatchTST —
    consider `norm=False` if long-horizon bias is observed.
    """
    def __init__(self, d_hidden: int = 128, d_mlp: int = 512, n_blocks: int = 4,
                 n_head: int = 8, patch_size: int = 4,
                 context_length: int = 21, prediction_length: int = 63,
                 attention_type: str = "kronecker_product", dropout: float = 0.0,
                 pe: str = "rope", norm: bool = True):
        super().__init__()
        assert pe in ("rope", "nope"), f"pe must be 'rope' or 'nope', got {pe!r}"
        self.patch_size        = patch_size
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.pe                = pe
        self.norm              = norm

        t_patches = math.ceil(context_length / patch_size)

        self.pos_emb = lambda x: torch.zeros_like(x).to(x.device)

        self.emb = nn.Sequential(
            nn.Conv1d(1, d_hidden, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emb_norm = nn.LayerNorm(d_hidden)

        # Input to transformer blocks: [B, H, W, Tp', d]
        # KroneckerAttention iterates dims 1..3 (H, W, Tp') → 3 modes.
        num_modes = 3
        rope_dims = [3] if pe == "rope" else []
        self.blocks = nn.ModuleList([
            TransformerBlock(d_hidden=d_hidden, d_mlp=d_mlp, n_head=n_head,
                             dropout=dropout, attention_type=attention_type,
                             num_modes=num_modes, rope_dims=rope_dims,
                             input_size=t_patches)
            for _ in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, prediction_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, T]
        bs, H, W, T = x.shape

        if self.norm:
            mu  = x.mean(dim=-1, keepdim=True)
            std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x_input = (x - mu) / std
        else:
            x_input = x

        if T % self.patch_size != 0:
            pad   = self.patch_size - (T % self.patch_size)
            x_pad = torch.cat([x_input, x_input[..., -1:].repeat(1, 1, 1, pad)], dim=-1)
        else:
            x_pad = x_input

        Tp = x_pad.shape[-1]
        h = x_pad.reshape(bs * H * W, Tp).unsqueeze(1)  # [B*H*W, 1, Tp]
        h = self.emb(h).transpose(1, 2)                 # [B*H*W, Tp', d]
        h = self.emb_norm(h)
        Tp2 = h.shape[1]
        h = h.view(bs, H, W, Tp2, h.shape[-1])          # [B, H, W, Tp', d]

        h = h + self.pos_emb(h)

        for block in self.blocks:
            h = block(h)

        logits = self.head(h.mean(dim=3))                # [B, H, W, pred]

        if self.norm:
            return (logits * std) + mu
        return logits


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


def _to_grid(iv: np.ndarray) -> np.ndarray:
    """
    Reshape [N, 170] → [N, H=17 (delta), W=10 (tau)] using F-order.

    CSV columns are sorted (T outer, D inner): col k = i_T·17 + i_D.
    F-order reshape with shape (17, 10): result[i_D, i_T] = flat[i_D + 17·i_T] = col k. ✓
    Mirror-reshape lives in compare_models.load_hot.
    """
    return iv.reshape(-1, H_DELTA, W_TAU, order="F")


def load_splits(csv_path: str, dataset: str, target_space: str,
                seq_len: int, pred_len: int):
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = _slice_dataset(df, dataset)

    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    n_iv = len(iv_cols)
    if n_iv != N_IV_EXPECTED:
        raise ValueError(
            f"HOT expects {N_IV_EXPECTED} = {H_DELTA}×{W_TAU} iv_ columns; "
            f"found {n_iv}. Update H_DELTA/W_TAU if the data spec changed."
        )

    iv_raw     = df[iv_cols].to_numpy(dtype=np.float32)
    dates_full = df["date"].to_numpy(dtype="datetime64[D]")
    data, dates = _apply_target_space(iv_raw, dates_full, target_space)

    T = len(data)
    n_train = int(T * TRAIN_FRAC)
    n_test  = int(T * TEST_FRAC)
    n_val   = T - n_train - n_test

    b1 = [0,           n_train - seq_len,  T - n_test - seq_len]
    b2 = [n_train,     n_train + n_val,    T]

    scaler = StandardScaler().fit(data[b1[0]:b2[0]])
    iv = scaler.transform(data).astype(np.float32)

    iv_grid = _to_grid(iv)   # [T, H, W]

    def _windows(start, end):
        sl = iv_grid[start:end]
        n  = len(sl) - seq_len - pred_len + 1
        # X: [N, H, W, seq_len]  y: [N, H, W, pred_len]
        X = np.stack([sl[i : i+seq_len].transpose(1, 2, 0)         for i in range(n)])
        y = np.stack([sl[i+seq_len : i+seq_len+pred_len].transpose(1, 2, 0) for i in range(n)])
        return X.astype(np.float32), y.astype(np.float32)

    X_tr, y_tr = _windows(b1[0], b2[0])
    X_va, y_va = _windows(b1[1], b2[1])
    X_te, _y   = _windows(b1[2], b2[2])

    test_slice_dates = dates[b1[2]:b2[2]]
    test_start_dates = np.array([test_slice_dates[i + seq_len] for i in range(len(X_te))])

    return X_tr, y_tr, X_va, y_va, X_te, test_start_dates, dict(
        T=T, n_iv=n_iv, n_train=n_train, n_val=n_val, n_test=n_test,
        train_end_date=str(dates[n_train - 1]),
        train_windows=len(X_tr), val_windows=len(X_va), test_windows=len(X_te),
    )


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
        "model":          "HOT",
        "dataset":        args.dataset,
        "target_space":   args.target_space,
        "csv_path":       args.csv_path,
        "seq_len":        args.seq_len,
        "pred_len":       args.pred_len,
        "d_hidden":       args.d_hidden,
        "d_mlp":          args.d_mlp,
        "n_blocks":       args.n_blocks,
        "n_head":         args.n_head,
        "patch_size":     args.patch_size,
        "attention_type": args.attention_type,
        "pe":             args.pe,
        "norm":           args.norm,
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
        "h_delta":        H_DELTA,
        "w_tau":          W_TAU,
        "n_train":        info["n_train"],
        "n_val":          info["n_val"],
        "n_test":         info["n_test"],
        "train_end_date": info["train_end_date"],
        "git_commit":     _git_commit(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train HOT on SPX IV surface")
    ap.add_argument("--csv_path",        default="SPX_surfaces.csv")
    ap.add_argument("--dataset",         default="full", choices=DATASET_CHOICES,
                    help="full = entire CSV; precovid = dates <= 2019-12-31")
    ap.add_argument("--target_space",    default="level", choices=TARGET_SPACE_CHOICES,
                    help="level = train on raw IV (default); logdiff = train on "
                         "log(IV)[1:]-log(IV)[:-1]. logdiff loses one day at the front.")
    ap.add_argument("--seq_len",         type=int,   default=21)
    ap.add_argument("--pred_len",        type=int,   default=63)
    ap.add_argument("--d_hidden",        type=int,   default=128)
    ap.add_argument("--d_mlp",           type=int,   default=512)
    ap.add_argument("--n_blocks",        type=int,   default=4)
    ap.add_argument("--n_head",          type=int,   default=8)
    ap.add_argument("--patch_size",      type=int,   default=4)
    ap.add_argument("--attention_type",  default="kronecker_product",
                    choices=["kronecker_product", "kronecker_sum"])
    ap.add_argument("--pe",              default="rope",
                    choices=["rope", "nope"],
                    help="Positional encoding: 'rope' applies RoPE on the temporal "
                         "dim (legacy default); 'nope' disables it.")
    ap.add_argument("--norm",            default="on", choices=["on", "off"],
                    help="Per-cell window norm/denorm inside HOT.forward. "
                         "'on' = legacy. 'off' = pass scaled inputs through directly "
                         "(typically reduces long-horizon bias on this dataset).")
    ap.add_argument("--dropout",         type=float, default=0.1)
    ap.add_argument("--epochs",          type=int,   default=100)
    ap.add_argument("--batch_size",      type=int,   default=32)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--weight_decay",    type=float, default=1e-2)
    ap.add_argument("--patience",        type=int,   default=15)
    ap.add_argument("--device",          default="auto")
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--out_dir",         default=None)
    ap.add_argument("--predict_only",    action="store_true",
                    help="Skip training; load best_model.pt + config.json from --out_dir, "
                         "run inference, write pred.npy.")
    ap.add_argument("--loss",            default="mse",
                    choices=["mse", "huber_scaled"])
    ap.add_argument("--huber_delta",     type=float, default=1.0)
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
                  "d_hidden", "d_mlp", "n_blocks", "n_head", "patch_size", "attention_type",
                  "pe", "norm", "dropout", "batch_size", "loss", "huber_delta"):
            if k in cfg:
                setattr(args, k, cfg[k])
        print(f"[predict_only] {cfg_path}")

    if args.out_dir is None:
        loss_suffix = {
            "mse":          "",
            "huber_scaled": f"_losshuberscaled_d{args.huber_delta:g}",
        }[args.loss]
        norm_suffix = "" if args.norm == "on" else "_nonorm"
        args.out_dir = (f"HOT/results/"
                        f"{args.dataset}_{args.target_space}_SPX_IV_"
                        f"{args.seq_len}_{args.pred_len}"
                        f"_HOT_tensor_dh{args.d_hidden}_nb{args.n_blocks}"
                        f"_nh{args.n_head}_ps{args.patch_size}_{args.attention_type}"
                        f"_pe{args.pe}_ep{args.epochs}{loss_suffix}{norm_suffix}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Dataset    : {args.dataset}")
    print(f"Target     : {args.target_space}")
    print(f"Device     : {device}")
    print(f"Output dir : {args.out_dir}")
    print(f"Attention  : {args.attention_type}  pe={args.pe}  norm={args.norm}")

    print("\nLoading data...")
    X_tr, y_tr, X_va, y_va, X_te, test_dates, info = load_splits(
        args.csv_path, args.dataset, args.target_space, args.seq_len, args.pred_len)
    print(f"  T={info['T']}  n_iv={info['n_iv']}  "
          f"train={info['train_windows']}  "
          f"val={info['val_windows']}  test={info['test_windows']} windows")
    print(f"  Train ends {info['train_end_date']}")
    print(f"  X shape (per split): {X_tr.shape}  [N, H={H_DELTA}, W={W_TAU}, seq]")

    model = HOT(d_hidden=args.d_hidden, d_mlp=args.d_mlp, n_blocks=args.n_blocks,
                n_head=args.n_head, patch_size=args.patch_size,
                context_length=args.seq_len, prediction_length=args.pred_len,
                attention_type=args.attention_type, dropout=args.dropout,
                pe=args.pe, norm=(args.norm == "on")).to(device)
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
    # preds: [N, H, W, pred_len]

    np.save(os.path.join(args.out_dir, "pred.npy"),        preds)
    np.save(os.path.join(args.out_dir, "start_dates.npy"), test_dates)
    print(f"  pred.npy        shape={preds.shape}  (HOT format [N, H={H_DELTA}, W={W_TAU}, pred])")
    print(f"  start_dates.npy range={test_dates[0]} → {test_dates[-1]}")
    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
