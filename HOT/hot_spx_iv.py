#!/usr/bin/env python3
"""
HOT (Higher-Order Transformer) training script for SPX IV surface forecasting.

Treats the 400 IV features as a 20×20 structured tensor (moneyness × tau)
and applies Kronecker-product attention across both spatial axes.

Requires:  pip install einops

Outputs (in --out_dir):
    pred.npy          [N_test, H=20, W=20, pred_len]  HOT tensor format
    start_dates.npy   [N_test]                          datetime64[D]
    train_log.csv     epoch, train_loss, val_loss
    best_model.pt     checkpoint of best validation weights
"""

import argparse
import csv
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
N_IV       = 400
H_MONO     = 20   # moneyness axis (HOT H dim)
W_TAU      = 20   # tau axis       (HOT W dim)


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

    Input:  [B, H=20, W=20, context_length=21]
    Output: [B, H=20, W=20, prediction_length=63]

    Internally normalises each (H,W) point over the context window.
    """
    def __init__(self, d_hidden: int = 128, d_mlp: int = 512, n_blocks: int = 4,
                 n_head: int = 8, patch_size: int = 4,
                 context_length: int = 21, prediction_length: int = 63,
                 attention_type: str = "kronecker_product", dropout: float = 0.0,
                 pe: str = "rope"):
        super().__init__()
        assert pe in ("rope", "nope"), f"pe must be 'rope' or 'nope', got {pe!r}"
        self.patch_size        = patch_size
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.pe                = pe

        t_patches = math.ceil(context_length / patch_size)

        # 'nope' and 'rope' both set the additive pos_emb to zero. RoPE is applied
        # inside the attention layer along the temporal dim (rope_dims=[3]).
        self.pos_emb = lambda x: torch.zeros_like(x).to(x.device)

        self.emb = nn.Sequential(
            nn.Conv1d(1, d_hidden, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emb_norm = nn.LayerNorm(d_hidden)

        # Input to transformer blocks: [B, H=20, W=20, Tp', d]
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

        mu  = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - mu) / std

        if T % self.patch_size != 0:
            pad   = self.patch_size - (T % self.patch_size)
            x_pad = torch.cat([x_norm, x_norm[..., -1:].repeat(1, 1, 1, pad)], dim=-1)
        else:
            x_pad = x_norm

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
        return (logits * std) + mu


# ─── Data ─────────────────────────────────────────────────────────────────────

def _to_grid(iv: np.ndarray) -> np.ndarray:
    """
    Reshape [N, 400] → [N, H_MONO=20, W_TAU=20] using F-order.

    CSV columns are sorted (tau outer, moneyness inner): col k = i_tau*20 + i_mono.
    F-order reshape: result[i_mono, i_tau] = flat[i_mono + 20*i_tau] = flat[k]. ✓
    """
    return iv.reshape(-1, H_MONO, W_TAU, order="F")


def load_splits(csv_path: str, seq_len: int, pred_len: int):
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    # CSV column order = (tau outer, moneyness inner). Required for the F-order
    # reshape `iv.reshape(-1, 20, 20, order='F')` → [i_mono, i_tau] mapping.
    # Do NOT sort — alphabetical order scrambles the surface.
    iv_cols = [c for c in df.columns if c.startswith("iv_")]
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

    # Convert to grid: [T, H, W]
    iv_grid = _to_grid(iv)

    def _windows(start, end):
        sl = iv_grid[start:end]  # [T_slice, H, W]
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
    ap = argparse.ArgumentParser(description="Train HOT on SPX IV surface")
    ap.add_argument("--csv_path",        default="SPX_surfaces.csv")
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
                         "dim (matches legacy ts_tensor.py default); 'nope' disables it.")
    ap.add_argument("--dropout",         type=float, default=0.1)
    ap.add_argument("--epochs",          type=int,   default=100)
    ap.add_argument("--batch_size",      type=int,   default=32)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--weight_decay",    type=float, default=1e-2)
    ap.add_argument("--patience",        type=int,   default=15)
    ap.add_argument("--device",          default="auto")
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--out_dir",         default=None)
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():           device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)

    if args.out_dir is None:
        args.out_dir = (f"HOT/results/SPX_IV_{args.seq_len}_{args.pred_len}"
                        f"_HOT_tensor_dh{args.d_hidden}_nb{args.n_blocks}"
                        f"_nh{args.n_head}_ps{args.patch_size}_{args.attention_type}"
                        f"_pe{args.pe}_ep{args.epochs}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Output dir : {args.out_dir}")
    print(f"Attention  : {args.attention_type}")

    print("\nLoading data...")
    X_tr, y_tr, X_va, y_va, X_te, test_dates, info = load_splits(
        args.csv_path, args.seq_len, args.pred_len)
    print(f"  T={info['T']}  train={info['train_windows']}  "
          f"val={info['val_windows']}  test={info['test_windows']} windows")
    print(f"  X shape (per split): {X_tr.shape}  [N, H={H_MONO}, W={W_TAU}, seq]")

    model = HOT(d_hidden=args.d_hidden, d_mlp=args.d_mlp, n_blocks=args.n_blocks,
                n_head=args.n_head, patch_size=args.patch_size,
                context_length=args.seq_len, prediction_length=args.pred_len,
                attention_type=args.attention_type, dropout=args.dropout,
                pe=args.pe).to(device)
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
    # preds: [N, H, W, pred_len]
    preds = np.concatenate(preds, axis=0).astype(np.float32)

    np.save(os.path.join(args.out_dir, "pred.npy"),        preds)
    np.save(os.path.join(args.out_dir, "start_dates.npy"), test_dates)
    print(f"  pred.npy        shape={preds.shape}  (HOT format [N, H, W, pred])")
    print(f"  start_dates.npy range={test_dates[0]} → {test_dates[-1]}")
    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
