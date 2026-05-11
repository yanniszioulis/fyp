"""
HOT (Higher-Order Transformer) for structured IV surface forecasting.

Treats the IV grid as an [H × W] structured tensor (e.g. moneyness × tau)
and applies Kronecker attention across both spatial axes plus the
temporal patches. H and W are inferred at runtime from the input tensor;
the model parameters do not depend on either axis size.

Requires:  pip install einops

Hyperparameters (passed to HOT.__init__):
    d_hidden           int    — transformer hidden dim. Default: 128.
    d_mlp              int    — SwiGLU feed-forward inner dim. Default: 512.
    n_blocks           int    — number of transformer blocks. Default: 4.
    n_head             int    — number of attention heads per Kronecker mode.
                                Default: 8.
    patch_size         int    — temporal patch size for the patcher conv.
                                Default: 4.
    context_length     int    — input window length (lookback). Default: 21.
    prediction_length  int    — forecast horizon length. Default: 63.
    attention_type     str    — 'kronecker_product' or 'kronecker_sum'.
                                Default: 'kronecker_product'.
    dropout            float  — dropout in attention / feed-forward / head.
                                Default: 0.0.
    pe                 str    — temporal positional encoding: 'rope' applies
                                RoPE on the temporal dim; 'nope' disables it.
                                Default: 'rope'.
    norm               bool   — per-cell window norm/denorm inside forward()
                                (RevIN-style). Default: True.
    spatial_pe         str    — 2D PE on the (H, W) grid: 'none', 'lape'
                                (learned [h_max, w_max, d]), or 'sin2d'
                                (fixed sinusoidal; needs d_hidden % 4 == 0).
                                Default: 'none'.
    h_max              int    — max H supported by the spatial PE. Default: 32.
    w_max              int    — max W supported by the spatial PE. Default: 32.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


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


class SpatialPE(nn.Module):
    """
    2D spatial positional embedding for [B, H, W, T, d] tensors.

    Returns a [1, H, W, 1, d] tensor that broadcast-adds to the input.

    Modes:
      none   identity; no parameters. (HOT.forward skips the add.)
      lape   learned [h_max, w_max, d] parameter (LAPE-style, Omranpour et al.).
             Initialised to 0.02·randn so it's a gentle injection on top of the
             unit-scale post-LayerNorm patch embeddings.
      sin2d  fixed sinusoidal 2D PE: first d/2 channels encode the H index,
             last d/2 encode the W index. Requires d divisible by 4. No params.

    Without this, KroneckerAttention is permutation-equivariant per spatial
    axis: shuffling rows produces shuffled outputs. With LAPE the grid gains
    a per-cell positional fingerprint independent of content.
    """
    def __init__(self, mode: str, d_hidden: int, h_max: int = 32, w_max: int = 32):
        super().__init__()
        self.mode  = mode
        self.h_max = h_max
        self.w_max = w_max
        if mode == "none":
            return
        if mode == "lape":
            self.pe = nn.Parameter(0.02 * torch.randn(h_max, w_max, d_hidden))
        elif mode == "sin2d":
            self.register_buffer("pe", self._build_sin2d(h_max, w_max, d_hidden),
                                 persistent=False)
        else:
            raise ValueError(f"Unknown spatial_pe mode {mode!r}; "
                             f"choose from 'none', 'lape', 'sin2d'.")

    @staticmethod
    def _build_sin2d(h_max: int, w_max: int, d: int) -> torch.Tensor:
        if d % 4 != 0:
            raise ValueError(f"sin2d PE requires d_hidden divisible by 4, got {d}")
        d_half = d // 2

        def _sincos(n: int, dim: int) -> torch.Tensor:
            pos = torch.arange(n, dtype=torch.float).unsqueeze(1)            # [n, 1]
            div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                            * (-math.log(10000.0) / dim))                    # [dim/2]
            out = torch.zeros(n, dim)
            out[:, 0::2] = torch.sin(pos * div)
            out[:, 1::2] = torch.cos(pos * div)
            return out

        pe = torch.zeros(h_max, w_max, d)
        pe[:, :, :d_half] = _sincos(h_max, d_half).unsqueeze(1)               # broadcast over W
        pe[:, :, d_half:] = _sincos(w_max, d_half).unsqueeze(0)               # broadcast over H
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[1], x.shape[2]
        if H > self.h_max or W > self.w_max:
            raise ValueError(f"Input H={H}, W={W} exceeds spatial-PE capacity "
                             f"({self.h_max}, {self.w_max}); raise h_max/w_max in HOT()")
        return self.pe[:H, :W].unsqueeze(0).unsqueeze(3)                      # [1, H, W, 1, d]


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


class HOT(nn.Module):
    """
    Higher-Order Transformer for structured IV surface forecasting.

    Input:  [B, H, W, context_length]
    Output: [B, H, W, prediction_length]

    H and W are inferred at runtime from the input tensor; the model
    parameters do not depend on either axis size (Kronecker attention
    pools over the spatial dims dynamically).

    If `norm=True`, normalises each (H, W) point over the context window
    inside forward() and denormalises the prediction with the same stats.
    This strips per-cell level information in the same way RevIN does.
    """
    def __init__(self, d_hidden: int = 128, d_mlp: int = 512, n_blocks: int = 4,
                 n_head: int = 8, patch_size: int = 4,
                 context_length: int = 21, prediction_length: int = 63,
                 attention_type: str = "kronecker_product", dropout: float = 0.0,
                 pe: str = "rope", norm: bool = True,
                 spatial_pe: str = "none", h_max: int = 32, w_max: int = 32):
        super().__init__()
        assert pe in ("rope", "nope"), f"pe must be 'rope' or 'nope', got {pe!r}"
        self.patch_size        = patch_size
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.pe                = pe
        self.norm              = norm
        self.spatial_pe        = spatial_pe
        self.has_spatial_pe    = (spatial_pe != "none")

        t_patches = math.ceil(context_length / patch_size)

        self.pos_emb = SpatialPE(spatial_pe, d_hidden, h_max=h_max, w_max=w_max)

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
        Tp_iv = h.shape[1]
        h = h.view(bs, H, W, Tp_iv, h.shape[-1])        # [B, H, W, Tp', d]

        if self.has_spatial_pe:
            h = h + self.pos_emb(h)

        for block in self.blocks:
            h = block(h)

        logits = self.head(h.mean(dim=3))               # [B, H, W, pred]

        if self.norm:
            return (logits * std) + mu
        return logits
