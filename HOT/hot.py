"""
HOT (Higher-Order Transformer) for structured IV surface forecasting.

Treats the IV grid as an [H × W] structured tensor (e.g. moneyness × tau)
and applies Kronecker attention across both spatial axes plus the
temporal patches. H and W are inferred at runtime from the input tensor;
the model parameters do not depend on either axis size.

Requires:  pip install einops

Ablation knobs to revisit later (not in the tuning grid yet):
    - `head_type`: 'flatten' vs 'mean'. Default 'flatten'.

Deviations from the reference HOT implementation:
    - RevIN-style per-cell window normalisation inside forward() (RevIN-style).
    - Conv1d patcher embedding over the temporal axis (PatchTST-style).
    - PatchTST-style flatten+linear head over all temporal patches is the
      default (`head_type='flatten'`); the mean-pool head is kept for
      ablation (`head_type='mean'`).
    - Unified `pe` flag: 'rope' applies RoPE on H, W, and temporal axes;
      'nope' disables all positional encoding. The separate `spatial_pe` /
      `h_max` / `w_max` interface and the SpatialPE module have been removed.

Hyperparameters (passed to HOT.__init__):
    d_hidden           int    — transformer hidden dim. Default: 128.
                                SwiGLU feed-forward inner dim is always
                                derived as `d_mlp = 4 * d_hidden`.
    n_blocks           int    — number of transformer blocks. Default: 4.
    n_head             int    — number of attention heads per Kronecker mode.
                                Default: 2.
    patch_size         int    — temporal patch size for the patcher conv.
                                Default: 4.
    context_length     int    — input window length (lookback). Default: 21.
    prediction_length  int    — forecast horizon length. Default: 63.
    attention_type     str    — 'kronecker_product' or 'kronecker_sum'.
                                Default: 'kronecker_product'.
    dropout            float  — dropout in encoder blocks: residual dropout
                                around attention and FFN sublayers, dropout
                                inside the FFN, attention output projection,
                                and patcher embedding. Default: 0.0.
    attn_dropout       float  — dropout applied to attention weights after
                                softmax, inside Kronecker attention. Default: 0.0.
    head_dropout       float  — dropout in the prediction head, before the
                                final Linear projection to pred_len. Default: 0.0.
    pe                 str    — positional encoding: 'rope' applies RoPE on
                                the H, W, and temporal axes; 'nope' disables
                                all positional encoding. Default: 'rope'.
    norm               bool   — joint per-window normalisation: strip the
                                surface-wide mean and std across (H, W, T),
                                apply, then add back at the output.
                                Preserves cross-cell structure within a
                                window while removing the overall vol
                                level/scale. Default: True.
    head_type          str    — 'flatten' (default) applies a Flatten+Linear
                                head over [Tp, d]; 'mean' averages over Tp
                                before the linear head. (PatchTST uses the 'flatten' approach, while the reference HOT uses 'mean'.)
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


class KroneckerAttention(nn.Module):
    def __init__(self, num_modes: int, d_model: int, n_head: int,
                 dropout: float = 0., attn_dropout: float = 0.,
                 rotary_emb=None, mode: str = "product", rope_dims: list = []):
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
        self.att_dropout  = nn.Dropout(attn_dropout)
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
    def __init__(self, d_hidden: int, d_mlp: int, n_head: int,
                 dropout: float = 0., attn_dropout: float = 0.,
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
        self.attention   = KroneckerAttention(num_modes, d_hidden, n_head,
                                              dropout=dropout,
                                              attn_dropout=attn_dropout,
                                              rotary_emb=rotary_emb,
                                              mode=mode,
                                              rope_dims=rope_dims)
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
    def __init__(self, d_hidden: int = 128, n_blocks: int = 4,
                 n_head: int = 2, patch_size: int = 4,
                 context_length: int = 21, prediction_length: int = 63,
                 attention_type: str = "kronecker_product",
                 dropout: float = 0.0, attn_dropout: float = 0.0,
                 head_dropout: float = 0.0,
                 pe: str = "rope", norm: bool = True,
                 head_type: str = "flatten"):
        super().__init__()
        assert pe in ("rope", "nope"), f"pe must be 'rope' or 'nope', got {pe!r}"
        for name, val in [("dropout", dropout), ("attn_dropout", attn_dropout), ("head_dropout", head_dropout)]:
            if not (0.0 <= val < 1.0):
                raise ValueError(f"{name} must be in [0, 1), got {val}")
        d_mlp = 4 * d_hidden
        self.patch_size        = patch_size
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.pe                = pe
        self.norm              = norm
        self.head_type         = head_type

        t_patches = math.ceil(context_length / patch_size)

        self.emb = nn.Sequential(
            nn.Conv1d(1, d_hidden, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emb_norm = nn.LayerNorm(d_hidden)

        # Input to transformer blocks: [B, H, W, Tp', d]
        # KroneckerAttention iterates dims 1..3 (H, W, Tp') → 3 modes.
        num_modes = 3
        rope_dims = [1, 2, 3] if pe == "rope" else []
        self.blocks = nn.ModuleList([
            TransformerBlock(d_hidden=d_hidden, d_mlp=d_mlp, n_head=n_head,
                             dropout=dropout, attn_dropout=attn_dropout,
                             attention_type=attention_type,
                             num_modes=num_modes, rope_dims=rope_dims,
                             input_size=t_patches)
            for _ in range(n_blocks)
        ])

        if head_type == "flatten":
            self.head = nn.Sequential(
                nn.LayerNorm(d_hidden),
                nn.Flatten(start_dim=-2),
                nn.Dropout(head_dropout),
                nn.Linear(d_hidden * t_patches, prediction_length),
            )
        elif head_type == "mean":
            self.head = nn.Sequential(
                nn.LayerNorm(d_hidden),
                nn.Dropout(head_dropout),
                nn.Linear(d_hidden, prediction_length),
            )
        else:
            raise ValueError(f"head_type must be 'flatten' or 'mean', got {head_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, T]
        bs, H, W, T = x.shape

        if self.norm:
            mu  = x.mean(dim=(1, 2, 3), keepdim=True)
            std = torch.sqrt(x.var(dim=(1, 2, 3), keepdim=True, unbiased=False) + 1e-5)
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

        for block in self.blocks:
            h = block(h)

        if self.head_type == "flatten":
            logits = self.head(h)                       # head includes Flatten over [Tp, d]
        else:
            logits = self.head(h.mean(dim=3))           # [B, H, W, pred]

        if self.norm:
            return (logits * std) + mu
        return logits
