"""
PatchTST — channel-independent patch-based Transformer for multivariate
time-series forecasting, with optional RevIN normalisation.

Hyperparameters (passed to PatchTST.__init__):
    c_in           int    — number of input channels (e.g. IV cells).
    seq_len        int    — input window length (lookback).
    pred_len       int    — forecast horizon length.
    patch_len      int    — patch length used to tokenise each channel.
                            Default: 7.
    stride         int    — stride between consecutive patches. Default: 7.
    d_model        int    — transformer model dim. Default: 128.
    n_heads        int    — number of self-attention heads. Default: 16.
    n_layers       int    — number of transformer encoder layers. Default: 3.
    d_ff           int    — feed-forward inner dim. Default: 256.
    attn_dropout   float  — dropout on attention weights. Default: 0.
    dropout        float  — dropout in feed-forward / projections. Default: 0.
    head_dropout   float  — dropout in the prediction head. Default: 0.
    res_attention  bool   — pass attention scores residually between layers.
                            Default: True.
    revin          bool   — joint per-window RevIN norm/denorm around the
                            model: strips mean/std jointly over (L, C),
                            preserving cross-channel structure within the
                            window. Default: True.
    affine         bool   — learnable affine in RevIN (only if revin=True).
                            Default: False.
    padding_patch  str    — 'end' replicates the last value `stride` times
                            before unfolding (adds +1 patch). Default: 'end'.
    decomposition  bool   — DLinear-style trend/residual moving-average
                            decomposition. When True, two independent
                            backbones+heads run on the trend and residual
                            components and their outputs are summed.
                            Default: False.
    kernel_size    int    — moving-average kernel for decomposition (must
                            be odd). Only used if decomposition=True.
                            Default: 25.
    store_attn     bool   — if True, each encoder layer saves the most recent
                            attention weights to its `.attn` attribute
                            (shape [B*C, n_heads, patch_num, patch_num])
                            so attention maps can be plotted post-hoc.
                            Default: False.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RevIN(nn.Module):
    """
    Joint RevIN: per-window mean/std reduction over all non-batch axes
    (L and C jointly). Strips overall window level/scale while preserving
    cross-channel structure within the window. The `num_features` argument
    is retained for API compatibility but only affects affine parameter
    shape (which is now scalar regardless).
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            # Joint RevIN: one scalar pair, broadcasting across all positions.
            self.affine_weight = nn.Parameter(torch.ones(1))
            self.affine_bias   = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(mode)

    def _get_statistics(self, x: Tensor):
        dims = tuple(range(1, x.ndim))
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


class _Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.dims)


class _MovingAvg(nn.Module):
    """Boundary-padded 1-D moving average preserving sequence length."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, C]
        x = torch.cat([
            x[:, :1].expand(-1, self.pad, -1),
            x,
            x[:, -1:].expand(-1, self.pad, -1),
        ], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class _SeriesDecomp(nn.Module):
    """Split a series into (residual, trend) via moving-average smoothing."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: Tensor):  # [B, T, C] -> (res, trend)
        trend = self.moving_avg(x)
        return x - trend, trend


def _positional_encoding(q_len: int, d_model: int) -> nn.Parameter:
    """Learnable 'zeros'-initialised positional encoding."""
    W = torch.empty((q_len, d_model))
    nn.init.uniform_(W, -0.02, 0.02)
    return nn.Parameter(W, requires_grad=True)


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
            output, attn, scores = self.sdp_attn(q_s, k_s, v_s, prev=prev)
        else:
            output, attn = self.sdp_attn(q_s, k_s, v_s)
            scores = None
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        output = self.to_out(output)
        return output, attn, scores


class _TSTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int = 256, attn_dropout: float = 0.,
                 dropout: float = 0., res_attention: bool = True,
                 store_attn: bool = False):
        super().__init__()
        self.res_attention = res_attention
        self.store_attn = store_attn
        self.self_attn = _MultiheadAttention(d_model, n_heads, attn_dropout, dropout, res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(d_model), _Transpose(1, 2))
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(d_model), _Transpose(1, 2))

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):
        src2, attn, scores = self.self_attn(src, prev=prev)
        if self.store_attn:
            self.attn = attn.detach()
        src = self.norm_attn(src + self.dropout_attn(src2))
        src2 = self.ff(src)
        src = self.norm_ffn(src + self.dropout_ffn(src2))
        if self.res_attention:
            return src, scores
        return src


class _TSTEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 attn_dropout: float, dropout: float, n_layers: int,
                 res_attention: bool, store_attn: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            _TSTEncoderLayer(d_model, n_heads, d_ff, attn_dropout, dropout,
                             res_attention, store_attn)
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
    def __init__(self, patch_num: int, patch_len: int, d_model: int,
                 n_heads: int, d_ff: int, attn_dropout: float, dropout: float,
                 n_layers: int, res_attention: bool, store_attn: bool = False):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.W_P   = nn.Linear(patch_len, d_model)
        self.W_pos = _positional_encoding(patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _TSTEncoder(d_model, n_heads, d_ff,
                                   attn_dropout, dropout, n_layers,
                                   res_attention, store_attn)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, patch_len, patch_num]
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)        # [B, C, patch_num, patch_len]
        x = self.W_P(x)                  # [B, C, patch_num, d_model]

        u = x.reshape(-1, x.shape[2], x.shape[3])  # [B*C, patch_num, d_model]
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)                        # [B*C, patch_num, d_model]

        z = z.reshape(-1, n_vars, z.shape[-2], z.shape[-1])
        return z.permute(0, 1, 3, 2)


class _FlattenHead(nn.Module):
    def __init__(self, nf: int, target_window: int, head_dropout: float = 0.):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear  = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear(self.flatten(x)))


class _PatchTSTBackbone(nn.Module):
    """RevIN → patch → channel-independent transformer → flatten head."""
    def __init__(self, c_in: int, seq_len: int, pred_len: int,
                 patch_len: int, stride: int,
                 d_model: int, n_heads: int, n_layers: int, d_ff: int,
                 attn_dropout: float, dropout: float, head_dropout: float,
                 res_attention: bool, revin: bool, affine: bool,
                 padding_patch: str, store_attn: bool):
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

        self.backbone = _TSTiEncoder(patch_num, patch_len, d_model, n_heads,
                                     d_ff, attn_dropout, dropout, n_layers,
                                     res_attention, store_attn)
        nf = d_model * patch_num
        self.head = _FlattenHead(nf, pred_len, head_dropout)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, C]
        if self.revin:
            x = self.revin_layer(x, "norm")
        z = x.permute(0, 2, 1)
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)              # [B, C, patch_len, patch_num]

        z = self.backbone(z)
        z = self.head(z)
        z = z.permute(0, 2, 1)
        if self.revin:
            z = self.revin_layer(z, "denorm")
        return z


class PatchTST(nn.Module):
    """
    Channel-independent PatchTST with RevIN, optional trend/residual
    decomposition and optional attention-weight capture.

    Input:  [B, seq_len, C]  (scaled)
    Output: [B, pred_len, C] (scaled)

    padding_patch: 'end' replicates the last value `stride` times before unfolding,
    yielding patch_num+1 patches. Matches legacy run_longExp.py default.

    decomposition: when True, splits the input into trend + residual via a
    moving-average decomposition and runs two independent backbones+heads
    (one per component), summing their outputs.

    store_attn: when True, each encoder layer caches its attention weights
    in `<layer>.attn` after every forward pass. With decomposition the two
    sub-models are reachable via `model.model_trend` and `model.model_res`;
    without decomposition use `model.model`.
    """
    def __init__(self, c_in: int, seq_len: int, pred_len: int,
                 patch_len: int = 7, stride: int = 7,
                 d_model: int = 128, n_heads: int = 16, n_layers: int = 3,
                 d_ff: int = 256, attn_dropout: float = 0., dropout: float = 0.,
                 head_dropout: float = 0., res_attention: bool = True,
                 revin: bool = True, affine: bool = False,
                 padding_patch: str = "end",
                 decomposition: bool = False, kernel_size: int = 25,
                 store_attn: bool = False):
        super().__init__()
        self.decomposition = decomposition

        backbone_kwargs = dict(
            c_in=c_in, seq_len=seq_len, pred_len=pred_len,
            patch_len=patch_len, stride=stride,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            attn_dropout=attn_dropout, dropout=dropout, head_dropout=head_dropout,
            res_attention=res_attention, revin=revin, affine=affine,
            padding_patch=padding_patch, store_attn=store_attn,
        )

        if decomposition:
            self.decomp_layer = _SeriesDecomp(kernel_size)
            self.model_trend = _PatchTSTBackbone(**backbone_kwargs)
            self.model_res   = _PatchTSTBackbone(**backbone_kwargs)
        else:
            self.model = _PatchTSTBackbone(**backbone_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.decomposition:
            res, trend = self.decomp_layer(x)
            return self.model_res(res) + self.model_trend(trend)
        return self.model(x)
