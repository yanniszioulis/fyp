"""Shared framework for the IV-surface forecasting family.

Model-agnostic pieces every model imports: the Config, the attention primitives,
the instance-norm that defines the target, and the loss / windowing utilities.

Shared contract: each model takes a window of standardised log-IV z (B, L, M, T),
centres each cell on today (instance_norm), and predicts netDelta (B, n_horizons, M, T)
— the per-cell change from today to each horizon (ẑ = z_today + netDelta, so a zero
prediction is the random-walk null). Training minimises surface_loss (uniform MSE on
the standardised change).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# config
@dataclass
class Config:
    """Shared hyperparameters; models ignore fields they don't use. build_model in
    train.py overrides M / T / L / horizons and the coordinate grids from the parsed
    dataset; the defaults match the current SPX grid so the smoke tests run standalone.
    """
    M: int = 11                      # moneyness points
    T: int = 10                      # maturity points
    L: int = 63                      # input window length (trading days); fixed project-wide
    horizons: Tuple[int, ...] = (1, 5, 10, 21)   # forecast horizons (days ahead)
    d: int = 48                      # embedding dim (must be divisible by n_heads)
    n_heads: int = 4
    n_layers: int = 2
    d_ff_mult: int = 2               # FFN hidden = d_ff_mult * d
    d_head_hidden: int = 48          # hidden width of the per-cell output head MLP
    dropout: float = 0.1
    # financial grid coordinates (M strikes, ATM=0 at i=5; T maturities); only SANTA + ablations use them
    k_grid: Tuple[float, ...] = tuple(
        round(-0.10 + 0.02 * i, 6) for i in range(11)
    )
    tau_grid_years: Tuple[float, ...] = tuple(
        round(d_ / 365.0, 6)
        for d_ in (30, 45, 60, 90, 120, 150, 180, 240, 300, 365)
    )

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)

    @property
    def d_head(self) -> int:
        return self.d // self.n_heads


# instance normalisation — defines the forecasting target
def instance_norm(z: torch.Tensor):
    """Centre each cell's window on today (reversible instance norm).

    z (B, L, M, T) → u centred history (u[:, -1] == 0), L0 today's level (B, M, T),
    s_tilde per-cell window std (B, M, T), kept as an amplitude cue (not divided out).
    Centring on today means forecasting the future level == forecasting the change.
    """
    L0 = z[:, -1, :, :]                            # today (B, M, T)
    u = z - L0.unsqueeze(1)                        # centred history (B, L, M, T)
    s_tilde = z.std(dim=1)                         # per-cell window scale (B, M, T)
    return u, L0, s_tilde


# multi-head self-attention with explicit Q/K/V
class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA over the sequence axis (-2) of a (Bx, N, d) tensor.

    Bx is the flattened non-attended batch; N is the attended axis (M, T or L).
    One operator is learned per axis; its softmax weights are still input-dependent.
    """
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d, self.h, self.dk = d, n_heads, d // n_heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.collect_attn = False     # set True to stash maps for diagnostics
        self._attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Bx, N, d)
        Bx, N, _ = x.shape
        q = self.W_q(x).view(Bx, N, self.h, self.dk).permute(0, 2, 1, 3)  # (Bx,h,N,dk)
        k = self.W_k(x).view(Bx, N, self.h, self.dk).permute(0, 2, 1, 3)
        v = self.W_v(x).view(Bx, N, self.h, self.dk).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (Bx,h,N,N)
        attn = torch.softmax(scores, dim=-1)
        if self.collect_attn:
            self._attn = attn.detach()
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)                          # (Bx,h,N,dk)
        out = out.permute(0, 2, 1, 3).reshape(Bx, N, self.d)  # merge heads
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d: int, d_ff_mult: int, dropout: float):
        super().__init__()
        d_ff = d * d_ff_mult
        self.net = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d)
        )

    def forward(self, x): return self.net(x)


class SubBlock(nn.Module):
    """Pre-LN transformer sub-block on (Bx, N, d): x += MHSA(LN(x)); x += FFN(LN(x)).
    The single attention unit every model is built from; only the attended axis differs.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.attn = MultiHeadSelfAttention(cfg.d, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.ff = FeedForward(cfg.d, cfg.d_ff_mult, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


# reconstruction + loss
def reconstruct(z_today: torch.Tensor, netDelta: torch.Tensor) -> torch.Tensor:
    """ẑ_{t+h} = z_today + netDelta_h.  z_today (B,M,T); netDelta (B,Hh,M,T)."""
    return z_today.unsqueeze(1) + netDelta                     # (B,Hh,M,T)


def surface_loss(netDelta: torch.Tensor,
                 z_today: torch.Tensor,
                 z_future: torch.Tensor,
                 gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Uniform MSE on cumulative standardised changes.

    netDelta/z_future (B,Hh,M,T), z_today (B,M,T), gamma (Hh,) horizon weights.
    netDelta == 0 is the random-walk null; per-cell standardisation upstream already
    equalises cell scales, so no explicit per-cell weight.
    """
    B, Hh, M, T = netDelta.shape
    Dz = z_future - z_today.unsqueeze(1)                       # (B,Hh,M,T) target change
    sq = (netDelta - Dz) ** 2
    per_h = sq.mean(dim=(0, 2, 3))                             # (Hh,) mean over batch+cells
    if gamma is None:
        gamma = torch.ones(Hh, device=netDelta.device)
    gamma = gamma / gamma.sum() * Hh                           # normalise so mean weight = 1
    return (gamma * per_h).mean()


def rw_loss(z_today: torch.Tensor, z_future: torch.Tensor,
            gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Random-walk baseline loss = surface_loss with netDelta == 0 (predict no change)."""
    netDelta = torch.zeros_like(z_future)
    return surface_loss(netDelta, z_today, z_future, gamma)


def destandardise(z: torch.Tensor, mu_cell: torch.Tensor,
                  sd_cell: torch.Tensor) -> torch.Tensor:
    """Map standardised log-IV back to log-IV. mu_cell, sd_cell: (M,T) train stats.
    z: (..., M, T). Returns log-IV in the same shape. (Then exp() for vol if desired.)"""
    return z * sd_cell + mu_cell


# window builder (honours the chronological split + H-day embargo externally)
def build_windows(Z: torch.Tensor, horizons: Tuple[int, ...], L: int
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sliding windows from Z (S, M, T) chronological standardised log-IV.

    Returns windows (Nsamp, L, M, T), z_today (Nsamp, M, T), z_future (Nsamp, Hh, M, T),
    and base_idx (Nsamp,) the index t of each sample. Z must already be standardised
    with TRAIN-only mu/sd.
    """
    S, M, T = Z.shape
    Hmax = max(horizons)
    samples_w, samples_today, samples_fut, idxs = [], [], [], []
    for t in range(L - 1, S - Hmax):
        samples_w.append(Z[t - L + 1: t + 1])                          # (L,M,T)
        samples_today.append(Z[t])                                     # (M,T)
        samples_fut.append(torch.stack([Z[t + h] for h in horizons]))  # (Hh,M,T)
        idxs.append(t)
    return (torch.stack(samples_w), torch.stack(samples_today),
            torch.stack(samples_fut), torch.tensor(idxs))
