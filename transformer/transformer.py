"""VanillaTransformer — day-token encoder baseline (flattened surface per day)."""

from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from surface_core import Config, SubBlock, MultiHeadSelfAttention, instance_norm


class VanillaTransformer(nn.Module):
    """Day-token encoder: project each day's flattened surface to a token, attend over days."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_cells = cfg.M * cfg.T
        self._n_cells = n_cells

        # day projection
        self.day_proj = nn.Linear(n_cells, cfg.d)

        # lag embedding over days
        self.emb_lag = nn.Embedding(cfg.L, cfg.d)
        self.register_buffer("lag_idx", torch.arange(cfg.L))
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList([SubBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d)

        # level/scale re-injection
        self.level_proj = nn.Linear(n_cells, cfg.d)
        self.scale_proj = nn.Linear(n_cells, cfg.d)

        # surface-wide head
        self.head = nn.Linear(cfg.d, n_cells * cfg.n_horizons)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, L, M, T = z.shape
        assert (L, M, T) == (cfg.L, cfg.M, cfg.T), "window shape mismatch"

        # centre each cell's window on today
        u, L0, s_tilde = instance_norm(z)                          # u: (B,L,M,T)

        # flatten centred surface in tau-major order (T outer, M inner) to match the csv columns
        u_flat = u.permute(0, 1, 3, 2).contiguous().reshape(B, L, M * T)  # (B,L,110)

        # day projection: 110-vector → d per day
        x = self.day_proj(u_flat)                                  # (B, L, d)

        # additive lag embedding
        x = x + self.emb_lag(self.lag_idx)[None, :, :]
        x = self.emb_drop(x)

        # temporal attention stack
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)                                       # (B, L, d)

        # today's representation: the last day-token
        r = x[:, -1, :]                                            # (B, d)

        # re-inject flattened level + scale
        L0_flat = L0.permute(0, 2, 1).contiguous().reshape(B, M * T)
        s_flat  = s_tilde.permute(0, 2, 1).contiguous().reshape(B, M * T)
        r = r + self.level_proj(L0_flat) + self.scale_proj(s_flat)

        # surface-wide head, reshape to canonical
        out = self.head(r)                                         # (B, M*T*Hh)
        out = out.reshape(B, T, M, cfg.n_horizons)                 # tau-major flat → (T,M,H)
        netDelta = out.permute(0, 3, 2, 1).contiguous()            # (B, H, M, T)
        return netDelta

    def enable_attn_collection(self, on: bool = True):
        for m in self.modules():
            if isinstance(m, MultiHeadSelfAttention):
                m.collect_attn = on


def print_param_breakdown(model: VanillaTransformer):
    """Print where the parameter budget goes (the surface-wide head dominates at large n_horizons)."""
    groups = {
        "day_proj":   model.day_proj,
        "emb_lag":    model.emb_lag,
        "backbone":   model.layers,
        "final_ln":   model.final_ln,
        "level_proj": model.level_proj,
        "scale_proj": model.scale_proj,
        "head":       model.head,
    }
    total = sum(p.numel() for p in model.parameters())
    print(f"VanillaTransformer parameters: {total:,}")
    for name, mod in groups.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<14s} {n:>9,}  ({n/total*100:5.2f}%)")


# smoke test
if __name__ == "__main__":
    from surface_core import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config(M=11, T=10, L=63, horizons=tuple(range(1, 22)),
                 d=16, n_heads=4, n_layers=2, d_ff_mult=1,
                 d_head_hidden=24, dropout=0.1)
    model = VanillaTransformer(cfg)
    print_param_breakdown(model)

    S = 300
    Z = torch.randn(S, cfg.M, cfg.T).cumsum(0) * 0.02
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)

    windows, z_today, z_future, _ = build_windows(Z, cfg.horizons, cfg.L)
    bs = 64
    xb, tb, fb = windows[:bs], z_today[:bs], z_future[:bs]
    netDelta = model(xb)
    print("netDelta:", tuple(netDelta.shape))

    loss = surface_loss(netDelta, tb, fb)
    base = rw_loss(tb, fb)
    print(f"model loss = {loss.item():.4f}   RW loss = {base.item():.4f}")

    loss.backward()
    print("backward OK:",
          all(p.grad is not None for p in model.parameters() if p.requires_grad))

    model.enable_attn_collection(True)
    _ = model(xb[:2])
    cmap = model.layers[0].attn._attn   # (B, n_heads, L, L)
    print("temporal attn map:", tuple(cmap.shape))
