"""
transformer.py
===================================================================================
VanillaTransformer — the architectural floor for the SANTA family ablation series.

The structural change vs SANTA-Temporal is one move: the day becomes the token
instead of the cell. Everything not on the diff list below stays byte-for-byte
identical to SANTA-Temporal.

Diff to SANTA-Temporal
----------------------
  1) Delete the CoordinateEmbedding for moneyness and √τ; the vanilla model has
     no cell-geometry information.
  2) Tokenisation: flatten the centred surface for each day in tau-major order
     (matching the SPX_surfaces.csv channel storage) and project
     `Linear(M·T → d)` per day. The per-cell value embedding `Linear(1 → d)`
     becomes a single per-day projection — implicit cross-cell mixing happens
     inside this one linear map.
  3) Temporal positional encoding (lag emb) is added to (B, L, d) — one vector
     per day, not per cell — but is otherwise unchanged.
  4) Attention runs once over the L day-tokens, not M·T times per cell. The
     SubBlock module is the same one SANTA uses for its temporal block.
  5) Head: surface-wide expansion. The model has compressed cells into a single
     d-vector, so the head must reconstruct the whole surface:
         Linear(d → M·T × n_horizons), reshape to (M·T, n_horizons).
     Re-injection of today's level + per-cell window scale: each 110-vector is
     projected to d via a small `Linear(M·T → d)` and ADDED to today's
     d-representation before the surface-wide head. This is a deviation from a
     literal concat → Linear((d+2·M·T) → M·T·H) only because that literal
     concat has a (M·T)² ≈ 510k weight matrix floor at n_horizons=21 that
     dominates any reasonable parameter budget regardless of d. The information
     available to the head is the same — only the encoding differs. The "one
     big projection out" stays a single Linear from a d-vector to the flattened
     surface.

What stays exactly the same
---------------------------
  - Instance-norm centring on today's level; per-cell window scale s̃ kept.
  - Target: predict netDelta (residual on today); reconstruction
    ẑ_{t+h} = z_today + netDelta_h, identical to SANTA.
  - Loss: imported `surface_loss` from santa.py (uniform MSE on cumulative
    standardised changes); same horizons, same γ.
  - L, n_horizons, n_heads, n_layers, dropout, optimiser, early-stop, seeds —
    all driven from the shared SANTAConfig.

Param target & accounting
-------------------------
At `n_layers=2, n_heads=4, d_ff_mult=1, d=16` the model lands at ~48.9k —
within the SANTA family budget. The surface-wide head dominates: each unit of
d costs ~2.7k extra params (a 110*21=2310-row matrix), so increasing d quickly
blows past the others. The "head bottleneck" is the structural cost of the
vanilla-transformer design and is intentional — it's part of what makes this
the floor: reduce-then-forecast through a d-dim bottleneck.

A `print_param_breakdown` helper is provided so the smoke test can show where
the budget is going.
"""
from __future__ import annotations
import os
import sys

# Make SANTA/ importable when invoked standalone. train.py already adds the
# transformer/ and SANTA/ folders to sys.path so this insert is only needed
# for `python transformer/transformer.py` smoke runs.
_HERE  = os.path.dirname(os.path.abspath(__file__))
_SANTA = os.path.join(os.path.dirname(_HERE), "SANTA")
if _SANTA not in sys.path:
    sys.path.insert(0, _SANTA)

import torch
import torch.nn as nn

# Re-export the shared Config so callers write VanillaTransformer(Config(...))
# the same way they write SANTA(Config(...)). The fields the vanilla model
# does NOT use (d_head_hidden, k_grid, tau_grid_years) are ignored.
from santa import (                                           # noqa: E402
    Config,
    SubBlock,
    MultiHeadSelfAttention,
)


# ----------------------------------------------------------------------------------
# Vanilla day-token transformer
# ----------------------------------------------------------------------------------
class VanillaTransformer(nn.Module):
    """Day-token encoder transformer.

    Input  : (B, L, M, T) standardised log-IV window — the same contract as
             the SANTA family, so the trainer-side adapter is the SANTA one.
    Output : (B, n_horizons, M, T) netDelta — predicted residual change from
             today to each horizon, in standardised log-IV units.

    Internal sequence:
      1) Instance-norm centring on today's slice (per cell);
      2) Flatten centred surface to (B, L, M·T) in tau-major order;
      3) Day projection Linear(M·T → d) gives a (B, L, d) token stream;
      4) Add learned lag embedding (positional encoding over days);
      5) Stack of SubBlocks (pre-LN MHSA + FFN) over the L axis;
      6) Final LayerNorm; today's representation = last day-token;
      7) Re-inject flattened level + scale via small projections, added in d;
      8) Surface-wide head Linear(d → M·T · H) and reshape to (M, T, H).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_cells = cfg.M * cfg.T
        self._n_cells = n_cells

        # --- Step 2 day projection (Linear in) -----------------------------------
        # Single Linear from the flattened centred surface to d. This is where
        # all cross-cell information enters the model (no separate spatial
        # attention, no coordinate embeddings).
        self.day_proj = nn.Linear(n_cells, cfg.d)

        # --- Step 4 lag (positional) embedding -----------------------------------
        # Same learned table SANTA uses (size L × d). Added to (B, L, d).
        self.emb_lag = nn.Embedding(cfg.L, cfg.d)
        self.register_buffer("lag_idx", torch.arange(cfg.L))
        self.emb_drop = nn.Dropout(cfg.dropout)

        # --- Step 5 backbone -----------------------------------------------------
        # Stack of pre-LN transformer SubBlocks. Identical to SANTA's
        # FactoredLayer.block_time except attention runs once on (B, L, d)
        # rather than 110 times batched into (B*M*T, L, d).
        self.layers = nn.ModuleList([SubBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d)

        # --- Step 7 level/scale re-injection -------------------------------------
        # See module docstring. Project each 110-vector to d and add to the
        # day-rep before the head, instead of literal concat → big linear.
        self.level_proj = nn.Linear(n_cells, cfg.d)
        self.scale_proj = nn.Linear(n_cells, cfg.d)

        # --- Step 8 surface-wide head --------------------------------------------
        # The single "big projection out" the spec calls for. Expands the
        # (level+scale-augmented) d-vector to the whole flattened surface ×
        # horizons. Reshaped at forward time to (B, H, M, T) for the loss.
        self.head = nn.Linear(cfg.d, n_cells * cfg.n_horizons)

    # -- Step 1 instance norm (kept identical to SANTA) --------------------------
    def _instance_norm(self, z: torch.Tensor):
        """z: (B, L, M, T) standardised log-IV (chronological; -1 = today).
        Returns u (centred history (B,L,M,T)), L0 (today (B,M,T)), s̃ ((B,M,T))."""
        if self.cfg.centre_on == "last":
            L0 = z[:, -1, :, :]
        elif self.cfg.centre_on == "mean":
            L0 = z.mean(dim=1)
        else:
            raise ValueError(self.cfg.centre_on)
        u = z - L0.unsqueeze(1)                        # centred history
        s_tilde = z.std(dim=1)                          # per-cell window scale
        return u, L0, s_tilde

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, L, M, T = z.shape
        assert (L, M, T) == (cfg.L, cfg.M, cfg.T), "window shape mismatch"

        # Step 1 — centre each cell's window on today.
        u, L0, s_tilde = self._instance_norm(z)        # u: (B,L,M,T)

        # Step 2 — flatten centred surface in TAU-MAJOR order (T outer, M inner)
        # to match the data CSV's column storage (parse_grid lays columns
        # tau-outer, money-inner). The model receives an "ordered" 110-vector
        # so positionally-adjacent inputs are surface-neighbours along the
        # money axis at a fixed tau — a free, free of any geometry leak.
        u_flat = u.permute(0, 1, 3, 2).contiguous().reshape(B, L, M * T)  # (B,L,110)

        # Step 3 — day projection: 110-vector → d per day.
        x = self.day_proj(u_flat)                       # (B, L, d)

        # Step 4 — additive lag embedding (positional over days).
        x = x + self.emb_lag(self.lag_idx)[None, :, :]
        x = self.emb_drop(x)                            # (B, L, d)

        # Step 5 — temporal attention stack.
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)                            # (B, L, d)

        # Step 6 — today's representation: the last day-token. The remaining L−1
        # tokens have done their job by attending into this one.
        r = x[:, -1, :]                                 # (B, d)

        # Step 7 — re-inject flattened level + scale. Same tau-major order so
        # the projections see the same neighbour structure as the input proj.
        L0_flat = L0.permute(0, 2, 1).contiguous().reshape(B, M * T)
        s_flat  = s_tilde.permute(0, 2, 1).contiguous().reshape(B, M * T)
        r = r + self.level_proj(L0_flat) + self.scale_proj(s_flat)

        # Step 8 — surface-wide head, reshape to canonical (B, Hh, M, T).
        out = self.head(r)                              # (B, M*T*Hh)
        # The head's output is laid out as flat (M*T) × H; we kept tau-major
        # ordering coming in, so undo it the same way going out.
        out = out.reshape(B, T, M, cfg.n_horizons)      # tau-major flat → (T, M, H)
        netDelta = out.permute(0, 3, 2, 1).contiguous() # (B, H, M, T)
        return netDelta

    # Mirror SANTA's diagnostic switch so eval scripts can grab attention maps.
    def enable_attn_collection(self, on: bool = True):
        for m in self.modules():
            if isinstance(m, MultiHeadSelfAttention):
                m.collect_attn = on


# ----------------------------------------------------------------------------------
# Helper: parameter breakdown, useful when sizing d for the budget.
# ----------------------------------------------------------------------------------
def print_param_breakdown(model: VanillaTransformer):
    groups = {
        "day_proj":    model.day_proj,
        "emb_lag":     model.emb_lag,
        "backbone":    model.layers,
        "final_ln":    model.final_ln,
        "level_proj":  model.level_proj,
        "scale_proj":  model.scale_proj,
        "head":        model.head,
    }
    total = sum(p.numel() for p in model.parameters())
    print(f"VanillaTransformer parameters: {total:,}")
    for name, mod in groups.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<14s} {n:>9,}  ({n/total*100:5.2f}%)")


# ----------------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from santa import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    # Match the trainer's at-runtime config: M=11, T=10 (current dataset).
    cfg = Config(M=11, T=10, L=63, horizons=tuple(range(1, 22)),
                 d=16, n_heads=4, n_layers=2, d_ff_mult=1,
                 d_head_hidden=24, dropout=0.1)
    model = VanillaTransformer(cfg)
    print_param_breakdown(model)

    # Fake standardised series (300 days × 11×10 surface).
    S = 300
    Z = torch.randn(S, cfg.M, cfg.T).cumsum(0) * 0.02
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)

    windows, z_today, z_future, _ = build_windows(Z, cfg.horizons, cfg.L)
    print("windows :", tuple(windows.shape))
    print("z_today :", tuple(z_today.shape))
    print("z_future:", tuple(z_future.shape))

    bs = 64
    xb, tb, fb = windows[:bs], z_today[:bs], z_future[:bs]
    netDelta = model(xb)
    print("netDelta:", tuple(netDelta.shape))

    loss = surface_loss(netDelta, tb, fb)
    base = rw_loss(tb, fb)
    print(f"model loss = {loss.item():.4f}   RW loss = {base.item():.4f}")

    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters()
                  if p.requires_grad)
    print("backward OK, all params have grads:", grad_ok)

    # Pull a temporal attention map (block 0).
    model.enable_attn_collection(True)
    _ = model(xb[:2])
    cmap = model.layers[0].attn._attn   # (B, n_heads, L, L)
    print("temporal attn map:", tuple(cmap.shape))
