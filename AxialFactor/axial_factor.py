"""
AxialFactor — low-rank factor model for IV surface forecasting, with a
gated per-cell residual bypass.

Two parallel paths produce the forecast (in normalised space):

  factor path    : a small number of latent factors extracted from the
                   lookback surface by separable axial cross-attention,
                   evolved over the horizon by a per-factor linear
                   temporal map, projected back to surfaces by fixed
                   per-factor spatial loadings. Rank-F, horizon-
                   independent loadings.
  residual path  : a per-cell linear forecast from the normalised
                   lookback, using temporal weights shared across all
                   cells, modulated by a horizon-dependent sigmoid gate.
                   Lets short horizons read off recent residual
                   dynamics directly; the gate decays for long horizons
                   so the factor path takes over.

The two paths are summed (factor + gated residual) before
de-normalisation.

Factor structure
----------------
IV surface dynamics are dominated by a few interpretable shape modes —
the first 3–4 principal components of the daily surface explain over
95% of the variance (level, slope, skew, butterfly). AxialFactor bakes
this in: F latent factors carry the surface state through time, and
each factor has its own temporal dynamics. The model never represents
the surface as 150 independent cell-channels (as DLinear does) nor as
a generic [H, W, T] tensor with full attention (as HOT does) — it
commits to the factor decomposition.

Separable axial cross-attention (extraction)
--------------------------------------------
Factor extraction pools the cell grid in two stages — first over the
tenor axis H per moneyness column, then over moneyness W on the
H-pooled tokens. Both stages use the same F learned factor queries.
With one head, the implied per-factor cell loading is

    L_f[i, j] = α_f(i) · β_f(j)        — rank-1 in (i, j)

With n_heads attention heads summed, the loading is rank-n_heads
separable. This is the inductive bias matching IV surface geometry:
level, slope, skew are rank-1 separable in (moneyness, tenor);
curvature is rank-2.

Per-cell window normalisation
-----------------------------
With norm="cell_mean" (default), every (W, H) cell has its own
lookback mean subtracted before tokenisation and added back at the
output. The model therefore sees only deviations from each cell's
recent average — it forecasts dynamics, not absolute levels. This
strips the dominant low-frequency drift (which is large per cell but
trivial to predict if you know it) and frees the factor capacity to
model surface deformation.

norm="cell_full" additionally divides by the per-cell lookback std,
yielding a scale-free input. norm="none" disables normalisation.

Per-factor temporal evolution (AR(1))
-------------------------------------
Each factor's d_model embedding is collapsed to a scalar amplitude by
a shared linear head, then evolved over the horizon as a per-factor
AR(1) process:

    a[b, f]      = head(f[b, f, :])
    g[b, f, p]   = ρ[f]^p · (a[b, f] - μ[f]) + μ[f]

Two scalar parameters per factor (ρ_f, μ_f), plus one shared head.

This replaces the free [F, d_model, P] per-factor linear map that
earlier versions of the model used (`_PerFactorTemporal`, kept in
this file for ablation). That free parameterisation consistently
gradient-collapsed: across Ax3 and Ax4 runs the cross-factor top-1
horizon-shape cosine similarities were 0.997 and exactly ±1.000 —
every factor had drifted to the same dominant data-residual shape.
Orthogonality penalties on factor queries and on spatial loadings
did not prevent this because they constrain parameter geometry, not
the function class.

AR(1)'s 2-params-per-factor budget makes that collapse structurally
hard: for four factors to all express the same horizon shape they'd
need ρ_0 = ρ_1 = ρ_2 = ρ_3 exactly, which is not an attractor unless
the data genuinely has a single timescale. If the AR fix succeeds,
the four ρ_f values diverge during training and the per-factor
trajectories are distinct by construction.

Reconstruction (fixed per-factor spatial loadings)
--------------------------------------------------
Reconstruction is a static factor projection:

    spatial_loadings : [F, W, H]    — one learned loading map per factor
    y[b, p, w, h]    = Σ_f g[b, f, p] · spatial_loadings[f, w, h]
                       + out_bias[w, h]

The loadings carry no horizon dependence — each factor's spatial
footprint is fixed; only the factor's amplitude g[b, f, p] is
horizon-dependent. This deliberately throws away the cell-specific,
horizon-specific routing that a cross-attention reconstruction would
provide, on the diagnosis that the cross-attention version was a
generalisation leak (memorising train-window-specific cell × horizon
shapes that don't transfer to val). A static loading is the literal
form of the PCA prior the rest of the model leans on.

A per-cell [W, H] bias is added at the end so the model can learn a
typical surface offset without leaking it into horizon-dependent
capacity.

Residual bypass
---------------
A small parallel path lets the model use the recent normalised
lookback directly, bypassing the factor bottleneck for short horizons.
It is a single shared [L, pred_len] temporal matrix applied to every
cell's normalised lookback:

    y_resid[b, p, w, h] = Σ_l x_n[b, l, w, h] · residual_temporal[l, p]

The temporal weights are shared across all cells — the path can only
learn one "typical residual decay shape" common to all (W, H). It
cannot memorise cell-specific horizon shapes, so it cannot overfit
the way a per-cell head [L, P, W, H] would.

The output is scaled by a horizon-dependent sigmoid gate ∈ (0, 1)^P,
parameterised through a small learned horizon embedding so similar
horizons get smoothly-related gates. The gate is free to learn its
own shape; the inductive expectation (not a constraint) is that it
stays near 1 for short horizons and decays for long horizons, letting
the factor path own the long-horizon forecast.

This is essentially DLinear's seasonal head added as a gated residual.
It targets the h=1 weakness of a pure factor model — where the
bottleneck is too lossy to keep the short-term residual signal —
without compromising the factor path's long-horizon role.

Initialisation
--------------
The init has a single goal: at step 0, the model's normalised-space
forecast is approximately zero, so with norm="cell_mean" the raw
output is approximately the per-cell lookback mean broadcast across
the horizon — a flat baseline (not persistence, not mean-reversion).

This is achieved by:
  pe_W, pe_H            ~ orthogonal init scaled to 0.1 row-norm
  factor_queries        ~ orthogonal init scaled to 0.1 row-norm
  temporal rho_raw        = 0.5  (ρ ≈ 0.46; small enough that g≈0 at
                                  init but nonzero so the amplitude
                                  head is trainable from step 1)
  temporal mu             = 0
  amplitude_head          default nn.Linear, bias = 0
  temporal horizon_emb  ~ trunc_normal(std=0.02)   (AR per-horizon gate)
  temporal gate_linear    default                    (initial AR gate ≈ 0.5)
  spatial_loadings      ~ trunc_normal(std=0.02)
  out_bias                = 0
  residual_temporal     ~ trunc_normal(std=0.002)  — small but non-zero
                                                     so the gate gets
                                                     nonzero gradient
                                                     from step 1
  horizon_emb           ~ trunc_normal(std=0.02)   (residual gate)
  gate_linear             default                    (initial gate ≈ 0.5)

All learnable parameters of the forward receive nonzero gradient from
step 1.

Input:  [B, seq_len,  W, H]
Output: [B, pred_len, W, H]
"""

import torch
import torch.nn as nn


class _FactorExtractor(nn.Module):
    """Separable axial cross-attention from cell tokens to F factor tokens.

    Two cross-attention stages share the same F factor queries:
        τ-stage: for each moneyness column w, the factor queries attend
                 into the H cell tokens at that column. Pools the tenor
                 axis. Output: [B, F, W, d_model].
        m-stage: for each factor f, the same factor query attends into
                 the W tokens from the τ-stage at factor index f. Pools
                 the moneyness axis. Output: [B, F, d_model].

    The per-factor framing of the m-stage (one query per factor over
    that factor's own W tokens, rather than F queries jointly over a
    flattened factor×W set) preserves the factor identity carried
    through from the τ-stage.
    """

    def __init__(self, n_factors: int, d_model: int, n_heads: int):
        super().__init__()
        # Orthogonal init requires F ≤ d_model so the QR factor has F
        # orthonormal columns. Crash loudly if violated rather than
        # silently degrade.
        assert n_factors <= d_model, (
            f"orthogonal init requires n_factors <= d_model, "
            f"got n_factors={n_factors}, d_model={d_model}"
        )
        self.n_factors = n_factors
        self.d_model = d_model
        # Shared factor queries reused in every spatial slice of both
        # axial stages.
        self.factor_queries = nn.Parameter(torch.empty(n_factors, d_model))
        self.attn_tau = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_m   = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        with torch.no_grad():
            # Orthogonal init: the F factor queries are mutually
            # orthonormal at step 0, so the four queries produce
            # genuinely distinct attention distributions over cells from
            # the first forward pass. This breaks the symmetric basin
            # (queries collapse to a single vector aligned with the
            # dominant surface direction — PC1) that ordinary random
            # init falls into. Verified by probe 1.5 on the broken Ax2
            # run (effective rank 1.099, all pairwise cos-sim +0.999).
            #
            # Row-norm scale (0.1) is chosen to match the original
            # trunc_normal(std=0.02) magnitude: that init had row norm
            # ≈ 0.02·sqrt(d_model) = 0.08 for d_model=16. Going smaller
            # makes attention near-uniform; going larger makes queries
            # dominate cell tokens. 0.1 sits in the same regime.
            g = torch.randn(d_model, n_factors)
            q, _ = torch.linalg.qr(g)                # [d_model, n_factors]
            self.factor_queries.copy_(q.T * 0.1)     # [n_factors, d_model]

    def forward(self, cell_tokens: torch.Tensor, W: int, H: int) -> torch.Tensor:
        # cell_tokens: [B, W*H, d_model]   (W-outer, H-inner ordering)
        B, _, D = cell_tokens.shape
        F = self.n_factors

        # τ-stage: pool over H per moneyness column.
        # [B, W*H, d] -> [B, W, H, d] -> [B*W, H, d]   (merge W into batch).
        tau_in = cell_tokens.view(B, W, H, D).reshape(B * W, H, D)
        # Broadcast the F factor queries over the (B*W) spatial slices.
        Q_tau = self.factor_queries.unsqueeze(0).expand(B * W, F, D)
        tau_out, _ = self.attn_tau(Q_tau, tau_in, tau_in, need_weights=False)
        # tau_out: [B*W, F, d] -> [B, W, F, d] -> [B, F, W, d].
        tau_out = tau_out.view(B, W, F, D).transpose(1, 2)

        # m-stage: per factor, pool over W. Merge (B, F) into the batch
        # axis so each (b, f) slice's single query is factor_queries[f].
        m_in = tau_out.reshape(B * F, W, D)                          # [B*F, W, d]
        Q_m = (self.factor_queries.unsqueeze(0)
                                  .expand(B, F, D)
                                  .reshape(B * F, 1, D))             # [B*F, 1, d]
        m_out, _ = self.attn_m(Q_m, m_in, m_in, need_weights=False)
        factor_tokens = m_out.view(B, F, D)                          # [B, F, d]

        return self.norm(factor_tokens)


class _PerFactorTemporal(nn.Module):
    """Per-factor linear map d_model -> pred_len.  (UNUSED IN Ax5+.)

    Kept in the file for ablation purposes — AxialFactor now uses
    `_PerFactorAR` below. The free [F, d_model, P] parameterisation
    here had 336 params/factor (at d=16, P=21) and consistently
    gradient-collapsed onto a single shared horizon shape across
    factors (cross-factor top-1 trajectory cos-sim 0.997–1.000 on
    Ax3/Ax4 runs), regardless of orthogonality penalties on queries or
    loadings.

        g[b, f, p] = Σ_d f[b, f, d] · W[f, d, p]   + b[f, p]
    """

    def __init__(self, n_factors: int, d_model: int, pred_len: int,
                 init_scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_factors, d_model, pred_len))
        self.bias   = nn.Parameter(torch.zeros(n_factors, pred_len))
        with torch.no_grad():
            # Small init: the reconstructor's pre-bias output is then
            # near zero at step 0, so y ≈ per-cell lookback mean under
            # norm='cell_mean'.
            nn.init.trunc_normal_(self.weight, std=init_scale)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [B, F, d_model]   ->   g: [B, F, pred_len]
        return torch.einsum('bfd,fdp->bfp', f, self.weight) + self.bias


class _PerFactorAR(nn.Module):
    """Per-factor AR(1) horizon forecast.

    Each factor token is collapsed to a current-amplitude scalar via a
    shared linear head, then evolved over the horizon as an AR(1)
    process with per-factor persistence and long-run mean:

        a[b, f]      = head(factor_tokens[b, f, :])          # [B, F]
        g[b, f, h]   = ρ[f]^h · (a[b, f] - μ[f]) + μ[f]      # [B, F, P]

    Parameter count: 2 per factor (ρ_f, μ_f), plus one shared
    (d_model → 1) Linear head used for all factors.

    Why parametric and not free [d_model, P]: with the free map (see
    `_PerFactorTemporal` above), gradient descent collapses all
    factors onto the dominant data residual shape r(p) — verified on
    Ax3 and Ax4 (cross-factor top-1 horizon-shape cos-sim 0.997 and
    ±1.000 respectively). AR(1) has only 2 params per factor;
    collapse to a shared shape requires ρ_0 = ρ_1 = ρ_2 = ρ_3
    exactly, which is not an attractor unless the data really has one
    timescale. Distinct factor dynamics are now structurally enforced
    rather than penalty-encouraged.

    Persistence ρ_f is parameterised as tanh(rho_raw[f]) ∈ (-1, 1).
    Long-run mean μ_f is unconstrained.

    Init note: rho_raw is initialised at 0.5 (giving ρ ≈ 0.46), NOT 0.
    With ρ = 0, ∂g/∂a = ρ^h = 0 for all h, so the amplitude head's
    gradient would be exactly zero from step 1 — a gradient-stranding
    bug. A small positive initial ρ keeps the head trainable from the
    first batch; the warm-init test still passes (max diff < 0.1)
    because the amplitude head's own output is small at init (default
    nn.Linear weight, bias=0) and the spatial loadings are also small.

    Per-horizon gate
    ~~~~~~~~~~~~~~~~
    The AR output is multiplied by a learned per-horizon sigmoid gate
    (structurally identical to the residual path's gate). Lets the
    model attenuate the AR contribution at horizons where the per-cell
    residual path is a better predictor — empirically, at h=1, where
    truth is close to persistence and the residual path matches it
    naturally. Without this gate, the un-gated AR forecast in Ax5
    regressed h=1 MSE to 2.14× DLinear (vs 1.64× in Ax3/Ax4) by
    contributing a wrong-direction signal at short horizons that the
    rest of the model couldn't suppress. With the gate, the model can
    learn to use AR selectively (typically near zero at short
    horizons, near one at long).

    The gate is parameterised through a small horizon embedding
    (gate_dim=16) + linear projection so similar horizons get
    smoothly-related gates. Initial gate ≈ 0.5 uniformly; the model
    learns the actual shape.
    """

    def __init__(self, n_factors: int, d_model: int, pred_len: int,
                 gate_dim: int = 16):
        super().__init__()
        self.n_factors = n_factors
        self.pred_len = pred_len
        self.gate_dim = gate_dim

        # Single shared head: d_model → 1 scalar amplitude per factor.
        # Sharing is fine because differentiation across factors is the
        # upstream cross-attention's job, not this head's.
        self.amplitude_head = nn.Linear(d_model, 1)

        # Per-factor AR(1) parameters. rho_raw=0.5 gives ρ≈0.46 (see
        # docstring init note); mu=0 gives near-zero long-run mean so
        # the warm start is "predict per-cell lookback mean" under
        # norm='cell_mean'.
        self.rho_raw = nn.Parameter(torch.full((n_factors,), 0.5))
        self.mu      = nn.Parameter(torch.zeros(n_factors))

        # Per-horizon gate on the AR output. Symmetric to
        # _ResidualPath.gate: small horizon embedding + linear →
        # sigmoid → [P] in (0, 1). Lets the model attenuate the AR
        # contribution at horizons where the per-cell residual path is
        # a better predictor (Ax5 regressed at h=1 from 1.64× DLinear
        # to 2.14× DLinear because the un-gated AR forecast contributed
        # a wrong-direction signal at h=1; this gate is the targeted
        # fix). Expected post-training pattern: gate ≈ 0 at short
        # horizons, ≈ 1 at long horizons — mirror of the residual gate.
        self.horizon_emb = nn.Parameter(torch.empty(pred_len, gate_dim))
        self.gate_linear = nn.Linear(gate_dim, 1)

        # Horizon indices [1, 2, ..., P] as a non-persistent buffer so
        # `ρ ** horizons` produces ρ^h for every h in one shot.
        self.register_buffer(
            "horizons",
            torch.arange(1, pred_len + 1).float(),
            persistent=False,
        )

        with torch.no_grad():
            # Zero head bias so a[b, f] = W_a @ token (no constant
            # offset) — matches the previous module's "near-zero g at
            # init" property.
            self.amplitude_head.bias.zero_()
            # Gate horizon embedding small ⇒ gate logits small ⇒ initial
            # gate ≈ 0.5 uniformly across horizons. Identical to the
            # residual-path gate init.
            nn.init.trunc_normal_(self.horizon_emb, std=0.02)

    def rho(self) -> torch.Tensor:
        # [F] in (-1, 1).
        return torch.tanh(self.rho_raw)

    def gate(self) -> torch.Tensor:
        # [pred_len] in (0, 1).
        return torch.sigmoid(self.gate_linear(self.horizon_emb).squeeze(-1))

    def forward(self, factor_tokens: torch.Tensor) -> torch.Tensor:
        # factor_tokens: [B, F, d_model]
        # Step 1: collapse each factor token to a scalar amplitude.
        a = self.amplitude_head(factor_tokens).squeeze(-1)           # [B, F]

        # Step 2: closed-form AR(1) horizon forecast.
        #     g[b, f, h] = ρ[f]^h · (a[b, f] - μ[f]) + μ[f]
        rho = self.rho()                                             # [F]
        rho_pow = rho.unsqueeze(-1) ** self.horizons                 # [F, P]
        deviation = (a - self.mu).unsqueeze(-1)                      # [B, F, 1]
        g = deviation * rho_pow.unsqueeze(0) + self.mu.view(1, -1, 1)

        # Step 3: per-horizon gate (broadcast over batch and factor).
        g = g * self.gate().view(1, 1, -1)
        return g                                                     # [B, F, P]


class _FactorReconstructor(nn.Module):
    """Static factor projection from factor amplitudes to surfaces.

    Each factor has a fixed learned spatial loading map (no horizon
    dependence). The forecast is the sum over factors of
    amplitude × loading, plus a per-cell bias:

        y[b, p, w, h] = Σ_f g[b, f, p] · spatial_loadings[f, w, h]
                        + out_bias[w, h]

    This replaces the spec's cross-attention reconstruction with the
    literal static factor model that the PCA prior suggests — the
    diagnosis being that the cross-attention version was a
    generalisation leak (it could memorise train-specific cell ×
    horizon shapes the loadings cannot).
    """

    def __init__(self, n_factors: int, W: int, H: int):
        super().__init__()
        self.n_factors = n_factors
        self.W = W
        self.H = H
        # One [W, H] loading map per factor.
        self.spatial_loadings = nn.Parameter(torch.empty(n_factors, W, H))
        # Per-cell offset (typical surface shape); kept out of the
        # horizon-dependent path.
        self.out_bias = nn.Parameter(torch.zeros(W, H))
        with torch.no_grad():
            nn.init.trunc_normal_(self.spatial_loadings, std=0.02)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        # g: [B, F, pred_len]   ->   y: [B, pred_len, W, H]
        y = torch.einsum("bfp,fwh->bpwh", g, self.spatial_loadings)
        return y + self.out_bias


class _ResidualPath(nn.Module):
    """Per-cell residual channel with shared temporal weights and a
    horizon-dependent gate.

    Each cell's normalised lookback x_n[b, :, w, h] is mapped to its
    forecast through a single shared [seq_len, pred_len] temporal
    matrix — every cell uses the same weights, so the path can only
    express one "typical residual decay shape" common to all (W, H).
    The output is scaled by a learned horizon-dependent sigmoid gate
    so the residual contribution can fade out as h grows, letting the
    factor path own the long-horizon forecast.

    The gate is parameterised through a small horizon embedding +
    linear projection so similar horizons get smoothly-related gates
    (without enforcing monotone decay — the model is free to learn the
    actual shape).
    """

    def __init__(self, seq_len: int, pred_len: int, gate_dim: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Shared per-cell temporal weights. Small init so the residual
        # path is near zero at step 0 (preserving the warm start) while
        # still nonzero — the gate would otherwise receive zero
        # gradient at step 1 since dL/d(gate) ∝ y_residual_pre_gate.
        self.temporal = nn.Parameter(torch.empty(seq_len, pred_len))
        # Horizon embedding + linear gate -> [P] in (0, 1).
        self.horizon_emb = nn.Parameter(torch.empty(pred_len, gate_dim))
        self.gate_linear = nn.Linear(gate_dim, 1)
        with torch.no_grad():
            nn.init.trunc_normal_(self.temporal,    std=0.002)
            nn.init.trunc_normal_(self.horizon_emb, std=0.02)
            # gate_linear: default init; with horizon_emb std=0.02 the
            # logits are small-noise and the initial gate is ≈ 0.5
            # uniformly across horizons.

    def gate(self) -> torch.Tensor:
        # [pred_len] in (0, 1).
        return torch.sigmoid(self.gate_linear(self.horizon_emb).squeeze(-1))

    def forward(self, x_n: torch.Tensor) -> torch.Tensor:
        # x_n: [B, L, W, H]   (normalised lookback)
        # Shared per-cell temporal projection (no spatial mixing):
        #     y[b, p, w, h] = Σ_l x_n[b, l, w, h] · temporal[l, p]
        y = torch.einsum("blwh,lp->bpwh", x_n, self.temporal)        # [B, P, W, H]
        # Gate is broadcast over (B, W, H).
        return self.gate().view(1, -1, 1, 1) * y


class AxialFactor(nn.Module):
    """Low-rank factor model for IV surface forecasting.

    A few latent factors are extracted from the lookback surface via
    separable axial cross-attention, evolved over the horizon by a
    per-factor linear temporal map, then projected back to surfaces by
    cross-attention from per-cell queries into the lifted factors.

    Input:  [B, seq_len,  W, H]
    Output: [B, pred_len, W, H]

    Args:
        seq_len             int   — lookback length L.
        pred_len            int   — forecast horizon P.
        W                   int   — moneyness axis size.
        H                   int   — tenor axis size.
        n_factors           int   — number of latent factors F. Default 4
                                    (matches the 3–4 PCs that explain
                                    >95% of IV surface variance; a couple
                                    extra channels give slack for surface
                                    deformations beyond the dominant
                                    modes). Range: F >= 1.
        d_model             int   — embedding dim. Default 64. Must be
                                    divisible by n_heads.
        n_heads             int   — heads per cross-attention call.
                                    Default 4. With n_heads heads, the
                                    implied per-factor cell loading is
                                    rank-n_heads separable in
                                    (moneyness, tenor).
        norm                str   — per-cell window normalisation mode.
                                    One of:
                                      'none'      — no normalisation;
                                      'cell_mean' — subtract per-cell
                                                    lookback mean (added
                                                    back at output).
                                                    Default.
                                      'cell_full' — subtract per-cell
                                                    lookback mean and
                                                    divide by per-cell
                                                    lookback std
                                                    (reversed at output).
                                    Default 'cell_mean'.
        temporal_init_scale float — DEPRECATED no-op. Was the std of the
                                    free per-factor temporal weight init
                                    in the previous `_PerFactorTemporal`
                                    module. The current `_PerFactorAR`
                                    module initialises ρ/μ explicitly
                                    (see "Initialisation" in the module
                                    docstring); this kwarg is kept for
                                    backwards-compatible call signature
                                    from train.py and has no effect.
    """

    _VALID_NORMS = ("none", "cell_mean", "cell_full")

    def __init__(self, seq_len: int, pred_len: int, W: int, H: int,
                 n_factors: int = 4, d_model: int = 64, n_heads: int = 4,
                 norm: str = "cell_mean",
                 temporal_init_scale: float = 0.02):
        super().__init__()
        if norm not in self._VALID_NORMS:
            raise ValueError(
                f"norm must be one of {self._VALID_NORMS}, got {norm!r}"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must divide d_model ({d_model})"
            )
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")

        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.W = W
        self.H = H
        self.n_factors = n_factors
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_mode = norm

        # Per-cell lookback -> d_model embedding (one token per cell).
        self.embedder = nn.Linear(seq_len, d_model)

        # Separable 2D positional encoding: pe[i, j] = pe_W[i] + pe_H[j].
        # Reused by the reconstructor's cell queries so the spatial
        # geometry is tied between read and write.
        self.pe_W = nn.Parameter(torch.empty(W, d_model))
        self.pe_H = nn.Parameter(torch.empty(H, d_model))
        self.embed_norm = nn.LayerNorm(d_model)
        with torch.no_grad():
            nn.init.trunc_normal_(self.pe_W, std=0.02)
            nn.init.trunc_normal_(self.pe_H, std=0.02)

        self.extractor = _FactorExtractor(n_factors, d_model, n_heads)
        # Temporal evolution: per-factor AR(1) with 2 scalars/factor
        # (`_PerFactorAR`). Replaces the free [F, d_model, P] map
        # (`_PerFactorTemporal`, kept above for ablation) which
        # consistently gradient-collapsed onto a single shared
        # horizon shape across factors on Ax3/Ax4 runs. `temporal_init_scale`
        # is now a no-op (the AR module initialises rho/mu explicitly);
        # the kwarg is kept for backwards-compatible call signature.
        self.temporal  = _PerFactorAR(n_factors, d_model, pred_len)
        self.reconstructor = _FactorReconstructor(n_factors, W, H)

        # Gated per-cell residual bypass. Shares one [L, pred_len]
        # temporal map across all cells; the horizon gate decides how
        # much of this residual signal each horizon uses.
        self.residual_path = _ResidualPath(seq_len, pred_len)

    def _pe_flat(self) -> torch.Tensor:
        # pe_2d[w, h] = pe_W[w] + pe_H[h], flattened W-outer, H-inner so
        # the index w*H + h matches the cell_tokens reshape order. Only
        # used on the input side (the static reconstructor doesn't carry
        # a positional encoding).
        pe_2d = self.pe_W.unsqueeze(1) + self.pe_H.unsqueeze(0)      # [W, H, d]
        return pe_2d.reshape(self.W * self.H, self.d_model)          # [W*H, d]

    def factor_orthogonality_loss(self) -> torch.Tensor:
        """Penalty on factor-query collinearity.

        Returns the sum of squared off-diagonal entries of the
        unit-normalised Gram matrix of factor_queries. Zero iff the
        queries are mutually orthogonal; grows quadratically with
        pairwise cosine similarity.

        Designed to be added to the MSE training loss with weight 1e-2.
        Without it, training collapses all factor queries onto the
        dominant data direction — verified by probe 1.5 on the broken
        Ax2 run (AxialFactor/63_21/2026-05-17T14-50-25Z/: effective
        rank 1.099, all pairwise cos-sim +0.999).

        Penalty on queries alone is necessary but *not sufficient*:
        even with orthogonal queries, the downstream attention +
        temporal + loading pipeline still funnels reconstruction onto
        PC1 with sign flips (verified by probe 4 on the Ax3 run at
        AxialFactor/63_21/2026-05-17T16-03-24Z/). See
        `loading_orthogonality_loss` for the complementary penalty.
        """
        Q = self.extractor.factor_queries                            # [F, d]
        Q_n = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
        G = Q_n @ Q_n.T                                              # [F, F]
        eye = torch.eye(self.n_factors, device=Q.device, dtype=Q.dtype)
        off_diag = G - eye
        return (off_diag ** 2).sum()

    def loading_orthogonality_loss(self) -> torch.Tensor:
        """Penalty on spatial-loading collinearity.

        Sum of squared off-diagonal entries of the unit-normalised Gram
        matrix of the flattened spatial loadings ([F, W*H]). Zero iff
        the F loadings are mutually orthogonal as flat surface
        patterns; grows quadratically with pairwise cosine similarity.

        Two loadings that look like +PC1 and −PC1 sit at cos-sim −1
        (highly penalised — they are parallel up to sign), which is
        exactly the failure mode probe 4 on the Ax3 run found
        (AxialFactor/63_21/2026-05-17T16-03-24Z/: all four loadings
        best-match PC1 with |cos-sim| 0.96–0.98; stacked-loading
        effective rank 1.72 out of 4).

        Used together with `factor_orthogonality_loss` (same weight
        1e-2): queries-only decorrelated cleanly but did not propagate
        downstream, because the attention W_Q/W_K/W_V/W_O + temporal
        map + loadings together provide enough gauge freedom to
        re-collapse onto a single surface mode. Penalising the loadings
        directly removes that degree of freedom at the structural exit
        of the model.
        """
        L = self.reconstructor.spatial_loadings                      # [F, W, H]
        L_flat = L.reshape(self.n_factors, -1)                       # [F, W*H]
        L_n = L_flat / (L_flat.norm(dim=-1, keepdim=True) + 1e-8)
        G = L_n @ L_n.T                                              # [F, F]
        eye = torch.eye(self.n_factors, device=L.device, dtype=L.dtype)
        off_diag = G - eye
        return (off_diag ** 2).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, W, H]
        B, L, W, H = x.shape

        # Step 1: per-cell window normalisation.
        if self.norm_mode == "cell_mean":
            mu  = x.mean(dim=1, keepdim=True)                        # [B, 1, W, H]
            std = None
            x_n = x - mu
        elif self.norm_mode == "cell_full":
            mu  = x.mean(dim=1, keepdim=True)                        # [B, 1, W, H]
            std = x.std(dim=1, keepdim=True, unbiased=False) + 1e-5  # [B, 1, W, H]
            x_n = (x - mu) / std
        else:  # "none"
            mu = std = None
            x_n = x

        # Step 2: cell embedding (per-cell linear projection of the
        # lookback) + 2D PE. Cells are flattened W-outer, H-inner.
        cell_seqs   = x_n.reshape(B, L, W * H).transpose(1, 2)       # [B, W*H, L]
        cell_tokens = self.embedder(cell_seqs)                       # [B, W*H, d]
        pe_flat     = self._pe_flat()                                # [W*H, d]
        cell_tokens = cell_tokens + pe_flat                          # broadcast over B
        cell_tokens = self.embed_norm(cell_tokens)                   # [B, W*H, d]

        # Step 3: separable axial cross-attention -> factor tokens.
        factor_tokens = self.extractor(cell_tokens, W, H)            # [B, F, d]

        # Step 4: per-factor temporal map.
        g = self.temporal(factor_tokens)                             # [B, F, P]

        # Step 5a: factor path -> per-(batch, horizon) surface via
        # fixed per-factor spatial loadings (no horizon-conditional
        # routing).
        y_factor = self.reconstructor(g)                             # [B, P, W, H]

        # Step 5b: gated residual bypass — per-cell linear from x_n with
        # cell-shared temporal weights, scaled by a per-horizon gate.
        y_residual = self.residual_path(x_n)                         # [B, P, W, H]

        y_n = y_factor + y_residual                                  # [B, P, W, H]

        # Step 6: reverse Step 1.
        if self.norm_mode == "cell_mean":
            return y_n + mu
        if self.norm_mode == "cell_full":
            return y_n * std + mu
        return y_n


if __name__ == "__main__":
    torch.manual_seed(0)
    L, P, Wm, Ht = 63, 21, 15, 10
    model = AxialFactor(seq_len=L, pred_len=P, W=Wm, H=Ht)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AxialFactor params: {n_params:,}")

    # Factor-query orthogonality at init: Q @ Q.T ≈ c · I for c = scale².
    Q = model.extractor.factor_queries.detach()
    gram = Q @ Q.T
    diag_mean = float(gram.diag().mean().item())
    off_diag_max = float(
        (gram - torch.eye(model.n_factors) * diag_mean).abs().max().item()
    )
    print(f"  factor_queries gram diag mean: {diag_mean:.6f}  "
          f"max |off-diag|: {off_diag_max:.2e}")
    assert off_diag_max < 1e-5, (
        f"factor_queries not orthogonal at init: max |off-diag|={off_diag_max}"
    )

    ortho_init = model.factor_orthogonality_loss().item()
    print(f"  factor_orthogonality_loss at init: {ortho_init:.2e}")
    assert ortho_init < 1e-6, (
        f"orthogonality loss not zero at init: {ortho_init}"
    )

    # Spatial loadings are init'd as trunc_normal(std=0.02), i.e. random
    # vectors in R^{W*H=150}; they're not orthogonal at init, but
    # pairwise cos-sims should be small and the penalty modest. The
    # check is just a smoke test that the method is computable and
    # doesn't return something pathological.
    loading_init = model.loading_orthogonality_loss().item()
    print(f"  loading_orthogonality_loss at init: {loading_init:.4f}")
    assert loading_init < 1.0, (
        f"loading orthogonality loss unexpectedly large at init: {loading_init}"
    )

    # AR persistences at init: rho_raw=0.5 → ρ ≈ 0.46 for every factor.
    # Symmetry across factors is broken upstream by the orthogonal
    # factor_queries init — the four factors start at the same ρ but
    # see different content, so training should pull the ρ values apart.
    rho_init = model.temporal.rho().detach()
    mu_init  = model.temporal.mu.detach()
    print(f"  AR rho at init: {rho_init.tolist()}")
    print(f"  AR mu  at init: {mu_init.tolist()}")
    assert (rho_init.abs() < 0.6).all(), (
        f"AR persistence too large at init: {rho_init.tolist()}"
    )

    # AR gate at init: horizon_emb std=0.02 ⇒ logits ~ small noise ⇒
    # sigmoid output ≈ 0.5 ± small jitter across horizons.
    ar_gate_init = model.temporal.gate().detach()
    print(f"  AR gate at init: min={ar_gate_init.min().item():.3f}  "
          f"max={ar_gate_init.max().item():.3f}  "
          f"mean={ar_gate_init.mean().item():.3f}")
    assert 0.3 < ar_gate_init.mean().item() < 0.7, (
        f"AR gate not near 0.5 at init: mean={ar_gate_init.mean().item()}"
    )

    x = torch.randn(4, L, Wm, Ht)
    y = model(x)
    assert y.shape == (4, P, Wm, Ht), f"unexpected output shape {tuple(y.shape)}"

    # Gradient flow check.
    loss = y.sum()
    loss.backward()
    issues = []
    for name, p in model.named_parameters():
        if p.grad is None:
            issues.append((name, "None"))
        elif p.grad.abs().sum().item() == 0.0:
            issues.append((name, "all-zero"))
    # Stricter per-slice checks on the most gradient-strandable groups.
    fq_grad   = model.extractor.factor_queries.grad
    rho_grad  = model.temporal.rho_raw.grad
    mu_grad   = model.temporal.mu.grad
    amp_grad  = model.temporal.amplitude_head.weight.grad
    ar_he_grad = model.temporal.horizon_emb.grad
    ar_gl_grad = model.temporal.gate_linear.weight.grad
    rt_grad   = model.residual_path.temporal.grad
    he_grad   = model.residual_path.horizon_emb.grad
    if fq_grad is None or (fq_grad.abs().sum(dim=-1) == 0).any():
        issues.append(("extractor.factor_queries", "row(s) zero"))
    if rho_grad is None or (rho_grad.abs() == 0).any():
        issues.append(("temporal.rho_raw", "factor(s) with zero grad"))
    if mu_grad is None or (mu_grad.abs() == 0).any():
        issues.append(("temporal.mu", "factor(s) with zero grad"))
    if amp_grad is None or amp_grad.abs().sum().item() == 0.0:
        issues.append(("temporal.amplitude_head.weight", "all-zero grad"))
    if ar_he_grad is None or (ar_he_grad.abs().sum(dim=-1) == 0).any():
        issues.append(("temporal.horizon_emb (AR gate)", "row(s) zero"))
    if ar_gl_grad is None or ar_gl_grad.abs().sum().item() == 0.0:
        issues.append(("temporal.gate_linear.weight (AR gate)", "all-zero grad"))
    if rt_grad is None or (rt_grad.abs().sum(dim=0) == 0).any():
        issues.append(("residual_path.temporal", "horizon column(s) zero"))
    if he_grad is None or (he_grad.abs().sum(dim=-1) == 0).any():
        issues.append(("residual_path.horizon_emb", "row(s) zero"))
    if issues:
        for name, why in issues:
            print(f"  gradient issue: {name}: {why}")
        raise AssertionError("gradient-flow check failed")

    # Init mean-broadcast check: with norm='cell_mean', forward(x) at
    # init should be ~ x.mean(L) broadcast across all P horizons.
    model.zero_grad()
    with torch.no_grad():
        y2 = model(x)
        expected = x.mean(dim=1, keepdim=True).expand(-1, P, -1, -1)
        max_diff = (y2 - expected).abs().max().item()
    print(f"  init max |y - x.mean(L) broadcast|: {max_diff:.4f}")
    assert max_diff < 0.1, (
        f"init mean-broadcast deviation too large: {max_diff:.4f}"
    )

    print("AxialFactor sanity checks passed")
