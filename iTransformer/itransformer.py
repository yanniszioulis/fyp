"""
iTransformer for IV-surface forecasting.

Cell-as-token attention with per-cell linear time handling. Built
after five rounds on AxialFactor showed that the factor-decomposition
framework had a local ceiling at ~10 % above DLinear on this dataset.
Pivots to a different architectural class entirely: keep DLinear's
strong per-cell time map, but bolt on cross-cell self-attention as the
mechanism for cells to share information — the piece DLinear lacks.

Motivation
----------
DLinear's per-cell linear (lookback → horizon) is the strong baseline.
Adding attention over time was overkill on this data (HOT confirmed).
Removing the per-cell time map and forcing everything through a factor
bottleneck was also overkill (AxialFactor confirmed). The remaining
under-exploited structure in the dataset is **cross-cell relationships**
— how moneyness × τ cells co-move at a given moment. iTransformer
captures exactly that:

  tokens = cells          (not time, not patches, not factors)
  attention = cross-cell  (full self-attention over the W·H grid)
  time = per-cell linear  (same shape as DLinear's per-channel map)

The bet: factor structure of IV surfaces lives in cross-cell
relationships that DLinear cannot exploit; iTransformer's attention
discovers that structure without imposing a rigid factor bottleneck.

Architectural choices
---------------------
* **Per-cell RevIN (default).** Each cell's lookback mean / std are
  stripped before tokenisation and restored on the forecast. This is
  the iTransformer-canonical normalisation (per-variable, not the joint
  per-window variant used by DLinear / PatchTST). With this on, the
  model implicitly starts by predicting "per-cell lookback mean
  broadcast across horizons" — the same warm-start the no-RevIN
  variant needed an explicit `head_bias` warm-start to express. Pass
  `revin=False` to disable; then `train.py` warm-starts the head bias
  to the per-cell-per-horizon training-target mean instead.
* **Fixed 2D sinusoidal positional encoding, split-half.** The first
  d_model/2 dimensions sinusoidally encode moneyness index; the
  second d_model/2 dimensions encode τ index. Pe is built once at
  construction and registered as a non-persistent buffer (zero
  learnable PE parameters). This bakes in the geometric prior that
  cells close on the (m, τ) grid behave similarly (the τ grid is
  log-spaced, so grid-adjacency on H is log-τ adjacency — the natural
  metric for term-structure dynamics).
* **Full cross-cell self-attention.** Every cell attends to every
  cell. With W·H = 150 tokens, attention is 150² = 22 500 weights per
  head per layer — cheap. No axial restriction (factor models
  imposed that), no windowing (HOT's Kronecker form imposed that), no
  patches.
* **Two transformer blocks**, pre-norm, GELU FFN inner-dim ratio 2
  (instead of the usual 4). Conservative defaults for ~1 200 training
  windows.
* **Per-cell output head.** Each cell has its own [d_model, pred_len]
  weight matrix and [pred_len] bias. The shared embedding +
  cross-cell attention build a context-aware token per cell; the
  per-cell head then decodes that token into a cell-specific forecast
  trajectory. This is the iTransformer ↔ DLinear hybrid: cross-cell
  information sharing in the attention, per-cell expressiveness in
  the head. Weights are trunc_normal(std=head_init_scale=0.001); the
  bias is zero-init at construction but **warm-started in train.py to
  the per-cell-per-horizon training mean before the optimiser is
  built** — the model starts by predicting the historical mean per
  cell per horizon, a reasonable baseline that is particularly
  accurate in calm regimes.

Pipeline
--------
   x [B, L, W, H]
     ──(permute & reshape: W·H tokens, each carrying its L-length lookback)──▶
   tokens [B, W·H, L]
     ──(embed: per-cell Linear(L → d_model), shared across cells)──▶
   tokens [B, W·H, d_model]
     ──(+ fixed 2D sinusoidal PE)──▶
     ──(input LayerNorm)──▶
     ──(N transformer blocks: pre-norm self-attn + pre-norm FFN)──▶
   tokens [B, W·H, d_model]
     ──(head_norm)──▶
     ──(per-cell head: [d_model → pred_len] map and bias, one per cell)──▶
   forecast [B, W·H, P]
     ──(reshape & permute back to grid)──▶
   forecast [B, P, W, H]

Hyperparameters (passed to ITransformer.__init__)
-------------------------------------------------
    seq_len          int   — lookback length L (e.g. 63).
    pred_len         int   — forecast horizon P (e.g. 21).
    W                int   — moneyness axis size.
    H                int   — τ axis size.
    d_model          int   — token embedding dim. Default 64. Must be
                              divisible by 4 (split-half 2D PE then
                              sin/cos pairs within each half) and by
                              n_heads.
    n_blocks         int   — transformer blocks. Default 2.
    n_heads          int   — attention heads per block. Default 4.
    ffn_ratio        int   — FFN inner-dim multiplier. Default 2
                              (smaller than the usual 4× for the small
                              training-set size).
    dropout          float — dropout in FFN and on attention output.
                              Default 0.1.
    head_init_scale  float — std of trunc_normal init for the output
                              head's weight. Default 0.001 so the
                              initial forecast is ~zero and the head
                              learns absolute level cleanly from
                              gradient signal.

Input:  [B, seq_len,  W, H]
Output: [B, pred_len, W, H]

Cell-flattening convention: cell (w, h) maps to flat index w*H + h
(W-outer, H-inner) in both directions of the reshape so the round-trip
is identity.
"""

import math

import torch
import torch.nn as nn


def _build_2d_geometric_pe(W: int, H: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal 2D PE, split-half encoding.

    First d_model/2 dims encode moneyness index (W-axis); second half
    encode τ index (H-axis). Within each half: standard NLP-style
    alternating sin/cos at geometric frequencies (base 10 000).

    Dot products of PE vectors decay monotonically with grid distance —
    the inductive bias matching IV surface geometry: cells close on the
    (m, τ) grid (the τ grid being log-spaced, so H-adjacency is log-τ
    adjacency) are likely to behave similarly.

    Returns: [W, H, d_model]
    """
    if d_model % 4 != 0:
        raise ValueError(
            f"d_model must be divisible by 4 (split-half then sin/cos pairs); "
            f"got {d_model}"
        )
    d_half = d_model // 2

    pe = torch.zeros(W, H, d_model)

    # Frequencies for the sin/cos pairs within each half.
    div = torch.exp(
        torch.arange(0, d_half, 2, dtype=torch.float)
        * (-math.log(10000.0) / d_half)
    )                                                              # [d_half/2]

    # Moneyness (W) encoding in dims [0, d_half).
    position_w = torch.arange(W, dtype=torch.float).unsqueeze(1)   # [W, 1]
    pe_w = torch.zeros(W, d_half)
    pe_w[:, 0::2] = torch.sin(position_w * div)
    pe_w[:, 1::2] = torch.cos(position_w * div)
    pe[:, :, :d_half] = pe_w.unsqueeze(1).expand(-1, H, -1)

    # Tau (H) encoding in dims [d_half, d_model).
    position_h = torch.arange(H, dtype=torch.float).unsqueeze(1)   # [H, 1]
    pe_h = torch.zeros(H, d_half)
    pe_h[:, 0::2] = torch.sin(position_h * div)
    pe_h[:, 1::2] = torch.cos(position_h * div)
    pe[:, :, d_half:] = pe_h.unsqueeze(0).expand(W, -1, -1)

    return pe


class _PerCellRevIN(nn.Module):
    """Per-cell Reversible Instance Normalisation, iTransformer-canonical.

    For each (B, w, h) cell, strip the per-window lookback mean and std
    before tokenisation; restore them on the forecast at the end of the
    forward pass.

    Differs from DLinear's / PatchTST's `JointRevIN` (and HOT's `norm`
    flag), which strip a *single* scalar mean/std jointly over (L, C):
    those preserve cross-channel structure within a window but cannot
    handle wide per-cell level differences (e.g. ATM ≈ 12 % vs deep
    wings ≈ 30 %). The per-cell variant is what iTransformer uses
    natively and is well-suited here because each cell is already its
    own token in the architecture.

    With affine=False (default), this is a pure pre/post operation —
    nothing learnable. With affine=True, per-cell γ/β are applied after
    normalisation and inverted before denormalisation (the standard
    Kim et al. 2021 form).

    Statistics are stored on the module after `forward(x, "norm")`;
    `forward(y, "denorm")` retrieves them. Both inputs are
    [B, T, W*H] (T = lookback for normalise, T = pred_len for
    denormalise).
    """

    def __init__(self, n_cells: int, eps: float = 1e-5,
                 affine: bool = False):
        super().__init__()
        self.n_cells = n_cells
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(n_cells))
            self.affine_bias   = nn.Parameter(torch.zeros(n_cells))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: [B, T, n_cells]
        if mode == "norm":
            self.mean  = x.mean(dim=1, keepdim=True).detach()        # [B, 1, n_cells]
            self.stdev = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()                                                # [B, 1, n_cells]
            z = (x - self.mean) / self.stdev
            if self.affine:
                z = z * self.affine_weight + self.affine_bias
            return z
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
            return x * self.stdev + self.mean
        raise NotImplementedError(mode)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block (self-attention + GELU FFN)."""

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int,
                 dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention.
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        # Pre-norm FFN.
        x = x + self.ffn(self.norm2(x))
        return x


class ITransformer(nn.Module):
    """Cell-as-token Transformer for IV-surface forecasting.

    Input:  [B, seq_len,  W, H]
    Output: [B, pred_len, W, H]
    """

    def __init__(self, seq_len: int, pred_len: int, W: int, H: int,
                 d_model: int = 64, n_blocks: int = 2, n_heads: int = 4,
                 ffn_ratio: int = 2, dropout: float = 0.1,
                 head_init_scale: float = 0.001,
                 revin: bool = True, revin_affine: bool = False,
                 revin_eps: float = 1e-5):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must divide d_model ({d_model})"
            )
        if d_model % 4 != 0:
            raise ValueError(
                f"d_model must be divisible by 4 for the split-half 2D PE; "
                f"got {d_model}"
            )

        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.W = W
        self.H = H
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.revin = revin

        # Per-cell RevIN. Strips each cell's own lookback mean / std
        # before tokenisation; reverses on the forecast at the very end
        # of forward(). This is the iTransformer-canonical normalisation
        # (per-variable rather than the joint per-window variant used
        # in DLinear / PatchTST). It also folds in the "predict per-cell
        # lookback mean" warm-start that v2 previously needed an
        # explicit head_bias warm-start for: with zero head_bias and
        # tiny head_weight, the model's normalised-space forecast is
        # ~0, so the de-normalised forecast is approximately the
        # per-cell lookback mean broadcast across horizons.
        if self.revin:
            self.revin_layer = _PerCellRevIN(
                n_cells=W * H, eps=revin_eps, affine=revin_affine,
            )

        # Per-cell lookback embedding. Shared parameters across cells —
        # cross-cell variation must come from the PE + attention.
        self.embed = nn.Linear(seq_len, d_model)

        # Fixed 2D sinusoidal PE, registered as a buffer (no gradient).
        pe_2d = _build_2d_geometric_pe(W, H, d_model)                # [W, H, d]
        self.register_buffer("pe_2d", pe_2d, persistent=False)

        self.input_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, ffn_ratio, dropout)
            for _ in range(n_blocks)
        ])

        # Output head: pre-norm + per-cell [d_model → pred_len] map.
        #
        # Each of the W·H cells has its own [d_model, pred_len] weight
        # matrix and [pred_len] bias. The shared embedding + cross-cell
        # attention build a context-aware token per cell; the per-cell
        # head then decodes that token into a cell-specific forecast.
        # This is the iTransformer ↔ DLinear hybrid: cross-cell info
        # sharing in the attention, per-cell expressiveness in the head.
        #
        # Param count: W·H · (d_model·pred_len + pred_len). At W=15,
        # H=10, d_model=8, P=21 this is 150·(168+21) = 28 350 params —
        # the model's largest single group. If overfitting emerges
        # (train/val gap > ~1.5×), the first fix is a higher weight
        # decay on `head_weight` specifically (see notes in train.py).
        #
        # head_weight init: trunc_normal(std=head_init_scale) so the
        # forecast at step 0 is ~ head_bias (and tiny noise around it).
        # head_bias is zero-init at construction; train.py warm-starts
        # it to the per-cell-per-horizon training mean before the
        # optimiser is built, so the model starts predicting the
        # historical mean per cell per horizon — a reasonable baseline
        # particularly accurate in calm regimes.
        self.head_norm = nn.LayerNorm(d_model)
        self.head_weight = nn.Parameter(
            torch.empty(W * H, d_model, pred_len)
        )
        self.head_bias = nn.Parameter(torch.zeros(W * H, pred_len))
        with torch.no_grad():
            nn.init.trunc_normal_(self.head_weight, std=head_init_scale)
            # head_bias stays at zero here — train.py warm-starts it.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, W, H]
        B, L, W, H = x.shape

        # Step 1: cell tokenisation. Each cell (w, h) becomes a token
        # whose features are its L-length lookback. W-outer, H-inner
        # flattening so cell (w, h) lands at flat index w*H + h.
        tokens = x.permute(0, 2, 3, 1).reshape(B, W * H, L)          # [B, W*H, L]

        # Step 1b: per-cell RevIN — normalise each cell's lookback by
        # its own mean/std. RevIN buffers the stats on `revin_layer`
        # so step 6b can invert them on the forecast.
        if self.revin:
            # `_PerCellRevIN` expects [B, T, n_cells] with T being the
            # axis to reduce. Our cell-token tensor is [B, n_cells, L],
            # so transpose before/after the call.
            tokens_n = self.revin_layer(tokens.transpose(1, 2), "norm")
            tokens = tokens_n.transpose(1, 2)                        # [B, W*H, L]

        # Step 2: per-cell lookback embedding (shared across cells).
        tokens = self.embed(tokens)                                  # [B, W*H, d]

        # Step 3: add fixed 2D sinusoidal PE (broadcast over batch).
        pe_flat = self.pe_2d.reshape(W * H, self.d_model)            # [W*H, d]
        tokens = tokens + pe_flat

        # Step 4: input LayerNorm before the transformer stack.
        tokens = self.input_norm(tokens)

        # Step 5: N pre-norm transformer blocks (full cross-cell attn).
        for block in self.blocks:
            tokens = block(tokens)

        # Step 6: per-cell forecast head. Each cell has its own
        # [d_model, pred_len] weight matrix; the einsum applies the
        # n-th cell's weights to the n-th cell's token.
        tokens = self.head_norm(tokens)                              # [B, W*H, d]
        forecast = (
            torch.einsum("bnd,ndp->bnp", tokens, self.head_weight)
            + self.head_bias                                         # [B, W*H, P]
        )

        # Step 6b: invert the per-cell RevIN. Pre-norm shape is
        # [B, pred_len, n_cells], so transpose to align with how the
        # mean/std were captured.
        if self.revin:
            forecast = self.revin_layer(
                forecast.transpose(1, 2), "denorm"
            ).transpose(1, 2)                                        # [B, W*H, P]

        # Step 7: round-trip back to the grid. Matches the flattening
        # convention from step 1.
        forecast = forecast.reshape(B, W, H, self.pred_len).permute(0, 3, 1, 2)
        return forecast                                              # [B, P, W, H]


if __name__ == "__main__":
    torch.manual_seed(0)
    L, P, Wm, Ht = 63, 21, 15, 10
    model = ITransformer(seq_len=L, pred_len=P, W=Wm, H=Ht)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ITransformer params: {n_params:,}")

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
    if issues:
        for name, why in issues:
            print(f"  gradient issue: {name}: {why}")
        raise AssertionError("gradient-flow check failed")

    # Init forecast scale.
    #   - With revin=True (default): the output is denormalised, so its
    #     scale tracks the input's per-cell mean/std (with x ~ N(0, 1)
    #     here, that's order 1). What we check instead is that the
    #     normalised-space forecast is approximately the input's per-
    #     cell lookback mean broadcast across horizons (the "predict
    #     per-cell mean" warm-start that RevIN provides for free).
    #   - With revin=False: head_weight is small-init'd and head_bias
    #     is zero, so the forecast itself is small.
    model.zero_grad()
    with torch.no_grad():
        y2 = model(x)
        max_abs = y2.abs().max().item()
        mean_abs = y2.abs().mean().item()
    print(f"  init forecast |max|={max_abs:.4f}  |mean|={mean_abs:.4f}")
    if model.revin:
        # Compare against per-cell lookback mean broadcast across P.
        expected = x.mean(dim=1, keepdim=True).expand(-1, P, -1, -1)
        diff = (y2 - expected).abs()
        print(f"  init |y - mu_cell broadcast|: max={diff.max().item():.4f}  "
              f"mean={diff.mean().item():.4f}")
        assert diff.max().item() < 0.5, (
            "RevIN warm-init broken: forecast deviates from per-cell "
            f"lookback mean by {diff.max().item():.4f}"
        )
    else:
        assert max_abs < 0.5, (
            "init forecast too large; small-init head not working: "
            f"max={max_abs}"
        )

    # head_bias must be exactly zero at construction. Under RevIN it
    # stays zero (RevIN handles the level warm-start). Without RevIN,
    # train.py warm-starts it to per-cell-per-horizon training means.
    assert model.head_bias.abs().max().item() == 0.0, (
        "head_bias should be zero at construction; "
        "warm-start (no-RevIN path) happens in train.py"
    )

    # PE check: pe_2d should be a buffer, not a parameter.
    pe_params = [
        name for name, _ in model.named_parameters() if "pe_2d" in name
    ]
    assert len(pe_params) == 0, (
        f"pe_2d should be a buffer, not a parameter: {pe_params}"
    )

    print("ITransformer sanity checks passed")
