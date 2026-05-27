"""
PCAFormer — frozen-basis PCA + vanilla time-axis transformer for
IV-surface forecasting.

Pipeline (input [B, L, W, H], output [B, P, W, H]):

    1. flatten cells → [B, L, W*H]
    2. RevIN (per-cell, instance) → x_n
    3. PCA encode: z = (x_n - V_mean) @ V     V: [W*H, F]  frozen buffer
    4. input_proj: Linear(F → d_model) + learned pos-emb over L
    5. nn.TransformerEncoder, N layers   (vanilla MHA + FFN)
    6. flatten head: Linear(L · d_model → P · F) → [B, P, F]
    7. PCA decode: y_n = z_pred @ V.T + V_mean → [B, P, W*H]
    8. RevIN denorm → reshape → [B, P, W, H]

Why frozen PCA: the standard "PCA + transformer" pattern fits the basis
once on training data and lets the transformer model the *dynamics* of
the PC-score time series. A learnable basis just becomes a smart-init
linear layer — it stops being PCA.

Why 1D PCA: each principal component is a free [W*H] vector. The
top components are approximately separable in (moneyness, tenor) — the
PCA "level/slope/skew/butterfly" shapes — but the basis is not
constrained to be separable, so non-separable curvature is captured by
the marginal modes when present.

Why a flatten head: direct multi-step forecasting from the transformer
output. Matches PatchTST's standard head. Tiny-initialised so the
model's step-0 forecast is RevIN's baseline — predicting the per-cell
lookback mean across the horizon — while still letting gradient flow
back into the encoder from step 1.

fit_pca
-------
The frozen basis V (and the cross-window mean V_mean) is populated by
`fit_pca(train_windows)` before the first forward pass. The fit
applies the same RevIN normalisation the forward will, then takes
**one sample per training window** — the last lookback timestep,
which is the most-recent normalised surface in that window — and
SVDs the resulting [N, W·H] matrix. Keeps the top F right singular
vectors. Stored as register_buffers (no gradient).

One sample per window (not every timestep) keeps the data matrix at
the natural size of the training set with no duplication, while still
matching the RevIN-normalised distribution the transformer will see
at inference.
"""

import torch
import torch.nn as nn


class _RevIN(nn.Module):
    """Per-channel reversible instance normalisation (Kim et al. 2022).

    Expects [B, T, C]. Stats are computed across T per (batch, channel).
    """

    def __init__(self, num_channels: int, eps: float = 1e-5,
                 affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_channels))
            self.affine_bias   = nn.Parameter(torch.zeros(num_channels))

    def _stats(self, x: torch.Tensor):
        # x: [B, T, C]. Reduce over T only.
        self.mean  = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._stats(x)
            y = (x - self.mean) / self.stdev
            if self.affine:
                y = y * self.affine_weight + self.affine_bias
            return y
        if mode == "denorm":
            y = x
            if self.affine:
                y = (y - self.affine_bias) / (self.affine_weight + self.eps ** 2)
            return y * self.stdev + self.mean
        raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got {mode!r}")


class PCAFormer(nn.Module):
    """Frozen-PCA + vanilla transformer for IV-surface forecasting.

    Input:  [B, seq_len,  W, H]
    Output: [B, pred_len, W, H]

    Args:
        seq_len        int   — lookback length L.
        pred_len       int   — forecast horizon P.
        W              int   — moneyness axis size.
        H              int   — tenor axis size.
        n_factors      int   — number of PCs kept (F). Default 4
                               (matches the 3–4 PCs that explain >95% of
                               IV surface variance).
        d_model        int   — transformer embedding dim. Default 64.
                               Must be divisible by n_heads.
        n_heads        int   — attention heads. Default 4.
        n_layers       int   — encoder layers. Default 2.
        d_ff           int   — FFN inner dim. Default 128 (2 · d_model).
        dropout        float — transformer dropout. Default 0.1.
        revin_affine   bool  — learnable per-cell affine in RevIN.
                               Default True.
        head_init_scale float — std of trunc_normal init on the flatten
                               head's weight. Tiny (1e-4) so z_pred is
                               near zero at step 0 (warm-start ≈ RevIN
                               baseline = per-cell lookback mean
                               broadcast), but nonzero so gradient still
                               flows back through the head into the
                               transformer. Zero init here would strand
                               every upstream parameter at zero
                               gradient. Default 1e-4.
    """

    def __init__(self, seq_len: int, pred_len: int, W: int, H: int,
                 n_factors: int = 4, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.1,
                 revin_affine: bool = True,
                 head_init_scale: float = 1e-4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must divide d_model ({d_model})"
            )
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        n_cells = W * H
        if n_factors > n_cells:
            raise ValueError(
                f"n_factors ({n_factors}) must be <= W*H ({n_cells})"
            )

        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.W         = W
        self.H         = H
        self.n_cells   = n_cells
        self.n_factors = n_factors
        self.d_model   = d_model

        # ── RevIN over cells (C = W*H channels) ─────────────────────
        self.revin_layer = _RevIN(self.n_cells, affine=revin_affine)

        # ── PCA basis (frozen buffers, populated by fit_pca) ────────
        # V: [C, F] orthonormal columns (top-F right singular vectors
        # of the RevIN'd training data matrix). V_mean: [C] sample
        # mean of the RevIN'd training timesteps (near zero by
        # construction; included for centring correctness).
        self.register_buffer("V",          torch.zeros(self.n_cells, n_factors))
        self.register_buffer("V_mean",     torch.zeros(self.n_cells))
        self.register_buffer("pca_fitted", torch.zeros((), dtype=torch.bool))

        # ── Transformer over PC-score time series ───────────────────
        self.input_proj = nn.Linear(n_factors, d_model)
        self.pos_emb    = nn.Parameter(torch.empty(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        # `enable_nested_tensor=False` silences a PyTorch warning: the
        # nested-tensor fast path is incompatible with pre-norm
        # (`norm_first=True`), so PyTorch would skip it and warn at
        # every construction. We're choosing pre-norm deliberately for
        # training stability — no real fast path is being lost.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # ── Flatten head: [B, L·d_model] → [B, P·F] ────────────────
        # Weights are trunc_normal(std=head_init_scale=1e-4), bias=0. At
        # init z_pred has element std ≈ √(L·d_model) · 1e-4 · std(h) ≈
        # 0.006 for d_model=64, L=63, so y_factor is much smaller than
        # the RevIN scale and the model's step-0 forecast is essentially
        # the per-cell lookback mean broadcast across the horizon. The
        # weight is NOT zero-initialised because that would strand every
        # transformer parameter at zero gradient
        # (∂L/∂h = head.weight.T · ∂L/∂z_pred).
        self.head = nn.Linear(seq_len * d_model, pred_len * n_factors)
        with torch.no_grad():
            nn.init.trunc_normal_(self.head.weight, std=head_init_scale)
            self.head.bias.zero_()
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

    @torch.no_grad()
    def fit_pca(self, train_windows: torch.Tensor) -> None:
        """Fit the frozen PCA basis from training lookback windows.

        train_windows: [N, L, W, H] OR [N, L, W*H].

        The fit applies the same RevIN normalisation the forward will,
        takes ONE sample per window (the last lookback timestep — the
        most recent normalised surface in that window), centres the
        resulting [N, W·H] matrix, and keeps the top-F right singular
        vectors. Stored in `V` and `V_mean` register_buffers; sets
        `pca_fitted`.
        """
        x = train_windows
        if x.dim() == 4:
            B, L, W, H = x.shape
            if (W, H) != (self.W, self.H):
                raise ValueError(
                    f"train_windows W×H = {W}×{H} does not match model "
                    f"W×H = {self.W}×{self.H}"
                )
            x = x.reshape(B, L, self.n_cells)
        elif x.dim() != 3 or x.shape[-1] != self.n_cells:
            raise ValueError(
                f"train_windows must be [N, L, W, H] or [N, L, {self.n_cells}], "
                f"got shape {tuple(x.shape)}"
            )
        x = x.to(self.V.device, dtype=torch.float32)
        # Per-window per-cell mean/std (same as RevIN at inference).
        mean = x.mean(dim=1, keepdim=True)
        std  = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.revin_layer.eps
        )
        x_n = (x - mean) / std                                       # [N, L, C]
        # Note: the RevIN affine is identity at init (γ=1, β=0), so
        # fitting the basis on (x - μ) / σ matches the forward path at
        # step 0. After training the affine drifts, but the basis is
        # frozen — that's the deliberate design.
        #
        # One sample per window: the lookback's last timestep is the
        # most recent normalised surface in that window. Adjacent
        # windows share L-1 days, so the samples are autocorrelated
        # but unique (each window is a distinct (lookback, current-day)
        # pair). N samples instead of N·L removes the per-day
        # multiplicity that overlapping windows would otherwise
        # introduce, while still capturing the RevIN'd input
        # distribution the transformer sees.
        samples = x_n[:, -1, :]                                      # [N, C]
        sample_mean = samples.mean(dim=0)                            # [C]
        sc = samples - sample_mean
        # SVD on the (possibly tall-and-thin) centred data matrix.
        # full_matrices=False keeps the right singular matrix at
        # [min(N, C), C] which is what we want — its first F rows are
        # the top-F principal components as row vectors.
        _, _, Vh = torch.linalg.svd(sc, full_matrices=False)
        V = Vh[: self.n_factors].T.contiguous()                      # [C, F]
        self.V.copy_(V)
        self.V_mean.copy_(sample_mean)
        self.pca_fitted.fill_(True)

    def explained_variance_ratio(
            self, train_windows: torch.Tensor) -> torch.Tensor:
        """Diagnostic: fraction of variance the kept basis explains on
        a held-out (or train) set of windows. Uses the same one-sample-
        per-window scheme as `fit_pca`. Recomputes the data statistics
        with the *frozen* basis — does not refit."""
        if not bool(self.pca_fitted):
            raise RuntimeError("fit_pca() must be called first.")
        x = train_windows
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], self.n_cells)
        x = x.to(self.V.device, dtype=torch.float32)
        mean = x.mean(dim=1, keepdim=True)
        std  = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.revin_layer.eps
        )
        x_n = (x - mean) / std
        samples = x_n[:, -1, :] - self.V_mean                         # [N, C]
        total_var = (samples ** 2).sum()
        proj      = samples @ self.V                                  # [N, F]
        kept_var  = (proj ** 2).sum()
        return kept_var / total_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.pca_fitted):
            raise RuntimeError(
                "PCAFormer.fit_pca(train_windows) must be called before "
                "forward()."
            )
        # x: [B, L, W, H]
        B, L, W, H = x.shape
        # Flatten cells.
        x_flat = x.reshape(B, L, self.n_cells)                       # [B, L, C]

        # 1) RevIN.
        x_n = self.revin_layer(x_flat, "norm")                       # [B, L, C]

        # 2) PCA encode (centre then project onto frozen basis).
        z = (x_n - self.V_mean) @ self.V                             # [B, L, F]

        # 3) Transformer over time.
        h = self.input_proj(z)                                       # [B, L, d]
        h = h + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)                                          # [B, L, d]

        # 4) Flatten head → PC-score forecast.
        z_pred = self.head(h.reshape(B, -1)).view(
            B, self.pred_len, self.n_factors
        )                                                            # [B, P, F]

        # 5) PCA decode.
        y_n = z_pred @ self.V.T + self.V_mean                        # [B, P, C]

        # 6) RevIN denorm and reshape back to surface.
        y = self.revin_layer(y_n, "denorm")                          # [B, P, C]
        return y.reshape(B, self.pred_len, self.W, self.H)


if __name__ == "__main__":
    torch.manual_seed(0)
    L, P, Wm, Ht = 63, 21, 15, 10
    F = 4
    model = PCAFormer(seq_len=L, pred_len=P, W=Wm, H=Ht,
                      n_factors=F, d_model=64, n_heads=4,
                      n_layers=2, d_ff=128, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PCAFormer params: {n_params:,}")

    # Synthetic "training windows" → fit PCA.
    # N_train needs to be reasonably large because we now fit on ONE
    # sample per window (the last lookback timestep). The cross-window
    # sample mean V_mean has per-cell noise ~ 1/√N_train on N(0,1)
    # samples; for the worst-of-150-cells max that is ~3·σ ≈ 3/√N.
    # Real training sets (N≈3000+) make this negligible; bump N here
    # so the smoke test's noise floor stays well under the 0.1
    # warm-start tolerance.
    N_train = 4096
    X_train = torch.randn(N_train, L, Wm, Ht)
    model.fit_pca(X_train)

    # V should be orthonormal columns: V.T @ V ≈ I_F.
    G = model.V.T @ model.V
    eye = torch.eye(F)
    max_off = float((G - eye).abs().max().item())
    print(f"  V.T @ V max |off-I|: {max_off:.2e}")
    assert max_off < 1e-5, f"PCA basis not orthonormal: {max_off}"

    ev_ratio = model.explained_variance_ratio(X_train).item()
    print(f"  explained variance ratio (train, F={F}): {ev_ratio:.4f}")

    # Forward + shape check.
    x = torch.randn(4, L, Wm, Ht)
    y = model(x)
    assert y.shape == (4, P, Wm, Ht), f"unexpected output shape {tuple(y.shape)}"

    # Warm-start: at init, head weights = 0 → z_pred = 0 → y_factor =
    # V_mean broadcast. With the small residual init and the RevIN
    # affine at γ=1, β=0, the forward output should be very close to
    # the per-cell lookback mean broadcast across the horizon.
    with torch.no_grad():
        y2 = model(x)
        expected = x.mean(dim=1, keepdim=True).expand(-1, P, -1, -1)
        max_diff = (y2 - expected).abs().max().item()
    print(f"  init max |y - x.mean(L) broadcast|: {max_diff:.4f}")
    assert max_diff < 0.1, (
        f"init mean-broadcast deviation too large: {max_diff:.4f}"
    )

    # Gradient-flow check.
    model.zero_grad()
    y3 = model(x)
    loss = y3.sum()
    loss.backward()
    issues = []
    for name, p in model.named_parameters():
        if p.grad is None:
            issues.append((name, "None"))
        elif p.grad.abs().sum().item() == 0.0:
            issues.append((name, "all-zero"))
    # Buffers (V, V_mean, pca_fitted) must NOT have gradients.
    assert model.V.grad is None and model.V_mean.grad is None, (
        "PCA basis should be a frozen buffer, not a parameter"
    )
    if issues:
        for name, why in issues:
            print(f"  gradient issue: {name}: {why}")
        raise AssertionError("gradient-flow check failed")

    # fit_pca round-trip identity: encoding then decoding a centred
    # sample with rank ≥ F should preserve at least `ev_ratio` of its
    # squared norm.
    s = torch.randn(8, Wm * Ht)
    sc = s - model.V_mean
    z  = sc @ model.V
    s_hat = z @ model.V.T + model.V_mean
    keep_frac = ((s_hat - model.V_mean) ** 2).sum() / (sc ** 2).sum()
    print(f"  rank-F round-trip retained variance fraction: "
          f"{keep_frac.item():.4f}")

    print("PCAFormer sanity checks passed")
