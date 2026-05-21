"""
ConvLSTM for IV-surface forecasting.

PyTorch re-implementation of the convolutional LSTM of Medvedev & Wang
(2022) — "Multistep forecast of the implied volatility surface using
deep learning", Journal of Futures Markets 42(4), 645-667.

Idea
----
A fully-connected LSTM has to flatten each day's implied-volatility
surface into a vector, which destroys the spatial layout of the
moneyness x maturity grid. Medvedev & Wang instead keep each day's
surface as a small single-channel image and run a stacked ConvLSTM
(Shi et al. 2015) over the lookback window: the convolution kernels
slide across the surface and learn relationships between nearby
moneyness-maturity cells, while the LSTM gating carries information
across time. An average-pooling layer after each ConvLSTM layer
summarises neighbouring cells (the paper attributes capture of the IV
mean-reversion to this), and a final flatten -> dense layer maps the
encoded state to the whole multi-step forecast.

Architecture (Medvedev & Wang 2022, Section 3.5.2 / Figure 4)
-------------------------------------------------------------
    surface sequence  [B, L, 1, W, H]
      -> ConvLSTM layer 1  (16 kernels, 4x4)   return full sequence
      -> dropout 0.25
      -> average-pool 2x2  (per timestep)
      -> ConvLSTM layer 2  (8 kernels, 3x3)    return last hidden state
      -> average-pool 2x2
      -> dropout 0.25
      -> flatten -> dense -> reshape
    forecast          [B, P, W, H]

Differences from the paper (deliberate)
---------------------------------------
* PyTorch port of the original Keras model.
* Grid. The paper bins the surface into 20 log-moneyness groups x 4
  quarterly contract months (80 cells). This project's surface is
  W moneyness x H tau (default 15 x 10 = 150 cells). The ConvLSTM
  recipe — 2 layers, 16/8 kernels, 4x4 then 3x3, average pooling,
  0.25 dropout, flatten+dense head — is kept; only W and H differ.
* The ConvLSTM cell has no peephole connections, matching Keras's
  `ConvLSTM2D` (the layer the paper used); Shi et al.'s original
  formulation includes peephole terms.
* The forget-gate bias is initialised to 1 (Keras `unit_forget_bias`).
* The paper min-max scales inputs to [-1, 1] before training. That is
  a data-pipeline step, not part of the model, and is intentionally
  left out here — this file is the model only, not wired to train.py.
* Window: the paper fixes 30-day in -> 30-day out; here seq_len ->
  pred_len come from the project config.

Training choices in the paper, for whoever wires this up later: MSE
loss, Adam (lr 1e-3), batch size 32, early stopping. The tanh hidden
activation and sigmoid gates are built into the ConvLSTM cell below.

Hyperparameters (passed to ConvLSTM.__init__)
---------------------------------------------
    seq_len          int   — lookback length L.
    pred_len         int   — forecast horizon P.
    W                int   — moneyness axis size.
    H                int   — tau axis size.
    hidden_channels  tuple — kernels (filters) per ConvLSTM layer.
                              Default (16, 8) — Medvedev & Wang.
    kernel_sizes     tuple — conv kernel size per ConvLSTM layer.
                              Default (4, 3) — Medvedev & Wang.
    pool             int   — average-pool kernel/stride after each
                              ConvLSTM layer. Default 2 (paper). Set
                              to 1 to disable pooling.
    dropout          float — dropout rate between layers. Default
                              0.25 (paper).

Input:  [B, seq_len,  W, H]
Output: [B, pred_len, W, H]
"""

import torch
import torch.nn as nn


class _ConvLSTMCell(nn.Module):
    """One ConvLSTM cell (Shi et al. 2015), no peephole connections.

    The input-to-hidden and hidden-to-hidden convolutions are each
    fused into a single conv producing 4 * hidden_channels feature
    maps — the input, forget, cell and output gate pre-activations.
    Inputs are zero-padded with an explicit `nn.ZeroPad2d` so the
    W x H grid size is preserved. This is numerically identical to the
    conv's `padding='same'` (stride 1) but avoids the PyTorch warning
    that 'same' raises for even kernel sizes such as the paper's 4x4.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        # "same" padding for a stride-1 conv: pad a total of
        # kernel_size - 1, split as evenly as possible (the extra cell
        # for even kernels goes on the high side — matching how
        # PyTorch's own padding='same' splits it).
        lo = (kernel_size - 1) // 2
        hi = kernel_size - 1 - lo
        self.pad = nn.ZeroPad2d((lo, hi, lo, hi))
        self.conv_x = nn.Conv2d(in_channels, 4 * hidden_channels,
                                kernel_size, bias=True)
        self.conv_h = nn.Conv2d(hidden_channels, 4 * hidden_channels,
                                kernel_size, bias=False)
        # unit_forget_bias: open the forget gate at init so the cell
        # state carries through early in training (Keras default).
        with torch.no_grad():
            nn.init.zeros_(self.conv_x.bias)
            self.conv_x.bias[hidden_channels:2 * hidden_channels] = 1.0

    def forward(self, x, state):
        # x: [B, in_channels, W, H]; state = (h, c), each [B, hidden, W, H]
        h, c = state
        gates = self.conv_x(self.pad(x)) + self.conv_h(self.pad(h))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class _ConvLSTMLayer(nn.Module):
    """Runs a ConvLSTM cell across the time axis of a [B, T, C, W, H]
    sequence. Hidden and cell states start at zero.

    return_sequences=True  -> stack of hidden states [B, T, hidden, W, H]
    return_sequences=False -> last hidden state only  [B, hidden, W, H]
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int, return_sequences: bool):
        super().__init__()
        self.cell = _ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels
        self.return_sequences = return_sequences

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, W, H = x.shape
        h = x.new_zeros(B, self.hidden_channels, W, H)
        c = x.new_zeros(B, self.hidden_channels, W, H)
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            if self.return_sequences:
                outputs.append(h)
        if self.return_sequences:
            return torch.stack(outputs, dim=1)       # [B, T, hidden, W, H]
        return h                                     # [B, hidden, W, H]


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM surface forecaster (Medvedev & Wang 2022).

    Input:  [B, seq_len,  W, H]
    Output: [B, pred_len, W, H]
    """

    def __init__(self, seq_len: int, pred_len: int, W: int, H: int,
                 hidden_channels=(16, 8), kernel_sizes=(4, 3),
                 pool: int = 2, dropout: float = 0.25):
        super().__init__()
        if len(hidden_channels) != 2 or len(kernel_sizes) != 2:
            raise ValueError(
                "hidden_channels and kernel_sizes must each have 2 "
                "entries (the model is a 2-layer ConvLSTM stack)"
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.W = W
        self.H = H
        self.pool = pool

        # ConvLSTM layer 1 returns the full sequence of hidden states
        # so layer 2 can run over it; layer 2 returns the last hidden
        # state, which encodes the lookback window.
        self.layer1 = _ConvLSTMLayer(
            1, hidden_channels[0], kernel_sizes[0], return_sequences=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.AvgPool2d(pool) if pool > 1 else nn.Identity()

        self.layer2 = _ConvLSTMLayer(
            hidden_channels[0], hidden_channels[1], kernel_sizes[1],
            return_sequences=False,
        )
        self.pool2 = nn.AvgPool2d(pool) if pool > 1 else nn.Identity()
        self.drop2 = nn.Dropout(dropout)

        # Grid size after each average-pool (kernel = stride = pool,
        # no padding -> floor division).
        def _pooled(n: int) -> int:
            return n // pool if pool > 1 else n
        W1, H1 = _pooled(W), _pooled(H)
        W2, H2 = _pooled(W1), _pooled(H1)
        if min(W1, H1, W2, H2) < 1:
            raise ValueError(
                f"pool={pool} shrinks the {W}x{H} grid below 1x1 over "
                f"two layers (layer1 -> {W1}x{H1}, layer2 -> {W2}x{H2}); "
                f"use a smaller pool"
            )

        # Flatten the final encoded state and map it to the whole
        # multi-step forecast in one dense layer (the paper's
        # flatten -> dense head).
        self.head = nn.Linear(hidden_channels[1] * W2 * H2,
                               pred_len * W * H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, W, H]
        B = x.shape[0]

        # Add the single (implied-volatility) channel: [B, L, 1, W, H].
        seq = x.unsqueeze(2)

        # ConvLSTM layer 1 -> full sequence of hidden states.
        seq = self.layer1(seq)                       # [B, L, C1, W, H]
        seq = self.drop1(seq)

        # Average-pool each timestep's feature map. AvgPool2d wants a
        # 4-D tensor, so fold time into the batch axis and unfold after.
        if self.pool > 1:
            B_, L_, C1, Wg, Hg = seq.shape
            seq = self.pool1(seq.reshape(B_ * L_, C1, Wg, Hg))
            seq = seq.reshape(B_, L_, C1, seq.shape[-2], seq.shape[-1])

        # ConvLSTM layer 2 -> last hidden state, then pool + dropout.
        enc = self.layer2(seq)                       # [B, C2, W1, H1]
        enc = self.pool2(enc)                        # [B, C2, W2, H2]
        enc = self.drop2(enc)

        # Flatten + dense -> reshape to the forecast grid.
        out = self.head(enc.flatten(1))              # [B, P*W*H]
        return out.reshape(B, self.pred_len, self.W, self.H)


if __name__ == "__main__":
    torch.manual_seed(0)
    L, P, Wm, Ht = 63, 21, 15, 10
    model = ConvLSTM(seq_len=L, pred_len=P, W=Wm, H=Ht)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvLSTM params: {n_params:,}")

    x = torch.randn(4, L, Wm, Ht)
    y = model(x)
    assert y.shape == (4, P, Wm, Ht), f"unexpected output shape {tuple(y.shape)}"

    # Gradient-flow check.
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

    # Forget-gate bias must start open (=1) on both ConvLSTM layers.
    for idx, layer in enumerate((model.layer1, model.layer2), start=1):
        c = layer.cell.hidden_channels
        f_bias = layer.cell.conv_x.bias[c:2 * c]
        assert torch.allclose(f_bias, torch.ones_like(f_bias)), (
            f"layer {idx} forget-gate bias not initialised to 1"
        )

    # Pooling can be disabled and an odd grid still round-trips.
    nopool = ConvLSTM(seq_len=L, pred_len=P, W=Wm, H=Ht, pool=1)
    assert nopool(x).shape == (4, P, Wm, Ht), "pool=1 path broken"

    print("ConvLSTM sanity checks passed")
