"""
DynGWN (Graph-WaveNet) model — WaveNet-style dilated temporal convolutions
with graph convolution at each step (static grid adjacency + learned
adaptive adjacency).

Hyperparameters (passed to DynGWN.__init__):
    num_iv          int    — number of IV cells (graph nodes for the surface).
    dropout         float  — dropout in graph-conv layers. Default: 0.3.
    in_dim          int    — input channels per node. Default: 1.
    seq_len         int    — input window length (lookback). Default: 21.
    pred_len        int    — forecast horizon length. Default: 63.
    nhid            int    — residual/dilation channels (skip = nhid·2,
                             end = nhid·4). Default: 32.
    kernel_size     int    — temporal conv kernel. Default: 2.
    blocks          int    — number of WaveNet blocks. Default: 4.
    layers          int    — dilated conv layers per block. Default: 2.
    static_supports list   — list of pre-computed adjacency tensors (e.g. the
                             4-neighbor (tau, moneyness) grid). The learned
                             adaptive adjacency is always appended internally.
                             Default: None (adaptive only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _nconv(nn.Module):
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncvl,vw->ncwl", (x, A)).contiguous()


class _linear(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _gcn(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float,
                 support_len: int = 1, order: int = 2):
        super().__init__()
        self.nconv = _nconv()
        self.mlp   = _linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order   = order

    def forward(self, x: torch.Tensor, support: list) -> torch.Tensor:
        out = [x]
        for a in support:
            a  = a.to(x.device)
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class DynGWN(nn.Module):
    """
    WaveNet + adaptive graph convolution.

    Input:  [B, in_dim=1, num_iv, seq_len]  (padded +1 inside forward)
    Output: [B, pred_len, num_iv, 1]         (in scaled space)
    """
    def __init__(self, num_iv: int, dropout: float = 0.3,
                 in_dim: int = 1, seq_len: int = 21, pred_len: int = 63,
                 nhid: int = 32, kernel_size: int = 2,
                 blocks: int = 4, layers: int = 2,
                 static_supports: list = None):
        super().__init__()
        self.blocks  = blocks
        self.layers  = layers
        self.num_iv  = num_iv

        skip_channels = nhid * 2
        end_channels  = nhid * 4
        order         = 2

        self.static_supports = static_supports or []
        for i, sup in enumerate(self.static_supports):
            self.register_buffer(f"static_support_{i}", sup, persistent=False)
        support_len = len(self.static_supports) + 1   # +1 for adaptive

        self.start_conv = nn.Conv2d(in_dim, nhid, kernel_size=(1, 1))

        # Adaptive adjacency node vectors (rank-5 low-rank embedding)
        self.nodevec1 = nn.Parameter(torch.randn(num_iv, 5))
        self.nodevec2 = nn.Parameter(torch.randn(5, num_iv))

        self.filter_convs = nn.ModuleList()
        self.gate_convs   = nn.ModuleList()
        self.skip_convs   = nn.ModuleList()
        self.gconv        = nn.ModuleList()

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(nhid, nhid, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(
                    nn.Conv2d(nhid, nhid, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.skip_convs.append(
                    nn.Conv2d(nhid, skip_channels, kernel_size=(1, 1)))

                if (i + 1) * (b + 1) - 1 < blocks * layers - 1:
                    self.gconv.append(
                        _gcn(nhid, nhid, dropout, support_len=support_len, order=order))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.receptive_field = receptive_field
        end_kernel = seq_len + 1 - receptive_field + 1   # +1 for the input pad
        assert end_kernel >= 1, (
            f"Temporal kernel {end_kernel} < 1; reduce blocks/layers or increase seq_len")

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels,
                                    kernel_size=(1, end_kernel), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len,
                                    kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim, num_iv, seq_len]
        x = F.pad(x, (1, 0, 0, 0))        # [B, in_dim, num_iv, seq_len+1]
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))

        x    = self.start_conv(x)
        skip = 0
        adp  = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        statics = [getattr(self, f"static_support_{i}")
                   for i in range(len(self.static_supports))]
        new_supports = statics + [adp]

        gcn_idx = 0
        for i in range(self.blocks * self.layers):
            residual = x
            f = torch.tanh(self.filter_convs[i](residual))
            g = torch.sigmoid(self.gate_convs[i](residual))
            x = f * g

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            if i < self.blocks * self.layers - 1:
                x = self.gconv[gcn_idx](x, new_supports)
                gcn_idx += 1
                x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)          # [B, pred_len, num_iv, 1]
