"""
Graph WaveNet (GWN) — Wu, Pan, Long, Jiang, Zhang (IJCAI 2019).

    "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
    https://arxiv.org/abs/1906.00121

Reference implementation:
    https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py

WaveNet-style dilated temporal convolutions with gated activations and skip
connections, interleaved with diffusion graph convolution. The graph
adjacency is a combination of (a) any pre-computed static supports passed
in and (b) a learned low-rank adaptive adjacency  softmax(ReLU(E1 @ E2)).

Constructor arguments:
    num_nodes        int    — number of graph nodes.
    dropout          float  — dropout in graph-conv layers. Default 0.3.
    supports         list   — pre-computed adjacency tensors (e.g. the
                              4-neighbor (tau, moneyness) grid). The learned
                              adaptive adjacency is appended internally when
                              addaptadj is True. Default None.
    gcn_bool         bool   — use diffusion graph conv in the residual path
                              (otherwise use a 1x1 residual conv). Default True.
    addaptadj        bool   — include the learned adaptive adjacency in the
                              support list. Default True.
    aptinit          Tensor — optional (N, N) initialization for the adaptive
                              adjacency; the node embeddings are initialized
                              from its truncated SVD. Default None (random init).
    in_dim           int    — input channels per node. Default 1.
    seq_len          int    — input window length (lookback). Default 21.
    pred_len         int    — forecast horizon length. Default 63.
    residual_channels int   — residual-path width. Default 32.
    dilation_channels int   — dilated-conv width. Default 32.
    skip_channels    int    — skip-path width. Default 256.
    end_channels     int    — penultimate head width. Default 512.
    kernel_size      int    — temporal conv kernel. Default 2.
    blocks           int    — number of WaveNet blocks. Default 4.
    layers           int    — dilated conv layers per block. Default 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncvl,vw->ncwl", (x, A)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float,
                 support_len: int = 1, order: int = 2):
        super().__init__()
        self.nconv = nconv()
        self.mlp   = linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order   = order

    def forward(self, x: torch.Tensor, support: list) -> torch.Tensor:
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GWN(nn.Module):
    """
    Graph WaveNet.

    Input:  [B, in_dim, num_nodes, seq_len]
    Output: [B, pred_len, num_nodes, 1]   (one-shot multi-horizon forecast)
    """
    def __init__(self,
                 num_nodes: int,
                 dropout: float = 0.3,
                 supports: list = None,
                 gcn_bool: bool = True,
                 addaptadj: bool = True,
                 aptinit: torch.Tensor = None,
                 in_dim: int = 1,
                 seq_len: int = 21,
                 pred_len: int = 63,
                 residual_channels: int = 32,
                 dilation_channels: int = 32,
                 skip_channels: int = 256,
                 end_channels: int = 512,
                 kernel_size: int = 2,
                 blocks: int = 4,
                 layers: int = 2):
        super().__init__()
        self.blocks    = blocks
        self.layers    = layers
        self.num_nodes = num_nodes
        self.gcn_bool  = gcn_bool
        self.addaptadj = addaptadj
        self.dropout   = dropout

        order = 2
        rank  = 10

        # Register static supports as buffers so they follow the module to
        # any device without manual .to(device) inside the gcn forward.
        supports = supports or []
        self._num_static_supports = len(supports)
        for i, sup in enumerate(supports):
            self.register_buffer(f"static_support_{i}", sup, persistent=False)

        supports_len = self._num_static_supports
        if gcn_bool and addaptadj:
            supports_len += 1
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, rank))
                self.nodevec2 = nn.Parameter(torch.randn(rank, num_nodes))
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = m[:, :rank] @ torch.diag(p[:rank] ** 0.5)
                initemb2 = torch.diag(p[:rank] ** 0.5) @ n[:, :rank].t()
                self.nodevec1 = nn.Parameter(initemb1)
                self.nodevec2 = nn.Parameter(initemb2)

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))

        self.filter_convs   = nn.ModuleList()
        self.gate_convs     = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs     = nn.ModuleList()
        self.bn             = nn.ModuleList()
        self.gconv          = nn.ModuleList()

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                if gcn_bool:
                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout,
                            support_len=supports_len, order=order))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.receptive_field = receptive_field
        end_kernel = max(seq_len, receptive_field) - receptive_field + 1
        assert end_kernel >= 1, (
            f"Temporal kernel {end_kernel} < 1; reduce blocks/layers or increase seq_len")

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels,
                                    kernel_size=(1, end_kernel), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len,
                                    kernel_size=(1, 1), bias=True)

    def _static_supports(self) -> list:
        return [getattr(self, f"static_support_{i}")
                for i in range(self._num_static_supports)]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [B, in_dim, num_nodes, seq_len]
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x    = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self._static_supports() + [adp]

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

            if self.gcn_bool:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self._static_supports())
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)          # [B, pred_len, num_nodes, 1]
