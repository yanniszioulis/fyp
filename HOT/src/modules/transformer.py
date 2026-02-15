import math
import torch
from torch import nn
import torch.nn.functional as F

from .attention import KroneckerAttention, DividedAttention, MultiScaleAttention, StandardAttention
from .embeddings import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, d_hidden: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_hidden, d_mlp):
        super().__init__()
        self.w1 = nn.Linear(d_hidden, d_mlp, bias=False)
        self.w2 = nn.Linear(d_mlp, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_mlp, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_hidden, 
        d_mlp,
        n_head, 
        dropout=0.,
        attention_type='kronecker_product', #kronecker_sum, divided_space_time, multiscale, full
        num_modes=1,
        rope_dims=[],
        input_size=7
    ):
        super().__init__()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        rotary_emb = None
        if len(rope_dims) > 0:
            rotary_emb = RotaryEmbedding(d_hidden // n_head, max_position_embeddings=input_size)

        if 'kronecker' in attention_type:
            mode = attention_type.split('_')[1]
            self.attention = KroneckerAttention(
                num_modes,
                d_hidden, 
                n_head, 
                dropout,
                rotary_emb,
                mode,
                rope_dims,
            )
        elif attention_type == 'divided_space_time':
            self.attention = DividedAttention(d_hidden, n_head, dropout, rotary_emb)
        elif attention_type == 'multiscale':
            self.attention = MultiScaleAttention(d_hidden, input_size, n_head, dropout)
        else:
            self.attention = StandardAttention(d_hidden, n_head, dropout, rotary_emb)
        self.feedforward = SwiGLUFeedForward(d_hidden, d_mlp)

    def forward(self, X):
        h = self.attention(self.norm1(X))
        h = X + self.drop1(h)
        return h + self.drop2(self.feedforward(self.norm2(h)))