import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    """Patch Embedding with 2d/3d convolution"""
    def __init__(self, n_channels, d_hidden, patch_size, order=3, dropout=0.):
        super().__init__()
        conv = nn.Conv3d if order == 3 else nn.Conv2d
        norm = nn.BatchNorm3d if order == 3 else nn.BatchNorm2d
        self.conv = nn.Sequential(
            conv(n_channels, d_hidden, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            norm(d_hidden),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        h = self.conv(x.float())  
        if len(h.shape) == 5:
            return h.permute(0, 2, 3, 4, 1)          # (B, W, H, T, D)
        return h.permute(0, 2, 3, 1)


class LAPE(nn.Module):
    def __init__(self, max_size=512, d_model=128, order=1):
        super().__init__()
        self.max_size = max_size
        self.d_model = d_model
        size = [max_size] * order
        self.pe = nn.Parameter(torch.randn(*size, d_model), requires_grad=True)

    def forward(self, x):
        bs = x.shape[0]
        slices = []
        for n in x.shape[1:-1]:
            assert n <= self.max_size, "Input size exceeded max_size"
            slices += [slice(0, n)]
        assert x.shape[-1] == self.d_model, "Invalid hidden size"
        return self.pe[tuple(slices)].unsqueeze(0).repeat(bs, 1, 1)


class SinPE1D(nn.Module):
    def __init__(self, max_size=512, d_model=128):
        super().__init__()
        self.max_size = max_size
        self.d_model = d_model
        
        self.pe = nn.Parameter(torch.zeros(max_size, d_model), requires_grad=False)
        pos = torch.arange(0., max_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * (math.log(1.0/10000.0) / d_model)
        )
        self.pe[:, 0:d_model:2] = torch.sin(pos * div_term)
        self.pe[:, 1:d_model:2] = torch.cos(pos * div_term)
    
    def forward(self, x):
        bs, n, d = x.shape
        assert n <= self.max_size, "Input size exceeded max_size"
        assert d == self.d_model, "Invalid hidden size"
        return self.pe[:n].unsqueeze(0).repeat(bs, 1, 1)


class SinPE2D(nn.Module):
    def __init__(self, max_size=512, d_model=128):
        super().__init__()
        self.max_size = max_size
        self.d_model = d_model
        
        self.pe = nn.Parameter(torch.zeros(max_size, max_size, d_model), requires_grad=False)
        dim = d_model // 2
        pos_w = torch.arange(0., max_size).unsqueeze(1)
        pos_h = torch.arange(0., max_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., dim, 2) *
                         (math.log(1.0/10000.0) / dim))
        self.pe[:, :, 0:dim:2] = torch.sin(pos_w * div_term).unsqueeze(1).repeat(1, max_size, 1)
        self.pe[:, :, 1:dim:2] = torch.cos(pos_w * div_term).unsqueeze(1).repeat(1, max_size, 1)
        self.pe[:, :, dim::2] = torch.sin(pos_h * div_term).unsqueeze(0).repeat(max_size, 1, 1)
        self.pe[:, :, dim+1::2] = torch.cos(pos_h * div_term).unsqueeze(0).repeat(max_size, 1, 1)
    
    def forward(self, x):
        bs, h, w, d = x.shape
        assert h <= self.max_size and w <= self.max_size, "Input size exceeded max_size"
        assert d == self.d_model, "Invalid hidden size"
        return self.pe[:h, :w].unsqueeze(0).repeat(bs, 1, 1, 1)

class SinPE3D(nn.Module):
    def __init__(self, max_size=512, d_model=128):
        super().__init__()
        self.max_size = max_size
        self.d_model = d_model
        
        self.pe = nn.Parameter(torch.zeros(max_size, max_size, max_size, d_model), requires_grad=False)
        dim = d_model // 3
        pos_w = torch.arange(0., max_size).unsqueeze(1)
        pos_h = torch.arange(0., max_size).unsqueeze(1)
        pos_d = torch.arange(0., max_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., dim, 2) *
                         (math.log(1.0/10000.0) / dim))
        div_term2 = torch.exp(torch.arange(0., dim + 1, 2) *
                         (math.log(1.0/10000.0) / (dim + 1)))
        self.pe[:, :, :, 0:dim:2] = torch.sin(pos_w * div_term).unsqueeze(0).unsqueeze(0).repeat(max_size, max_size, 1, 1)
        self.pe[:, :, :, 1:dim:2] = torch.cos(pos_w * div_term).unsqueeze(0).unsqueeze(0).repeat(max_size, max_size, 1, 1)
        self.pe[:, :, :, dim:dim*2:2] = torch.sin(pos_h * div_term).unsqueeze(0).unsqueeze(0).repeat(max_size, max_size, 1, 1)
        self.pe[:, :, :, dim+1:dim*2:2] = torch.cos(pos_h * div_term).unsqueeze(0).unsqueeze(0).repeat(max_size, max_size, 1, 1)
        self.pe[:, :, :, 2*dim::2] = torch.sin(pos_d * div_term2).unsqueeze(0).unsqueeze(0).repeat(max_size, max_size, 1, 1)
        self.pe[:, :, :, 2*dim+1::2] = torch.cos(pos_d * div_term2).unsqueeze(0).unsqueeze(0).repeat(max_size, max_size, 1, 1)
    
    def forward(self, x):
        bs, h, w, t, d = x.shape
        assert h <= self.max_size and w <= self.max_size and t <= self.max_size, "Input size exceeded max_size"
        assert d == self.d_model, "Invalid hidden size"
        return self.pe[:h, :w, :t].unsqueeze(0).repeat(bs, 1, 1, 1, 1)
        

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.max_seq_len_cached = 0
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.max_seq_len_cached < seq_len:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
            )

            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(
                "cos_cached", emb.cos().to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin().to(dtype), persistent=False
            ) # seq_len, dim

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    
    def forward(self, x):
        bs, l, nh, dh = x.shape
        self._set_cos_sin_cache(seq_len=l, device=x.device, dtype=x.dtype)
        cos = self.cos_cached[:l].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:l].unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self.rotate_half(x) * sin)