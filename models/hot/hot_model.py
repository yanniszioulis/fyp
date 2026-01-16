"""
Higher-Order Transformer (HOT) model for IV surface forecasting.

Uses Kronecker-structured attention to factorize attention across
(time, tau, logm) dimensions for efficient 3D surface modeling.
Predicts correction from baseline to target surface.
"""

from typing import Optional
import os
import copy
import numpy as np
import math

from models.base_model import BaseModel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    from einops import einsum, rearrange
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for temporal dimension."""
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
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device if device is None else device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.max_seq_len_cached < seq_len:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(
                "cos_cached", emb.cos().to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin().to(dtype), persistent=False
            )

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


class KroneckerAttention(nn.Module):
    """
    Kronecker-structured attention for 3D data (time, tau, logm).
    
    Factorizes attention across dimensions to reduce complexity from
    O((T×τ×m)²) to O(T² + τ² + m²).
    """
    def __init__(
        self, 
        num_modes: int,  # Should be 3 for (time, tau, logm)
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
        rotary_emb: Optional[RotaryEmbedding] = None,
        mode: str = 'product',  # 'product' or 'sum'
        rope_dims: list = []  # Which dimensions to apply RoPE (e.g., [0] for time)
    ):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.rotary_emb = rotary_emb
        self.rope_dims = rope_dims
        self.mode = mode
        
        # Projections for each mode (time, tau, logm)
        self.query_proj = nn.Linear(d_model, d_model * num_modes)
        self.key_proj = nn.Linear(d_model, d_model * num_modes)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.att_dropout = nn.Dropout(p=dropout)
        self.proj_dropout = nn.Dropout(p=dropout)
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)

    def compute_attention(self, query, key, value, dim, use_rope=True):
        """
        Compute attention along a specific dimension.
        
        Args:
            query: (batch, ..., l, d_model) where l is the length along dim
            key: (batch, ..., l, d_model)
            value: (batch, ..., l, n_head, d_head)
            dim: dimension index to compute attention along
            use_rope: whether to apply rotary embedding
        """
        def pool(x):
            # Pool other dimensions, keep the attention dimension
            # TODO: Verify this pooling strategy is correct for our use case
            return einsum(x, 'bs ... l nh dh -> bs l nh dh')
        
        l = query.shape[dim]
        q = query.transpose(dim, -2)   # (bs ... l d)
        k = key.transpose(dim, -2)      # (bs ... l d)
        v = value.transpose(dim, -3)    # (bs ... l nh dh)

        q = q.unflatten(dim=-1, sizes=(self.n_head, self.d_head))  # (bs ... l nh dh)
        k = k.unflatten(dim=-1, sizes=(self.n_head, self.d_head))  # (bs ... l nh dh)

        q = self.q_norm(pool(q))
        k = self.k_norm(pool(k))
        
        if use_rope and self.rotary_emb is not None:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        att = einsum(q, k, 'bs l1 nh d, bs l2 nh d -> bs l1 l2 nh') / math.sqrt(q.shape[3])
        att = self.att_dropout(F.softmax(att, dim=2))
        h = einsum(att, v, 'bs l1 l2 nh, bs ... l2 nh d -> bs ... l1 nh d')
        return h.transpose(dim, -3), att

    def forward(self, X):
        """
        Forward pass with Kronecker attention.
        
        Args:
            X: (batch, context_length, n_tau, n_logm, d_model)
        
        Returns:
            output: (batch, context_length, n_tau, n_logm, d_model)
        """
        # Split projections for each mode (time, tau, logm)
        query = self.query_proj(X).split(self.d_model, dim=-1)
        key = self.key_proj(X).split(self.d_model, dim=-1)
        value = self.value_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head))
        
        if self.mode == 'product':
            # Sequential attention along each dimension
            # TODO: Verify order matters? Should we do time -> tau -> logm or different order?
            for idx, dim in enumerate(range(1, len(X.shape) - 1)):  # Skip batch and feature dims
                value, _ = self.compute_attention(
                    query=query[idx],
                    key=key[idx],
                    value=value,
                    dim=dim,
                    use_rope=self.rotary_emb is not None and dim in self.rope_dims
                )
            value = value.flatten(start_dim=-2)
        elif self.mode == 'sum':
            # Sum attention across dimensions
            # TODO: Is sum mode useful for IV surfaces? May need experimentation
            res = 0
            for idx, dim in enumerate(range(1, len(X.shape) - 1)):
                v, _ = self.compute_attention(
                    query=query[idx],
                    key=key[idx],
                    value=value,
                    dim=dim,
                    use_rope=self.rotary_emb is not None and dim in self.rope_dims
                )
                res += v
            value = res.flatten(start_dim=-2)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self.proj_dropout(self.out_proj(value))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU activation for feedforward network."""
    def __init__(self, d_hidden, d_mlp):
        super().__init__()
        self.w1 = nn.Linear(d_hidden, d_mlp, bias=False)
        self.w2 = nn.Linear(d_mlp, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_mlp, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HOTTransformerBlock(nn.Module):
    """Transformer block with Kronecker attention."""
    def __init__(
        self,
        d_hidden: int,
        d_mlp: int,
        n_head: int,
        dropout: float = 0.0,
        attention_type: str = 'kronecker_product',
        num_modes: int = 3,  # time, tau, logm
        rope_dims: list = [0],  # Apply RoPE to time dimension (index 1 in shape)
        max_context_length: int = 100,
    ):
        super().__init__()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        
        rotary_emb = None
        if len(rope_dims) > 0:
            # TODO: Should we use RoPE for time dimension? Or learnable positional embeddings?
            rotary_emb = RotaryEmbedding(
                d_hidden // n_head,
                max_position_embeddings=max_context_length
            )

        if 'kronecker' in attention_type:
            mode = attention_type.split('_')[1] if '_' in attention_type else 'product'
            self.attention = KroneckerAttention(
                num_modes,
                d_hidden,
                n_head,
                dropout,
                rotary_emb,
                mode,
                rope_dims,
            )
        else:
            raise ValueError(f"Only kronecker attention supported, got: {attention_type}")
        
        self.feedforward = SwiGLUFeedForward(d_hidden, d_mlp)

    def forward(self, X):
        """X: (batch, context_length, n_tau, n_logm, d_model)"""
        h = self.attention(self.norm1(X))
        h = X + self.drop1(h)
        return h + self.drop2(self.feedforward(self.norm2(h)))


class SurfaceEmbedding(nn.Module):
    """
    Embed 2D IV surface into d_model dimensions.
    
    Options:
    1. Linear projection per surface point (simple, preserves structure)
    2. Conv2D with patches (reduces resolution, may lose fine details)
    3. Learnable embeddings per (tau, logm) location (most flexible)
    
    TODO: Experiment with different embedding strategies
    """
    def __init__(
        self,
        d_model: int,
        n_tau: int,
        n_logm: int,
        embedding_type: str = 'linear',  # 'linear', 'conv2d', 'learnable'
        patch_size: Optional[int] = None,
    ):
        super().__init__()
        self.embedding_type = embedding_type
        self.n_tau = n_tau
        self.n_logm = n_logm
        
        if embedding_type == 'linear':
            # Simple linear projection: each surface point -> d_model
            self.embed = nn.Linear(1, d_model)
        elif embedding_type == 'conv2d':
            # 2D convolution with patches
            # TODO: Determine optimal patch size (e.g., 2x2, 4x4)
            if patch_size is None:
                patch_size = 2
            self.embed = nn.Sequential(
                nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )
            # Update n_tau, n_logm after patching
            self.n_tau = n_tau // patch_size
            self.n_logm = n_logm // patch_size
        elif embedding_type == 'learnable':
            # Learnable embeddings per (tau, logm) location
            # TODO: This might be too many parameters? Consider if needed
            self.embed = nn.Parameter(torch.randn(n_tau, n_logm, d_model))
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

    def forward(self, x):
        """
        Args:
            x: (batch, context_length, n_tau, n_logm) - raw IV values
        
        Returns:
            embedded: (batch, context_length, n_tau, n_logm, d_model)
        """
        batch, context_length, n_tau, n_logm = x.shape
        
        if self.embedding_type == 'linear':
            # (batch, context_length, n_tau, n_logm, 1) -> (batch, context_length, n_tau, n_logm, d_model)
            x = x.unsqueeze(-1)
            embedded = self.embed(x)
        elif self.embedding_type == 'conv2d':
            # Reshape for conv2d: (batch*context, 1, n_tau, n_logm)
            x_reshaped = x.view(batch * context_length, 1, n_tau, n_logm)
            embedded = self.embed(x_reshaped)  # (batch*context, d_model, n_tau', n_logm')
            # Reshape back: (batch, context_length, n_tau', n_logm', d_model)
            embedded = embedded.permute(0, 2, 3, 1).contiguous()
            embedded = embedded.view(batch, context_length, self.n_tau, self.n_logm, -1)
        elif self.embedding_type == 'learnable':
            # Broadcast learnable embeddings
            embedded = self.embed.unsqueeze(0).unsqueeze(0)  # (1, 1, n_tau, n_logm, d_model)
            embedded = embedded.repeat(batch, context_length, 1, 1, 1)
            # Add value information? Or just use learnable?
            # TODO: Consider adding x as a bias or using it somehow
            embedded = embedded + x.unsqueeze(-1) * 0.01  # Small contribution from actual values
        
        return embedded


class PositionalEmbedding3D(nn.Module):
    """
    Positional embeddings for 3D data (time, tau, logm).
    
    Options:
    1. Learnable embeddings for each dimension
    2. Sinusoidal for tau/logm (ordered), learnable for time
    3. RoPE for time (handled in attention), learnable for tau/logm
    
    TODO: Experiment with different positional encoding strategies
    """
    def __init__(
        self,
        max_context_length: int,
        n_tau: int,
        n_logm: int,
        d_model: int,
        pe_type: str = 'learnable',  # 'learnable', 'sinusoidal', 'none'
    ):
        super().__init__()
        self.pe_type = pe_type
        
        if pe_type == 'learnable':
            # Learnable embeddings for each dimension
            # TODO: Should we combine them additively or use separate embeddings?
            self.pe_time = nn.Parameter(torch.randn(max_context_length, d_model))
            self.pe_tau = nn.Parameter(torch.randn(n_tau, d_model))
            self.pe_logm = nn.Parameter(torch.randn(n_logm, d_model))
        elif pe_type == 'sinusoidal':
            # Sinusoidal for tau/logm (they're ordered), learnable for time
            # TODO: Implement sinusoidal embeddings if needed
            self.pe_time = nn.Parameter(torch.randn(max_context_length, d_model))
            # Could add sinusoidal here
            self.pe_tau = nn.Parameter(torch.randn(n_tau, d_model))
            self.pe_logm = nn.Parameter(torch.randn(n_logm, d_model))
        elif pe_type == 'none':
            # No positional embeddings (rely on RoPE in attention)
            self.pe_time = None
            self.pe_tau = None
            self.pe_logm = None
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

    def forward(self, x):
        """
        Args:
            x: (batch, context_length, n_tau, n_logm, d_model)
        
        Returns:
            x with positional embeddings added
        """
        if self.pe_type == 'none':
            return x
        
        batch, context_length, n_tau, n_logm, d_model = x.shape
        
        # Add positional embeddings
        # TODO: Should we add them separately or combine? Current: additive
        if self.pe_time is not None:
            pe_t = self.pe_time[:context_length].unsqueeze(1).unsqueeze(1)  # (context, 1, 1, d_model)
            x = x + pe_t
        
        if self.pe_tau is not None:
            pe_tau = self.pe_tau.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, n_tau, 1, d_model)
            x = x + pe_tau
        
        if self.pe_logm is not None:
            pe_logm = self.pe_logm.unsqueeze(0).unsqueeze(0).unsqueeze(1)  # (1, 1, 1, n_logm, d_model)
            x = x + pe_logm
        
        return x


class _HOTSurfaceModel(nn.Module):
    """
    HOT model for IV surface forecasting.
    
    Architecture:
    1. Embed each surface in context: (batch, context, tau, logm) -> (batch, context, tau, logm, d_model)
    2. Add positional embeddings
    3. Apply HOT transformer blocks with Kronecker attention
    4. Pool over time dimension to get single representation
    5. Output head to predict correction surface
    """
    def __init__(
        self,
        context_length: int,
        n_tau: int,
        n_logm: int,
        d_model: int = 256,
        d_mlp: int = 1024,
        n_blocks: int = 4,
        n_head: int = 8,
        dropout: float = 0.1,
        embedding_type: str = 'linear',
        pe_type: str = 'learnable',
        attention_mode: str = 'kronecker_product',
        use_rope: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.n_tau = n_tau
        self.n_logm = n_logm
        self.d_model = d_model
        
        # Surface embedding
        self.surface_embed = SurfaceEmbedding(
            d_model=d_model,
            n_tau=n_tau,
            n_logm=n_logm,
            embedding_type=embedding_type,
        )
        
        # Update n_tau, n_logm if using conv2d embedding
        if embedding_type == 'conv2d':
            self.n_tau = self.surface_embed.n_tau
            self.n_logm = self.surface_embed.n_logm
        
        # Positional embeddings
        self.pos_emb = PositionalEmbedding3D(
            max_context_length=context_length,
            n_tau=self.n_tau,
            n_logm=self.n_logm,
            d_model=d_model,
            pe_type=pe_type,
        )
        
        # HOT transformer blocks
        rope_dims = [1] if use_rope else []  # Apply RoPE to time dimension (index 1)
        self.blocks = nn.ModuleList([
            HOTTransformerBlock(
                d_hidden=d_model,
                d_mlp=d_mlp,
                n_head=n_head,
                dropout=dropout,
                attention_type=attention_mode,
                num_modes=3,  # time, tau, logm
                rope_dims=rope_dims,
                max_context_length=context_length,
            )
            for _ in range(n_blocks)
        ])
        
        # Pool over time dimension
        # TODO: Should we use mean, last, or learnable pooling?
        self.temporal_pool = 'mean'  # 'mean', 'last', 'learnable'
        
        # Output head: predict correction surface
        # TODO: Should output head be simple linear or more complex?
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)  # Output single value per (tau, logm) location
        )

    def forward(self, x):
        """
        Args:
            x: (batch, context_length, n_tau, n_logm) - normalized IV surfaces
        
        Returns:
            correction: (batch, n_tau, n_logm) - predicted correction surface
        """
        # Embed surfaces
        h = self.surface_embed(x)  # (batch, context, n_tau, n_logm, d_model)
        
        # Add positional embeddings
        h = self.pos_emb(h)
        
        # Apply HOT transformer blocks
        for block in self.blocks:
            h = block(h)
        
        # Pool over time dimension
        if self.temporal_pool == 'mean':
            h = h.mean(dim=1)  # (batch, n_tau, n_logm, d_model)
        elif self.temporal_pool == 'last':
            h = h[:, -1, :, :, :]  # (batch, n_tau, n_logm, d_model)
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")
        
        # Output head: predict correction
        correction = self.head(h).squeeze(-1)  # (batch, n_tau, n_logm)
        
        return correction


class HOTSurfaceModel(BaseModel):
    """
    Higher-Order Transformer for IV surface forecasting.
    
    Uses Kronecker-structured attention to efficiently model 3D IV surface data
    (time, tau, logm). Predicts correction from baseline to target surface.
    """
    
    def __init__(
        self,
        name: str = "hot_surface",
        d_model: int = 256,
        d_mlp: int = 1024,
        n_blocks: int = 4,
        n_head: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 50,
        patience: int = 10,
        min_delta: float = 0.0,
        use_amp: bool = False,
        baseline_decay: float = 1.0,  # -1 for persistence, 0.0-1.0 for exponential
        embedding_type: str = 'linear',  # 'linear', 'conv2d', 'learnable'
        pe_type: str = 'learnable',  # 'learnable', 'sinusoidal', 'none'
        attention_mode: str = 'kronecker_product',  # 'kronecker_product', 'kronecker_sum'
        use_rope: bool = True,  # Use RoPE for time dimension
        device: Optional[str] = None,
    ):
        super().__init__(name=name)
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and einops are required for HOTSurfaceModel. "
                "Install with: pip install torch einops"
            )

        self.requires_normalization = False
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_blocks = n_blocks
        self.n_head = n_head
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.use_amp = use_amp
        self.baseline_decay = baseline_decay
        self.embedding_type = embedding_type
        self.pe_type = pe_type
        self.attention_mode = attention_mode
        self.use_rope = use_rope
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.n_tau = None
        self.n_logm = None
        self.context_length = None
        self.mean = None
        self.std = None
        self.mean_corr = None
        self.std_corr = None

    def _compute_baseline(self, X: np.ndarray) -> np.ndarray:
        """
        Compute baseline surface from context.
        
        Args:
            X: (n_samples, context_length, n_tau, n_logm)
        
        Returns:
            baseline: (n_samples, n_tau, n_logm)
        """
        if self.baseline_decay == -1:
            # Persistence: use last surface
            return X[:, -1, :, :].copy()
        elif self.baseline_decay == 0.0:
            # Uniform average
            return X.mean(axis=1)
        else:
            # Exponential-weighted average
            # Shifted formula: w_i = (1-decay)^(context_length-1-i) / sum
            context_length = X.shape[1]
            weights = np.array([(1 - self.baseline_decay) ** (context_length - 1 - i) 
                              for i in range(context_length)])
            weights = weights / weights.sum()
            baseline = np.einsum('nctm,c->ntm', X, weights)
            return baseline

    def fit(self, X_train, y_train, context_length, horizon, tau_grid=None, logm_grid=None, **kwargs):
        """
        Train the HOT model.
        
        Args:
            X_train: (n_samples, context_length, n_tau, n_logm)
            y_train: (n_samples, n_tau, n_logm)
            context_length: number of context surfaces
            horizon: prediction horizon (not used directly, but stored)
            tau_grid: tau grid (not used, but kept for interface consistency)
            logm_grid: logm grid (not used, but kept for interface consistency)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        
        n_samples, context_len, n_tau, n_logm = X_train.shape
        self.context_length = context_length
        self.n_tau = n_tau
        self.n_logm = n_logm
        
        # Compute baseline and corrections
        baseline_train = self._compute_baseline(X_train)
        y_correction = y_train - baseline_train
        
        # Normalization stats
        X_train_flat = X_train.reshape(-1, n_tau, n_logm)
        self.mean = X_train_flat.mean(axis=0, keepdims=True)
        self.std = X_train_flat.std(axis=0, keepdims=True)
        self.std = np.maximum(self.std, 1e-8)
        
        y_correction_flat = y_correction.reshape(-1, n_tau, n_logm)
        self.mean_corr = y_correction_flat.mean(axis=0, keepdims=True)
        self.std_corr = y_correction_flat.std(axis=0, keepdims=True)
        self.std_corr = np.maximum(self.std_corr, 1e-8)
        
        # Normalize
        X_train_norm = (X_train - self.mean) / self.std
        y_correction_norm = (y_correction - self.mean_corr) / self.std_corr
        
        # Clip to prevent extreme values
        X_train_norm = np.clip(X_train_norm, -5.0, 5.0)
        y_correction_norm = np.clip(y_correction_norm, -5.0, 5.0)
        
        # Initialize model
        self.model = _HOTSurfaceModel(
            context_length=context_length,
            n_tau=n_tau,
            n_logm=n_logm,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            n_blocks=self.n_blocks,
            n_head=self.n_head,
            dropout=self.dropout,
            embedding_type=self.embedding_type,
            pe_type=self.pe_type,
            attention_mode=self.attention_mode,
            use_rope=self.use_rope,
        ).to(self.device)
        
        # Initialize weights
        # TODO: May need to tune initialization strategy
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
        
        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scaler = GradScaler(enabled=self.use_amp and self.device.startswith("cuda"))
        
        # Convert to tensors
        X_t = torch.tensor(X_train_norm, dtype=torch.float32).to(self.device)
        y_corr_t = torch.tensor(y_correction_norm, dtype=torch.float32).to(self.device)
        baseline_train_t = torch.tensor(baseline_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Dataset
        dataset = TensorDataset(X_t, y_corr_t, baseline_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_x, batch_y_corr, batch_baseline, batch_y_true in dataloader:
                optimizer.zero_grad()
                
                with autocast(
                    device_type="cuda" if self.use_amp and self.device.startswith("cuda") else "cpu",
                    enabled=self.use_amp and self.device.startswith("cuda")
                ):
                    # Predict normalized correction
                    pred_correction_norm = self.model(batch_x)
                    
                    # Denormalize
                    mean_corr_t = torch.tensor(self.mean_corr, dtype=torch.float32).to(self.device)
                    std_corr_t = torch.tensor(self.std_corr, dtype=torch.float32).to(self.device)
                    pred_correction = pred_correction_norm * std_corr_t + mean_corr_t
                    
                    # Final prediction
                    pred = batch_baseline + pred_correction
                    
                    # Loss: RMSE on surfaces (not corrections)
                    loss = torch.sqrt(torch.mean((pred - batch_y_true) ** 2))
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        self.is_fitted = True

    def predict_horizon(self, X, horizon=1):
        """
        Predict future surface.
        
        Args:
            X: (n_samples, context_length, n_tau, n_logm)
            horizon: prediction horizon (not used, kept for interface)
        
        Returns:
            predictions: (n_samples, n_tau, n_logm)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Normalize
        X_norm = (X - self.mean) / self.std
        X_norm = np.clip(X_norm, -5.0, 5.0)
        
        # Compute baseline
        baseline = self._compute_baseline(X)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
            pred_correction_norm = self.model(X_t)
            
            # Denormalize
            mean_corr_t = torch.tensor(self.mean_corr, dtype=torch.float32).to(self.device)
            std_corr_t = torch.tensor(self.std_corr, dtype=torch.float32).to(self.device)
            pred_correction = (pred_correction_norm.cpu().numpy() * self.std_corr + self.mean_corr)
        
        # Final prediction
        predictions = baseline + pred_correction
        
        return predictions

    def predict(self, X):
        """Alias for predict_horizon."""
        return self.predict_horizon(X, horizon=1)
