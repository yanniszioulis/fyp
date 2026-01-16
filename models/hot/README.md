# Higher-Order Transformer (HOT) for IV Surface Forecasting

This module adapts the Higher-Order Transformer architecture with Kronecker-structured attention for forecasting implied volatility (IV) surfaces.

## Overview

HOT uses **Kronecker attention** to factorize attention across multiple dimensions, making it efficient for 3D data. For IV surfaces, we have three dimensions:
- **Time**: Context length (5-63 days)
- **Tau**: Time to maturity (20 values)
- **Logm**: Log-moneyness (21 values)

Instead of flattening to 1D and using standard attention (O((T×τ×m)²) complexity), HOT factorizes attention across dimensions (O(T² + τ² + m²) complexity).

## Architecture

```
Input: (batch, context_length, n_tau, n_logm)
  ↓
Surface Embedding: Linear/Conv2D/Learnable per surface point
  → (batch, context_length, n_tau, n_logm, d_model)
  ↓
Positional Embeddings: Add to time, tau, logm dimensions
  ↓
HOT Transformer Blocks: Kronecker attention across (time, tau, logm)
  ↓
Temporal Pooling: Mean/Last over time dimension
  → (batch, n_tau, n_logm, d_model)
  ↓
Output Head: Linear projection
  → (batch, n_tau, n_logm)  [correction surface]
```

## Key Components

### 1. Surface Embedding (`SurfaceEmbedding`)
Embeds each 2D surface into `d_model` dimensions. Options:
- **`linear`**: Simple linear projection per surface point (preserves structure)
- **`conv2d`**: 2D convolution with patches (reduces resolution)
- **`learnable`**: Learnable embeddings per (tau, logm) location

**TODO**: Experiment to determine optimal embedding strategy.

### 2. Positional Embeddings (`PositionalEmbedding3D`)
Adds positional information for all three dimensions:
- **Time**: Learnable or RoPE (handled in attention)
- **Tau**: Learnable or sinusoidal (maturity is ordered)
- **Logm**: Learnable or sinusoidal (moneyness is ordered)

**TODO**: Experiment with different positional encoding strategies.

### 3. Kronecker Attention (`KroneckerAttention`)
Factorizes attention across dimensions:
- Computes attention separately along time, tau, and logm
- Combines via Kronecker product or sum
- Supports RoPE for time dimension

**Modes:**
- **`kronecker_product`**: Sequential attention along each dimension
- **`kronecker_sum`**: Sum attention across dimensions

**TODO**: Verify if attention order matters (time→tau→logm vs other orders).

### 4. HOT Transformer Blocks (`HOTTransformerBlock`)
Standard transformer block with:
- Kronecker attention
- Layer normalization
- SwiGLU feedforward network
- Residual connections

### 5. Output Head
Simple linear projection to predict correction surface.

**TODO**: Consider if more complex output head (e.g., MLP) would help.

## Usage

```python
from models.hot import HOTSurfaceModel

model = HOTSurfaceModel(
    name="hot_surface",
    d_model=256,
    d_mlp=1024,
    n_blocks=4,
    n_head=8,
    dropout=0.1,
    embedding_type='linear',  # or 'conv2d', 'learnable'
    pe_type='learnable',      # or 'sinusoidal', 'none'
    attention_mode='kronecker_product',  # or 'kronecker_sum'
    use_rope=True,            # Use RoPE for time dimension
    baseline_decay=-1,        # -1 for persistence, 0.0-1.0 for exponential
)

model.fit(X_train, y_train, context_length=21, horizon=5)
predictions = model.predict_horizon(X_test, horizon=5)
```

## Design Decisions (TODO)

The following design decisions need experimentation:

1. **Embedding Strategy**: Linear vs Conv2D vs Learnable
2. **Positional Embeddings**: Learnable vs Sinusoidal vs None (rely on RoPE)
3. **Attention Order**: Does the order of attention (time→tau→logm) matter?
4. **Attention Mode**: Product vs Sum for Kronecker combination
5. **Temporal Pooling**: Mean vs Last vs Learnable
6. **Output Head**: Simple linear vs MLP
7. **RoPE Usage**: Should we use RoPE for time dimension or learnable embeddings?
8. **Normalization**: Current per-location normalization - is this optimal?

## Integration

The model integrates with the existing pipeline:
- Inherits from `BaseModel`
- Uses correction-based prediction (baseline + correction)
- Supports exponential-weighted and persistence baselines
- Compatible with `run_model.py` and `tune_model.py`

## References

- Original HOT paper: [Higher-Order Transformers with Kronecker-Structured Attention](https://openreview.net/forum?id=QN0aXcKFkT)
- Implementation adapted from: `HOT/` directory
