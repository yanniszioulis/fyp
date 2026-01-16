# GNN Transformer Model

Graph Neural Network-based model for IV surface forecasting.

## Architecture

Combines:
1. **Spatial GNN Encoder**: Encodes each surface to a latent representation using graph structure
2. **Temporal Transformer**: Models sequences of latents over time
3. **Spatial GNN Decoder**: Decodes predicted latent back to surface

## Graph Structure

- **Nodes**: Each (tau, log-moneyness) point on the surface
- **Edges**: k-nearest neighbors in (tau, logm) space (default: k=8)
- **Graph type**: Undirected, k-NN

## Training

Uses correction-based prediction (same as transformer):
- Computes baseline from context surfaces
- Predicts correction from baseline to target
- Loss: RMSE on final surfaces

## Dependencies

- `torch` (required)
- `torch-geometric` (optional, but recommended for true GNN)
- `scikit-learn` (for k-NN graph construction)
- `scipy` (fallback for graph construction)

If `torch-geometric` is not available, the model falls back to MLP layers (no graph structure).

## Usage

```python
from models.gnn.gnn_model import GNNTransformerModel

model = GNNTransformerModel(
    name="gnn_test",
    d_latent=128,
    d_model=256,
    n_heads=8,
    n_layers_gnn=3,
    n_layers_temporal=4,
    dropout=0.1,
    baseline_decay=0.5,  # Moderate recency bias
    use_gat=False,  # Use GCN (faster) or GAT (more expressive)
    k_neighbors=8  # Number of neighbors per node
)

model.fit(X_train, y_train, context_length=21, horizon=5,
          tau_grid=tau_grid, logm_grid=logm_grid,
          X_val=X_val, y_val=y_val)

predictions = model.predict_horizon(X_test, horizon=5)
```

## Key Parameters

- `d_latent`: Latent dimension for encoded surfaces (default: 128)
- `d_model`: Transformer model dimension (default: 256)
- `n_layers_gnn`: Number of GNN layers in encoder/decoder (default: 3)
- `n_layers_temporal`: Number of transformer layers (default: 4)
- `use_gat`: Use Graph Attention Network instead of GCN (default: False)
- `k_neighbors`: Number of neighbors per node in graph (default: 8)
- `baseline_decay`: Baseline computation (-1=persistence, 1.0=mean, etc.)
