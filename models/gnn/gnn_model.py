"""
Graph Neural Network model for IV surface forecasting.

Uses spatial GNN to encode/decode surfaces, combined with temporal transformer
for sequence modeling. Follows correction-based prediction approach.
"""

from typing import Optional, Tuple
import os
import copy
import numpy as np

from models.base_model import BaseModel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    try:
        from torch_geometric.nn import GCNConv, GATConv, MessagePassing
        from torch_geometric.data import Data, Batch
        TORCH_GEOMETRIC_AVAILABLE = True
    except ImportError:
        TORCH_GEOMETRIC_AVAILABLE = False
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    TORCH_GEOMETRIC_AVAILABLE = False


def build_surface_graph(n_tau: int, n_logm: int, tau_grid: np.ndarray, 
                       logm_grid: np.ndarray, k_neighbors: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build graph structure for IV surface.
    
    Nodes: each (tau, logm) point
    Edges: k-nearest neighbors in (tau, logm) space
    
    Parameters:
    -----------
    n_tau : int
        Number of tau (maturity) points
    n_logm : int
        Number of log-moneyness points
    tau_grid : np.ndarray
        Tau values
    logm_grid : np.ndarray
        Log-moneyness values
    k_neighbors : int
        Number of neighbors per node
        
    Returns:
    --------
    edge_index : torch.Tensor, shape (2, n_edges)
        Edge connectivity
    node_positions : torch.Tensor, shape (n_nodes, 2)
        Node positions (tau, logm) for visualization
    """
    n_nodes = n_tau * n_logm
    
    # Create node positions
    node_positions = []
    node_to_idx = {}
    idx = 0
    for i, tau in enumerate(tau_grid):
        for j, logm in enumerate(logm_grid):
            node_positions.append([tau, logm])
            node_to_idx[(i, j)] = idx
            idx += 1
    node_positions = torch.tensor(node_positions, dtype=torch.float32)
    
    # Build k-nearest neighbor graph
    try:
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_nodes), metric='euclidean')
        knn.fit(node_positions.numpy())
        distances, indices = knn.kneighbors(node_positions.numpy())
    except ImportError:
        # Fallback: simple distance-based neighbors
        from scipy.spatial.distance import cdist
        distances = cdist(node_positions.numpy(), node_positions.numpy())
        indices = np.argsort(distances, axis=1)[:, :min(k_neighbors + 1, n_nodes)]
    
    # Create edge list (exclude self-loops)
    edges = []
    for i in range(n_nodes):
        for j in indices[i][1:]:  # Skip first (self)
            edges.append([i, j])
            edges.append([j, i])  # Undirected graph
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return edge_index, node_positions


class SpatialGNNEncoder(nn.Module):
    """
    Encodes IV surface to latent representation using graph structure.
    """
    
    def __init__(self, n_nodes: int, d_latent: int = 128, n_layers: int = 3, 
                 dropout: float = 0.1, use_gat: bool = False):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_latent = d_latent
        self.n_layers = n_layers
        
        # Input projection: each node gets 1 feature (IV value)
        self.input_proj = nn.Linear(1, d_latent)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_gat and TORCH_GEOMETRIC_AVAILABLE:
                # GAT: more expressive, can learn attention over neighbors
                self.gnn_layers.append(GATConv(d_latent, d_latent, heads=4, dropout=dropout, concat=False))
            elif TORCH_GEOMETRIC_AVAILABLE:
                # GCN: simpler, faster
                self.gnn_layers.append(GCNConv(d_latent, d_latent))
            else:
                # Fallback: simple MLP (no graph structure)
                self.gnn_layers.append(nn.Linear(d_latent, d_latent))
        
        self.dropout = nn.Dropout(dropout)
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_layers)])
        
        # Global pooling: aggregate node features to single latent
        self.pool = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent)
        )
        
    def forward(self, surface: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode surface to latent.
        
        Parameters:
        -----------
        surface : torch.Tensor, shape (batch, n_tau, n_logm)
            IV surface
        edge_index : torch.Tensor, shape (2, n_edges)
            Graph edge connectivity
            
        Returns:
        --------
        latent : torch.Tensor, shape (batch, d_latent)
            Encoded latent representation
        """
        batch_size = surface.shape[0]
        n_nodes = surface.shape[1] * surface.shape[2]
        
        # Flatten surface to nodes: (batch, n_nodes, 1)
        x = surface.reshape(batch_size, n_nodes, 1)
        
        # Project to latent dimension
        x = self.input_proj(x)  # (batch, n_nodes, d_latent)
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            # Reshape for GNN: (batch * n_nodes, d_latent)
            x_flat = x.reshape(-1, self.d_latent)
            
            if TORCH_GEOMETRIC_AVAILABLE and isinstance(gnn_layer, (GCNConv, GATConv)):
                # Create batch for all samples
                batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(n_nodes)
                
                # Expand edge_index for batch
                edge_index_batch = edge_index.clone()
                for b in range(1, batch_size):
                    edge_offset = b * n_nodes
                    edge_index_batch = torch.cat([
                        edge_index_batch,
                        edge_index + edge_offset
                    ], dim=1)
                
                # Apply GNN layer
                x_flat = gnn_layer(x_flat, edge_index_batch)
            else:
                # Fallback: simple MLP
                x_flat = gnn_layer(x_flat)
            
            x_flat = self.norm_layers[i](x_flat)
            x_flat = F.gelu(x_flat)
            x_flat = self.dropout(x_flat)
            
            # Reshape back: (batch, n_nodes, d_latent)
            x = x_flat.reshape(batch_size, n_nodes, self.d_latent)
        
        # Global pooling: mean pool + MLP
        x_pooled = x.mean(dim=1)  # (batch, d_latent)
        latent = self.pool(x_pooled)  # (batch, d_latent)
        
        return latent


class SpatialGNNDecoder(nn.Module):
    """
    Decodes latent representation back to IV surface.
    """
    
    def __init__(self, n_nodes: int, d_latent: int = 128, n_layers: int = 3,
                 dropout: float = 0.1, use_gat: bool = False):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_latent = d_latent
        self.n_layers = n_layers
        
        # Broadcast latent to all nodes
        self.latent_proj = nn.Linear(d_latent, d_latent)
        
        # GNN layers (reverse of encoder)
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_gat and TORCH_GEOMETRIC_AVAILABLE:
                self.gnn_layers.append(GATConv(d_latent, d_latent, heads=4, dropout=dropout, concat=False))
            elif TORCH_GEOMETRIC_AVAILABLE:
                self.gnn_layers.append(GCNConv(d_latent, d_latent))
            else:
                self.gnn_layers.append(nn.Linear(d_latent, d_latent))
        
        self.dropout = nn.Dropout(dropout)
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_layers)])
        
        # Output projection: node features -> IV value
        self.output_proj = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent, 1)
        )
        
    def forward(self, latent: torch.Tensor, edge_index: torch.Tensor, 
                n_tau: int, n_logm: int) -> torch.Tensor:
        """
        Decode latent to surface.
        
        Parameters:
        -----------
        latent : torch.Tensor, shape (batch, d_latent)
            Latent representation
        edge_index : torch.Tensor, shape (2, n_edges)
            Graph edge connectivity
        n_tau : int
            Number of tau points
        n_logm : int
            Number of log-moneyness points
            
        Returns:
        --------
        surface : torch.Tensor, shape (batch, n_tau, n_logm)
            Decoded IV surface
        """
        batch_size = latent.shape[0]
        n_nodes = n_tau * n_logm
        
        # Broadcast latent to all nodes
        x = self.latent_proj(latent)  # (batch, d_latent)
        x = x.unsqueeze(1).expand(-1, n_nodes, -1)  # (batch, n_nodes, d_latent)
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_flat = x.reshape(-1, self.d_latent)
            
            if TORCH_GEOMETRIC_AVAILABLE and isinstance(gnn_layer, (GCNConv, GATConv)):
                batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(n_nodes)
                edge_index_batch = edge_index.clone()
                for b in range(1, batch_size):
                    edge_offset = b * n_nodes
                    edge_index_batch = torch.cat([
                        edge_index_batch,
                        edge_index + edge_offset
                    ], dim=1)
                
                x_flat = gnn_layer(x_flat, edge_index_batch)
            else:
                x_flat = gnn_layer(x_flat)
            
            x_flat = self.norm_layers[i](x_flat)
            x_flat = F.gelu(x_flat)
            x_flat = self.dropout(x_flat)
            
            x = x_flat.reshape(batch_size, n_nodes, self.d_latent)
        
        # Output projection
        surface_flat = self.output_proj(x)  # (batch, n_nodes, 1)
        surface = surface_flat.reshape(batch_size, n_tau, n_logm)  # (batch, n_tau, n_logm)
        
        return surface


class TemporalTransformer(nn.Module):
    """
    Simple temporal transformer for sequence of latents.
    """
    
    def __init__(self, d_latent: int, context_length: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.context_length = context_length
        self.d_latent = d_latent
        self.d_model = d_model
        
        # Project latent to transformer dimension
        self.input_proj = nn.Linear(d_latent, d_model)
        self.positional = nn.Parameter(torch.zeros(context_length, d_model))
        self.pre_encoder_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_latent)
        
    def forward(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict future latent from sequence of latents.
        
        Parameters:
        -----------
        latent_sequence : torch.Tensor, shape (batch, context_length, d_latent)
            Sequence of encoded surfaces
            
        Returns:
        --------
        future_latent : torch.Tensor, shape (batch, d_latent)
            Predicted future latent
        """
        x = self.input_proj(latent_sequence)  # (batch, context_length, d_model)
        x = x + self.positional.unsqueeze(0)
        x = self.pre_encoder_norm(x)
        x = self.encoder(x)  # (batch, context_length, d_model)
        x = x[:, -1, :]  # Take last timestep
        future_latent = self.output_proj(x)  # (batch, d_latent)
        return future_latent


class GNNTransformerModel(BaseModel):
    """
    GNN-based model for IV surface forecasting.
    
    Combines spatial GNN (for surface structure) with temporal transformer
    (for sequence modeling). Uses correction-based prediction.
    """
    
    def __init__(self,
                 name: str = "gnn_transformer",
                 d_latent: int = 128,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers_gnn: int = 3,
                 n_layers_temporal: int = 4,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 use_amp: bool = False,
                 baseline_decay: float = 1.0,
                 use_gat: bool = False,
                 k_neighbors: int = 8,
                 device: Optional[str] = None):
        super().__init__(name=name)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("Warning: torch_geometric not available. GNN will use MLP fallback.")
            print("Install with: pip install torch-geometric")
        
        self.requires_normalization = False
        self.d_latent = d_latent
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers_gnn = n_layers_gnn
        self.n_layers_temporal = n_layers_temporal
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.use_amp = use_amp
        self.baseline_decay = baseline_decay
        self.use_gat = use_gat
        self.k_neighbors = k_neighbors
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.n_tau = None
        self.n_logm = None
        self.context_length = None
        self.tau_grid = None
        self.logm_grid = None
        self.edge_index = None
        self.mean = None
        self.std = None
        self.mean_corr = None
        self.std_corr = None
    
    def _compute_baseline(self, X: np.ndarray) -> np.ndarray:
        """Compute baseline from context surfaces (same as transformer)."""
        n_samples, context_length, n_tau, n_logm = X.shape
        
        if self.baseline_decay is None or self.baseline_decay == -1:
            return X[:, -1, :, :].copy()
        
        if context_length == 1:
            return X[:, 0, :, :]
        
        indices = np.arange(context_length)
        normalized_indices = indices / max(1, context_length - 1)
        weights = np.exp((self.baseline_decay - 1.0) * normalized_indices)
        weights = weights / weights.sum()
        
        baseline = np.average(X, axis=1, weights=weights)
        return baseline
    
    def fit(self, X_train, y_train=None, context_length: Optional[int] = None,
            X_val=None, y_val=None, tau_grid: Optional[np.ndarray] = None,
            logm_grid: Optional[np.ndarray] = None, **kwargs):
        """Train the GNN model."""
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train are required")
        
        if X_train.ndim != 4:
            raise ValueError(f"Expected X_train shape (n_samples, context, n_tau, n_logm), got {X_train.shape}")
        
        n_samples, context_length_inferred, n_tau, n_logm = X_train.shape
        if context_length is None:
            context_length = context_length_inferred
        elif context_length != context_length_inferred:
            raise ValueError(f"context_length mismatch: provided {context_length}, but X_train has {context_length_inferred}")
        
        self.context_length = context_length
        self.n_tau = n_tau
        self.n_logm = n_logm
        
        # Get tau/logm grids (create default if not provided)
        if tau_grid is None:
            tau_grid = np.linspace(0.1, 2.0, n_tau)
        if logm_grid is None:
            logm_grid = np.linspace(-0.5, 0.5, n_logm)
        
        self.tau_grid = tau_grid
        self.logm_grid = logm_grid
        
        # Build graph structure
        self.edge_index, _ = build_surface_graph(n_tau, n_logm, tau_grid, logm_grid, self.k_neighbors)
        self.edge_index = self.edge_index.to(self.device)
        
        # Compute baseline and corrections
        baseline_train = self._compute_baseline(X_train)
        y_correction = y_train - baseline_train
        
        # Normalization
        if True:  # Always normalize
            # X normalization
            X_flat = X_train.reshape(-1, n_tau * n_logm)
            self.mean = X_flat.mean(axis=0, keepdims=True).reshape(1, 1, n_tau, n_logm)
            self.std = X_flat.std(axis=0, keepdims=True).reshape(1, 1, n_tau, n_logm)
            self.std = np.maximum(self.std, 1e-8)
            
            # Correction normalization
            y_corr_flat = y_correction.reshape(-1, n_tau * n_logm)
            self.mean_corr = y_corr_flat.mean(axis=0, keepdims=True).reshape(1, n_tau, n_logm)
            self.std_corr = y_corr_flat.std(axis=0, keepdims=True).reshape(1, n_tau, n_logm)
            self.std_corr = np.maximum(self.std_corr, 1e-8)
        
        # Create model
        n_nodes = n_tau * n_logm
        encoder = SpatialGNNEncoder(n_nodes, self.d_latent, self.n_layers_gnn, self.dropout, self.use_gat)
        decoder = SpatialGNNDecoder(n_nodes, self.d_latent, self.n_layers_gnn, self.dropout, self.use_gat)
        temporal = TemporalTransformer(self.d_latent, context_length, self.d_model, 
                                       self.n_heads, self.n_layers_temporal, self.dropout)
        
        self.model = nn.ModuleDict({
            'encoder': encoder,
            'decoder': decoder,
            'temporal': temporal
        }).to(self.device)
        
        # Initialize weights
        self._initialize_weights()
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scaler = GradScaler(enabled=self.use_amp and self.device.startswith("cuda"))
        
        # Convert to tensors
        baseline_train_t = torch.tensor(baseline_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        mean_corr_t = torch.tensor(self.mean_corr, dtype=torch.float32).to(self.device)
        std_corr_t = torch.tensor(self.std_corr, dtype=torch.float32).to(self.device)
        
        # Training loop
        best_state = None
        best_val = float("inf")
        epochs_no_improve = 0
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            indices = torch.randperm(n_samples)
            total_loss = 0.0
            
            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                batch_X = torch.tensor((X_train[batch_idx] - self.mean) / self.std, dtype=torch.float32).to(self.device)
                batch_baseline = baseline_train_t[batch_idx]
                batch_y_true = y_train_t[batch_idx]
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type="cuda" if self.use_amp and self.device.startswith("cuda") else "cpu", 
                             enabled=self.use_amp and self.device.startswith("cuda")):
                    # Encode each surface in context
                    context_latents = []
                    for t in range(context_length):
                        latent_t = self.model['encoder'](batch_X[:, t, :, :], self.edge_index)
                        context_latents.append(latent_t)
                    context_latents = torch.stack(context_latents, dim=1)  # (batch, context_length, d_latent)
                    
                    # Temporal prediction
                    future_latent = self.model['temporal'](context_latents)  # (batch, d_latent)
                    
                    # Decode to correction
                    pred_correction_norm = self.model['decoder'](future_latent, self.edge_index, n_tau, n_logm)
                    
                    # Denormalize correction
                    pred_correction = pred_correction_norm * std_corr_t + mean_corr_t
                    
                    # Final prediction
                    pred_surface = batch_baseline + pred_correction
                    
                    # Loss
                    loss = torch.sqrt(torch.mean((pred_surface - batch_y_true) ** 2))
                
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered")
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val, y_val, mean_corr_t, std_corr_t)
                
                if val_loss + self.min_delta < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if self.patience > 0 and epochs_no_improve >= self.patience:
                        break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True
        return self
    
    def _evaluate(self, X_val, y_val, mean_corr_t, std_corr_t):
        """Evaluate on validation set."""
        self.model.eval()
        baseline_val = self._compute_baseline(X_val)
        baseline_val_t = torch.tensor(baseline_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(X_val), self.batch_size):
                batch_X = torch.tensor((X_val[i:i+self.batch_size] - self.mean) / self.std, 
                                      dtype=torch.float32).to(self.device)
                batch_baseline = baseline_val_t[i:i+self.batch_size]
                batch_y_true = y_val_t[i:i+self.batch_size]
                
                with autocast(device_type="cuda" if self.use_amp and self.device.startswith("cuda") else "cpu",
                             enabled=self.use_amp and self.device.startswith("cuda")):
                    context_latents = []
                    for t in range(self.context_length):
                        latent_t = self.model['encoder'](batch_X[:, t, :, :], self.edge_index)
                        context_latents.append(latent_t)
                    context_latents = torch.stack(context_latents, dim=1)
                    
                    future_latent = self.model['temporal'](context_latents)
                    pred_correction_norm = self.model['decoder'](future_latent, self.edge_index, 
                                                                self.n_tau, self.n_logm)
                    pred_correction = pred_correction_norm * std_corr_t + mean_corr_t
                    pred_surface = batch_baseline + pred_correction
                    
                    loss = torch.sqrt(torch.mean((pred_surface - batch_y_true) ** 2))
                    val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
    
    def predict_horizon(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Predict future surface."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if X.ndim != 4:
            raise ValueError(f"Expected X shape (n_samples, context, n_tau, n_logm), got {X.shape}")
        
        n_samples, context_length, n_tau, n_logm = X.shape
        if context_length != self.context_length:
            raise ValueError(f"context_length mismatch: expected {self.context_length}, got {context_length}")
        
        baseline = self._compute_baseline(X)
        baseline_t = torch.tensor(baseline, dtype=torch.float32).to(self.device)
        
        X_norm = (X - self.mean) / self.std
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        
        mean_corr_t = torch.tensor(self.mean_corr, dtype=torch.float32).to(self.device)
        std_corr_t = torch.tensor(self.std_corr, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_t[i:i+self.batch_size]
                batch_baseline = baseline_t[i:i+self.batch_size]
                
                with autocast(device_type="cuda" if self.use_amp and self.device.startswith("cuda") else "cpu",
                             enabled=self.use_amp and self.device.startswith("cuda")):
                    context_latents = []
                    for t in range(context_length):
                        latent_t = self.model['encoder'](batch_X[:, t, :, :], self.edge_index)
                        context_latents.append(latent_t)
                    context_latents = torch.stack(context_latents, dim=1)
                    
                    future_latent = self.model['temporal'](context_latents)
                    pred_correction_norm = self.model['decoder'](future_latent, self.edge_index, 
                                                                self.n_tau, self.n_logm)
                    pred_correction = pred_correction_norm * std_corr_t + mean_corr_t
                    pred_surface = batch_baseline + pred_correction
                    
                    predictions.append(pred_surface.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict (default horizon=1)."""
        return self.predict_horizon(X, horizon=1)
