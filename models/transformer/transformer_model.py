"""
Transformer model for forecasting IV surfaces via correction prediction.
Predicts correction from exponential-weighted baseline to target surface.
"""

from typing import Optional
import os
import copy
import numpy as np

from models.base_model import BaseModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class _SurfaceTransformer(nn.Module):
    def __init__(self,
                 n_features: int,
                 context_length: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float,
                 pool: str):
        super().__init__()
        self.context_length = context_length
        self.pool = pool

        self.input_proj = nn.Linear(n_features, d_model)
        self.positional = nn.Parameter(torch.zeros(context_length, d_model))
        # Add layer norm before encoder to stabilize inputs
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

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features)
        )

    def forward(self, x):
        # x: (batch, context_length, n_features)
        x = self.input_proj(x)
        # Check for NaN after input projection
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            raise ValueError(f"NaN/Inf after input_proj: input range=[{x.min().item():.6f}, {x.max().item():.6f}]")
        
        x = x + self.positional.unsqueeze(0)
        # Check for NaN after positional encoding
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            raise ValueError(f"NaN/Inf after positional encoding")
        
        # Normalize before encoder to stabilize
        x = self.pre_encoder_norm(x)
        # Check for NaN after pre-encoder norm
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print(f"Error: NaN/Inf after pre_encoder_norm")
            print(f"  Input stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            raise ValueError(f"NaN/Inf after pre_encoder_norm")
        
        x = self.encoder(x)
        # Check for NaN after encoder
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print(f"Error: NaN/Inf after encoder")
            print(f"  Input to encoder stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            # Try to get more info about encoder layers
            raise ValueError(f"NaN/Inf after encoder - check attention mechanism or feedforward network")
        
        if self.pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, -1, :]
        
        x = self.head(x)
        # Check for NaN after head
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            raise ValueError(f"NaN/Inf after head")
        
        return x


class TransformerSurfaceModel(BaseModel):
    """
    Temporal transformer that predicts correction from exponential-weighted
    baseline to target surface. Each surface is treated as one token.
    """

    def __init__(self,
                 name: str = "transformer_surface",
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 pool: str = "last",
                 normalize: bool = True,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 use_amp: bool = False,
                 baseline_decay: float = 1.0,
                 device: Optional[str] = None):
        super().__init__(name=name)
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TransformerSurfaceModel. Install with: pip install torch"
            )

        self.requires_normalization = False
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pool = pool
        self.normalize = normalize
        self.patience = patience
        self.min_delta = min_delta
        self.use_amp = use_amp
        self.baseline_decay = baseline_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.n_tau = None
        self.n_logm = None
        self.n_features = None
        self.context_length = None
        self.mean = None  # For X normalization
        self.std = None   # For X normalization
        self.mean_corr = None  # For correction normalization
        self.std_corr = None   # For correction normalization

    def _initialize_weights(self):
        """Initialize model weights to prevent NaN outputs."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Use smaller initialization to prevent overflow
                torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, torch.nn.LayerNorm):
                # LayerNorm should be initialized to 1 and 0
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.constant_(module.weight, 1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
        
        # Initialize positional encoding with small values
        if hasattr(self.model, 'positional'):
            torch.nn.init.normal_(self.model.positional, mean=0.0, std=0.01)

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        # X: (n_samples, context, n_tau, n_logm)
        n_samples, context_length, n_tau, n_logm = X.shape
        return X.reshape(n_samples, context_length, n_tau * n_logm)

    def _compute_baseline(self, X: np.ndarray) -> np.ndarray:
        """
        Compute exponential-weighted baseline from context surfaces.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, context_length, n_tau, n_logm)
            Context surfaces
            
        Returns:
        --------
        baseline : np.ndarray, shape (n_samples, n_tau, n_logm)
            Weighted average baseline surface
        """
        n_samples, context_length, n_tau, n_logm = X.shape
        
        # Use persistence (last surface) if baseline_decay is None or -1
        if self.baseline_decay is None or self.baseline_decay == -1:
            return X[:, -1, :, :].copy()  # Persistence: just use last surface
        
        if context_length == 1:
            return X[:, 0, :, :]
        
        # Exponential-weighted average
        indices = np.arange(context_length)
        normalized_indices = indices / max(1, context_length - 1)
        weights = np.exp((self.baseline_decay - 1.0) * normalized_indices)
        weights = weights / weights.sum()
        
        baseline = np.average(X, axis=1, weights=weights)
        return baseline

    def _normalize_array(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Model must be fitted before normalization")
        return (X - self.mean) / (self.std + 1e-8)

    def _denormalize_array(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Model must be fitted before denormalization")
        return X * (self.std + 1e-8) + self.mean

    def fit(self, X_train, y_train=None, context_length: Optional[int] = None,
            X_val=None, y_val=None, **kwargs):
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train are required for TransformerSurfaceModel")

        if X_train.ndim != 4:
            raise ValueError(f"Expected X_train shape (n_samples, context, n_tau, n_logm), got {X_train.shape}")

        n_samples, context_length_inferred, n_tau, n_logm = X_train.shape
        if context_length is None:
            context_length = context_length_inferred
        elif context_length != context_length_inferred:
            raise ValueError(
                f"context_length mismatch: provided {context_length}, "
                f"but X_train has {context_length_inferred}"
            )

        self.context_length = context_length
        self.n_tau = n_tau
        self.n_logm = n_logm
        self.n_features = n_tau * n_logm

        baseline_train = self._compute_baseline(X_train)
        y_correction = y_train - baseline_train

        # Check for NaN/Inf in inputs
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            raise ValueError(f"X_train contains NaN or Inf values")
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            raise ValueError(f"y_train contains NaN or Inf values")
        if np.any(np.isnan(baseline_train)) or np.any(np.isinf(baseline_train)):
            raise ValueError(f"baseline_train contains NaN or Inf values")
        if np.any(np.isnan(y_correction)) or np.any(np.isinf(y_correction)):
            raise ValueError(f"y_correction contains NaN or Inf values")

        X_flat = self._flatten(X_train).astype(np.float32, copy=False)
        y_correction_flat = y_correction.reshape(n_samples, self.n_features).astype(np.float32, copy=False)

        if self.normalize:
            # X uses its own stats, corrections use their own stats (like test_overfit_transformer.py)
            flat_for_stats = X_flat.reshape(-1, self.n_features)
            self.mean = flat_for_stats.mean(axis=0, keepdims=True)
            self.std = flat_for_stats.std(axis=0, keepdims=True)
            self.std = np.maximum(self.std, 1e-8)
            X_flat = (X_flat - self.mean) / self.std
            # Clip extreme values to prevent overflow in transformer
            X_flat = np.clip(X_flat, -10.0, 10.0)
            
            # Corrections normalized with their own statistics
            self.mean_corr = y_correction_flat.mean(axis=0, keepdims=True)
            self.std_corr = y_correction_flat.std(axis=0, keepdims=True)
            self.std_corr = np.maximum(self.std_corr, 1e-8)
            
            # Check for NaN/Inf in normalization stats
            if np.any(np.isnan(self.mean_corr)) or np.any(np.isnan(self.std_corr)):
                raise ValueError(f"Normalization stats contain NaN: mean_corr has NaN={np.any(np.isnan(self.mean_corr))}, std_corr has NaN={np.any(np.isnan(self.std_corr))}")
            if np.any(np.isinf(self.mean_corr)) or np.any(np.isinf(self.std_corr)):
                raise ValueError(f"Normalization stats contain Inf: mean_corr has Inf={np.any(np.isinf(self.mean_corr))}, std_corr has Inf={np.any(np.isinf(self.std_corr))}")
            
            y_correction_flat = (y_correction_flat - self.mean_corr) / self.std_corr
            # Clip extreme values more aggressively to prevent overflow
            y_correction_flat = np.clip(y_correction_flat, -5.0, 5.0)
            
            # Check for NaN/Inf after normalization
            if np.any(np.isnan(y_correction_flat)) or np.any(np.isinf(y_correction_flat)):
                raise ValueError(f"y_correction_flat contains NaN or Inf after normalization")

        # Include baseline and true targets in dataset for loss computation
        baseline_train_flat = baseline_train.reshape(n_samples, self.n_features).astype(np.float32)
        y_train_flat = y_train.reshape(n_samples, self.n_features).astype(np.float32)
        
        # Final check before creating dataset
        if np.any(np.isnan(baseline_train_flat)) or np.any(np.isnan(y_train_flat)):
            raise ValueError(f"baseline_train_flat or y_train_flat contains NaN before dataset creation")
        dataset = TensorDataset(
            torch.tensor(X_flat, dtype=torch.float32),
            torch.tensor(y_correction_flat, dtype=torch.float32),
            torch.tensor(baseline_train_flat, dtype=torch.float32),
            torch.tensor(y_train_flat, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            if X_val.ndim != 4:
                raise ValueError(f"Expected X_val shape (n_samples, context, n_tau, n_logm), got {X_val.shape}")
            
            baseline_val = self._compute_baseline(X_val)
            y_correction_val = y_val - baseline_val
            
            X_val_flat = self._flatten(X_val).astype(np.float32, copy=False)
            y_correction_val_flat = y_correction_val.reshape(X_val_flat.shape[0], self.n_features).astype(np.float32, copy=False)
            if self.normalize:
                # Use training stats for normalization
                X_val_flat = (X_val_flat - self.mean) / self.std
                X_val_flat = np.clip(X_val_flat, -5.0, 5.0)
                y_correction_val_flat = (y_correction_val_flat - self.mean_corr) / self.std_corr
                y_correction_val_flat = np.clip(y_correction_val_flat, -5.0, 5.0)
            baseline_val_flat = baseline_val.reshape(X_val.shape[0], self.n_features).astype(np.float32)
            y_val_flat = y_val.reshape(X_val.shape[0], self.n_features).astype(np.float32)
            val_dataset = TensorDataset(
                torch.tensor(X_val_flat, dtype=torch.float32),
                torch.tensor(y_correction_val_flat, dtype=torch.float32),
                torch.tensor(baseline_val_flat, dtype=torch.float32),
                torch.tensor(y_val_flat, dtype=torch.float32)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = _SurfaceTransformer(
            n_features=self.n_features,
            context_length=self.context_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            pool=self.pool
        ).to(self.device)
        
        # Initialize weights properly to avoid NaN
        self._initialize_weights()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        # Loss is RMSE on surfaces (not MSE on corrections), matching test_overfit_transformer.py
        scaler = GradScaler(enabled=self.use_amp and self.device.startswith("cuda"))
        
        # Convert normalization stats to tensors for loss computation
        mean_corr_t = torch.tensor(self.mean_corr, dtype=torch.float32).to(self.device) if self.normalize else None
        std_corr_t = torch.tensor(self.std_corr, dtype=torch.float32).to(self.device) if self.normalize else None

        best_state = None
        best_val = float("inf")
        epochs_no_improve = 0

        self.model.train()
        for _ in range(self.num_epochs):
            for batch_x, batch_y_correction, batch_baseline, batch_y_true in loader:
                batch_x = batch_x.to(self.device)
                batch_y_correction = batch_y_correction.to(self.device)
                batch_baseline = batch_baseline.to(self.device)
                batch_y_true = batch_y_true.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda" if self.use_amp and self.device.startswith("cuda") else "cpu", enabled=self.use_amp and self.device.startswith("cuda")):
                    pred_correction_norm = self.model(batch_x)  # Predicts normalized corrections
                    
                    # Check model output for NaN/Inf
                    if torch.any(torch.isnan(pred_correction_norm)) or torch.any(torch.isinf(pred_correction_norm)):
                        print(f"Error: Model output contains NaN/Inf")
                        print(f"  pred_correction_norm stats: min={pred_correction_norm.min().item():.6f}, max={pred_correction_norm.max().item():.6f}, mean={pred_correction_norm.mean().item():.6f}")
                        print(f"  batch_x stats: min={batch_x.min().item():.6f}, max={batch_x.max().item():.6f}, has_nan={torch.isnan(batch_x).any().item()}, has_inf={torch.isinf(batch_x).any().item()}")
                        raise ValueError("Model output contains NaN or Inf - check model initialization or input normalization")
                    
                    # Denormalize corrections and add to baseline to get surface predictions
                    if self.normalize:
                        # Check normalization tensors
                        if torch.any(torch.isnan(std_corr_t)) or torch.any(torch.isnan(mean_corr_t)):
                            print(f"Error: Normalization tensors contain NaN")
                            print(f"  std_corr_t: min={std_corr_t.min().item():.6f}, max={std_corr_t.max().item():.6f}, has_nan={torch.isnan(std_corr_t).any().item()}")
                            print(f"  mean_corr_t: min={mean_corr_t.min().item():.6f}, max={mean_corr_t.max().item():.6f}, has_nan={torch.isnan(mean_corr_t).any().item()}")
                            raise ValueError("Normalization tensors contain NaN")
                        pred_correction = pred_correction_norm * std_corr_t + mean_corr_t
                        # Check denormalized result
                        if torch.any(torch.isnan(pred_correction)) or torch.any(torch.isinf(pred_correction)):
                            print(f"Error: Denormalized corrections contain NaN/Inf")
                            print(f"  pred_correction_norm: min={pred_correction_norm.min().item():.6f}, max={pred_correction_norm.max().item():.6f}")
                            print(f"  std_corr_t: min={std_corr_t.min().item():.6f}, max={std_corr_t.max().item():.6f}")
                            print(f"  mean_corr_t: min={mean_corr_t.min().item():.6f}, max={mean_corr_t.max().item():.6f}")
                            raise ValueError("Denormalized corrections contain NaN or Inf")
                    else:
                        pred_correction = pred_correction_norm
                    
                    # Check baseline and targets before computing loss
                    if torch.any(torch.isnan(batch_baseline)) or torch.any(torch.isnan(batch_y_true)):
                        print(f"Error: batch_baseline or batch_y_true contains NaN")
                        print(f"  batch_baseline: min={batch_baseline.min().item():.6f}, max={batch_baseline.max().item():.6f}, has_nan={torch.isnan(batch_baseline).any().item()}")
                        print(f"  batch_y_true: min={batch_y_true.min().item():.6f}, max={batch_y_true.max().item():.6f}, has_nan={torch.isnan(batch_y_true).any().item()}")
                        raise ValueError("batch_baseline or batch_y_true contains NaN")
                    
                    # Compute RMSE on surfaces (not corrections)
                    pred_surface = batch_baseline + pred_correction
                    squared_errors = (pred_surface - batch_y_true) ** 2
                    loss = torch.sqrt(torch.mean(squared_errors))
                    
                if not torch.isfinite(loss):
                    print(f"Error: Non-finite loss value: {loss.item()}")
                    print(f"  Final check - pred_surface: min={pred_surface.min().item():.6f}, max={pred_surface.max().item():.6f}, has_nan={torch.isnan(pred_surface).any().item()}")
                    print(f"  Final check - batch_y_true: min={batch_y_true.min().item():.6f}, max={batch_y_true.max().item():.6f}, has_nan={torch.isnan(batch_y_true).any().item()}")
                    raise ValueError("Non-finite loss encountered during training")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            if val_loader is None:
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y_correction, batch_baseline, batch_y_true in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y_correction = batch_y_correction.to(self.device)
                    batch_baseline = batch_baseline.to(self.device)
                    batch_y_true = batch_y_true.to(self.device)
                    with autocast(device_type="cuda" if self.use_amp and self.device.startswith("cuda") else "cpu", enabled=self.use_amp and self.device.startswith("cuda")):
                        pred_correction_norm = self.model(batch_x)  # Predicts normalized corrections
                        
                        # Denormalize corrections and add to baseline
                        if self.normalize:
                            pred_correction = pred_correction_norm * std_corr_t + mean_corr_t
                        else:
                            pred_correction = pred_correction_norm
                        
                        # Compute RMSE on surfaces
                        pred_surface = batch_baseline + pred_correction
                        val_loss = torch.sqrt(torch.mean((pred_surface - batch_y_true) ** 2)).item()
                    val_losses.append(val_loss)
            self.model.train()

            if len(val_losses) > 0:
                avg_val = float(np.mean(val_losses))
                if avg_val + self.min_delta < best_val:
                    best_val = avg_val
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_horizon(X, horizon=1)

    def predict_horizon(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        if X.ndim != 4:
            raise ValueError(f"Expected X shape (n_samples, context, n_tau, n_logm), got {X.shape}")

        n_samples, context_length, n_tau, n_logm = X.shape
        if context_length != self.context_length:
            raise ValueError(
                f"context_length mismatch: model expects {self.context_length}, got {context_length}"
            )
        if n_tau != self.n_tau or n_logm != self.n_logm:
            raise ValueError(
                f"Surface shape mismatch: expected ({self.n_tau}, {self.n_logm}), got ({n_tau}, {n_logm})"
            )

        baseline = self._compute_baseline(X)
        X_flat = self._flatten(X).astype(np.float32, copy=False)
        if self.normalize:
            X_flat = (X_flat - self.mean) / self.std

        self.model.eval()
        with torch.no_grad():
            with autocast(device_type="cuda" if self.use_amp and self.device.startswith("cuda") else "cpu", enabled=self.use_amp and self.device.startswith("cuda")):
                pred_correction_flat_norm = self.model(torch.tensor(X_flat, dtype=torch.float32).to(self.device))
            pred_correction_flat_norm = pred_correction_flat_norm.cpu().numpy()

        if self.normalize:
            # Denormalize corrections using correction stats (not X stats)
            pred_correction_flat = pred_correction_flat_norm * self.std_corr + self.mean_corr
        else:
            pred_correction_flat = pred_correction_flat_norm

        pred_correction = pred_correction_flat.reshape(n_samples, n_tau, n_logm)
        predictions = baseline + pred_correction

        return predictions

    def save_checkpoint(self, path: str):
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "n_tau": self.n_tau,
            "n_logm": self.n_logm,
            "n_features": self.n_features,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "pool": self.pool,
            "normalize": self.normalize,
            "use_amp": self.use_amp,
            "baseline_decay": self.baseline_decay,
            "mean": self.mean,
            "std": self.std,
            "mean_corr": self.mean_corr,
            "std_corr": self.std_corr
        }, path)
