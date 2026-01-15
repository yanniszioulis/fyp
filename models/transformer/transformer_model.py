"""
Transformer model for forecasting IV surfaces (temporal transformer).
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
    from torch.cuda.amp import autocast, GradScaler
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
                 pool: str,
                 use_causal: bool):
        super().__init__()
        self.context_length = context_length
        self.pool = pool
        self.use_causal = use_causal

        self.input_proj = nn.Linear(n_features, d_model)
        self.positional = nn.Parameter(torch.zeros(context_length, d_model))

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
        x = x + self.positional.unsqueeze(0)
        if self.use_causal:
            mask = torch.triu(
                torch.ones(self.context_length, self.context_length, device=x.device),
                diagonal=1
            ).bool()
            x = self.encoder(x, mask=mask)
        else:
            x = self.encoder(x)
        if self.pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, -1, :]
        return self.head(x)


class TransformerSurfaceModel(BaseModel):
    """
    Temporal transformer that treats each surface as one token.
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
                 normalize_mode: str = "per_point",
                 patience: int = 10,
                 min_delta: float = 0.0,
                 use_amp: bool = False,
                 use_causal: bool = True,
                 delta_mode: bool = False,
                 input_delta: bool = False,
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
        self.normalize_mode = normalize_mode
        self.patience = patience
        self.min_delta = min_delta
        self.use_amp = use_amp
        self.use_causal = use_causal
        self.delta_mode = delta_mode
        self.input_delta = input_delta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.n_tau = None
        self.n_logm = None
        self.n_features = None
        self.context_length = None
        self.mean = None
        self.std = None
        self.delta_mean = None
        self.delta_std = None
        self.anchor_mean = None
        self.anchor_std = None
        self.epochs_trained = 0

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        # X: (n_samples, context, n_tau, n_logm)
        n_samples, context_length, n_tau, n_logm = X.shape
        return X.reshape(n_samples, context_length, n_tau * n_logm)

    def _build_input_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Build model input sequence.
        If input_delta is True, use consecutive deltas plus last surface as anchor.
        """
        if not self.input_delta:
            return X
        # X shape: (n_samples, context, n_tau, n_logm)
        n_samples, context_length, n_tau, n_logm = X.shape
        if context_length < 2:
            return X
        # Consecutive deltas for first context_length-1 tokens
        deltas = X[:, :-1, :, :] - X[:, 1:, :, :]
        # Last token is the level anchor (last surface)
        last_surface = X[:, -1:, :, :]
        return np.concatenate([deltas, last_surface], axis=1)

    def _normalize_array(self, X: np.ndarray, mean: Optional[np.ndarray] = None,
                         std: Optional[np.ndarray] = None) -> np.ndarray:
        mean = self.mean if mean is None else mean
        std = self.std if std is None else std
        if mean is None or std is None:
            raise ValueError("Model must be fitted before normalization")
        return (X - mean) / (std + 1e-8)

    def _denormalize_array(self, X: np.ndarray, mean: Optional[np.ndarray] = None,
                           std: Optional[np.ndarray] = None) -> np.ndarray:
        mean = self.mean if mean is None else mean
        std = self.std if std is None else std
        if mean is None or std is None:
            raise ValueError("Model must be fitted before denormalization")
        return X * (std + 1e-8) + mean

    def _compute_stats(self, data_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.normalize_mode == "per_point":
            mean = data_flat.mean(axis=0, keepdims=True)
            std = data_flat.std(axis=0, keepdims=True)
        elif self.normalize_mode == "global":
            mean = data_flat.mean()
            std = data_flat.std()
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        return mean, std

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

        X_input = self._build_input_sequence(X_train)
        X_flat = self._flatten(X_input).astype(np.float32, copy=False)
        y_flat = y_train.reshape(n_samples, self.n_features).astype(np.float32, copy=False)

        if self.delta_mode:
            last_surface = X_train[:, -1, :, :].reshape(n_samples, self.n_features)
            y_flat = y_flat - last_surface

        if self.normalize:
            if self.input_delta:
                # Stats for delta tokens/targets
                delta_tokens = X_train[:, :-1, :, :] - X_train[:, 1:, :, :]
                delta_flat = delta_tokens.reshape(-1, self.n_features)
                self.delta_mean, self.delta_std = self._compute_stats(delta_flat)
                # Stats for anchor token (last surface)
                anchor_flat = X_train[:, -1, :, :].reshape(-1, self.n_features)
                self.anchor_mean, self.anchor_std = self._compute_stats(anchor_flat)
                # Normalize input tokens
                X_tokens = X_input.reshape(n_samples, self.context_length, self.n_features)
                X_tokens[:, :-1, :] = self._normalize_array(
                    X_tokens[:, :-1, :], mean=self.delta_mean, std=self.delta_std
                )
                X_tokens[:, -1, :] = self._normalize_array(
                    X_tokens[:, -1, :], mean=self.anchor_mean, std=self.anchor_std
                )
                X_flat = X_tokens.reshape(n_samples, self.context_length, self.n_features)
                # Normalize delta targets with delta stats
                y_flat = self._normalize_array(y_flat, mean=self.delta_mean, std=self.delta_std)
            else:
                flat_for_stats = X_flat.reshape(-1, self.n_features)
                self.mean, self.std = self._compute_stats(flat_for_stats)
                X_flat = self._normalize_array(X_flat)
                y_flat = self._normalize_array(y_flat)

        dataset = TensorDataset(
            torch.tensor(X_flat, dtype=torch.float32),
            torch.tensor(y_flat, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            if X_val.ndim != 4:
                raise ValueError(f"Expected X_val shape (n_samples, context, n_tau, n_logm), got {X_val.shape}")
            X_val_input = self._build_input_sequence(X_val)
            X_val_flat = self._flatten(X_val_input).astype(np.float32, copy=False)
            y_val_flat = y_val.reshape(X_val_flat.shape[0], self.n_features).astype(np.float32, copy=False)
            if self.delta_mode:
                last_surface_val = X_val[:, -1, :, :].reshape(X_val_flat.shape[0], self.n_features)
                y_val_flat = y_val_flat - last_surface_val
            if self.normalize:
                if self.input_delta and self.delta_mean is not None and self.anchor_mean is not None:
                    X_tokens = X_val_flat.reshape(X_val_flat.shape[0], self.context_length, self.n_features)
                    X_tokens[:, :-1, :] = self._normalize_array(
                        X_tokens[:, :-1, :], mean=self.delta_mean, std=self.delta_std
                    )
                    X_tokens[:, -1, :] = self._normalize_array(
                        X_tokens[:, -1, :], mean=self.anchor_mean, std=self.anchor_std
                    )
                    X_val_flat = X_tokens.reshape(X_val_flat.shape[0], self.context_length, self.n_features)
                    y_val_flat = self._normalize_array(y_val_flat, mean=self.delta_mean, std=self.delta_std)
                else:
                    X_val_flat = self._normalize_array(X_val_flat)
                    y_val_flat = self._normalize_array(y_val_flat)
            val_dataset = TensorDataset(
                torch.tensor(X_val_flat, dtype=torch.float32),
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
            pool=self.pool,
            use_causal=self.use_causal
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        scaler = GradScaler(enabled=self.use_amp and self.device.startswith("cuda"))

        best_state = None
        best_val = float("inf")
        epochs_no_improve = 0
        self.epochs_trained = 0

        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                    preds = self.model(batch_x)
                    loss = loss_fn(preds, batch_y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered during training")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            if val_loader is None:
                self.epochs_trained = epoch + 1
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                        preds = self.model(batch_x)
                        val_loss = loss_fn(preds, batch_y).item()
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
                        self.epochs_trained = epoch + 1
                        break
            self.epochs_trained = epoch + 1

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

        X_input = self._build_input_sequence(X)
        X_flat = self._flatten(X_input).astype(np.float32, copy=False)
        if self.normalize:
            if self.input_delta and self.delta_mean is not None and self.anchor_mean is not None:
                X_tokens = X_flat.reshape(n_samples, self.context_length, self.n_features)
                X_tokens[:, :-1, :] = self._normalize_array(
                    X_tokens[:, :-1, :], mean=self.delta_mean, std=self.delta_std
                )
                X_tokens[:, -1, :] = self._normalize_array(
                    X_tokens[:, -1, :], mean=self.anchor_mean, std=self.anchor_std
                )
                X_flat = X_tokens.reshape(n_samples, self.context_length, self.n_features)
            else:
                X_flat = self._normalize_array(X_flat)

        self.model.eval()
        with torch.no_grad():
            with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                preds = self.model(torch.tensor(X_flat, dtype=torch.float32).to(self.device))
            preds = preds.cpu().numpy()

        if self.normalize:
            if self.delta_mode and self.delta_mean is not None:
                preds = self._denormalize_array(preds, mean=self.delta_mean, std=self.delta_std)
            else:
                preds = self._denormalize_array(preds)

        if self.delta_mode:
            last_surface = X[:, -1, :, :].reshape(n_samples, n_tau, n_logm)
            preds = preds.reshape(n_samples, n_tau, n_logm)
            preds = preds + last_surface
            return preds

        return preds.reshape(n_samples, n_tau, n_logm)

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
            "normalize_mode": self.normalize_mode,
            "use_amp": self.use_amp,
            "use_causal": self.use_causal,
            "delta_mode": self.delta_mode,
            "input_delta": self.input_delta,
            "mean": self.mean,
            "std": self.std,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
            "anchor_mean": self.anchor_mean,
            "anchor_std": self.anchor_std
        }, path)
