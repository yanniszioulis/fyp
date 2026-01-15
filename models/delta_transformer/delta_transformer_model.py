"""
Delta transformer model for forecasting IV surfaces.
Delta-only inputs and delta-only targets.
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


class DeltaTransformerSurfaceModel(BaseModel):
    """
    Temporal transformer that predicts deltas relative to the last surface.
    """

    def __init__(self,
                 name: str = "delta_transformer_surface",
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 pool: str = "last",
                 patience: int = 10,
                 min_delta: float = 0.0,
                 use_amp: bool = False,
                 use_causal: bool = True,
                 device: Optional[str] = None):
        super().__init__(name=name)
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DeltaTransformerSurfaceModel. Install with: pip install torch"
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
        self.normalize = True
        self.normalize_mode = "per_point"
        self.patience = patience
        self.min_delta = min_delta
        self.use_amp = use_amp
        self.use_causal = use_causal
        self.max_grad_norm = 1.0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.n_tau = None
        self.n_logm = None
        self.n_features = None
        self.context_length = None
        self.input_context_length = None
        self.delta_token_mean = None
        self.delta_token_std = None
        self.delta_target_mean = None
        self.delta_target_std = None
        self.epochs_trained = 0

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        # X: (n_samples, context, n_tau, n_logm)
        n_samples, context_length, n_tau, n_logm = X.shape
        return X.reshape(n_samples, context_length, n_tau * n_logm)

    def _compute_deltas(self, X: np.ndarray) -> Optional[np.ndarray]:
        # X: (n_samples, context, n_tau, n_logm)
        if X.shape[1] < 2:
            return None
        return X[:, 1:, :, :] - X[:, :-1, :, :]

    def _validate_delta_context(self, context_length: int):
        if context_length < 2:
            raise ValueError("Delta transformer requires context_length >= 2")

    def _build_input_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Build delta-only input sequence: consecutive deltas.
        """
        deltas = self._compute_deltas(X)
        if deltas is None:
            raise ValueError("Delta transformer requires context_length >= 2")
        return deltas

    def _compute_baseline(self, X: np.ndarray) -> np.ndarray:
        return X.mean(axis=1)

    def _normalize_array(self, X: np.ndarray, mean: Optional[np.ndarray] = None,
                         std: Optional[np.ndarray] = None) -> np.ndarray:
        if mean is None or std is None:
            raise ValueError("Model must be fitted before normalization")
        return (X - mean) / (std + 1e-8)

    def _denormalize_array(self, X: np.ndarray, mean: Optional[np.ndarray] = None,
                           std: Optional[np.ndarray] = None) -> np.ndarray:
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
        log_interval = kwargs.pop("log_interval", None)
        log_train_val = kwargs.pop("log_train_val", False)
        log_loss_only = kwargs.pop("log_loss_only", False)
        debug_delta_stats = kwargs.pop("debug_delta_stats", False)
        use_val_for_early_stopping = kwargs.pop("use_val_for_early_stopping", True)
        horizon = kwargs.get("horizon", 1)
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train are required for DeltaTransformerSurfaceModel")

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

        self.input_context_length = context_length
        self.context_length = context_length - 1
        self._validate_delta_context(context_length)
        self.n_tau = n_tau
        self.n_logm = n_logm
        self.n_features = n_tau * n_logm

        X_input = self._build_input_sequence(X_train)
        X_flat = self._flatten(X_input).astype(np.float32, copy=False)
        y_flat = y_train.reshape(n_samples, self.n_features).astype(np.float32, copy=False)

        baseline = self._compute_baseline(X_train).reshape(n_samples, self.n_features)
        y_flat = y_flat - baseline

        if self.normalize:
            delta_flat = X_input.reshape(-1, self.n_features)
            self.delta_token_mean, self.delta_token_std = self._compute_stats(delta_flat)
            X_flat = self._normalize_array(X_flat, mean=self.delta_token_mean, std=self.delta_token_std)
            self.delta_target_mean, self.delta_target_std = self._compute_stats(y_flat)
            y_flat = self._normalize_array(y_flat, mean=self.delta_target_mean, std=self.delta_target_std)

        dataset = TensorDataset(
            torch.tensor(X_flat, dtype=torch.float32),
            torch.tensor(y_flat, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        if use_val_for_early_stopping and X_val is not None and y_val is not None:
            if X_val.ndim != 4:
                raise ValueError(f"Expected X_val shape (n_samples, context, n_tau, n_logm), got {X_val.shape}")
            self._validate_delta_context(X_val.shape[1])
            X_val_input = self._build_input_sequence(X_val)
            X_val_flat = self._flatten(X_val_input).astype(np.float32, copy=False)
            y_val_flat = y_val.reshape(X_val_flat.shape[0], self.n_features).astype(np.float32, copy=False)
            baseline_val = self._compute_baseline(X_val).reshape(X_val_flat.shape[0], self.n_features)
            y_val_flat = y_val_flat - baseline_val
            if self.normalize:
                X_val_flat = self._normalize_array(
                    X_val_flat, mean=self.delta_token_mean, std=self.delta_token_std
                )
                y_val_flat = self._normalize_array(
                    y_val_flat, mean=self.delta_target_mean, std=self.delta_target_std
                )
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

        def _loss_fn(preds, targets):
            return nn.functional.mse_loss(preds, targets)

        scaler = GradScaler(enabled=self.use_amp and self.device.startswith("cuda"))

        best_state = None
        best_val = float("inf")
        epochs_no_improve = 0
        self.epochs_trained = 0

        self.is_fitted = True
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_losses = []
            epoch_grad_norms = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                    preds = self.model(batch_x)
                    loss = _loss_fn(preds, batch_y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered during training")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())
                if torch.isfinite(grad_norm):
                    epoch_grad_norms.append(float(grad_norm))

            if val_loader is None:
                self.epochs_trained = epoch + 1
                if log_loss_only and log_interval and (epoch + 1) % log_interval == 0:
                    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                    avg_grad_norm = None
                    if epoch_grad_norms:
                        avg_grad_norm = float(np.mean(epoch_grad_norms))
                    if avg_grad_norm is None:
                        print(f"  epoch={epoch + 1} loss={avg_loss:.6f}")
                    else:
                        print(
                            f"  epoch={epoch + 1} loss={avg_loss:.6f} "
                            f"grad_norm={avg_grad_norm:.6f}"
                        )
                if log_train_val and log_interval and (epoch + 1) % log_interval == 0:
                    self._log_epoch_metrics(
                        X_train, y_train, X_val, y_val, horizon, epoch + 1,
                        epoch_losses=epoch_losses,
                        epoch_grad_norms=epoch_grad_norms
                    )
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                        preds = self.model(batch_x)
                        val_loss = _loss_fn(preds, batch_y).item()
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
            if log_train_val and log_interval and (epoch + 1) % log_interval == 0:
                self._log_epoch_metrics(
                    X_train, y_train, X_val, y_val, horizon, epoch + 1,
                    epoch_losses=epoch_losses,
                    epoch_grad_norms=epoch_grad_norms
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

        if debug_delta_stats:
            token_std = self.delta_token_std
            target_std = self.delta_target_std
            if token_std is not None:
                print(
                    "delta_token_std:",
                    f"mean={float(np.mean(token_std)):.6f} std={float(np.std(token_std)):.6f} "
                    f"min={float(np.min(token_std)):.6f} max={float(np.max(token_std)):.6f}"
                )
            if target_std is not None:
                print(
                    "delta_target_std:",
                    f"mean={float(np.mean(target_std)):.6f} std={float(np.std(target_std)):.6f} "
                    f"min={float(np.min(target_std)):.6f} max={float(np.max(target_std)):.6f}"
                )

            n_eval = min(256, X_flat.shape[0])
            if n_eval > 0:
                X_eval = torch.tensor(X_flat[:n_eval], dtype=torch.float32).to(self.device)
                y_eval = y_flat[:n_eval]
                self.model.eval()
                with torch.no_grad():
                    with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                        preds_eval = self.model(X_eval).cpu().numpy()
                mse_norm = float(np.mean((preds_eval - y_eval[:n_eval]) ** 2))
                if self.normalize and self.delta_target_mean is not None and self.delta_target_std is not None:
                    preds_denorm = self._denormalize_array(
                        preds_eval, mean=self.delta_target_mean, std=self.delta_target_std
                    )
                    y_denorm = self._denormalize_array(
                        y_eval[:n_eval], mean=self.delta_target_mean, std=self.delta_target_std
                    )
                    mse_denorm = float(np.mean((preds_denorm - y_denorm) ** 2))
                else:
                    mse_denorm = mse_norm
                print(
                    f"delta_mse normalized={mse_norm:.6f} denormalized={mse_denorm:.6f}"
                )
        return self

    def _log_epoch_metrics(self, X_train, y_train, X_val, y_val, horizon: int, epoch: int,
                           epoch_losses=None, epoch_grad_norms=None):
        if X_train is None or y_train is None:
            return
        from evaluation.metrics import compute_all_metrics

        def _top_delta_rmse(X, y_true, y_pred, top_pct: float = 20.0):
            if X is None or y_true is None or y_pred is None:
                return None
            baseline = self._compute_baseline(X)
            delta_true = y_true - baseline
            delta_pred = y_pred - baseline
            target = delta_true
            errors = delta_pred - delta_true
            threshold = np.percentile(np.abs(target), 100.0 - top_pct)
            mask = np.abs(target) >= threshold
            if not np.any(mask):
                return None
            mse = np.mean((errors[mask]) ** 2)
            return float(np.sqrt(mse))

        train_preds = self.predict_horizon(X_train, horizon=horizon)
        train_metrics = compute_all_metrics(y_train, train_preds)
        train_top_rmse = _top_delta_rmse(X_train, y_train, train_preds)
        val_metrics = None
        val_top_rmse = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_preds = self.predict_horizon(X_val, horizon=horizon)
            val_metrics = compute_all_metrics(y_val, val_preds)
            val_top_rmse = _top_delta_rmse(X_val, y_val, val_preds)
        avg_loss = None
        if epoch_losses:
            avg_loss = float(np.mean(epoch_losses))
        avg_grad_norm = None
        if epoch_grad_norms:
            avg_grad_norm = float(np.mean(epoch_grad_norms))

        extra_bits = []
        if avg_loss is not None:
            extra_bits.append(f"loss={avg_loss:.6f}")
        if avg_grad_norm is not None:
            extra_bits.append(f"grad_norm={avg_grad_norm:.6f}")
        extra = ""
        if extra_bits:
            extra = " " + " ".join(extra_bits)

        if val_metrics is None:
            if train_top_rmse is None:
                print(f"  epoch={epoch} train_iv_rmse={train_metrics['iv_rmse']:.6f}{extra}")
            else:
                print(
                    f"  epoch={epoch} train_iv_rmse={train_metrics['iv_rmse']:.6f} "
                    f"train_top20_rmse={train_top_rmse:.6f}{extra}"
                )
        else:
            if train_top_rmse is None or val_top_rmse is None:
                print(
                    f"  epoch={epoch} train_iv_rmse={train_metrics['iv_rmse']:.6f} "
                    f"val_iv_rmse={val_metrics['iv_rmse']:.6f}{extra}"
                )
            else:
                print(
                    f"  epoch={epoch} train_iv_rmse={train_metrics['iv_rmse']:.6f} "
                    f"train_top20_rmse={train_top_rmse:.6f} "
                    f"val_iv_rmse={val_metrics['iv_rmse']:.6f} "
                    f"val_top20_rmse={val_top_rmse:.6f}{extra}"
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_horizon(X, horizon=1)

    def predict_horizon(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        if X.ndim != 4:
            raise ValueError(f"Expected X shape (n_samples, context, n_tau, n_logm), got {X.shape}")

        n_samples, context_length, n_tau, n_logm = X.shape
        self._validate_delta_context(context_length)
        if self.input_context_length is not None and context_length != self.input_context_length:
            raise ValueError(
                f"context_length mismatch: model expects {self.input_context_length}, got {context_length}"
            )
        if n_tau != self.n_tau or n_logm != self.n_logm:
            raise ValueError(
                f"Surface shape mismatch: expected ({self.n_tau}, {self.n_logm}), got ({n_tau}, {n_logm})"
            )

        X_input = self._build_input_sequence(X)
        X_flat = self._flatten(X_input).astype(np.float32, copy=False)
        if self.normalize:
            X_flat = self._normalize_array(
                X_flat, mean=self.delta_token_mean, std=self.delta_token_std
            )

        self.model.eval()
        with torch.no_grad():
            with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                preds = self.model(torch.tensor(X_flat, dtype=torch.float32).to(self.device))
            preds = preds.cpu().numpy()

        if self.normalize:
            preds = self._denormalize_array(
                preds, mean=self.delta_target_mean, std=self.delta_target_std
            )

        baseline = self._compute_baseline(X).reshape(n_samples, n_tau, n_logm)
        preds = preds.reshape(n_samples, n_tau, n_logm)
        preds = preds + baseline
        return preds

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
            "input_context_length": self.input_context_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "pool": self.pool,
            "normalize": self.normalize,
            "normalize_mode": self.normalize_mode,
            "use_amp": self.use_amp,
            "use_causal": self.use_causal,
            "max_grad_norm": self.max_grad_norm,
            "delta_token_mean": self.delta_token_mean,
            "delta_token_std": self.delta_token_std,
            "delta_target_mean": self.delta_target_mean,
            "delta_target_std": self.delta_target_std,
        }, path)
"""
Delta transformer model for forecasting IV surfaces.
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


class DeltaTransformerSurfaceModel(BaseModel):
    """
    Temporal transformer that predicts deltas relative to the last surface.
    """

    def __init__(self,
                 name: str = "delta_transformer_surface",
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
                 input_delta: bool = True,
                 use_anchor_token: bool = False,
                 scale_deltas: bool = False,
                 delta_scale_eps: float = 1e-6,
                 delta_scale_factor: float = 10.0,
                 delta_loss_weighting: bool = False,
                 delta_loss_alpha: float = 0.0,
                 loss_scale: float = 1.0,
                 max_grad_norm: float = 1.0,
                 device: Optional[str] = None):
        super().__init__(name=name)
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DeltaTransformerSurfaceModel. Install with: pip install torch"
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
        self.input_delta = input_delta
        self.use_anchor_token = use_anchor_token
        self.scale_deltas = scale_deltas
        self.delta_scale_eps = delta_scale_eps
        self.delta_scale_factor = delta_scale_factor
        self.delta_loss_weighting = delta_loss_weighting
        self.delta_loss_alpha = delta_loss_alpha
        self.loss_scale = loss_scale
        self.max_grad_norm = max_grad_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.n_tau = None
        self.n_logm = None
        self.n_features = None
        self.context_length = None
        self.input_context_length = None
        self.delta_mean = None
        self.delta_std = None
        self.delta_target_mean = None
        self.delta_target_std = None
        self.anchor_mean = None
        self.anchor_std = None
        self.epochs_trained = 0

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        # X: (n_samples, context, n_tau, n_logm)
        n_samples, context_length, n_tau, n_logm = X.shape
        return X.reshape(n_samples, context_length, n_tau * n_logm)

    def _compute_deltas(self, X: np.ndarray) -> Optional[np.ndarray]:
        # X: (n_samples, context, n_tau, n_logm)
        if X.shape[1] < 2:
            return None
        return X[:, 1:, :, :] - X[:, :-1, :, :]

    def _validate_delta_context(self, context_length: int):
        if (self.input_delta or self.scale_deltas) and context_length < 2:
            raise ValueError("input_delta requires context_length >= 2")

    def _compute_delta_scale(self, X: np.ndarray) -> np.ndarray:
        deltas = self._compute_deltas(X)
        if deltas is None:
            raise ValueError("input_delta requires context_length >= 2")
        scale = deltas.std(axis=(1, 2, 3), keepdims=True)
        return scale + self.delta_scale_eps

    def _apply_delta_scale_to_tokens(self, X_tokens: np.ndarray, delta_scale: np.ndarray) -> np.ndarray:
        # X_tokens shape: (n_samples, context, n_tau, n_logm)
        X_tokens = X_tokens.copy()
        if self.use_anchor_token:
            X_tokens[:, :-1, :, :] = X_tokens[:, :-1, :, :] / delta_scale
        else:
            X_tokens[:, :, :, :] = X_tokens[:, :, :, :] / delta_scale
        return X_tokens

    def _build_input_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Build model input sequence.
        If input_delta is True, use consecutive deltas plus last surface as anchor.
        """
        if not self.input_delta:
            return X
        # X shape: (n_samples, context, n_tau, n_logm)
        deltas = self._compute_deltas(X)
        if deltas is None:
            return X
        # Consecutive deltas for first context_length-1 tokens
        # Last token is either the level anchor or zeros (anchor removed)
        if self.use_anchor_token:
            last_token = X[:, -1:, :, :]
            return np.concatenate([deltas, last_token], axis=1)
        return deltas

    def _normalize_array(self, X: np.ndarray, mean: Optional[np.ndarray] = None,
                         std: Optional[np.ndarray] = None) -> np.ndarray:
        if mean is None or std is None:
            raise ValueError("Model must be fitted before normalization")
        return (X - mean) / (std + 1e-8)

    def _denormalize_array(self, X: np.ndarray, mean: Optional[np.ndarray] = None,
                           std: Optional[np.ndarray] = None) -> np.ndarray:
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
        log_interval = kwargs.pop("log_interval", None)
        log_train_val = kwargs.pop("log_train_val", False)
        log_loss_only = kwargs.pop("log_loss_only", False)
        use_val_for_early_stopping = kwargs.pop("use_val_for_early_stopping", True)
        horizon = kwargs.get("horizon", 1)
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train are required for DeltaTransformerSurfaceModel")

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

        self.input_context_length = context_length
        self.context_length = context_length if self.use_anchor_token else context_length - 1
        self._validate_delta_context(context_length)
        self.n_tau = n_tau
        self.n_logm = n_logm
        self.n_features = n_tau * n_logm

        X_input = self._build_input_sequence(X_train)
        delta_scale = None
        if self.scale_deltas:
            delta_scale = self._compute_delta_scale(X_train)
            if self.input_delta:
                X_input = self._apply_delta_scale_to_tokens(X_input, delta_scale)
        X_flat = self._flatten(X_input).astype(np.float32, copy=False)
        y_flat = y_train.reshape(n_samples, self.n_features).astype(np.float32, copy=False)

        last_surface = X_train[:, -1, :, :].reshape(n_samples, self.n_features)
        y_flat = y_flat - last_surface
        if self.scale_deltas and delta_scale is not None:
            y_flat = y_flat / delta_scale.reshape(n_samples, 1)

        if self.normalize:
            if self.input_delta:
                # Stats for delta tokens
                delta_tokens = self._compute_deltas(X_train)
                if delta_tokens is None:
                    raise ValueError("input_delta requires context_length >= 2")
                if self.scale_deltas and delta_scale is not None:
                    delta_tokens = delta_tokens / delta_scale
                delta_flat = delta_tokens.reshape(-1, self.n_features)
                self.delta_mean, self.delta_std = self._compute_stats(delta_flat)
                if self.use_anchor_token:
                    # Stats for anchor token (last surface)
                    anchor_flat = X_train[:, -1, :, :].reshape(-1, self.n_features)
                    self.anchor_mean, self.anchor_std = self._compute_stats(anchor_flat)
                # Normalize input tokens
                X_tokens = X_input.reshape(n_samples, self.context_length, self.n_features)
                if self.use_anchor_token:
                    X_tokens[:, :-1, :] = self._normalize_array(
                        X_tokens[:, :-1, :], mean=self.delta_mean, std=self.delta_std
                    )
                    X_tokens[:, -1, :] = self._normalize_array(
                        X_tokens[:, -1, :], mean=self.anchor_mean, std=self.anchor_std
                    )
                else:
                    X_tokens[:, :, :] = self._normalize_array(
                        X_tokens[:, :, :], mean=self.delta_mean, std=self.delta_std
                    )
                X_flat = X_tokens.reshape(n_samples, self.context_length, self.n_features)
                # Normalize delta targets with target-specific stats
                self.delta_target_mean, self.delta_target_std = self._compute_stats(y_flat)
                y_flat = self._normalize_array(
                    y_flat, mean=self.delta_target_mean, std=self.delta_target_std
                )
            else:
                flat_for_stats = X_flat.reshape(-1, self.n_features)
                self.delta_mean, self.delta_std = self._compute_stats(flat_for_stats)
                X_flat = self._normalize_array(X_flat, mean=self.delta_mean, std=self.delta_std)
                y_flat = self._normalize_array(y_flat, mean=self.delta_mean, std=self.delta_std)

        if self.delta_scale_factor != 1.0:
            y_flat = y_flat * self.delta_scale_factor

        dataset = TensorDataset(
            torch.tensor(X_flat, dtype=torch.float32),
            torch.tensor(y_flat, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if use_val_for_early_stopping and X_val is not None and y_val is not None:
            if X_val.ndim != 4:
                raise ValueError(f"Expected X_val shape (n_samples, context, n_tau, n_logm), got {X_val.shape}")
            if self.input_delta and X_val.shape[1] < 2:
                raise ValueError("input_delta requires context_length >= 2")
            X_val_input = self._build_input_sequence(X_val)
            delta_scale_val = None
            if self.scale_deltas:
                delta_scale_val = self._compute_delta_scale(X_val)
                if self.input_delta:
                    X_val_input = self._apply_delta_scale_to_tokens(X_val_input, delta_scale_val)
            X_val_flat = self._flatten(X_val_input).astype(np.float32, copy=False)
            y_val_flat = y_val.reshape(X_val_flat.shape[0], self.n_features).astype(np.float32, copy=False)
            last_surface_val = X_val[:, -1, :, :].reshape(X_val_flat.shape[0], self.n_features)
            y_val_flat = y_val_flat - last_surface_val
            if self.scale_deltas and delta_scale_val is not None:
                y_val_flat = y_val_flat / delta_scale_val.reshape(X_val_flat.shape[0], 1)
            if self.normalize:
                if self.input_delta and self.delta_mean is not None:
                    X_tokens = X_val_flat.reshape(X_val_flat.shape[0], self.context_length, self.n_features)
                    if self.use_anchor_token and self.anchor_mean is not None:
                        X_tokens[:, :-1, :] = self._normalize_array(
                            X_tokens[:, :-1, :], mean=self.delta_mean, std=self.delta_std
                        )
                        X_tokens[:, -1, :] = self._normalize_array(
                            X_tokens[:, -1, :], mean=self.anchor_mean, std=self.anchor_std
                        )
                    else:
                        X_tokens[:, :, :] = self._normalize_array(
                            X_tokens[:, :, :], mean=self.delta_mean, std=self.delta_std
                        )
                    X_val_flat = X_tokens.reshape(X_val_flat.shape[0], self.context_length, self.n_features)
                    target_mean = self.delta_target_mean or self.delta_mean
                    target_std = self.delta_target_std or self.delta_std
                    y_val_flat = self._normalize_array(
                        y_val_flat, mean=target_mean, std=target_std
                    )
                else:
                    X_val_flat = self._normalize_array(X_val_flat, mean=self.delta_mean, std=self.delta_std)
                    y_val_flat = self._normalize_array(y_val_flat, mean=self.delta_mean, std=self.delta_std)
            if self.delta_scale_factor != 1.0:
                y_val_flat = y_val_flat * self.delta_scale_factor
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

        def _loss_fn(preds, targets):
            if self.delta_loss_weighting and self.delta_loss_alpha > 0.0:
                weights = 1.0 + self.delta_loss_alpha * torch.abs(targets)
                loss = (weights * (preds - targets) ** 2).mean()
            else:
                loss = nn.functional.mse_loss(preds, targets)
            return loss * self.loss_scale

        scaler = GradScaler(enabled=self.use_amp and self.device.startswith("cuda"))

        best_state = None
        best_val = float("inf")
        epochs_no_improve = 0
        self.epochs_trained = 0

        self.is_fitted = True
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_losses = []
            epoch_grad_norms = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                    preds = self.model(batch_x)
                    loss = _loss_fn(preds, batch_y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered during training")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())
                if torch.isfinite(grad_norm):
                    epoch_grad_norms.append(float(grad_norm))

            if val_loader is None:
                self.epochs_trained = epoch + 1
                if log_loss_only and log_interval and (epoch + 1) % log_interval == 0:
                    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                    avg_grad_norm = None
                    if epoch_grad_norms:
                        avg_grad_norm = float(np.mean(epoch_grad_norms))
                    if avg_grad_norm is None:
                        print(f"  epoch={epoch + 1} loss={avg_loss:.6f}")
                    else:
                        print(
                            f"  epoch={epoch + 1} loss={avg_loss:.6f} "
                            f"grad_norm={avg_grad_norm:.6f}"
                        )
                if log_train_val and log_interval and (epoch + 1) % log_interval == 0:
                    self._log_epoch_metrics(
                        X_train, y_train, X_val, y_val, horizon, epoch + 1,
                        epoch_losses=epoch_losses,
                        epoch_grad_norms=epoch_grad_norms
                    )
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                        preds = self.model(batch_x)
                        val_loss = _loss_fn(preds, batch_y).item()
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
            if log_train_val and log_interval and (epoch + 1) % log_interval == 0:
                self._log_epoch_metrics(
                    X_train, y_train, X_val, y_val, horizon, epoch + 1,
                    epoch_losses=epoch_losses,
                    epoch_grad_norms=epoch_grad_norms
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        return self

    def _log_epoch_metrics(self, X_train, y_train, X_val, y_val, horizon: int, epoch: int,
                           epoch_losses=None, epoch_grad_norms=None):
        if X_train is None or y_train is None:
            return
        from evaluation.metrics import compute_all_metrics

        def _top_delta_rmse(X, y_true, y_pred, top_pct: float = 20.0):
            if X is None or y_true is None or y_pred is None:
                return None
            baseline = X[:, -1, :, :]
            delta_true = y_true - baseline
            delta_pred = y_pred - baseline
            target = delta_true
            errors = delta_pred - delta_true
            threshold = np.percentile(np.abs(target), 100.0 - top_pct)
            mask = np.abs(target) >= threshold
            if not np.any(mask):
                return None
            mse = np.mean((errors[mask]) ** 2)
            return float(np.sqrt(mse))

        train_preds = self.predict_horizon(X_train, horizon=horizon)
        train_metrics = compute_all_metrics(y_train, train_preds)
        train_top_rmse = _top_delta_rmse(X_train, y_train, train_preds)
        val_metrics = None
        val_top_rmse = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_preds = self.predict_horizon(X_val, horizon=horizon)
            val_metrics = compute_all_metrics(y_val, val_preds)
            val_top_rmse = _top_delta_rmse(X_val, y_val, val_preds)
        avg_loss = None
        if epoch_losses:
            avg_loss = float(np.mean(epoch_losses))
        avg_grad_norm = None
        if epoch_grad_norms:
            avg_grad_norm = float(np.mean(epoch_grad_norms))

        extra_bits = []
        if avg_loss is not None:
            extra_bits.append(f"loss={avg_loss:.6f}")
        if avg_grad_norm is not None:
            extra_bits.append(f"grad_norm={avg_grad_norm:.6f}")
        extra = ""
        if extra_bits:
            extra = " " + " ".join(extra_bits)

        if val_metrics is None:
            if train_top_rmse is None:
                print(f"  epoch={epoch} train_iv_rmse={train_metrics['iv_rmse']:.6f}{extra}")
            else:
                print(
                    f"  epoch={epoch} train_iv_rmse={train_metrics['iv_rmse']:.6f} "
                    f"train_top20_rmse={train_top_rmse:.6f}{extra}"
                )
        else:
            if train_top_rmse is None or val_top_rmse is None:
                print(
                    f"  epoch={epoch} train_iv_rmse={train_metrics['iv_rmse']:.6f} "
                    f"val_iv_rmse={val_metrics['iv_rmse']:.6f}{extra}"
                )
            else:
                print(
                    f"  epoch={epoch} train_iv_rmse={train_metrics['iv_rmse']:.6f} "
                    f"train_top20_rmse={train_top_rmse:.6f} "
                    f"val_iv_rmse={val_metrics['iv_rmse']:.6f} "
                    f"val_top20_rmse={val_top_rmse:.6f}{extra}"
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_horizon(X, horizon=1)

    def predict_horizon(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        if X.ndim != 4:
            raise ValueError(f"Expected X shape (n_samples, context, n_tau, n_logm), got {X.shape}")

        n_samples, context_length, n_tau, n_logm = X.shape
        self._validate_delta_context(context_length)
        if self.input_context_length is not None and context_length != self.input_context_length:
            raise ValueError(
                f"context_length mismatch: model expects {self.input_context_length}, got {context_length}"
            )
        if n_tau != self.n_tau or n_logm != self.n_logm:
            raise ValueError(
                f"Surface shape mismatch: expected ({self.n_tau}, {self.n_logm}), got ({n_tau}, {n_logm})"
            )

        X_input = self._build_input_sequence(X)
        delta_scale = None
        if self.scale_deltas:
            delta_scale = self._compute_delta_scale(X)
            if self.input_delta:
                X_input = self._apply_delta_scale_to_tokens(X_input, delta_scale)
        X_flat = self._flatten(X_input).astype(np.float32, copy=False)
        if self.normalize:
            if self.input_delta and self.delta_mean is not None:
                X_tokens = X_flat.reshape(n_samples, self.context_length, self.n_features)
                if self.use_anchor_token and self.anchor_mean is not None:
                    X_tokens[:, :-1, :] = self._normalize_array(
                        X_tokens[:, :-1, :], mean=self.delta_mean, std=self.delta_std
                    )
                    X_tokens[:, -1, :] = self._normalize_array(
                        X_tokens[:, -1, :], mean=self.anchor_mean, std=self.anchor_std
                    )
                else:
                    X_tokens[:, :, :] = self._normalize_array(
                        X_tokens[:, :, :], mean=self.delta_mean, std=self.delta_std
                    )
                X_flat = X_tokens.reshape(n_samples, self.context_length, self.n_features)
            else:
                X_flat = self._normalize_array(X_flat, mean=self.delta_mean, std=self.delta_std)

        self.model.eval()
        with torch.no_grad():
            with autocast(enabled=self.use_amp and self.device.startswith("cuda")):
                preds = self.model(torch.tensor(X_flat, dtype=torch.float32).to(self.device))
            preds = preds.cpu().numpy()

        if self.normalize:
            target_mean = self.delta_target_mean or self.delta_mean
            target_std = self.delta_target_std or self.delta_std
            if target_mean is not None and target_std is not None:
                preds = self._denormalize_array(preds, mean=target_mean, std=target_std)
            else:
                preds = self._denormalize_array(preds, mean=self.delta_mean, std=self.delta_std)

        if self.delta_scale_factor != 1.0:
            preds = preds / self.delta_scale_factor
        if self.scale_deltas and delta_scale is not None:
            preds = preds * delta_scale.reshape(n_samples, 1)
        last_surface = X[:, -1, :, :].reshape(n_samples, n_tau, n_logm)
        preds = preds.reshape(n_samples, n_tau, n_logm)
        preds = preds + last_surface
        return preds

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
            "input_context_length": self.input_context_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "pool": self.pool,
            "normalize": self.normalize,
            "normalize_mode": self.normalize_mode,
            "use_amp": self.use_amp,
            "use_causal": self.use_causal,
            "input_delta": self.input_delta,
            "use_anchor_token": self.use_anchor_token,
            "scale_deltas": self.scale_deltas,
            "delta_scale_eps": self.delta_scale_eps,
            "delta_scale_factor": self.delta_scale_factor,
            "delta_loss_weighting": self.delta_loss_weighting,
            "delta_loss_alpha": self.delta_loss_alpha,
            "loss_scale": self.loss_scale,
            "max_grad_norm": self.max_grad_norm,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
            "delta_target_mean": self.delta_target_mean,
            "delta_target_std": self.delta_target_std,
            "anchor_mean": self.anchor_mean,
            "anchor_std": self.anchor_std
        }, path)
