"""
Delta transformer model for forecasting IV surfaces.
"""

from typing import Optional

from models.transformer.transformer_model import TransformerSurfaceModel


class DeltaTransformerSurfaceModel(TransformerSurfaceModel):
    """
    Transformer that predicts deltas relative to the last surface.
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
        super().__init__(
            name=name,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=num_epochs,
            pool=pool,
            normalize=normalize,
            normalize_mode=normalize_mode,
            patience=patience,
            min_delta=min_delta,
            use_amp=use_amp,
            use_causal=use_causal,
            delta_mode=True,
            input_delta=input_delta,
            use_anchor_token=use_anchor_token,
            scale_deltas=scale_deltas,
            delta_scale_eps=delta_scale_eps,
            delta_scale_factor=delta_scale_factor,
            delta_loss_weighting=delta_loss_weighting,
            delta_loss_alpha=delta_loss_alpha,
            loss_scale=loss_scale,
            max_grad_norm=max_grad_norm,
            device=device,
        )
