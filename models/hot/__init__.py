"""
Higher-Order Transformers (HOT) for IV Surface Forecasting.

Adapts Kronecker-structured attention for 3D IV surface data (time, tau, logm).
"""

from .hot_model import HOTSurfaceModel

__all__ = ['HOTSurfaceModel']
