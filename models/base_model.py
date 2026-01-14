"""
Base class for all forecasting models.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Base class for all forecasting models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.requires_normalization = True  # Most models need normalization
    
    @abstractmethod
    def fit(self, X_train, y_train=None, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def predict_horizon(self, X, horizon=1):
        """
        Predict for specific horizon.
        Default implementation just calls predict (override if needed).
        """
        return self.predict(X)
