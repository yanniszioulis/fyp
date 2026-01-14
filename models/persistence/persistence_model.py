"""
Persistence baseline model: Forecast = Last observed surface.
"""

import numpy as np
from models.base_model import BaseModel


class PersistenceModel(BaseModel):
    """
    Persistence baseline: Forecast = Last observed surface
    
    For any horizon h:
        w_forecast(t+h) = w_true(t)
    
    No normalization needed - just copies values.
    """
    
    def __init__(self, name="persistence"):
        super().__init__(name=name)
        self.requires_normalization = False  # No normalization needed
    
    def fit(self, X_train, y_train=None, **kwargs):
        """
        Persistence doesn't need training - it's just copying
        
        Parameters:
        -----------
        X_train : array, shape (n_samples, context_length, n_tau, n_logm)
            Training sequences (not used, but kept for interface consistency)
        y_train : array, optional
            Training targets (not used)
        """
        # Persistence doesn't learn anything
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict by copying the last observed surface
        
        Parameters:
        -----------
        X : array, shape (n_samples, context_length, n_tau, n_logm)
            Input sequences
            
        Returns:
        --------
        predictions : array, shape (n_samples, n_tau, n_logm)
            Forecasted surfaces (copied from last timestep)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Copy last timestep from context
        # X shape: (n_samples, context_length, n_tau, n_logm)
        # Return: (n_samples, n_tau, n_logm)
        predictions = X[:, -1, :, :].copy()  # Last timestep
        
        return predictions
    
    def predict_horizon(self, X, horizon=1):
        """
        Predict for specific horizon (still just copies last surface)
        
        Parameters:
        -----------
        X : array, shape (n_samples, context_length, n_tau, n_logm)
            Input sequences
        horizon : int
            Days ahead to forecast (1, 5, 21, etc.)
            
        Returns:
        --------
        predictions : array, shape (n_samples, n_tau, n_logm)
            Forecasted surfaces (same for any horizon - just copy)
        """
        # For persistence, horizon doesn't matter - always copy last surface
        return self.predict(X)
