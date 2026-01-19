"""
ConvLSTM model for volatility surface forecasting.
Based on PI-ConvTF's SAConvLSTM with inter_channels='None' for vanilla ConvLSTM.
Uses RMSE loss for training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import os
from typing import Optional, Tuple

from models.base_model import BaseModel


# ConvLSTM Cell (adapted from PI-ConvTF's SAConvLSTMCell with inter_channels='None')
class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell - vanilla version without self-attention"""
    
    def __init__(self, input_channels, feature_channels, kernel_size, stride, padding, device):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        
        # Input-to-state convolutions
        self.Wxi = nn.Conv2d(in_channels=input_channels, out_channels=feature_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)
        self.Wxf = nn.Conv2d(in_channels=input_channels, out_channels=feature_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)
        self.Wxg = nn.Conv2d(in_channels=input_channels, out_channels=feature_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)
        self.Wxo = nn.Conv2d(in_channels=input_channels, out_channels=feature_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Hidden-to-state convolutions
        self.Whi = nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=1)
        self.Whf = nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=1)
        self.Whg = nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=1)
        self.Who = nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels, kernel_size=1)
    
    def forward(self, x, others):
        """Forward pass of ConvLSTM cell"""
        ct_1, ht_1 = others  # Only cell and hidden state (no memory for vanilla ConvLSTM)
        
        it = torch.sigmoid(self.Wxi(x) + self.Whi(ht_1))
        ft = torch.sigmoid(self.Wxf(x) + self.Whf(ht_1))
        gt = torch.tanh(self.Wxg(x) + self.Whg(ht_1))
        
        Ct = ft * ct_1 + it * gt
        ot = torch.sigmoid(self.Wxo(x) + self.Who(ht_1))
        Ht = ot * torch.tanh(Ct)
        
        return Ct, Ht
    
    def init_hidden(self, batch_size, image_size):
        """Initialize hidden and cell states"""
        return (
            torch.zeros(batch_size, self.feature_channels, image_size, image_size, device=self.device),
            torch.zeros(batch_size, self.feature_channels, image_size, image_size, device=self.device)
        )


# ConvLSTM Network (adapted from PI-ConvTF's SAConvLSTM)
class ConvLSTM(nn.Module):
    """ConvLSTM Network for sequential surface prediction"""
    
    def __init__(self, input_channels, feature_channels, kernel_size, stride, padding, device, last_conv, num_layers):
        """
        Parameters:
        -----------
        input_channels : int
            Input tensor channel dimension
        feature_channels : list
            List of filter numbers per layer
        kernel_size : list
            List of kernel_sizes to apply convolution operations on cell inputs per layer
        stride : list
            List of stride sizes to apply convolution operations on cell inputs per layer
        padding : list
            List of padding sizes to apply convolution operations on cell inputs per layer
        device : str
            Device to train model on ('cuda' or 'cpu')
        last_conv : list
            [last conv kernel size, last conv stride, last conv padding]
        num_layers : int
            Number of ConvLSTM layers
        """
        super(ConvLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        self.num_layers = num_layers
        last_kernel, last_stride, last_padding = last_conv[0], last_conv[1], last_conv[2]
        
        # Create list of ConvLSTM cells
        cell_list = []
        for i in range(num_layers):
            in_chan = input_channels if i == 0 else feature_channels[i-1]
            cell_list.append(
                ConvLSTMCell(in_chan, feature_channels[i], kernel_size[i], stride[i], padding[i], device)
            )
        
        self.cell_list = nn.ModuleList(cell_list)
        self.final_convout = nn.Conv2d(
            in_channels=feature_channels[num_layers-1],
            out_channels=1,
            kernel_size=last_kernel,
            stride=last_stride,
            padding=last_padding
        )
    
    def forward(self, x):
        """Forward pass through ConvLSTM network"""
        # Calculate hidden sizes after each layer's convolutions
        hidden_sizes = [
            math.floor((x.size(dim=-1) + 2*self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        ]
        for i in range(self.num_layers-1):
            hidden_sizes.append(
                math.floor((hidden_sizes[i] + 2*self.padding[i+1] - self.kernel_size[i+1]) / self.stride[i+1] + 1)
            )
        
        # Initialize hidden states
        hidden_states = self._init_hidden(x.size(dim=0), hidden_sizes)
        
        seq_len = x.size(dim=1)
        cur_layer_input = x
        
        # Process through each layer
        for layer_idx in range(self.num_layers):
            c, h = hidden_states[layer_idx]
            output_per_layer = []
            
            # Process each timestep in sequence
            for t in range(seq_len):
                c, h = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :], others=(c, h))
                output_per_layer.append(h)
            
            # Stack outputs for this layer
            layer_output = torch.stack(output_per_layer, dim=1)
            cur_layer_input = layer_output
        
        # Use last timestep's output from last layer
        out = layer_output[:, -1, :, :, :]
        out = self.final_convout(out)
        
        return out
    
    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states for all layers"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size[i]))
        return init_states


class ConvLSTMModel(BaseModel):
    """
    ConvLSTM model for volatility surface forecasting.
    
    Uses direct forecasting: separate model instance per horizon.
    
    Parameters:
    -----------
    name : str
        Model name
    num_layers : int, default 1
        Number of ConvLSTM layers
    filters : list, default [64]
        List of filter numbers per layer
    kernel_size : list, default [3]
        List of kernel sizes per layer
    strides : list, default [1]
        List of stride sizes per layer
    padding : list, default [0]
        List of padding sizes per layer
    last_conv_kernel : int, default 1
        Last convolution kernel size
    last_conv_stride : int, default 1
        Last convolution stride
    last_conv_padding : int, default 1
        Last convolution padding
    batch_size : int, default 32
        Training batch size
    epochs : int, default 100
        Number of training epochs
    learning_rate : float, default 0.001
        Initial learning rate
    lr_patience : int, default 5
        Patience for learning rate reduction
    lr_factor : float, default 0.5
        Factor for learning rate reduction
    device : str, optional
        Device to use ('cuda' or 'cpu'). If None, auto-detects.
    """
    
    def __init__(self, name="convlstm", num_layers=1, filters=[64], kernel_size=[3], strides=[1], padding=[0],
                 last_conv_kernel=1, last_conv_stride=1, last_conv_padding=1,
                 batch_size=32, epochs=100, learning_rate=0.001,
                 lr_patience=5, lr_factor=0.5, device=None):
        super().__init__(name=name)
        
        self.num_layers = num_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.last_conv_kernel = last_conv_kernel
        self.last_conv_stride = last_conv_stride
        self.last_conv_padding = last_conv_padding
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.requires_normalization = False  # PI-ConvTF doesn't normalize volatility data
        
        # Will be set during fitting
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.n_tau = None
        self.n_m = None
        self.fit_horizon = None  # Store horizon this instance was trained for
        self.best_model_state = None  # Store best model state
        self.best_val_loss = float('inf')
    
    def fit(self, X_train, y_train=None, **kwargs):
        """
        Train ConvLSTM model.
        
        Parameters:
        -----------
        X_train : array, shape (n_samples, context_length, n_tau, n_m)
            Training sequences
        y_train : array, shape (n_samples, n_tau, n_m)
            Training targets at the horizon specified in kwargs['horizon']
        **kwargs : dict
            Must include 'horizon' (int). May include 'X_val', 'y_val' for validation.
        """
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train must be provided and non-empty")
        
        if y_train is None:
            raise ValueError("y_train must be provided for direct forecasting")
        
        n_samples, context_length, n_tau, n_m = X_train.shape
        self.n_tau = n_tau
        self.n_m = n_m
        self.fit_horizon = kwargs.get('horizon', 1)
        
        # Reshape data: (n_samples, context_length, n_tau, n_m) -> (n_samples, context_length, 1, n_tau, n_m)
        # Add channel dimension
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)  # Add channel dim
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Add channel dim for consistency
        
        # Get validation data if available
        X_val = kwargs.get('X_val', None)
        y_val = kwargs.get('y_val', None)
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).unsqueeze(2)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        else:
            X_val_tensor = None
            y_val_tensor = None
        
        # Move to device
        X_train_tensor = X_train_tensor.to(self.device)
        y_train_tensor = y_train_tensor.to(self.device)
        if X_val_tensor is not None:
            X_val_tensor = X_val_tensor.to(self.device)
            y_val_tensor = y_val_tensor.to(self.device)
        
        # Initialize model
        self.model = ConvLSTM(
            input_channels=1,  # Single channel (volatility)
            feature_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self.padding,
            device=self.device,
            last_conv=[self.last_conv_kernel, self.last_conv_stride, self.last_conv_padding],
            num_layers=self.num_layers
        ).to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            threshold_mode='abs',
            threshold=0.01
        )
        
        # Loss function (RMSE Loss)
        # RMSE = sqrt(mean((pred - target)^2))
        class RMSELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
            
            def forward(self, pred, target):
                return torch.sqrt(self.mse(pred, target))
        
        loss_func = RMSELoss()
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        if X_val_tensor is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_x)
                loss = loss_func(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        predictions = self.model(batch_x)
                        loss = loss_func(predictions, batch_y)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                
                # Save best model
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_model_state = self.model.state_dict().copy()
                
                # Update learning rate
                self.scheduler.step(avg_val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"{self.name} Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                # No validation set - just save current model
                if avg_train_loss < self.best_val_loss:
                    self.best_val_loss = avg_train_loss
                    self.best_model_state = self.model.state_dict().copy()
                
                if (epoch + 1) % 10 == 0:
                    print(f"{self.name} Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.6f}")
        
        # Load best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        print(f"{self.name} training complete. Best {'val' if val_loader is not None else 'train'} loss: {self.best_val_loss:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.predict_horizon(X, horizon=self.fit_horizon if self.fit_horizon is not None else 1)
    
    def predict_horizon(self, X, horizon=1):
        """
        Make predictions for given horizon.
        
        Parameters:
        -----------
        X : array, shape (n_samples, context_length, n_tau, n_m)
            Input sequences
        horizon : int
            Prediction horizon (must match fit_horizon for direct forecasting)
        
        Returns:
        --------
        predictions : array, shape (n_samples, n_tau, n_m)
            Predicted surfaces
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if horizon != self.fit_horizon:
            raise ValueError(
                f"This model instance was trained for horizon={self.fit_horizon}, "
                f"but asked to predict for horizon={horizon}. "
                "For direct forecasting, each model instance is specific to a horizon."
            )
        
        self.model.eval()
        
        # Reshape input: add channel dimension
        X_tensor = torch.FloatTensor(X).unsqueeze(2)  # (n_samples, context_length, 1, n_tau, n_m)
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)  # (n_samples, 1, n_tau, n_m)
        
        # Remove channel dimension and convert to numpy
        predictions = predictions.squeeze(1).cpu().numpy()  # (n_samples, n_tau, n_m)
        
        return predictions
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        if self.model is None:
            raise ValueError("Model not initialized. Cannot save checkpoint.")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'best_val_loss': self.best_val_loss,
            'fit_horizon': self.fit_horizon,
            'n_tau': self.n_tau,
            'n_m': self.n_m,
            'hyperparameters': {
                'num_layers': self.num_layers,
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'last_conv_kernel': self.last_conv_kernel,
                'last_conv_stride': self.last_conv_stride,
                'last_conv_padding': self.last_conv_padding,
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore hyperparameters
        hp = checkpoint['hyperparameters']
        self.num_layers = hp['num_layers']
        self.filters = hp['filters']
        self.kernel_size = hp['kernel_size']
        self.strides = hp['strides']
        self.padding = hp['padding']
        self.last_conv_kernel = hp['last_conv_kernel']
        self.last_conv_stride = hp['last_conv_stride']
        self.last_conv_padding = hp['last_conv_padding']
        
        # Reinitialize model with restored hyperparameters
        self.model = ConvLSTM(
            input_channels=1,
            feature_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self.padding,
            device=self.device,
            last_conv=[self.last_conv_kernel, self.last_conv_stride, self.last_conv_padding],
            num_layers=self.num_layers
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.fit_horizon = checkpoint['fit_horizon']
        self.n_tau = checkpoint['n_tau']
        self.n_m = checkpoint['n_m']
        self.is_fitted = True
        
        print(f"Checkpoint loaded from {filepath}")
