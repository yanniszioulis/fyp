"""
SA-ConvLSTM model for volatility surface forecasting with Self-Attention Memory.
Based on PI-ConvTF's SAConvLSTM implementation.
Uses RMSE loss for training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import os
from typing import Optional, Tuple

from models.base_model import BaseModel


# Self-Attention Memory Module (from PI-ConvTF)
class SelfAttentionMemory(nn.Module):
    """Self-Attention Memory module for capturing long-range dependencies"""
    
    def __init__(self, input_channels, inter_channels):
        super(SelfAttentionMemory, self).__init__()
        
        self.input_channels = input_channels
        self.inter_channels = inter_channels
        
        # Query, Key, Value projections for hidden state
        self.Wq = nn.Conv2d(in_channels=input_channels, out_channels=inter_channels, kernel_size=1)
        self.Whk = nn.Conv2d(in_channels=input_channels, out_channels=inter_channels, kernel_size=1)
        self.Whv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        
        # Key, Value projections for memory
        self.Wmk = nn.Conv2d(in_channels=input_channels, out_channels=inter_channels, kernel_size=1)
        self.Wmv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        
        # Combine hidden and memory attention
        self.Wz = nn.Conv2d(in_channels=2*input_channels, out_channels=input_channels, kernel_size=1)
        
        # Memory update gates
        self.Wmzo = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmho = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmzg = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmhg = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmzi = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.Wmhi = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass of Self-Attention Memory"""
        ht, mt_1 = x  # Hidden state and previous memory
        feature_map_size = ht.size(dim=-1)
        
        # Compute Query from hidden state
        Q = self.Wq(ht)
        
        # Compute Keys and Values for hidden state and memory
        Kh, Vh = self.Whk(ht), self.Whv(ht)  # Hidden state K, V
        Km, Vm = self.Wmk(mt_1), self.Wmv(mt_1)  # Memory K, V
        
        # Flatten spatial dimensions for attention computation
        Q = torch.flatten(Q, start_dim=-2)
        QT = torch.transpose(Q, -2, -1)
        Kh = torch.flatten(Kh, start_dim=-2)
        Vh = torch.flatten(Vh, start_dim=-2)
        Km = torch.flatten(Km, start_dim=-2)
        Vm = torch.flatten(Vm, start_dim=-2)
        
        # Compute attention scores
        Ah = F.softmax(torch.matmul(QT, Kh), dim=-1)  # Attention over hidden state
        Am = F.softmax(torch.matmul(QT, Km), dim=-1)  # Attention over memory
        
        # Compute attention outputs
        Zh = torch.matmul(Vh, torch.transpose(Ah, -2, -1))
        Zm = torch.matmul(Vm, torch.transpose(Am, -2, -1))
        
        # Combine hidden and memory attention
        Z = self.Wz(torch.cat((Zh, Zm), dim=-2).view(
            ht.size(dim=0), ht.size(dim=1)*2, feature_map_size, feature_map_size))
        
        # Update memory using LSTM-like gates
        it = torch.sigmoid(self.Wmzi(Z) + self.Wmhi(ht))  # Input gate
        gt = torch.tanh(self.Wmzg(Z) + self.Wmhg(ht))     # Candidate memory
        Mt = (1 - it) * mt_1 + it * gt  # Update memory
        
        # Output gate for hidden state
        ot = torch.sigmoid(self.Wmzo(Z) + self.Wmho(ht))
        Ht = ot * Mt
        
        return Ht, Mt


# SA-ConvLSTM Cell with Self-Attention Memory
class SAConvLSTMCell(nn.Module):
    """SA-ConvLSTM Cell with Self-Attention Memory module"""
    
    def __init__(self, input_channels, feature_channels, inter_channels, kernel_size, stride, padding, device):
        super(SAConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.inter_channels = inter_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        
        # Input-to-state convolutions (same as ConvLSTM)
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
        
        # Self-Attention Memory module (if inter_channels is not 'None')
        if self.inter_channels != 'None' and self.inter_channels is not None:
            self.SAM = SelfAttentionMemory(input_channels=feature_channels, inter_channels=inter_channels)
    
    def forward(self, x, others):
        """Forward pass of SA-ConvLSTM cell"""
        ct_1, ht_1, mt_1 = others  # Cell, hidden, and memory state
        
        # Standard ConvLSTM gates
        it = torch.sigmoid(self.Wxi(x) + self.Whi(ht_1))
        ft = torch.sigmoid(self.Wxf(x) + self.Whf(ht_1))
        gt = torch.tanh(self.Wxg(x) + self.Whg(ht_1))
        
        Ct = ft * ct_1 + it * gt
        ot = torch.sigmoid(self.Wxo(x) + self.Who(ht_1))
        Ht = ot * torch.tanh(Ct)
        
        # Apply Self-Attention Memory if enabled
        if self.inter_channels != 'None' and self.inter_channels is not None:
            Ht, Mt = self.SAM((Ht, mt_1))
        else:
            Mt = mt_1  # No memory update if SA is disabled
        
        return Ct, Ht, Mt
    
    def init_hidden(self, batch_size, image_size):
        """Initialize hidden, cell, and memory states with zeros"""
        return (
            torch.zeros(batch_size, self.feature_channels, image_size, image_size, device=self.device),  # Cell state
            torch.zeros(batch_size, self.feature_channels, image_size, image_size, device=self.device),  # Hidden state
            torch.zeros(batch_size, self.feature_channels, image_size, image_size, device=self.device)   # Memory state
        )


# SA-ConvLSTM Network (adapted from PI-ConvTF's SAConvLSTM)
class SAConvLSTM(nn.Module):
    """SA-ConvLSTM Network for sequential surface prediction with Self-Attention Memory"""
    
    def __init__(self, input_channels, feature_channels, inter_channels, kernel_size, stride, padding, device, last_conv, num_layers):
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
        super(SAConvLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.inter_channels = inter_channels
        
        # Method to create timestep gradient hooks (will be set by SAConvLSTMModel)
        self._create_timestep_grad_hook = None
        self.feature_channels = feature_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        self.num_layers = num_layers
        last_kernel, last_stride, last_padding = last_conv[0], last_conv[1], last_conv[2]
        
        # Create list of SA-ConvLSTM cells
        cell_list = []
        for i in range(num_layers):
            in_chan = input_channels if i == 0 else feature_channels[i-1]
            inter_chan = inter_channels[i] if isinstance(inter_channels, list) else inter_channels
            cell_list.append(
                SAConvLSTMCell(in_chan, feature_channels[i], inter_chan, kernel_size[i], stride[i], padding[i], device)
            )
        
        self.cell_list = nn.ModuleList(cell_list)
        self.final_convout = nn.Conv2d(
            in_channels=feature_channels[num_layers-1],
            out_channels=1,
            kernel_size=last_kernel,
            stride=last_stride,
            padding=last_padding
        )
        # Initialize final layer bias to positive values (volatility is always positive)
        # This helps the model start in the right range
        if self.final_convout.bias is not None:
            nn.init.constant_(self.final_convout.bias, 0.5)  # Start with positive bias
    
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
        
        # Initialize hidden states (including memory for SA-ConvLSTM)
        hidden_states = self._init_hidden(x.size(dim=0), hidden_sizes)
        
        seq_len = x.size(dim=1)
        cur_layer_input = x
        
        # Store hidden states at each timestep for gradient diagnostics
        if hasattr(self, '_store_hidden_states'):
            self._hidden_states_by_timestep = [[] for _ in range(seq_len)]
        
        # Store timestep-wise hidden states for gradient tracking
        if hasattr(self, '_store_timestep_gradients'):
            self._h_timesteps = []  # Store h at each timestep for gradient hooks
        
        # Process through each layer
        for layer_idx in range(self.num_layers):
            c, h, m = hidden_states[layer_idx]  # Cell, hidden, and memory states
            output_per_layer = []
            
            # Reset gate activations for this layer if storing
            if hasattr(self, '_store_gates'):
                self.cell_list[layer_idx]._gate_activations = []
            
            # Track cell states for diagnostics (for last layer only)
            if hasattr(self, '_store_cell_states') and layer_idx == self.num_layers - 1:
                if not hasattr(self, '_cell_states_by_timestep'):
                    self._cell_states_by_timestep = []
            
            # Process each timestep in sequence
            for t in range(seq_len):
                c, h, m = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :], others=(c, h, m))
                output_per_layer.append(h)
                
                # Store hidden states for gradient tracking
                if hasattr(self, '_store_hidden_states'):
                    self._hidden_states_by_timestep[t].append((c.detach(), h.detach()))
                
                # Store cell states for diagnostics (for last layer only)
                if hasattr(self, '_store_cell_states') and layer_idx == self.num_layers - 1:
                    if t < len(self._cell_states_by_timestep):
                        self._cell_states_by_timestep[t] = c.detach()
                    else:
                        self._cell_states_by_timestep.append(c.detach())
                
                # Store h at each timestep for gradient hooks (for last layer only)
                # Use retain_grad() to keep gradients for intermediate tensors
                if hasattr(self, '_store_timestep_gradients') and layer_idx == self.num_layers - 1:
                    if not hasattr(self, '_h_timesteps'):
                        self._h_timesteps = []
                    
                    # Store h and retain gradient for this tensor
                    # This allows us to track gradients through intermediate values
                    if t < len(self._h_timesteps):
                        self._h_timesteps[t] = h
                    else:
                        self._h_timesteps.append(h)
                    
                    # Retain gradient so we can access it after backward pass
                    # This is necessary for intermediate tensors in the computation graph
                    if h.requires_grad:
                        h.retain_grad()
                    
                    # Register backward hook to capture gradients when they're computed
                    # Only register if h requires gradient (part of computation graph)
                    if hasattr(self, '_create_timestep_grad_hook') and self._create_timestep_grad_hook is not None:
                        if h.requires_grad:
                            if not hasattr(self, '_h_timestep_hooks'):
                                self._h_timestep_hooks = []
                            # Register backward hook (called during backward pass)
                            hook_handle = h.register_hook(self._create_timestep_grad_hook(t))
                            if t < len(self._h_timestep_hooks):
                                self._h_timestep_hooks[t] = hook_handle
                            else:
                                self._h_timestep_hooks.append(hook_handle)
            
            # Stack outputs for this layer
            layer_output = torch.stack(output_per_layer, dim=1)
            cur_layer_input = layer_output
            
            # Store gate activations per layer
            if hasattr(self, '_store_gates'):
                if not hasattr(self, '_all_gate_activations'):
                    self._all_gate_activations = []
                self._all_gate_activations.append(self.cell_list[layer_idx]._gate_activations)
        
        # Use last timestep's output from last layer
        out = layer_output[:, -1, :, :, :]
        out = self.final_convout(out)
        
        # NOTE: No ReLU here! After z-score normalization, values can be negative (mean≈0).
        # The model needs to predict freely in normalized space. We'll apply ReLU after
        # denormalization if needed to ensure non-negative volatility in original scale.
        
        return out
    
    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states for all layers"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size[i]))
        return init_states


class SAConvLSTMModel(BaseModel):
    """
    SA-ConvLSTM model for volatility surface forecasting with Self-Attention Memory.
    
    Uses direct forecasting: separate model instance per horizon.
    
    Parameters:
    -----------
    name : str
        Model name
    num_layers : int, default 1
        Number of SA-ConvLSTM layers
    filters : list, default [64]
        List of filter numbers per layer
    inter_channels : list or int or 'None', default 32
        List of inter_channels for Self-Attention Memory per layer.
        If single int, same value for all layers. If 'None', disables self-attention.
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
    patience : int, default 10
        Early stopping patience (number of epochs to wait for improvement)
    min_delta : float, default 0.0
        Minimum change to qualify as an improvement
    device : str, optional
        Device to use ('cuda' or 'cpu'). If None, auto-detects.
    """
    
    def __init__(self, name="sa_convlstm", num_layers=1, filters=[64], inter_channels=32,
                 kernel_size=[3], strides=[1], padding=[0],
                 last_conv_kernel=1, last_conv_stride=1, last_conv_padding=1,
                 batch_size=32, epochs=100, learning_rate=0.001,
                 lr_patience=5, lr_factor=0.5, patience=10, min_delta=0.0, device=None):
        super().__init__(name=name)
        
        self.num_layers = num_layers
        self.filters = filters
        # Handle inter_channels: convert single value to list, or keep as list
        if isinstance(inter_channels, (int, str)) and inter_channels != 'None':
            self.inter_channels = [inter_channels] * num_layers
        elif inter_channels == 'None' or inter_channels is None:
            self.inter_channels = ['None'] * num_layers
        else:
            self.inter_channels = inter_channels
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
        self.patience = patience
        self.min_delta = min_delta
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.requires_normalization = False  # PI-ConvTF doesn't normalize volatility data
        
        # Validate that hyperparameter lists match num_layers
        if len(self.filters) != self.num_layers:
            raise ValueError(
                f"filters list length ({len(self.filters)}) must match num_layers ({self.num_layers}). "
                f"Got filters={self.filters}, num_layers={self.num_layers}"
            )
        if len(self.kernel_size) != self.num_layers:
            raise ValueError(
                f"kernel_size list length ({len(self.kernel_size)}) must match num_layers ({self.num_layers}). "
                f"Got kernel_size={self.kernel_size}, num_layers={self.num_layers}"
            )
        if len(self.strides) != self.num_layers:
            raise ValueError(
                f"strides list length ({len(self.strides)}) must match num_layers ({self.num_layers}). "
                f"Got strides={self.strides}, num_layers={self.num_layers}"
            )
        if len(self.padding) != self.num_layers:
            raise ValueError(
                f"padding list length ({len(self.padding)}) must match num_layers ({self.num_layers}). "
                f"Got padding={self.padding}, num_layers={self.num_layers}"
            )
        if isinstance(self.inter_channels, list) and len(self.inter_channels) != self.num_layers:
            raise ValueError(
                f"inter_channels list length ({len(self.inter_channels)}) must match num_layers ({self.num_layers}). "
                f"Got inter_channels={self.inter_channels}, num_layers={self.num_layers}"
            )
        
        # Will be set during fitting
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.n_tau = None
        self.n_m = None
        self.fit_horizon = None  # Store horizon this instance was trained for
        self.best_model_state = None  # Store best model state
        self.best_val_loss = float('inf')
        
        # Normalization statistics (per-window z-score normalization)
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
        # Gradient diagnostics
        self.gradient_diagnostics_enabled = False  # Disabled by default
        self.gradient_stats = {}  # Store gradient statistics
        self.gradient_hooks = []  # Store hook handles
        
    def _setup_gradient_diagnostics(self):
        """Register hooks to track gradients at critical points"""
        if not self.gradient_diagnostics_enabled:
            return
            
        # Enable gate and hidden state storage in forward pass
        self.model._store_gates = True
        self.model._store_hidden_states = True
        self.model._store_timestep_gradients = True  # Enable timestep-wise gradient tracking
        self.model._store_cell_states = True  # Enable cell state tracking for diagnostics
        
        # Register hooks for final conv layer
        def make_grad_hook(name):
            def hook(grad):
                if grad is not None:
                    if name not in self.gradient_stats:
                        self.gradient_stats[name] = []
                    self.gradient_stats[name].append({
                        'norm': grad.norm().item(),
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'min': grad.min().item(),
                        'max': grad.max().item(),
                        'sparsity': (grad.abs() < 1e-6).float().mean().item()
                    })
                return grad
            return hook
        
        # Hook final conv layer
        self.model.final_convout.weight.register_hook(make_grad_hook('final_conv_weight'))
        if self.model.final_convout.bias is not None:
            self.model.final_convout.bias.register_hook(make_grad_hook('final_conv_bias'))
        
        # Hook ConvLSTM cells (input-to-state and hidden-to-state convolutions)
        for layer_idx in range(self.num_layers):
            cell = self.model.cell_list[layer_idx]
            
            # Hook input-to-state gates
            cell.Wxi.weight.register_hook(make_grad_hook(f'layer_{layer_idx}_Wxi'))
            cell.Wxf.weight.register_hook(make_grad_hook(f'layer_{layer_idx}_Wxf'))
            cell.Wxg.weight.register_hook(make_grad_hook(f'layer_{layer_idx}_Wxg'))
            cell.Wxo.weight.register_hook(make_grad_hook(f'layer_{layer_idx}_Wxo'))
            
            # Hook hidden-to-state gates (recurrent connections)
            cell.Whi.weight.register_hook(make_grad_hook(f'layer_{layer_idx}_Whi'))
            cell.Whf.weight.register_hook(make_grad_hook(f'layer_{layer_idx}_Whf'))
            cell.Whg.weight.register_hook(make_grad_hook(f'layer_{layer_idx}_Whg'))
            cell.Who.weight.register_hook(make_grad_hook(f'layer_{layer_idx}_Who'))
        
        # Register hooks for timestep-wise hidden state gradients
        # This will be used in forward pass to hook hidden states at each timestep
        def make_timestep_grad_hook(t):
            def hook(grad):
                if grad is not None:
                    key = f'h_timestep_{t}'
                    if key not in self.gradient_stats:
                        self.gradient_stats[key] = []
                    self.gradient_stats[key].append({
                        'norm': grad.norm().item(),
                        'mean': grad.mean().item(),
                        'std': grad.std().item()
                    })
                return grad
            return hook
        
        # Set hook creator on model for use in forward pass
        self.model._create_timestep_grad_hook = make_timestep_grad_hook
    
    def _collect_gradient_diagnostics(self, epoch, batch_idx=0):
        """Collect and print gradient diagnostic statistics"""
        if not self.gradient_diagnostics_enabled or len(self.gradient_stats) == 0:
            return
        
        # Collect gate saturation stats and activation distributions if available
        gate_stats = {}
        forget_gate_distribution = []
        if hasattr(self.model, '_all_gate_activations') and len(self.model._all_gate_activations) > 0:
            layer_activations = self.model._all_gate_activations[0]  # First (and only) layer
            for t, gates in enumerate(layer_activations):
                it_mean = gates['it'].mean().item()
                ft_mean = gates['ft'].mean().item()
                ot_mean = gates['ot'].mean().item()
                
                # Calculate saturation (gates close to 0 or 1)
                it_sat = ((gates['it'] < 0.01) | (gates['it'] > 0.99)).float().mean().item()
                ft_sat = ((gates['ft'] < 0.01) | (gates['ft'] > 0.99)).float().mean().item()
                ot_sat = ((gates['ot'] < 0.01) | (gates['ot'] > 0.99)).float().mean().item()
                
                # Calculate additional statistics
                it_std = gates['it'].std().item()
                ft_std = gates['ft'].std().item()
                ot_std = gates['ot'].std().item()
                
                # Collect forget gate distribution
                forget_gate_distribution.append(gates['ft'].cpu().numpy().flatten())
                
                gate_stats[t] = {
                    'it_mean': it_mean, 'it_std': it_std, 'it_sat': it_sat,
                    'ft_mean': ft_mean, 'ft_std': ft_std, 'ft_sat': ft_sat,
                    'ot_mean': ot_mean, 'ot_std': ot_std, 'ot_sat': ot_sat,
                    'it_min': gates['it'].min().item(), 'it_max': gates['it'].max().item(),
                    'ft_min': gates['ft'].min().item(), 'ft_max': gates['ft'].max().item(),
                    'ot_min': gates['ot'].min().item(), 'ot_max': gates['ot'].max().item()
                }
        
        # Print diagnostic report
        # print(f"\n{'='*70}")
        # print(f"GRADIENT DIAGNOSTIC REPORT - Epoch {epoch}, Batch {batch_idx}")
        # print(f"{'='*70}")
        
        # Final conv layer stats
        if 'final_conv_weight' in self.gradient_stats:
            stats = self.gradient_stats['final_conv_weight'][-1]
            # print(f"\nFinal Conv Layer:")
            # print(f"  Gradient norm: {stats['norm']:.6e}")
            # print(f"  Gradient mean: {stats['mean']:.6e}, std: {stats['std']:.6e}")
            # print(f"  Gradient range: [{stats['min']:.6e}, {stats['max']:.6e}]")
            # print(f"  Sparsity (|grad| < 1e-6): {stats['sparsity']:.2%}")
        
        # Layer-wise gradient stats (compare input-to-state vs hidden-to-state)
        print(f"\nConvLSTM Layer 0 (Recurrent Connections - Hidden-to-State):")
        for gate_name in ['Whi', 'Whf', 'Whg', 'Who']:
            key = f'layer_0_{gate_name}'
            if key in self.gradient_stats:
                stats = self.gradient_stats[key][-1]
                print(f"  {gate_name} (recurrent): norm={stats['norm']:.6e}, sparsity={stats['sparsity']:.2%}")
        
        print(f"\nConvLSTM Layer 0 (Input-to-State):")
        for gate_name in ['Wxi', 'Wxf', 'Wxg', 'Wxo']:
            key = f'layer_0_{gate_name}'
            if key in self.gradient_stats:
                stats = self.gradient_stats[key][-1]
                print(f"  {gate_name} (input): norm={stats['norm']:.6e}, sparsity={stats['sparsity']:.2%}")
        
        # Gate saturation by timestep (detailed)
        if gate_stats:
            print(f"\nGate Saturation by Timestep (Layer 0):")
            print(f"  {'Timestep':<10} {'Input Gate':<30} {'Forget Gate':<30} {'Output Gate':<30}")
            print(f"  {'-'*10} {'-'*30} {'-'*30} {'-'*30}")
            # Show every timestep for detailed analysis (or sample if too many)
            timesteps_to_show = sorted(gate_stats.keys())
            if len(timesteps_to_show) > 25:
                timesteps_to_show = timesteps_to_show[::len(timesteps_to_show)//20]  # Show ~20 timesteps
            for t in timesteps_to_show:
                gs = gate_stats[t]
                print(f"  {t:<10} "
                      f"mean={gs['it_mean']:.3f}±{gs['it_std']:.3f}, sat={gs['it_sat']:.1%}, range=[{gs['it_min']:.3f},{gs['it_max']:.3f}]  "
                      f"mean={gs['ft_mean']:.3f}±{gs['ft_std']:.3f}, sat={gs['ft_sat']:.1%}, range=[{gs['ft_min']:.3f},{gs['ft_max']:.3f}]  "
                      f"mean={gs['ot_mean']:.3f}±{gs['ot_std']:.3f}, sat={gs['ot_sat']:.1%}, range=[{gs['ot_min']:.3f},{gs['ot_max']:.3f}]")
            
            # Overall forget gate statistics
            if forget_gate_distribution:
                import numpy as np
                all_forget = np.concatenate(forget_gate_distribution)
                print(f"\nForget Gate Activation Distribution (All Timesteps):")
                print(f"  Mean: {all_forget.mean():.4f}, Std: {all_forget.std():.4f}")
                print(f"  Min: {all_forget.min():.4f}, Max: {all_forget.max():.4f}")
                print(f"  Saturation (<0.01 or >0.99): {(all_forget < 0.01).sum() / len(all_forget):.2%} near 0, "
                      f"{(all_forget > 0.99).sum() / len(all_forget):.2%} near 1, "
                      f"{(all_forget < 0.01).sum() / len(all_forget) + (all_forget > 0.99).sum() / len(all_forget):.2%} total")
                print(f"  Percentiles: 5%={np.percentile(all_forget, 5):.4f}, "
                      f"25%={np.percentile(all_forget, 25):.4f}, "
                      f"50%={np.percentile(all_forget, 50):.4f}, "
                      f"75%={np.percentile(all_forget, 75):.4f}, "
                      f"95%={np.percentile(all_forget, 95):.4f}")
        
        # Compare gradient magnitudes: recurrent vs input
        if 'layer_0_Whi' in self.gradient_stats and 'layer_0_Wxi' in self.gradient_stats:
            rec_grad = self.gradient_stats['layer_0_Whi'][-1]['norm']
            in_grad = self.gradient_stats['layer_0_Wxi'][-1]['norm']
            ratio = rec_grad / in_grad if in_grad > 0 else float('inf')
            print(f"\nRecurrent vs Input Gradient Ratio: {ratio:.6f}")
            print(f"  (If << 1, recurrent connections getting much weaker gradients)")
        
        # Cell state preservation analysis (check if information from t0 is preserved)
        print(f"\nCell State Preservation Analysis:")
        print(f"  (Checking if information from t0 is preserved in cell state)")
        if hasattr(self.model, '_cell_states_by_timestep') and len(self.model._cell_states_by_timestep) > 1:
            import torch
            C0 = self.model._cell_states_by_timestep[0]  # First timestep cell state
            C_last = self.model._cell_states_by_timestep[-1]  # Last timestep cell state
            seq_len = len(self.model._cell_states_by_timestep)
            
            # Compute statistics
            C0_mean = C0.mean().item()
            C0_std = C0.std().item()
            C_last_mean = C_last.mean().item()
            C_last_std = C_last.std().item()
            
            # Compute similarity metrics
            C_diff = (C_last - C0).abs()
            C_diff_mean = C_diff.mean().item()
            C_diff_max = C_diff.max().item()
            
            # Cosine similarity
            C0_flat = C0.flatten()
            C_last_flat = C_last.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(C0_flat.unsqueeze(0), C_last_flat.unsqueeze(0)).item()
            
            # Relative change (how much changed relative to initial magnitude)
            C0_norm = C0.norm().item()
            C_last_norm = C_last.norm().item()
            relative_change = C_diff.norm().item() / (C0_norm + 1e-8)
            
            print(f"  Sequence Length: {seq_len} timesteps")
            print(f"  C0 (first timestep): mean={C0_mean:.6f}, std={C0_std:.6f}, norm={C0_norm:.6f}")
            print(f"  C{seq_len-1} (last timestep): mean={C_last_mean:.6f}, std={C_last_std:.6f}, norm={C_last_norm:.6f}")
            print(f"  Absolute Difference (C_last - C0): mean={C_diff_mean:.6f}, max={C_diff_max:.6f}")
            print(f"  Cosine Similarity: {cosine_sim:.6f} (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
            print(f"  Relative Change: {relative_change:.6f} (0.0 = no change, 1.0 = completely different)")
            
            # Interpretation
            if cosine_sim > 0.9 and relative_change < 0.1:
                print(f"  → Cell state HIGHLY PRESERVED: Information from t0 is still present at t{seq_len-1}")
                print(f"    (This suggests old information is being preserved through the sequence)")
            elif cosine_sim > 0.7 and relative_change < 0.3:
                print(f"  → Cell state MODERATELY PRESERVED: Some information from t0 remains")
            elif cosine_sim > 0.5:
                print(f"  → Cell state PARTIALLY PRESERVED: Some similarity but significant change")
            else:
                print(f"  → Cell state LARGELY OVERWRITTEN: Information from t0 is mostly lost")
                print(f"    (High gradient at t0 might be misleading - gradients flow back but info is gone)")
            
            # Check intermediate states to see where change happens
            if seq_len > 10:
                # Sample a few intermediate states
                sample_indices = [seq_len // 4, seq_len // 2, 3 * seq_len // 4]
                print(f"\n  Intermediate Cell State Evolution:")
                print(f"    {'Timestep':<10} {'Mean':<15} {'Std':<15} {'Norm':<15} {'Cos Sim vs C0':<15}")
                print(f"    {'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
                print(f"    {0:<10} {C0_mean:<15.6f} {C0_std:<15.6f} {C0_norm:<15.6f} {1.000000:<15.6f}")
                for idx in sample_indices:
                    C_mid = self.model._cell_states_by_timestep[idx]
                    C_mid_mean = C_mid.mean().item()
                    C_mid_std = C_mid.std().item()
                    C_mid_norm = C_mid.norm().item()
                    C_mid_sim = torch.nn.functional.cosine_similarity(
                        C0_flat.unsqueeze(0), C_mid.flatten().unsqueeze(0)
                    ).item()
                    print(f"    {idx:<10} {C_mid_mean:<15.6f} {C_mid_std:<15.6f} {C_mid_norm:<15.6f} {C_mid_sim:<15.6f}")
                print(f"    {seq_len-1:<10} {C_last_mean:<15.6f} {C_last_std:<15.6f} {C_last_norm:<15.6f} {cosine_sim:<15.6f}")
        else:
            print(f"  Cell states not tracked (enable _store_cell_states flag)")
        
        # Timestep-wise gradient analysis (from registered hooks or retained gradients)
        print(f"\nTimestep-wise Gradient Analysis:")
        print(f"  (Tracking gradients through hidden states at each timestep)")
        timestep_grad_norms = []
        
        # First try to get from registered hooks
        for key in sorted(self.gradient_stats.keys()):
            if key.startswith('h_timestep_'):
                t = int(key.split('_')[-1])
                if len(self.gradient_stats[key]) > 0:
                    grad_norm = self.gradient_stats[key][-1]['norm']
                    timestep_grad_norms.append((t, grad_norm))
        
        # If no hooks, try to get from retained gradients
        if not timestep_grad_norms and hasattr(self.model, '_h_timesteps'):
            for t, h_t in enumerate(self.model._h_timesteps):
                if h_t.grad is not None:
                    grad_norm = h_t.grad.norm().item()
                    timestep_grad_norms.append((t, grad_norm))
        
        if timestep_grad_norms:
            print(f"  {'Timestep':<10} {'Gradient Norm':<20} {'Relative to Last':<20} {'Relative to t0':<20}")
            print(f"  {'-'*10} {'-'*20} {'-'*20} {'-'*20}")
            timestep_grad_norms.sort(key=lambda x: x[0])
            last_grad = timestep_grad_norms[-1][1] if timestep_grad_norms else 1.0
            first_grad = timestep_grad_norms[0][1] if timestep_grad_norms else 1.0
            for t, grad_norm in timestep_grad_norms:
                rel_to_last = grad_norm / last_grad if last_grad > 0 else float('inf')
                rel_to_first = grad_norm / first_grad if first_grad > 0 else float('inf')
                print(f"  {t:<10} {grad_norm:.6e}      {rel_to_last:.6e} ({rel_to_last*100:.2f}%)    {rel_to_first:.6e} ({rel_to_first*100:.2f}%)")
            
            # Calculate gradient decay factor
            if len(timestep_grad_norms) > 1:
                last_idx = timestep_grad_norms[-1][0]
                decay_factor = first_grad / last_grad if last_grad > 0 else 0
                print(f"\n  Gradient Decay Factor (t0/t{last_idx}): {decay_factor:.6e}")
                print(f"  (If << 1, gradients vanishing from early to late timesteps)")
                print(f"  (If >> 1, gradients increasing (unusual))")
                print(f"  (If ≈ 1, gradients stable through time)")
        else:
            print(f"  No timestep-wise gradients collected (hooks may not be registered or gradients not retained)")
        
        print(f"{'='*70}\n")
        
        # Clear stats for next batch (optional - comment out to accumulate)
        # self.gradient_stats = {}
    
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
        
        # # DEBUG: Print input data info (before normalization)
        # print(f"\n  [ConvLSTM.fit] Input data verification (BEFORE normalization):")
        # print(f"    X_train numpy shape: {X_train.shape}, dtype: {X_train.dtype}")
        # print(f"    X_train numpy range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        # print(f"    X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
        # print(f"    y_train numpy shape: {y_train.shape}, dtype: {y_train.dtype}")
        # print(f"    y_train numpy range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        # print(f"    y_train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
        
        # Z-score normalization per window
        # Compute statistics from training data only
        self.X_mean = X_train.mean()
        self.X_std = X_train.std()
        self.y_mean = y_train.mean()
        self.y_std = y_train.std()
        
        # Avoid division by zero
        if self.X_std < 1e-8:
            print(f"    WARNING: X_train std is very small ({self.X_std:.2e}), setting to 1.0")
            self.X_std = 1.0
        if self.y_std < 1e-8:
            print(f"    WARNING: y_train std is very small ({self.y_std:.2e}), setting to 1.0")
            self.y_std = 1.0
        
        # Normalize training data
        X_train_normalized = (X_train - self.X_mean) / self.X_std
        y_train_normalized = (y_train - self.y_mean) / self.y_std
        
        # print(f"\n  [ConvLSTM.fit] Normalization statistics:")
        # print(f"    X_mean: {self.X_mean:.6f}, X_std: {self.X_std:.6f}")
        # print(f"    y_mean: {self.y_mean:.6f}, y_std: {self.y_std:.6f}")
        # print(f"    X_train_normalized range: [{X_train_normalized.min():.4f}, {X_train_normalized.max():.4f}]")
        # print(f"    X_train_normalized mean: {X_train_normalized.mean():.4f}, std: {X_train_normalized.std():.4f}")
        # print(f"    y_train_normalized range: [{y_train_normalized.min():.4f}, {y_train_normalized.max():.4f}]")
        # print(f"    y_train_normalized mean: {y_train_normalized.mean():.4f}, std: {y_train_normalized.std():.4f}")
        
        # Reshape data: (n_samples, context_length, n_tau, n_m) -> (n_samples, context_length, 1, n_tau, n_m)
        # Add channel dimension
        X_train_tensor = torch.FloatTensor(X_train_normalized).unsqueeze(2)  # Add channel dim
        y_train_tensor = torch.FloatTensor(y_train_normalized).unsqueeze(1)  # Add channel dim for consistency
        
        # print(f"    X_train_tensor shape: {X_train_tensor.shape}")
        # print(f"    y_train_tensor shape: {y_train_tensor.shape}")
        
        # Get validation data if available
        X_val = kwargs.get('X_val', None)
        y_val = kwargs.get('y_val', None)
        if X_val is not None and y_val is not None:
            # Normalize validation data using training statistics
            X_val_normalized = (X_val - self.X_mean) / self.X_std
            y_val_normalized = (y_val - self.y_mean) / self.y_std
            X_val_tensor = torch.FloatTensor(X_val_normalized).unsqueeze(2)
            y_val_tensor = torch.FloatTensor(y_val_normalized).unsqueeze(1)
            # print(f"    X_val_tensor shape: {X_val_tensor.shape}")
            # print(f"    y_val_tensor shape: {y_val_tensor.shape}")
            # print(f"    X_val_normalized range: [{X_val_normalized.min():.4f}, {X_val_normalized.max():.4f}]")
            # print(f"    y_val_normalized range: [{y_val_normalized.min():.4f}, {y_val_normalized.max():.4f}]")
        else:
            X_val_tensor = None
            y_val_tensor = None
        
        # Move to device
        X_train_tensor = X_train_tensor.to(self.device)
        y_train_tensor = y_train_tensor.to(self.device)
        if X_val_tensor is not None:
            X_val_tensor = X_val_tensor.to(self.device)
            y_val_tensor = y_val_tensor.to(self.device)
        
        # print(f"    Moved tensors to device: {self.device}")
        # print(f"    X_train_tensor on device range: [{X_train_tensor.min().item():.4f}, {X_train_tensor.max().item():.4f}]")
        # print(f"    y_train_tensor on device range: [{y_train_tensor.min().item():.4f}, {y_train_tensor.max().item():.4f}]")
        # print(f"    (Note: These are normalized values, mean≈0, std≈1)")
        
        # Initialize model
        self.model = SAConvLSTM(
            input_channels=1,  # Single channel (volatility)
            feature_channels=self.filters,
            inter_channels=self.inter_channels,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self.padding,
            device=self.device,
            last_conv=[self.last_conv_kernel, self.last_conv_stride, self.last_conv_padding],
            num_layers=self.num_layers
        ).to(self.device)
        
        # Enable gradient diagnostics
        self._setup_gradient_diagnostics()
        
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
        
        # Training loop with early stopping
        self.best_val_loss = float('inf')
        self.best_model_state = None
        patience_counter = 0
        best_epoch = 0
        last_batch_x = None
        last_batch_y = None
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                # Store last batch for final debug output
                last_batch_x = batch_x
                last_batch_y = batch_y
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_x)
                
                # # Debug: Check shapes on first batch of first epoch
                # if epoch == 0 and len(train_losses) == 0:
                #     print(f"  DEBUG (Epoch 1, Batch 1): predictions shape: {predictions.shape}, batch_y shape: {batch_y.shape}")
                #     print(f"  DEBUG (Epoch 1, Batch 1): predictions min/max: {predictions.min().item():.4f}/{predictions.max().item():.4f}")
                #     print(f"  DEBUG (Epoch 1, Batch 1): batch_y min/max: {batch_y.min().item():.4f}/{batch_y.max().item():.4f}")
                
                
                loss = loss_func(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Collect gradient diagnostics (on first batch of every 10th epoch, or epoch 1)
                if self.gradient_diagnostics_enabled and (epoch % 10 == 0 or epoch == 0) and batch_idx == 0:
                    self._collect_gradient_diagnostics(epoch, batch_idx)
                
                self.optimizer.step()
                
                train_losses.append(loss.item())
                
                # Clear gradient stats after collection to avoid accumulation
                if self.gradient_diagnostics_enabled and (epoch % 10 == 0 or epoch == 0) and batch_idx == 0:
                    # Keep last entry for comparison, but clear older ones
                    for key in self.gradient_stats:
                        if len(self.gradient_stats[key]) > 1:
                            self.gradient_stats[key] = [self.gradient_stats[key][-1]]
            
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
                
                # Check for improvement
                improvement = self.best_val_loss - avg_val_loss
                if improvement > self.min_delta:
                    # Better model found
                    self.best_val_loss = avg_val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    best_epoch = epoch + 1
                else:
                    # No improvement
                    patience_counter += 1
                
                # Update learning rate
                self.scheduler.step(avg_val_loss)
                
                if (epoch + 1) % 5 == 0:
                    print(f"{self.name} Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (Best: {self.best_val_loss:.6f} @ epoch {best_epoch}, Patience: {patience_counter}/{self.patience})")
                
                # Early stopping
                if patience_counter >= self.patience:
                    print(f"\n{self.name} Early stopping triggered at epoch {epoch+1}")
                    print(f"  No improvement for {self.patience} epochs (best val loss: {self.best_val_loss:.6f} @ epoch {best_epoch})")
                    break
            else:
                # No validation set - just save current model
                if avg_train_loss < self.best_val_loss:
                    self.best_val_loss = avg_train_loss
                    self.best_model_state = self.model.state_dict().copy()
                
                # if (epoch + 1) % 5 == 0:
                #     print(f"{self.name} Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.6f}")
        
        # Load best model state (or keep final model for overfitting tests)
        use_final_model = kwargs.get('use_final_model', False)
        if use_final_model:
            print(f"{self.name} training complete. Using final model (train loss: {avg_train_loss:.6f})")
            # Don't load best state, keep final model
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if val_loader is not None:
                print(f"{self.name} training complete. Best val loss: {self.best_val_loss:.6f} @ epoch {best_epoch}")
            else:
                print(f"{self.name} training complete. Best train loss: {self.best_val_loss:.6f}")
        else:
            print(f"{self.name} training complete. Using final model")
        
        # Debug: Show final model predictions on a sample batch
        if last_batch_x is not None and last_batch_y is not None:
            self.model.eval()
            with torch.no_grad():
                final_predictions = self.model(last_batch_x.to(self.device))
        #         print(f"  DEBUG (Final Model): predictions shape: {final_predictions.shape}, batch_y shape: {last_batch_y.shape}")
        #         print(f"  DEBUG (Final Model): predictions min/max: {final_predictions.min().item():.4f}/{final_predictions.max().item():.4f}")
        #         print(f"  DEBUG (Final Model): batch_y min/max: {last_batch_y.min().item():.4f}/{last_batch_y.max().item():.4f}")
        else:
            # Fallback: evaluate on first batch of training data
            self.model.eval()
            with torch.no_grad():
                sample_batch_x, sample_batch_y = next(iter(train_loader))
                sample_batch_x = sample_batch_x.to(self.device)
                sample_batch_y = sample_batch_y.to(self.device)
                final_predictions = self.model(sample_batch_x)
                # print(f"  DEBUG (Final Model): predictions shape: {final_predictions.shape}, batch_y shape: {sample_batch_y.shape}")
                # print(f"  DEBUG (Final Model): predictions min/max: {final_predictions.min().item():.4f}/{final_predictions.max().item():.4f}")
                # print(f"  DEBUG (Final Model): batch_y min/max: {sample_batch_y.min().item():.4f}/{sample_batch_y.max().item():.4f}")
        
        self.is_fitted = True
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
        
        # Normalize input using training statistics
        if self.X_mean is None or self.X_std is None:
            raise ValueError("Model normalization statistics not found. Model may not have been fitted properly.")
        
        X_normalized = (X - self.X_mean) / self.X_std
        
        # Reshape input: add channel dimension
        X_tensor = torch.FloatTensor(X_normalized).unsqueeze(2)  # (n_samples, context_length, 1, n_tau, n_m)
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            predictions_normalized = self.model(X_tensor)  # (n_samples, 1, n_tau, n_m)
        
        # Remove channel dimension and convert to numpy
        predictions_normalized = predictions_normalized.squeeze(1).cpu().numpy()  # (n_samples, n_tau, n_m)
        
        # Denormalize predictions back to original scale
        predictions = predictions_normalized * self.y_std + self.y_mean
        
        # Apply ReLU to ensure non-negative volatility (volatility is always >= 0 in original scale)
        # This is safe to do after denormalization since we're back in original scale
        predictions = np.maximum(predictions, 0.0)
        
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
            'normalization_stats': {
                'X_mean': self.X_mean,
                'X_std': self.X_std,
                'y_mean': self.y_mean,
                'y_std': self.y_std
            },
            'hyperparameters': {
                'num_layers': self.num_layers,
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'last_conv_kernel': self.last_conv_kernel,
                'last_conv_stride': self.last_conv_stride,
                'last_conv_padding': self.last_conv_padding,
                'inter_channels': self.inter_channels,
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore hyperparameters
        hp = checkpoint['hyperparameters']
        self.num_layers = hp['num_layers']
        self.filters = hp['filters']
        self.inter_channels = hp.get('inter_channels', [32] * self.num_layers)  # Default for backward compatibility
        self.kernel_size = hp['kernel_size']
        self.strides = hp['strides']
        self.padding = hp['padding']
        self.last_conv_kernel = hp['last_conv_kernel']
        self.last_conv_stride = hp['last_conv_stride']
        self.last_conv_padding = hp['last_conv_padding']
        
        # Reinitialize model with restored hyperparameters
        self.model = SAConvLSTM(
            input_channels=1,
            feature_channels=self.filters,
            inter_channels=self.inter_channels,
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
        
        # Load normalization statistics if available
        if 'normalization_stats' in checkpoint:
            self.X_mean = checkpoint['normalization_stats']['X_mean']
            self.X_std = checkpoint['normalization_stats']['X_std']
            self.y_mean = checkpoint['normalization_stats']['y_mean']
            self.y_std = checkpoint['normalization_stats']['y_std']
        else:
            # For backward compatibility with old checkpoints
            print("  WARNING: Checkpoint does not contain normalization statistics. "
                  "Predictions may be incorrect if model was trained with normalization.")
            self.X_mean = None
            self.X_std = None
            self.y_mean = None
            self.y_std = None
        self.is_fitted = True
        
        print(f"Checkpoint loaded from {filepath}")
