#!/usr/bin/env python3
"""
Quick diagnostic to check ConvLSTM output dimensions and training behavior.
"""

import torch
import numpy as np
from models.convlstm.convlstm_model import ConvLSTM

# Test spatial dimensions
print("=" * 60)
print("Testing ConvLSTM Spatial Dimensions")
print("=" * 60)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

# Create a test input: (batch=1, seq_len=21, channels=1, height=20, width=20)
test_input = torch.randn(1, 21, 1, 20, 20).to(device)

# Test with padding=0, kernel=3 (should output 18x18, then last_conv padding=1 -> 20x20)
model = ConvLSTM(
    input_channels=1,
    feature_channels=[64],
    kernel_size=[3],
    stride=[1],
    padding=[0],
    device=device,
    last_conv=[1, 1, 1],  # kernel=1, stride=1, padding=1
    num_layers=1
).to(device)

print(f"\nInput shape: {test_input.shape}")
print(f"Expected after ConvLSTM layer (padding=0, kernel=3): (1, 21, 64, 18, 18)")
print(f"Expected after last_conv (padding=1, kernel=1): (1, 1, 20, 20)")

with torch.no_grad():
    output = model(test_input)

print(f"Actual output shape: {output.shape}")
print(f"Expected: (1, 1, 20, 20)")
print(f"Match: {output.shape == (1, 1, 20, 20)}")

if output.shape != (1, 1, 20, 20):
    print(f"\n⚠️  DIMENSION MISMATCH! This will cause training issues.")
    print(f"   Model outputs {output.shape[2]}x{output.shape[3]} but targets are 20x20")
else:
    print(f"\n✓ Spatial dimensions correct!")

# Test with different configurations
print("\n" + "=" * 60)
print("Testing Different Configurations")
print("=" * 60)

configs = [
    {"padding": [0], "last_conv_padding": 1, "name": "PI-ConvTF default (1 layer)"},
    {"padding": [1], "last_conv_padding": 0, "name": "Maintain size (1 layer)"},
    {"padding": [1, 1], "last_conv_padding": 0, "name": "Maintain size (2 layers)"},
]

for config in configs:
    try:
        model = ConvLSTM(
            input_channels=1,
            feature_channels=[64] * len(config["padding"]),
            kernel_size=[3] * len(config["padding"]),
            stride=[1] * len(config["padding"]),
            padding=config["padding"],
            device=device,
            last_conv=[1, 1, config["last_conv_padding"]],
            num_layers=len(config["padding"])
        ).to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        match = output.shape == (1, 1, 20, 20)
        status = "✓" if match else "✗"
        print(f"{status} {config['name']}: Output {output.shape} {'(CORRECT)' if match else '(WRONG)'}")
    except Exception as e:
        print(f"✗ {config['name']}: Error - {e}")

print("\n" + "=" * 60)
print("Recommendations for Overfitting:")
print("=" * 60)
print("1. Use learning_rate=0.001 (not 0.01)")
print("2. Increase patience or disable early stopping for small datasets")
print("3. Use batch_size <= n_samples (e.g., batch_size=7 for 7 samples)")
print("4. Increase epochs (e.g., 200-500 for overfitting test)")
print("5. Verify spatial dimensions match (20x20 output)")
