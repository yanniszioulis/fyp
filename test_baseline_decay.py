#!/usr/bin/env python3
"""Test baseline_decay weight calculation."""

import numpy as np

def compute_weights(context_length, baseline_decay):
    """Compute weights for exponential-weighted baseline."""
    indices = np.arange(context_length)
    normalized_indices = indices / max(1, context_length - 1)
    weights = np.exp((baseline_decay - 1.0) * normalized_indices)
    weights = weights / weights.sum()
    return weights, normalized_indices

# Test with context_length=5
context_length = 5
print('=' * 60)
print(f'Context length: {context_length}')
print('normalized_indices: 0=oldest, 1=newest')
print('=' * 60)
print()

for decay in [-1, 0.0, 0.5, 1.0, 2.0]:
    if decay == -1:
        print(f'decay={decay} (persistence): uses only last surface')
        print('  (special case - not computed here)')
    else:
        weights, norm_idx = compute_weights(context_length, decay)
        print(f'decay={decay}:')
        print(f'  normalized_indices: {norm_idx}')
        print(f'  raw exp values: {np.exp((decay - 1.0) * norm_idx)}')
        print(f'  normalized weights: {weights}')
        print(f'  Oldest weight: {weights[0]:.4f}, Newest weight: {weights[-1]:.4f}')
        print(f'  Ratio (newest/oldest): {weights[-1]/weights[0]:.4f}')
        if weights[-1] > weights[0]:
            print(f'  → Newest has MORE weight (recency bias)')
        else:
            print(f'  → Oldest has MORE weight (anti-recency)')
    print()

print('=' * 60)
print('Interpretation:')
print('  decay < 1.0: recency bias (newer surfaces weighted more)')
print('  decay = 1.0: uniform weights (all equal)')
print('  decay > 1.0: anti-recency (older surfaces weighted more)')
print('=' * 60)
