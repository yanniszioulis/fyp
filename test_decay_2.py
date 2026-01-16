#!/usr/bin/env python3
"""Test what decay=2.0 means."""

import numpy as np

def compute_weights(context_length, baseline_decay):
    """Compute weights for exponential-weighted baseline."""
    indices = np.arange(context_length)
    normalized_indices = indices / max(1, context_length - 1)  # 0=oldest, 1=newest
    weights = np.exp((baseline_decay - 1.0) * normalized_indices)
    weights = weights / weights.sum()
    return weights, normalized_indices

# Test with context_length=5
context_length = 5
print('=' * 70)
print(f'Context length: {context_length}')
print('normalized_indices: 0=oldest surface, 1=newest surface')
print('=' * 70)
print()

decay = 2.0
weights, norm_idx = compute_weights(context_length, decay)
print(f'baseline_decay = {decay}:')
print(f'  normalized_indices: {norm_idx}')
print(f'  Formula: exp(({decay} - 1.0) * normalized_indices) = exp(1.0 * normalized_indices)')
print()
print('  Raw exp values (before normalization):')
raw_weights = np.exp((decay - 1.0) * norm_idx)
for i, (ni, rw) in enumerate(zip(norm_idx, raw_weights)):
    print(f'    Surface {i} (normalized_idx={ni:.2f}): exp({decay-1.0}*{ni:.2f}) = {rw:.4f}')
print()
print(f'  Normalized weights (sum to 1.0):')
for i, (ni, w) in enumerate(zip(norm_idx, weights)):
    print(f'    Surface {i} (normalized_idx={ni:.2f}): {w:.4f} ({w*100:.1f}%)')
print()
print(f'  Oldest surface weight: {weights[0]:.4f} ({weights[0]*100:.1f}%)')
print(f'  Newest surface weight: {weights[-1]:.4f} ({weights[-1]*100:.1f}%)')
print(f'  Ratio (newest/oldest): {weights[-1]/weights[0]:.4f}x')
print()
print('  Interpretation:')
print(f'    → Newest surface gets {weights[-1]/weights[0]:.2f}x MORE weight than oldest')
print(f'    → This is STRONG RECENCY BIAS (newer surfaces weighted much more)')
print()

print('=' * 70)
print('Summary for different decay values:')
print('=' * 70)
for d in [0.0, 0.5, 1.0, 2.0]:
    if d == 1.0:
        print(f'  decay={d}: uniform weights (all equal)')
    else:
        w, _ = compute_weights(context_length, d)
        ratio = w[-1]/w[0]
        if ratio > 1.0:
            print(f'  decay={d}: recency bias (newest {ratio:.2f}x more than oldest)')
        else:
            print(f'  decay={d}: anti-recency (oldest {1/ratio:.2f}x more than newest)')
