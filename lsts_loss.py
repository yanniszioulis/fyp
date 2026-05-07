"""LSTS loss components for IV-surface forecasting.

Tensors end in (n_tau, n_k) -- shape inferred at runtime from the input.
Functions accept any leading shape via `...` indexing, so unbatched
(n_tau, n_k) and batched (B, n_tau, n_k) both work.
"""

import math

import torch


def compute_L_level(pred, true):
    return (pred.mean(dim=(-2, -1)) - true.mean(dim=(-2, -1))).pow(2).mean()


def compute_L_skew(pred, true, k):
    dk = k[1:] - k[:-1]
    skew_pred = (pred[..., :, 1:] - pred[..., :, :-1]) / dk
    skew_true = (true[..., :, 1:] - true[..., :, :-1]) / dk
    return (skew_pred - skew_true).pow(2).mean()


def compute_L_term(pred, true, tau):
    dtau = (tau[1:] - tau[:-1])[:, None]
    term_pred = (pred[..., 1:, :] - pred[..., :-1, :]) / dtau
    term_true = (true[..., 1:, :] - true[..., :-1, :]) / dtau
    return (term_pred - term_true).pow(2).mean()


def compute_L_curvature(pred, true):
    """MSE of second derivative along moneyness (uniform-grid form)."""
    d2_pred = pred[..., :, 2:] - 2 * pred[..., :, 1:-1] + pred[..., :, :-2]
    d2_true = true[..., :, 2:] - 2 * true[..., :, 1:-1] + true[..., :, :-2]
    return (d2_pred - d2_true).pow(2).mean()


def compute_lsts_components(pred, true, k, tau):
    """Returns dict of the four components for a single (pred, true) pair."""
    return {
        "level": compute_L_level(pred, true),
        "skew":  compute_L_skew(pred, true, k),
        "term":  compute_L_term(pred, true, tau),
        "curvature": compute_L_curvature(pred, true)
    }

