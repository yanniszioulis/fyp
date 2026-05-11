"""
VAR(p) — vector autoregression baseline for multivariate time-series
forecasting.

Model: x_t = c + Σ_{i=1..p} A_i x_{t-i} + ε_t, where x_t ∈ ℝ^K. Fit by
ordinary least squares with intercept on the training set.

Hyperparameters:
    p  int — VAR lag order (passed to `fit_var_p`). Choose externally.

API:
    fit_var_p(X, p)                 fit and return (c, A_list, E).
    make_step_window_fn(c, A_list)  closure that maps a [p, K] window to
                                    the next x_t one step ahead.
"""

from __future__ import annotations

import numpy as np


def fit_var_p(X: np.ndarray, p: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Fit VAR(p) by OLS with intercept on `X` of shape [T, K].

    Model: x_t = c + Σ_{i=1..p} A_i x_{t-i} + ε_t

    Returns:
        c        intercept [K]
        A_list   list of p coefficient matrices each [K, K]; A_list[i-1] is A_i
        E        residual matrix [T-p, K]
    """
    T, K = X.shape
    T_eff = T - p
    if T_eff <= K * p:
        raise ValueError(
            f"VAR({p}) needs T_eff > K*p (have T_eff={T_eff}, K*p={K*p}); "
            f"reduce p."
        )
    Z = np.empty((T_eff, p * K), dtype=np.float64)
    for i in range(p):
        # Block i corresponds to lag (i+1). Target row t (0..T_eff-1) is
        # X[p+t]; lag-(i+1) is X[p+t-(i+1)] = X[p-1-i+t].
        Z[:, i * K : (i + 1) * K] = X[p - 1 - i : T_eff + p - 1 - i]
    Y = X[p:]
    Z_full = np.hstack([Z, np.ones((T_eff, 1))])
    coef, *_ = np.linalg.lstsq(Z_full, Y, rcond=None)        # [pK+1, K]
    A_stacked = coef[:-1]                                     # [pK, K]
    c         = coef[-1]                                      # [K]
    A_list    = [A_stacked[i * K : (i + 1) * K] for i in range(p)]
    E         = Y - Z_full @ coef
    return c, A_list, E


def make_step_window_fn(c: np.ndarray, A_list: list[np.ndarray]):
    """Return a fn that consumes a [p, K] window (lag-1 = window[-1]) and emits x_next."""
    p = len(A_list)
    def step(window: np.ndarray) -> np.ndarray:
        x_next = c.copy()
        for i in range(p):
            x_next += window[p - 1 - i] @ A_list[i]
        return x_next
    return step
