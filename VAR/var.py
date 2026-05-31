"""
VAR — Gonçalves–Guidolin two-stage VAR baseline for the SPX IV surface.

Pipeline (Gonçalves & Guidolin, 2006):

  Stage 1 — for each day t INDEPENDENTLY, fit OLS across the 150 grid cells:
      ℓ(m, τ) = β₀ + β₁·M + β₂·M² + β₃·τ + β₄·(M·τ) + ε
      with M = k/√τ (time-adjusted moneyness; k = log-fwd-moneyness).
      Each day's surface is summarised by a 5-vector β_t. Because k spans
      only ±0.1 the M² lever arm is short, so β₂ is identified but noisy.

  Stage 2 — fit a VAR(p) on the 5-dim β series, TRAIN ONLY:
      β_t = c + Σ_{i=1..p} Φ_i β_{t-i} + u_t
      p chosen by BIC over p ∈ {1, …, max_lags}; OLS equation-by-equation
      via statsmodels.tsa.api.VAR. Parameters frozen after the train fit
      and reused unchanged on val/test (no refit).

  Forecast — for each test base date t, iterate the frozen VAR forward H
  steps to obtain β̂_{t+h}; reconstruct ℓ̂_{t+h} on every grid cell via
  the Stage-1 formula; restandardise into the trainer's space; score on
  the same windows the neural models use.

Channel order matches train.parse_grid (tau-outer / moneyness-inner) so
reconstructions align row-for-row with train.py's Yte without any
permutation.

Public API:
    gg_design_matrix(money_vals, tau_vals)        build the [C, 5] X
    gg_daily_fit(log_iv, X)                       [T, 5] β, [T] R²
    gg_fit_var(betas_train, max_lags)             (results, p, bic_table)
    gg_forecast_betas(results, betas, base_idx, H) [N, H, 5] paths
    gg_reconstruct(beta_paths, X)                 [N, H, C] log-IV
    run_var_baseline(data, pred_len, ...)         end-to-end driver
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Stage-1 design matrix + daily cross-sectional OLS
# ----------------------------------------------------------------------------
GG_DESIGN_COLS = ("intercept", "M", "M_sq", "tau", "M_tau")


def gg_design_matrix(money_vals: Sequence[float],
                     tau_vals:   Sequence[float]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the [C, 5] Stage-1 design matrix, in tau-outer / money-inner
    channel order so that row c corresponds to the cell at
    (tau_vals[c // n_money], money_vals[c % n_money]) — the same layout
    train.parse_grid uses.

    Columns are [1, M, M², τ, M·τ] with M = k/√τ.

    Returns:
        X     [C, 5]   constant across days (depends only on the grid)
        M_vec [C]      time-adjusted moneyness per cell
        T_vec [C]      maturity (years) per cell
    """
    n_money = len(money_vals)
    n_tau   = len(tau_vals)
    k_arr   = np.asarray(money_vals, dtype=np.float64)
    tau_arr = np.asarray(tau_vals,   dtype=np.float64)
    # tau-outer, money-inner: T_vec repeats each τ n_money times; K_vec tiles k.
    T_vec = np.repeat(tau_arr, n_money)                        # [C]
    K_vec = np.tile(k_arr, n_tau)                              # [C]
    M_vec = K_vec / np.sqrt(T_vec)                             # [C]
    X = np.column_stack([
        np.ones_like(M_vec),
        M_vec,
        M_vec ** 2,
        T_vec,
        M_vec * T_vec,
    ]).astype(np.float64)                                       # [C, 5]
    return X, M_vec, T_vec


def gg_daily_fit(log_iv: np.ndarray, X: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Stage-1: vectorised daily cross-sectional OLS, all days at once.

    For each row t of log_iv (length C), solve β_t = (XᵀX)⁻¹ Xᵀ ℓ_t.
    Returns (betas, r2) where betas is [T, 5] and r2 is [T] (per-day
    cross-sectional R² of the basis fit; diagnostic only).
    """
    Xpinv = np.linalg.pinv(X)                                  # [5, C]
    betas = (Xpinv @ log_iv.T).T                               # [T, 5]
    fitted = betas @ X.T                                        # [T, C]
    resid  = log_iv - fitted
    ss_res = (resid ** 2).sum(axis=1)
    ss_tot = ((log_iv - log_iv.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-18)
    return betas, r2


# ----------------------------------------------------------------------------
# Stage-2 VAR on β
# ----------------------------------------------------------------------------

def gg_fit_var(betas_train: np.ndarray, max_lags: int = 12):
    """Stage-2: VAR on β with BIC-selected lag, train-only.

    betas_train: [T_train, 5] — the slice of β_t covering only the training
    rows the scaler was fit on (no val/test leakage).

    Returns (results, p, bic_table) where `results` is the fitted
    statsmodels VARResults and bic_table maps p → BIC for diagnostics.
    """
    from statsmodels.tsa.api import VAR
    model = VAR(np.asarray(betas_train, dtype=np.float64))
    order = model.select_order(maxlags=max_lags)
    p = int(getattr(order, "bic", 1))
    if p < 1:
        # The project notes call out "in practice 1 or 2 wins"; if BIC
        # collapses to 0 we keep a VAR(1) floor so the model has dynamics.
        p = 1
    results = model.fit(p)
    bic_arr = order.ics.get("bic", [])
    bic_table = {int(i): float(v) for i, v in enumerate(bic_arr)}
    return results, p, bic_table


def gg_forecast_betas(results, betas: np.ndarray,
                      base_indices: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast β̂_{t+1..t+H} for every base date t in `base_indices`.

    betas:         [T, 5] FULL-sample β series (Stage-1 output for every
                   day; intra-day OLS has no temporal information so this
                   does NOT leak val/test info into the Stage-2 fit, which
                   is already frozen at this point).
    base_indices:  [N_base] integer row indices of "today" per window.
    horizon:       max forecast steps H. Returns paths of length H so the
                   caller can pick any subset of horizons.

    Returns: [N_base, H, 5]
    """
    p = int(results.k_ar)
    n = len(base_indices)
    paths = np.empty((n, horizon, betas.shape[1]), dtype=np.float64)
    for i, t in enumerate(base_indices):
        # last p observations ending at t (inclusive): β_{t-p+1..t}.
        last_p = betas[t - p + 1 : t + 1]                      # [p, 5]
        paths[i] = results.forecast(last_p, steps=horizon)
    return paths


def gg_reconstruct(beta_paths: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Reconstruct ℓ̂ on the full grid: ℓ̂_{t+h}(c) = X[c, :] · β̂_{t+h}.

    beta_paths: [N, H, 5]
    X:          [C, 5]
    Returns:    [N, H, C] raw log-IV predictions, channel order matches X.
    """
    return beta_paths @ X.T


# ----------------------------------------------------------------------------
# Scoring helpers (vol-points RMSE, direction hit, calm-vs-stressed split)
# ----------------------------------------------------------------------------

def _vol_points_rmse(pred_log_iv: np.ndarray, true_log_iv: np.ndarray) -> float:
    """RMSE in vol points (decimal): IV = exp(ℓ). Multiply by 100 for %."""
    pi = np.exp(pred_log_iv)
    ti = np.exp(true_log_iv)
    return float(np.sqrt(np.mean((pi - ti) ** 2)))


def _direction_hit_rate(pred_change: np.ndarray, true_change: np.ndarray,
                        eps: float = 1e-12) -> float:
    """Fraction of (window, horizon, cell) entries where sign(pred) == sign(true).
    Cells with |true_change| <= eps are excluded (ambiguous ties)."""
    valid = np.abs(true_change) > eps
    if not valid.any():
        return float("nan")
    return float((np.sign(pred_change[valid]) == np.sign(true_change[valid])).mean())


def _calm_vs_stressed_split(z_today_atm_1m: np.ndarray) -> np.ndarray:
    """Median-split each base date into 'calm' (False) vs 'stressed' (True)
    using the standardised ATM-1m log-IV level on the input window's last
    day. Robust, label-only — does not feed the model."""
    med = np.median(z_today_atm_1m)
    return z_today_atm_1m > med


def _atm_short_tau_channel(money_vals: Sequence[float],
                           tau_vals:   Sequence[float]) -> int:
    """Channel index for the ATM, shortest-τ cell. Used only for labelling
    windows into calm/stressed buckets — it's the closest thing we have
    to a VIX-like surface anchor on this grid."""
    n_money = len(money_vals)
    m_idx = int(np.argmin(np.abs(np.asarray(money_vals))))     # closest to k=0
    t_idx = int(np.argmin(np.asarray(tau_vals)))               # shortest τ
    return t_idx * n_money + m_idx


# ----------------------------------------------------------------------------
# End-to-end driver
# ----------------------------------------------------------------------------

def run_var_baseline(data: dict, pred_len: int,
                     max_lags: int = 12,
                     seed: int = 42,
                     out_dir: str | None = None,
                     report_horizons: Sequence[int] = (1, 5, 10, 21),
                     verbose: bool = True) -> dict:
    """End-to-end Gonçalves–Guidolin VAR baseline on a load_dataset() dict.

    Runs steps 1–5 of the spec exactly:
      1. De-standardise z back to raw log-IV using the train scaler.
      2. Daily Stage-1 OLS on all rows (intra-day only; no temporal info).
      3. VAR on the train slice of β with BIC-selected p ∈ [1, max_lags].
      4. Forecast β̂_{t+h} for h = 1..pred_len at every test base date.
      5. Reconstruct ℓ̂, restandardise, score vs the same Yte the neural
         models use — RMSE in both standardised log-IV (training space)
         and vol points (interpretable space), plus direction-of-change
         hit rate per horizon and a calm-vs-stressed decomposition.

    Writes preds.npy, metrics_test.json and hyperparams.json to `out_dir`
    when given. Returns the metrics dict either way.
    """
    np.random.seed(seed)
    grid = data["grid"]
    money_vals, tau_vals = grid.money_vals, grid.tau_vals
    n_tau, n_money = grid.n_tau, grid.n_money
    C = n_tau * n_money
    train_end = int(data["rows"]["train_end"])
    val_end   = int(data["rows"]["val_end"])
    N_rows    = int(data["rows"]["N"])
    Xte, Yte = data["test"]                                    # [N, P, C]
    L = int(Xte.shape[1])
    P = int(pred_len)
    if P > Yte.shape[1]:
        raise ValueError(f"pred_len={P} > Yte horizons={Yte.shape[1]}")

    # ------------------------------------------------------------------
    # Step 1 — recover raw log-IV
    # ------------------------------------------------------------------
    scaled = np.asarray(data["scaled_log_iv"], dtype=np.float64)   # [N_rows, C]
    mu  = np.asarray(data["scaler"]["mean"], dtype=np.float64)     # [C]
    sd  = np.asarray(data["scaler"]["std"],  dtype=np.float64)     # [C]
    log_iv = scaled * sd + mu                                      # [N_rows, C]

    # ------------------------------------------------------------------
    # Step 2 — daily cross-sectional OLS
    # ------------------------------------------------------------------
    # gg_design_matrix returns (X, M_vec, T_vec); only X is needed here.
    X_design, _, _ = gg_design_matrix(money_vals, tau_vals)
    betas, r2 = gg_daily_fit(log_iv, X_design)                     # [N_rows, 5], [N_rows]
    if verbose:
        print(f"[var]  Stage-1 daily OLS on {N_rows} days × {C} cells")
        print(f"        cross-sectional R² (full sample): "
              f"mean={r2.mean():.4f}  p10={np.quantile(r2,0.1):.4f}  "
              f"p50={np.median(r2):.4f}  p90={np.quantile(r2,0.9):.4f}")

    # ------------------------------------------------------------------
    # Step 3 — Stage-2 VAR on β, TRAIN ONLY
    # ------------------------------------------------------------------
    betas_train = betas[:train_end]
    var_results, p, bic_table = gg_fit_var(betas_train, max_lags=max_lags)
    if verbose:
        print(f"[var]  Stage-2 VAR on β: T_train={train_end}, K=5, "
              f"BIC-selected p={p}  (BIC over p=1..{max_lags}: "
              + ", ".join(f"{k}:{v:.2f}" for k, v in
                          sorted(bic_table.items()) if k >= 1) + ")")

    # ------------------------------------------------------------------
    # Step 4 — forecast β̂_{t+h} for each TEST base date, then reconstruct
    # ------------------------------------------------------------------
    # The base date for a test window starting at row s is t = s + L - 1
    # (the last input row). We reuse the exact test-window starts train.py
    # assigned (the shared whole-horizon-within-split rule) so the VAR
    # predictions align row-for-row with Yte and with the neural models.
    test_starts = data["test_starts"]
    base_indices = test_starts + L - 1
    assert len(base_indices) == Xte.shape[0], \
        f"Mismatch: var base_indices={len(base_indices)} vs Xte={Xte.shape[0]}"
    beta_paths = gg_forecast_betas(var_results, betas, base_indices, horizon=P)
    log_iv_hat = gg_reconstruct(beta_paths, X_design)              # [N, P, C]
    # Step 5 — restandardise into the trainer's space
    z_hat = (log_iv_hat - mu) / sd                                  # [N, P, C]

    # ------------------------------------------------------------------
    # Step 5 — scoring (identical windows, horizons, embargo as neural models)
    # ------------------------------------------------------------------
    Yte64 = Yte.astype(np.float64)                                  # [N, P, C]
    err_std = z_hat - Yte64
    mse_std  = float(np.mean(err_std ** 2))
    rmse_std = float(np.sqrt(mse_std))
    mae_std  = float(np.mean(np.abs(err_std)))

    true_log_iv = Yte64 * sd + mu                                   # [N, P, C]
    rmse_vol = _vol_points_rmse(log_iv_hat, true_log_iv)

    z_today = Xte[:, -1, :].astype(np.float64)                      # [N, C]
    true_change = Yte64 - z_today[:, None, :]                       # [N, P, C]
    pred_change = z_hat  - z_today[:, None, :]                      # [N, P, C]
    dir_hit_overall = _direction_hit_rate(pred_change, true_change)
    per_h_mse_std  = np.mean(err_std ** 2, axis=(0, 2))             # [P]
    per_h_dir_hit  = np.array([_direction_hit_rate(pred_change[:, h, :],
                                                    true_change[:, h, :])
                                for h in range(P)])

    atm_ch = _atm_short_tau_channel(money_vals, tau_vals)
    stressed = _calm_vs_stressed_split(z_today[:, atm_ch])          # [N] bool
    calm = ~stressed
    def _bucket(mask):
        if mask.sum() == 0:
            return None
        b_err = err_std[mask]
        return {
            "n_windows":        int(mask.sum()),
            "mse_std":          float(np.mean(b_err ** 2)),
            "rmse_std":         float(np.sqrt(np.mean(b_err ** 2))),
            "mae_std":          float(np.mean(np.abs(b_err))),
            "rmse_vol":         _vol_points_rmse(log_iv_hat[mask],
                                                  true_log_iv[mask]),
            "dir_hit_rate":     _direction_hit_rate(pred_change[mask],
                                                     true_change[mask]),
        }

    horizon_summary = {}
    for h in report_horizons:
        if 1 <= h <= P:
            err_h = err_std[:, h - 1, :]
            horizon_summary[f"h+{h}"] = {
                "rmse_std":     float(np.sqrt(np.mean(err_h ** 2))),
                "rmse_vol":     _vol_points_rmse(log_iv_hat[:, h - 1, :],
                                                  true_log_iv[:, h - 1, :]),
                "dir_hit_rate": float(per_h_dir_hit[h - 1]),
            }

    metrics = {
        "model":         "var",
        "var_p":         p,
        "bic_table":     bic_table,
        "stage1_r2": {
            "mean": float(r2.mean()),
            "p10":  float(np.quantile(r2, 0.1)),
            "p50":  float(np.median(r2)),
            "p90":  float(np.quantile(r2, 0.9)),
        },
        "pred_len":      P,
        "lookback":      L,
        "n_test":        int(Yte64.shape[0]),
        "test_mse":      mse_std,            # std-log-IV (training space)
        "test_rmse":     rmse_std,
        "test_mae":      mae_std,
        "test_rmse_vol": rmse_vol,           # interpretable vol points
        "dir_hit_rate":  dir_hit_overall,
        "per_horizon":   horizon_summary,
        "per_horizon_mse_std": per_h_mse_std.tolist(),
        "per_horizon_dir_hit": per_h_dir_hit.tolist(),
        "calm_vs_stressed": {
            "split_channel": int(atm_ch),
            "split_rule":    "median of standardised ATM-1m log-IV at base date",
            "calm":     _bucket(calm),
            "stressed": _bucket(stressed),
        },
        "space":         "standardized_log_iv",
    }

    if verbose:
        print(f"[var]  Stage-1 mean R²={r2.mean():.4f}; "
              f"VAR(p={p});  test RMSE std={rmse_std:.4f}  "
              f"vol={rmse_vol*100:.3f}pp  dir-hit={dir_hit_overall*100:.2f}%")
        print(f"        per-horizon RMSE std: " +
              "  ".join(f"h+{h}: {horizon_summary[f'h+{h}']['rmse_std']:.4f}"
                        for h in report_horizons if f"h+{h}" in horizon_summary))
        print(f"        per-horizon dir-hit %: " +
              "  ".join(f"h+{h}: {horizon_summary[f'h+{h}']['dir_hit_rate']*100:.1f}"
                        for h in report_horizons if f"h+{h}" in horizon_summary))
        cb = metrics["calm_vs_stressed"]["calm"]
        sb = metrics["calm_vs_stressed"]["stressed"]
        print(f"        calm    (n={cb['n_windows']}): "
              f"RMSE std={cb['rmse_std']:.4f}  vol={cb['rmse_vol']*100:.3f}pp  "
              f"dir-hit={cb['dir_hit_rate']*100:.2f}%")
        print(f"        stressed(n={sb['n_windows']}): "
              f"RMSE std={sb['rmse_std']:.4f}  vol={sb['rmse_vol']*100:.3f}pp  "
              f"dir-hit={sb['dir_hit_rate']*100:.2f}%")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "preds.npy"), z_hat.astype(np.float32))
        with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        hyper = {
            "model":          "var",
            "recipe":         "goncalves_guidolin_2006_two_stage",
            "stage1_basis":   list(GG_DESIGN_COLS),
            "stage1_M":       "k/sqrt(tau)",
            "stage2_var_p":   p,
            "bic_max_lags":   max_lags,
            "fit_space":      "raw_log_iv",
            "predict_space":  "raw_log_iv",
            "score_space":    "standardized_log_iv",
            "lookback":       L,
            "pred_len":       P,
            "seed":           seed,
            "n_train_rows":   train_end,
            "data_end":       data.get("data_end"),
            "first_date":     data.get("first_date"),
            "last_date":      data.get("last_date"),
            "test_first_target_date": data.get("test_first_target_date"),
            "grid": {
                "n_tau":      n_tau,
                "n_money":    n_money,
                "tau_vals":   list(tau_vals),
                "money_vals": list(money_vals),
            },
            "scaler": {
                "space":         "log_iv",
                "mean":          mu.tolist(),
                "std":           sd.tolist(),
                "channel_order": grid.iv_cols,
            },
        }
        with open(os.path.join(out_dir, "hyperparams.json"), "w") as f:
            json.dump(hyper, f, indent=2)
        if verbose:
            print(f"[var]  saved → {out_dir}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry: `python VAR/var.py --pred_len 21` for a standalone VAR run.
# ---------------------------------------------------------------------------
def _cli() -> None:
    import argparse
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    sys.path.insert(0, repo_root)                              # for train.load_dataset
    from train import LOOKBACK, MODEL_DIR, load_dataset

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_len", type=int, required=True,
                    choices=(1, 5, 10, 21, 42, 63))
    ap.add_argument("--csv_path", default=os.path.join(repo_root, "SPX_surfaces.csv"))
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--data_end",   type=str,   default="2023-12-29")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--max_lags",   type=int,   default=12)
    ap.add_argument("--out_subdir", type=str,   default="results",
                    help="Subdir under VAR/ for outputs; final path is "
                         "VAR/<out_subdir>/<lookback>_<pred_len>/")
    args = ap.parse_args()

    data_end = None if args.data_end.lower() == "none" else args.data_end
    data = load_dataset(args.csv_path, args.train_frac, args.val_frac,
                        LOOKBACK, args.pred_len, data_end=data_end)
    out_dir = os.path.join(repo_root, MODEL_DIR["var"], args.out_subdir,
                           f"{LOOKBACK}_{args.pred_len}")
    run_var_baseline(data, args.pred_len,
                     max_lags=args.max_lags, seed=args.seed,
                     out_dir=out_dir, verbose=True)


if __name__ == "__main__":
    _cli()
