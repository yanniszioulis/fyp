"""
VAR — Gonçalves–Guidolin two-stage VAR baseline for the SPX IV surface.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np


# design-matrix columns.
GG_DESIGN_COLS = ("intercept", "M", "M_sq", "tau", "M_tau")


# End-to-end driver

def run_var_baseline(data: dict, pred_len: int,
                     max_lags: int = 12,
                     seed: int = 42,
                     out_dir: str | None = None,
                     report_horizons: Sequence[int] = (1, 5, 10, 21),
                     verbose: bool = True) -> dict:
    """
    End-to-end Gonçalves–Guidolin VAR baseline.
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

    # recover raw log-IV
    scaled = np.asarray(data["scaled_log_iv"], dtype=np.float64)   # [N_rows, C]
    mu  = np.asarray(data["scaler"]["mean"], dtype=np.float64)     # [C]
    sd  = np.asarray(data["scaler"]["std"],  dtype=np.float64)     # [C]
    log_iv = scaled * sd + mu                                      # [N_rows, C]

   
    # daily cross-sectional OLS
    k_arr   = np.asarray(money_vals, dtype=np.float64)
    tau_arr = np.asarray(tau_vals,   dtype=np.float64)
    T_vec = np.repeat(tau_arr, n_money)                            # [C]
    M_vec = np.tile(k_arr, n_tau) / np.sqrt(T_vec)                 # [C]  M = k/√τ
    X_design = np.column_stack([
        np.ones_like(M_vec),
        M_vec,
        M_vec ** 2,
        T_vec,
        M_vec * T_vec,
    ]).astype(np.float64)                                          # [C, 5]

    Xpinv  = np.linalg.pinv(X_design)                              # [5, C]
    betas  = (Xpinv @ log_iv.T).T                                  # [N_rows, 5]
    fitted = betas @ X_design.T                                    # [N_rows, C]
    resid  = log_iv - fitted
    ss_res = (resid ** 2).sum(axis=1)
    ss_tot = ((log_iv - log_iv.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-18)                  # [N_rows]
    if verbose:
        print(f"[var]  Stage-1 daily OLS on {N_rows} days × {C} cells")
        print(f"        cross-sectional R² (full sample): "
              f"mean={r2.mean():.4f}  p10={np.quantile(r2,0.1):.4f}  "
              f"p50={np.median(r2):.4f}  p90={np.quantile(r2,0.9):.4f}")


    # Stage-2 VAR on β, TRAIN ONLY
    betas_train = betas[:train_end]
    from statsmodels.tsa.api import VAR
    var_model = VAR(np.asarray(betas_train, dtype=np.float64))
    order = var_model.select_order(maxlags=max_lags)
    p = int(getattr(order, "bic", 1))
    if p < 1:
        p = 1
    var_results = var_model.fit(p)
    bic_arr = order.ics.get("bic", [])
    bic_table = {int(i): float(v) for i, v in enumerate(bic_arr)}
    if verbose:
        print(f"[var]  Stage-2 VAR on β: T_train={train_end}, K=5, "
              f"BIC-selected p={p}  (BIC over p=1..{max_lags}: "
              + ", ".join(f"{k}:{v:.2f}" for k, v in
                          sorted(bic_table.items()) if k >= 1) + ")")


    # forecast β̂_{t+h} for each TEST base date, then reconstruct
    test_starts = data["test_starts"]
    base_indices = test_starts + L - 1
    assert len(base_indices) == Xte.shape[0], \
        f"Mismatch: var base_indices={len(base_indices)} vs Xte={Xte.shape[0]}"
    beta_paths = np.empty((len(base_indices), P, betas.shape[1]), dtype=np.float64)
    for i, t in enumerate(base_indices):
        last_p = betas[t - p + 1 : t + 1]                          # [p, 5]
        beta_paths[i] = var_results.forecast(last_p, steps=P)
    log_iv_hat = beta_paths @ X_design.T                           # [N, P, C]
    # restandardise into the trainer's space
    z_hat = (log_iv_hat - mu) / sd                                 # [N, P, C]


    # scoring 
    Yte64 = Yte.astype(np.float64)                                 # [N, P, C]
    err_std = z_hat - Yte64
    mse_std  = float(np.mean(err_std ** 2))
    rmse_std = float(np.sqrt(mse_std))
    mae_std  = float(np.mean(np.abs(err_std)))

    true_log_iv = Yte64 * sd + mu                                  # [N, P, C]

    pred_iv = np.exp(log_iv_hat)                                   # [N, P, C]
    true_iv = np.exp(true_log_iv)                                  # [N, P, C]
    rmse_vol = float(np.sqrt(np.mean((pred_iv - true_iv) ** 2)))

    z_today = Xte[:, -1, :].astype(np.float64)                     # [N, C]
    true_change = Yte64 - z_today[:, None, :]                      # [N, P, C]
    pred_change = z_hat  - z_today[:, None, :]                     # [N, P, C]

    eps = 1e-12
    dir_valid = np.abs(true_change) > eps                          # [N, P, C]
    dir_match = np.sign(pred_change) == np.sign(true_change)       # [N, P, C]
    dir_hit_overall = (float(dir_match[dir_valid].mean())
                       if dir_valid.any() else float("nan"))
    per_h_mse_std  = np.mean(err_std ** 2, axis=(0, 2))            # [P]
    per_h_dir_hit  = np.array([
        float(dir_match[:, h, :][dir_valid[:, h, :]].mean())
        if dir_valid[:, h, :].any() else float("nan")
        for h in range(P)
    ])

    # ATM / shortest-τ channel: closest to k=0 at the shortest maturity.
    m_idx = int(np.argmin(np.abs(np.asarray(money_vals))))         # closest to k=0
    t_idx = int(np.argmin(np.asarray(tau_vals)))                   # shortest τ
    atm_ch = t_idx * n_money + m_idx
    # median-split base dates into calm/stressed on the standardised ATM-1m level
    med = np.median(z_today[:, atm_ch])
    stressed = z_today[:, atm_ch] > med                            # [N] bool
    calm = ~stressed
    calm_vs_stressed = {}
    for name, mask in (("calm", calm), ("stressed", stressed)):
        if mask.sum() == 0:
            calm_vs_stressed[name] = None
            continue
        b_err   = err_std[mask]
        b_valid = dir_valid[mask]
        calm_vs_stressed[name] = {
            "n_windows":        int(mask.sum()),
            "mse_std":          float(np.mean(b_err ** 2)),
            "rmse_std":         float(np.sqrt(np.mean(b_err ** 2))),
            "mae_std":          float(np.mean(np.abs(b_err))),
            "rmse_vol":         float(np.sqrt(np.mean(
                                    (pred_iv[mask] - true_iv[mask]) ** 2))),
            "dir_hit_rate":     (float(dir_match[mask][b_valid].mean())
                                 if b_valid.any() else float("nan")),
        }

    horizon_summary = {}
    for h in report_horizons:
        if 1 <= h <= P:
            err_h = err_std[:, h - 1, :]
            horizon_summary[f"h+{h}"] = {
                "rmse_std":     float(np.sqrt(np.mean(err_h ** 2))),
                "rmse_vol":     float(np.sqrt(np.mean(
                                    (pred_iv[:, h - 1, :] - true_iv[:, h - 1, :]) ** 2))),
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
            "calm":     calm_vs_stressed["calm"],
            "stressed": calm_vs_stressed["stressed"],
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


# CLI entry: `python VAR/var.py --pred_len 21` for a standalone VAR run
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
