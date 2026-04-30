## Project Overview

SPX IV surface forecasting research project. Five models — VAR(1), DLinear,
PatchTST, HOT, DynGWN — each compressed into one self-contained training
script with the same input/output contract, dispatched via a single
`train.py`, and benchmarked by `compare_models.py`.

**Data**: SPX IV surface, 400 features (20 moneyness × 20 tau), ~4191 days.
**Task**: seq_len=21 → pred_len=63 (3-month horizon from 1-month context).
**Split**: canonical 70/10/20 (train/val/test) — identical across all models.

## Directory layout

```
fyp/
├── train.py                       # Master dispatcher
├── compare_models.py              # Unified evaluation harness
├── SPX_surfaces.csv               # Raw data (400 iv_* columns + date)
│
├── VAR1/var1_spx_iv.py            # VAR(1) baseline (ridge, optional tuning)
├── DLinear/dlinear_spx_iv.py      # Channel-independent DLinear (einsum)
├── PatchTST/patchtst_spx_iv.py    # PatchTST + RevIN, fully inlined
├── HOT/hot_spx_iv.py              # Kronecker-attention transformer, inlined
├── DynGWN/dyngwn_spx_iv.py        # Graph WaveNet, adaptive adjacency only
│
├── PatchTST-main/                 # Legacy reference impl (kept for old results)
├── DynGWN/main_dyngwn.py, ...     # Legacy reference impl (npz-based pipeline)
└── HOT/timeseries_main.py, ...    # Legacy reference impl (Lightning-based)
```

The standalone `*_spx_iv.py` scripts inline every model definition — there
is no shared model/util library between them. Each script can be deleted
without affecting the others.

## Data contract

Every standalone script implements the **same** pipeline:

1. Load `SPX_surfaces.csv`, sort iv_ columns alphabetically.
2. Canonical split: `n_train = int(T*0.70)`, `n_test = int(T*0.20)`.
3. Fit `StandardScaler` on `iv[:n_train]`, transform all rows.
4. Window: train uses `[0, n_train)`, val uses `[n_train-seq_len, n_train+n_val)`,
   test uses `[T-n_test-seq_len, T)`.
5. Test predictions saved as `pred.npy` in **scaled space**.
6. Test window start dates saved as `start_dates.npy` (datetime64[D]).

This invariant is what allows `compare_models.py` to align predictions by
date and compare them on a common ground truth.

## Output format

All `pred.npy` files are float32 in scaled space:

| Script | pred.npy shape | Loader in compare_models |
|--------|----------------|--------------------------|
| var1, dlinear, patchtst, dyngwn | `[N, 63, 400]` | `flat` |
| hot | `[N, 20, 20, 63]` (H_mono × W_tau) | `hot` (F-order reshape to flat) |

HOT is the only tensorized model; F-order reshape is correct because
CSV column k = i_tau\*20 + i_mono and HOT uses H=moneyness, W=tau.

## Running

### Master dispatcher

```bash
# All models, then comparison
python train.py --models all --device cuda --compare

# Subset
python train.py --models var1 dlinear patchtst

# Pass model-specific args (only routed if the model accepts them)
python train.py --models hot --attention_type kronecker_sum --epochs 200
python train.py --models dyngwn --nhid 64 --epochs 500
python train.py --models var1 --tune_ridge
```

### Individual scripts

Each script is fully standalone and can be invoked directly:

```bash
python VAR1/var1_spx_iv.py --tune_ridge
python DLinear/dlinear_spx_iv.py --epochs 100 --device cuda
python PatchTST/patchtst_spx_iv.py --d_model 128 --n_heads 16 --epochs 100
python HOT/hot_spx_iv.py --d_hidden 128 --attention_type kronecker_product
python DynGWN/dyngwn_spx_iv.py --nhid 32 --epochs 300
```

All scripts must be invoked **from the project root** (output paths are
relative to it). Each script auto-generates an output directory under its
respective `results/` subfolder and saves `pred.npy`, `start_dates.npy`,
`train_log.csv`, `best_model.pt`.

### Comparison

```bash
python compare_models.py
```

Auto-discovers the most recent results matching each glob pattern in
`MODELS`. Builds a fresh ground truth from CSV (does not depend on any
saved `true.npy`). Writes timestamped CSV + `latest.{csv,horizons.csv,summary.json}`
under `comparison_results/`.

Metrics: MSE, RMSE, MAE, RSE, signed bias, directional accuracy, Spearman IC
(mean & std across 400 features per window-step). Per-horizon breakdown at
t+{1, 5, 10, 21, 42, 63}.

## Implementation details

- **Train-only scaler**: every script fits `StandardScaler` on `iv[:n_train]`.
  Means and stds are not saved; they are re-derived deterministically.
- **HOT reshape**: `iv.reshape(-1, 20, 20, order="F")` maps CSV column k to
  `[i_mono, i_tau]`. Verified against the loader in `compare_models.load_hot`.
- **HOT positional encoding**: default `--pe rope` applies RoPE on the
  temporal dim (matches legacy `ts_tensor.py` default). `--pe nope` disables.
- **DynGWN graph_mode**: default `--graph_mode grid_plus_adaptive` adds a
  fixed 4-neighbor moneyness×tau adjacency on top of the learned adaptive
  one (matches legacy run). `--graph_mode adaptive_only` drops the grid.
  Receptive field with blocks=4, layers=2, kernel=2 is 13. The legacy
  `main_dyngwn.py` additionally supported `dynamic_gcn_bool` (correlation
  matrices over a sliding window) — **not ported** because legacy SPX runs
  used `dynamic_gcn_bool=False` anyway.
- **DynGWN training objective**: standalone trains scaled-space MSE.
  Legacy used `masked_mae` after `inverse_transform` — different objective,
  expect slightly different convergence.
- **PatchTST**: patch_len=stride=7 → 3 patches over seq_len=21. RevIN with
  affine=True. Residual scaled-dot-product attention with learnable scale.
- **VAR1**: global VAR(1) fit by ridge regression on `iv[:n_train]` (first 70%).
  Legacy `var_lag1_rollout.py` fit on the first 80% (train+val merged); the
  new fit is more apples-to-apples vs. neural models. `--tune_ridge` does a
  logspace search.

## Editing rules

- Don't reintroduce shared model code across scripts; the standalone-per-model
  contract is intentional. Each script must be deletable without affecting
  the others.
- Keep the data pipeline in every script byte-identical (sort iv_ cols,
  same border formula, same scaler). If you change one, change all.
- New result paths must match the glob patterns in `compare_models.MODELS`.
- HOT must save `[N, H, W, pred_len]`; flat models save `[N, pred_len, 400]`.
