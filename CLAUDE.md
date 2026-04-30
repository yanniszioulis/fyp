# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **time series forecasting research project** comparing multiple deep learning models for forecasting SPX (S&P 500) implied volatility (IV) surfaces. The codebase integrates 5+ different forecasting architectures (PatchTST, HOT, DLinear, DynGWN, VAR) to benchmark their performance on financial data.

**Primary data**: SPX IV surface (400-feature grid: 20 moneyness × 20 time-to-maturity levels)
**Task**: Multi-step ahead forecasting (seq_len=21, pred_len=63)
**Main output script**: `compare_models.py` - unified evaluation harness for all models

## Directory Structure

```
/fyp
├── compare_models.py           # Main evaluation script (model comparison)
├── var_lag1_rollout.py         # VAR(1) baseline implementation
├── SPX_surfaces.csv            # Raw data (400 IV features, ~2500 trading days)
├── requirements.txt            # Base dependencies (numpy, pandas, torch, scipy, etc.)
│
├── PatchTST-main/              # Patch-based Transformer (ICLR 2023)
│   ├── PatchTST_supervised/    # Main supervised learning implementation
│   │   ├── run_longExp.py      # Training entry point
│   │   ├── models/             # Model implementations (PatchTST, DLinear, NLinear)
│   │   ├── exp/                # Experiment runner (dataset loading, training loop)
│   │   └── data_provider/      # Custom data loaders
│   └── results/                # Saved predictions (pred.npy format)
│
├── HOT/                        # Higher-Order Transformers (TMLR 2025)
│   ├── timeseries_main.py      # Entry point (Lightning-based training)
│   ├── src/
│   │   ├── models/             # HOT implementations (ts.py, ts_tensor.py for surfaces)
│   │   ├── modules/            # Kronecker attention, positional encodings
│   │   └── ts_data.py          # Lightning DataModule for TS/surfaces
│   └── results/                # Saved predictions + dates
│
├── DynGWN/                     # Dynamic Graph Wavenet (ICAIF 2023)
│   ├── main_dyngwn.py          # Entry point with domain-specific defaults
│   ├── model.py                # Graph neural network + temporal modeling
│   ├── engine.py               # Training/validation loop
│   ├── generate_spx_iv_data.py # Data pipeline (CSV → windowed arrays)
│   └── results/                # Predictions + dates
│
├── DLinear/                    # Linear baseline (AAAI 2022)
│   ├── run_longExp.py          # Training script
│   ├── models/
│   │   ├── DLinear.py          # Simple decomposition + 2 linear layers
│   │   ├── Autoformer.py       # Baseline Transformer variants
│   │   └── Transformer.py
│   ├── exp/exp_main.py         # Experiment runner
│   └── Pyraformer/,FEDformer/  # Additional Transformer implementations
│
├── PI-ConvTF/                  # Placeholder (not actively used)
├── data_prep/                  # Data preprocessing utilities
└── var_lag1_results/           # VAR(1) outputs (created by script)
```

## Key Data Format & Pipeline

### Input Data: SPX_surfaces.csv
- **Shape**: [T=~2500 days, 400 IV features + metadata]
- **IV columns**: Named `iv_<moneyness>_<tau>` (e.g., `iv_0.9_0.04`)
- **Grid**: 20×20 structured (20 moneyness levels × 20 time-to-maturity)
- **Sorting**: Column order is (tau outer, moneyness inner) - critical for grid reshaping

### Standard Train/Val/Test Split
```
train: 0-70% → used to fit scaler
val:   70-80%
test:  80-100%
```
Each model window uses: `[seq_len=21 past steps] → [pred_len=63 future steps]`

### Output Formats
All models save to standardized numpy format in `results/` subdirectories:
- **pred.npy**: [N, pred_len=63, n_features] - predictions in scaled space
- **start_dates.npy**: [N] - datetime64[D] for each window (for alignment)
- **true.npy** (VAR only): Ground truth

## Running Models

### VAR(1) Baseline
```bash
python var_lag1_rollout.py \
  --csv_path SPX_surfaces.csv \
  --context_len 21 \
  --horizon_len 63 \
  --out_dir var_lag1_results \
  --tune_ridge                    # Optional: grid-search regularization
```
**Output**: `var_lag1_results/{pred.npy, true.npy, start_dates.npy}`

### PatchTST
```bash
cd PatchTST-main/PatchTST_supervised
python run_longExp.py \
  --is_training 1 \
  --model PatchTST \
  --data SPX_IV \
  --root_path ./dataset \
  --data_path SPX_surfaces.csv \
  --seq_len 21 \
  --pred_len 63 \
  --enc_in 400 --dec_in 400 --c_out 400 \
  --batch_size 32 \
  --train_epochs 50
```
**Default result path**: `results/SPX_IV_21_63_PatchTST_custom_ftM_sl21_...`

### HOT (Higher-Order Transformers)
```bash
cd HOT
python timeseries_main.py \
  --name spx_iv \
  --csv_path ../SPX_surfaces.csv \
  --d_hidden 128 \
  --d_mlp 512 \
  --num_blocks 4 \
  --num_heads 8 \
  --patch_size 4 \
  --attention_type kronecker_product \
  --num_epochs 50
```
**Note**: Automatically reshapes 400 features into [H=20, W=20] tensor using grid order from `generate_spx_iv_data.py`

### DynGWN
```bash
cd DynGWN
python main_dyngwn.py \
  --domain spx_iv \
  --data SPX_surfaces.csv \
  --epochs 500 \
  --batch_size 8 \
  --save_preds
```
**Domain defaults applied**: Sets SPX IV grid adjacency, normalizes, uses date-aligned windowing

### DLinear
```bash
cd DLinear
python run_longExp.py \
  --is_training 1 \
  --model DLinear \
  --data ETTm1 \
  --enc_in 400 --dec_in 400 --c_out 400 \
  --seq_len 21 --pred_len 63 \
  --individual           # Use per-channel linear layers
```

## Model Comparison

### Main Evaluation Script: `compare_models.py`
Unified benchmarking harness that loads predictions from all models and compares:

```bash
python compare_models.py \
  --csv_path SPX_surfaces.csv \
  --seq_len 21 --pred_len 63 \
  --patchtst_pred PatchTST-main/.../pred.npy \
  --hot_pred_product HOT/results/.../pred.npy \
  --hot_dates_product HOT/results/.../start_dates.npy \
  --dyngwn_pred DynGWN/results/.../pred.npy \
  --dyngwn_dates DynGWN/results/.../start_dates.npy
```

**Metrics computed**:
- **MSE, MAE, RSE**: Standard regression metrics
- **IC (Information Coefficient)**: Spearman rank correlation per timestep
- **Per-horizon breakdown**: Metrics for [t+1, t+5, t+10, t+21, t+42, t+63]

**Key alignment logic**: All predictions are date-aligned to PatchTST test set. If a model has missing dates, it raises an error.

## Architecture Patterns

### Shared Data Pipeline
- **StandardScaler**: Fit on train set, applied uniformly across all models
- **Data format**: [batch, seq_len/pred_len, n_features] or [batch, H, W, pred_len] (for HOT)
- **Handling NaN**: VAR(1) uses ridge regression to handle ill-conditioned matrices

### Model Base Classes
- **PatchTST/DLinear**: `torch.nn.Module` with custom `Exp_Main` runner (handles dataloading, training)
- **HOT**: PyTorch Lightning `LightningModule` for modular training
- **DynGWN**: Custom trainer loop in `engine.py`; graph construction in `main_dyngwn.py`

### Grid Representation
- **Linear models** (PatchTST, DLinear, VAR): Treat 400 features as flat vector
- **Graph models** (DynGWN): Grid adjacency [20×20] with self-loops, nearest-neighbor connectivity
- **Tensor models** (HOT): Reshape to [H=20, W=20] and apply structured attention (Kronecker products)

## Dependencies & Setup

```bash
pip install -r requirements.txt
# Base: numpy, pandas, matplotlib, scipy, torch, einops, scikit-learn

# Model-specific (optional, installed locally):
cd HOT && pip install -r requirements.txt    # PyTorch Lightning
cd DynGWN && pip install -r requirements.txt # Optuna (for hyperparameter tuning)
cd DLinear && pip install -r requirements.txt
cd PatchTST-main && pip install -r requirements.txt
```

## Development Notes

### Active Scripts
- `compare_models.py` - Recently updated; compares all model outputs
- `var_lag1_rollout.py` - Recently updated; VAR(1) baseline with ridge tuning
- Both accept CSV path and output directory arguments (no hardcoded paths)

### Data Access
- SPX data is loaded from `.csv` (not pickled) for transparency
- All models use consistent train/val/test split (70/10/20)
- Scaler parameters are not saved; re-fit on each run

### Key Decision Points
1. **Scaling strategy**: Train-only (fit on train, apply to all splits) - prevents data leakage
2. **Grid ordering**: `iv_columns` must be sorted by (tau, moneyness) for correct 20×20 reshaping
3. **Date alignment**: All models save `start_dates.npy` to enable cross-model comparison
4. **Horizon evaluation**: 63-step ahead forecasts evaluated on held-out test set (last 20%)

### Common Issues & Fixes
- **Missing IV columns**: Check CSV header; script expects `iv_<float>_<float>` pattern
- **Shape mismatch on HOT**: Verify 400 features; wrong grid dimensions will cause reshape errors
- **VAR divergence**: Use `--tune_ridge` to regularize; helps with ill-conditioned covariance
- **Date misalignment**: Ensure `start_dates.npy` files are saved; `compare_models.py` uses these for matching windows

## Code Style & Structure
- **Model implementations**: Keep in `models/` subdirs; follow `__init__` → `forward` pattern
- **Experiments**: Use `exp/exp_main.py` pattern (dataset, trainer, logger in one class)
- **Reproducibility**: Set random seeds in entry points; PyTorch Lightning handles device placement
