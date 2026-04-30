# SPX IV Surface Forecasting

Research project comparing forecasting models on the SPX implied-volatility
surface. Five models — VAR(1), DLinear, PatchTST, HOT, DynGWN — each
compressed into one self-contained training script with a uniform input/output
contract, dispatched via a single `train.py`, and benchmarked by
`compare_models.py`.

**Data**: SPX IV surface, 400 features (20 moneyness × 20 tau), 4191 days.
**Task**: seq_len=21 → pred_len=63 (1-month context → 3-month horizon).
**Split**: canonical 70/10/20 (train/val/test) — identical across every model.

## Directory layout

```
fyp/
├── train.py                       # Master dispatcher
├── compare_models.py              # Unified evaluation harness
├── SPX_surfaces.csv               # Raw data (400 iv_* columns + date)
│
├── VAR1/var1_spx_iv.py            # Plain OLS VAR(1)
├── DLinear/dlinear_spx_iv.py      # Channel-independent DLinear (einsum)
├── PatchTST/patchtst_spx_iv.py    # Patch-based Transformer + RevIN
├── HOT/hot_spx_iv.py              # Kronecker-attention transformer
├── DynGWN/dyngwn_spx_iv.py        # Graph WaveNet
│
└── comparison_results/            # Compare-models CSV/JSON outputs
```

Each `*_spx_iv.py` script inlines every model definition — there is no
shared model/util library. Each script can be deleted without affecting
the others.

## Data contract (every script)

1. Load `SPX_surfaces.csv`. **Use CSV column order as-is** — do NOT sort
   alphabetically (alphabetical sort scrambles the 20×20 surface and breaks
   cross-sectional alignment with `compare_models.py`).
2. Canonical split: `n_train = int(T*0.70)`, `n_test = int(T*0.20)`,
   `n_val = T - n_train - n_test`.
3. Fit `StandardScaler` on `iv[:n_train]`, transform all rows.
4. Window: train uses `[0, n_train)`, val uses `[n_train-seq_len, n_train+n_val)`,
   test uses `[T-n_test-seq_len, T)`.
5. Test predictions saved as `pred.npy` in **scaled space**.
6. Test window start dates saved as `start_dates.npy` (datetime64[D]).
7. Hyperparameters saved as `config.json`; checkpoint saved as `best_model.pt`.

## Output format

All `pred.npy` files are float32 in scaled space.

| Script | pred.npy shape | Loader in compare_models |
|--------|----------------|--------------------------|
| var1, dlinear, patchtst, dyngwn | `[N, 63, 400]` | `flat` |
| hot | `[N, 20, 20, 63]` | `hot` (F-order reshape to flat) |

HOT is the only tensorized model; F-order reshape is correct because the CSV
column `k = i_tau*20 + i_mono` and HOT uses `H=moneyness, W=tau`.

## Workflow

### Train

```bash
# Train every model. `hot` expands to BOTH kronecker_product and kronecker_sum
# unless --attention_type is given.
python train.py --models all --device cuda

# Subset
python train.py --models var1 dlinear patchtst

# Override a hyperparameter (only routed if the target model accepts it)
python train.py --models hot --attention_type kronecker_sum --epochs 200
python train.py --models dyngwn --nhid 64 --graph_mode adaptive_only
```

Each model script writes to its own `<Model>/results/<auto-named-dir>/`:
- `pred.npy`         — scaled-space predictions (gitignored)
- `start_dates.npy`  — alignment dates
- `config.json`      — full hyperparam record (used for `--predict_only`)
- `best_model.pt`    — checkpoint of best validation epoch (neural models only)
- `train_log.csv`    — per-epoch train/val loss

### Compare

```bash
python compare_models.py
```

For every model in `MODELS`, finds the most-recent matching results dir.
- If `pred.npy` is present → load it.
- Else if `config.json` (+ `best_model.pt` for neural) exists → call the
  script's `--predict_only` mode to regenerate `pred.npy`, then load it.
- Else → skip the model with a clear reason.

Persistence baseline `Persist(ref)` is computed live from the CSV inside
`build_reference()` — not a separate model entry.

Metrics (all in scaled space): MSE, RMSE, MAE, RSE, signed bias, directional
accuracy, Spearman rank-IC (mean and std over 400 features per window-step).
Per-horizon breakdown at t+{1, 5, 10, 21, 42, 63}.

### Predict-only (regenerate pred.npy from a checkpoint)

```bash
python <script> --predict_only --out_dir <existing-dir>
```

The script reads `config.json`, rebuilds the model with the saved
hyperparameters, loads `best_model.pt`, runs inference, writes `pred.npy`.
Used automatically by `compare_models.py` when a `pred.npy` is missing.

VAR1 has no checkpoint; `--predict_only` re-fits OLS from the CSV (~2 sec).

## Colab roundtrip

```bash
# On Colab
git pull
python train.py --models all --device cuda
git add -A && git commit -m "trained" && git push

# Locally
git pull
python compare_models.py    # auto-regens missing pred.npy from checkpoints
```

`pred.npy` files (~75 MB each) are gitignored; only the small artifacts
(`config.json`, `best_model.pt`, `start_dates.npy`, `train_log.csv`) move
through git, and predictions are reconstituted locally on demand.

## Model architectures — fidelity to legacy implementations

Each standalone script reproduces the architecture of its reference
implementation (PatchTST-main, HOT/src, DynGWN/model.py) module for module.
Architectural details:

- **VAR1**: global VAR(1) by ordinary least squares with intercept on the
  scaled training set. **No regularisation, no tuning** — kept simple to
  match the "no-tuning" treatment of the neural models.
- **DLinear**: channel-independent decomposition (boundary-padded moving
  average, `kernel_size=13`) + per-channel linear maps for trend and
  seasonality. Vectorised einsum over channels (~100× faster than the
  per-channel `ModuleList` of the original repo, mathematically identical).
- **PatchTST**: full backbone inlined — RevIN, channel-independent encoder,
  residual scaled-dot-product attention with learnable scale, BatchNorm
  pre/post sublayers, GELU FFN. Faithful to legacy defaults:
  `padding_patch='end'` (replicates the last value `stride` times before
  unfolding, gives `patch_num+1` patches), RevIN `affine=False`. The
  `--padding_patch none` and `--revin_affine 1` overrides exist for ablation.
- **HOT**: full block inlined — RotaryEmbedding (default `--pe rope`,
  applied on the temporal dim via `rope_dims=[3]`), RMSNorm, SwiGLU FFN,
  KroneckerAttention with `num_modes=3` (H, W, time-patches). Defaults
  match `HOT/src/models/ts_tensor.py`. `--pe nope` disables RoPE.
- **DynGWN**: WaveNet-style dilated temporal convolutions + graph
  convolution at each step, with adaptive adjacency (`nodevec1 @ nodevec2`)
  and optionally a fixed 4-neighbor moneyness×tau grid adjacency
  (`--graph_mode grid_plus_adaptive`, default; matches legacy run).
  `--graph_mode adaptive_only` drops the grid. Receptive field 13 with
  blocks=4, layers=2, kernel=2. The legacy `dynamic_gcn_bool` mode (sliding
  correlation matrices) is NOT ported because the SPX runs always had it off.
- **DynGWN training loss**: matches legacy `engine.py` —
  `masked_mae(scaler.inverse_transform(model_output), y_original)` in the
  ORIGINAL IV space. Inputs `x` are scaled; targets `y` are kept in
  original space; the model output is inverse-transformed inside the loss.
  `pred.npy` is still saved in scaled space (compare_models contract).

## Implementation details

- **Train-only scaler**: every script fits `StandardScaler` on `iv[:n_train]`.
  Means/stds are not persisted — they are re-derived deterministically from
  the CSV. DynGWN's training loop additionally retains the scaler in memory
  to inverse-transform model output for the masked-MAE loss.
- **HOT reshape**: `iv.reshape(-1, 20, 20, order='F')` maps CSV column k to
  `[i_mono, i_tau]`. Cross-checked against `compare_models.load_hot`.
- **No alphabetical sort of `iv_*` columns**. The CSV is already sorted as
  (tau outer, moneyness inner); alphabetical sort breaks this because
  `'iv_0.9105_0.04'` sorts before `'iv_0.9_0.04'` (`'1' < '_'`). Sorting was
  the original cause of a 4× IC collapse in early runs. Don't reintroduce.
- **Test windows from CSV**: `n_test = int(T*0.20)`, first prediction date
  index = `T - n_test`. Yields 776 test windows for T=4191.

## Editing rules

- Don't reintroduce shared model code across scripts. Each `_spx_iv.py`
  must be deletable without affecting the others.
- Keep the data pipeline in every script byte-identical (CSV column order,
  same border formula, same scaler). If you change one, change all five.
- New result paths must match the `regen_dir` glob in `compare_models.MODELS`.
- HOT must save `[N, H, W, pred_len]`; flat models save `[N, pred_len, 400]`.
- When training, always emit `config.json` alongside `best_model.pt` so
  `--predict_only` and `compare_models.py` auto-regen still work.
