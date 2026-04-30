# SPX IV Surface Forecasting

Research project on the SPX implied-volatility surface. Five reference
baselines (VAR(1), DLinear, PatchTST, HOT, DynGWN) plus our proposed model
**DCISM**, each compressed into one self-contained training script with a
uniform input/output contract, dispatched via a single `train.py`, and
benchmarked by `compare_models.py`.

**Data**: SPX IV surface, 400 features (20 moneyness × 20 tau), 4191 days.
**Task**: seq_len=21 → pred_len=63 (1-month context → 3-month horizon).
**Split**: canonical 70/10/20 (train/val/test) — identical across every model.

## Headline result

`DLinear` trained with **Huber loss** (`--loss huber_scaled --huber_delta 1.0`)
is the dominant model in the benchmark:

- Best overall Spearman IC (+0.591 vs runner-up DCISM-Huber +0.584)
- Best long-horizon IC at t+63 (+0.433 vs +0.421)
- Matches persistence at t+1 (IC 0.933)
- MSE 0.147 — within noise of the best (0.146 from DCISM-Huber)

The architectural conv-polish in DCISM helps when the loss is MAE, but its
value disappears once Huber is used: DLinear-Huber matches/exceeds DCISM-Huber
on every metric. **The right loss matters more than added architecture.**

See `thoughts.md` for the full ablation walkthrough and rejected hypotheses.

## Directory layout

```
fyp/
├── train.py                       # Master dispatcher
├── compare_models.py              # Unified evaluation harness
├── SPX_surfaces.csv               # Raw data (400 iv_* columns + date)
├── thoughts.md                    # Working brainstorm + ablation history
│
├── VAR1/var1_spx_iv.py            # Plain OLS VAR(1)
├── DLinear/dlinear_spx_iv.py      # Channel-independent DLinear (einsum)
├── PatchTST/patchtst_spx_iv.py    # Patch-based Transformer + RevIN
├── HOT/hot_spx_iv.py              # Kronecker-attention transformer
├── DynGWN/dyngwn_spx_iv.py        # Graph WaveNet
├── DCISM/                         # OUR proposed model
│   ├── dcism_spx_iv.py            # DLinear core + identity-init 2D conv polish
│   └── apply_bias_correction.py   # Post-hoc per-channel bias correction utility
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

# Loss-function ablation (DLinear & DCISM accept --loss / --huber_delta)
python train.py --models dlinear --loss huber_scaled --huber_delta 1.0
python train.py --models dcism   --loss mae_original
python train.py --models dlinear --loss mae_scaled        # uniform-weight MAE
```

Available `--loss` choices for DLinear and DCISM:
- `mse` (default) — MSE in scaled space
- `mae_original` — masked-MAE on inverse-transformed predictions vs raw IV;
  equivalent to std-weighted MAE in scaled space
- `mae_scaled` — plain MAE in scaled space (uniform per-channel weighting)
- `huber_scaled` — Huber/smooth-L1 in scaled space; quadratic for `|err| < δ`,
  linear above. Threshold via `--huber_delta` (default 1.0).

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

### Bias correction (post-hoc)

```bash
python DCISM/apply_bias_correction.py \
    --src_dir DCISM/results/SPX_IV_21_63_DCISMv0_k13_ck3_ep100_lossmae
```

Reads `<src_dir>` (must contain `config.json` + `best_model.pt`), fits a
per-channel additive offset on the validation set, applies it to test
predictions, and writes a sibling `<src_dir>_bc/` containing copies of
checkpoint/config plus `bias.npy` and bias-corrected `pred.npy`.

`compare_models.py` auto-runs this when a `*_bc` dir is missing `pred.npy`
but its source dir has the checkpoint.

> **Note:** in our benchmark a static per-channel offset fit on val
> *over-corrected* the test set due to regime drift between val (~2021-22)
> and test (2022-04 → 2025-06, includes the 2022-23 vol regime change).
> The utility is kept for ablation and for cleaner-distribution use cases,
> but is not part of the recommended pipeline. See `thoughts.md`.

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
- **DCISM** (proposed): DLinear core (boundary-padded MA decomposition +
  channel-independent einsum maps for trend and season components) followed
  by an identity-initialised 2D convolution (default `--conv_kernel 3`)
  applied to the predicted surface reshaped as `[H_mono=20, W_tau=20, pred_len]`.
  The polish layer starts as exactly the identity and only learns to deviate
  if the data supports it. ~1.14M parameters (≈ DLinear's 1.11M + ~35K conv).
  Optional `--loss {mse, mae_original, mae_scaled, huber_scaled}`.

## Recommended configuration (this benchmark)

```bash
python train.py --models dlinear --loss huber_scaled --huber_delta 1.0
```

Best overall IC, best long-horizon IC, matches persistence at t+1, MSE
within 0.5% of the best. `DCISMv0(Hub-1)` is statistically tied on MSE
but loses slightly on IC — not worth the architectural complexity once
Huber is in place.

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
