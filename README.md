# SPX IV Surface Forecasting

Research project on the S&P 500 implied-volatility surface. Per-day surface
is **170 cells** (10 maturities × 17 call-equivalent deltas Δ); models
forecast `pred_len=63` days from `seq_len=21` days of context.

## Data

```
SPX_surfaces.csv     date, iv_T30_D10 … iv_T730_D90   (170 IV cells)
SPX_dispersion.csv   date, disp_T30_D10 … disp_T730_D90
SPX_underlying.csv   date, underlying_price
```

Built from OptionMetrics IvyDB by `data_prep/preprocess_volatility_surface.py`.
Column ordering is **locked** (maturities outer, Δ inner) — do not re-sort.

| Dataset      | Date range                  | Rows |
|--------------|-----------------------------|------|
| `full`       | 2009-01-02 → 2025-08-29     | 4191 |
| `precovid`   | 2009-01-02 → 2019-12-31     | 2768 |

Datasets are not separate files: every script accepts
`--dataset {full,precovid}` and slices the CSV in-place. The 70/10/20 split
is computed **on the slice**, so `precovid` train ends 2016-09-12 and test
runs 2017-10 → 2019-10.

## Models

| Model    | Script                            | New-data status |
|----------|-----------------------------------|-----------------|
| VAR(1)   | `VAR1/var1_spx_iv.py`             | **ported**      |
| DLinear  | `DLinear/dlinear_spx_iv.py`       | needs port (170 + dataset flag) |
| PatchTST | `PatchTST/patchtst_spx_iv.py`     | needs port (170 + dataset flag) |
| HOT      | `HOT/hot_spx_iv.py`               | needs port (10×17 reshape + dataset flag) |
| DynGWN   | `DynGWN/dyngwn_spx_iv.py`         | deferred — uses different loss space |

All ported scripts derive `n_iv` from the CSV and respect the same I/O contract
(see below).

## Workflow

### Train

```bash
# Full dataset
python VAR1/var1_spx_iv.py --seq_len 21 --pred_len 63

# Precovid (2009 → 2019-12-31)
python VAR1/var1_spx_iv.py --dataset precovid --seq_len 21 --pred_len 63
```

`train.py` (master dispatcher) is currently calibrated to the old 400-cell
layout and will be re-wired alongside the model ports. For now invoke
each model script directly.

### Compare

```bash
python compare_models.py --dataset precovid --seq_len 21 --pred_len 63
```

For every model in the registry, finds the most-recent matching
`{dataset}_SPX_IV_{seq_len}_{pred_len}_*` dir.
- If `pred.npy` is present → load it.
- Else if `config.json` (+ `best_model.pt` for neural) exists → call the
  script's `--predict_only` mode to regenerate `pred.npy`, then load it.
- Else → skip the model with a clear reason (verbose).

`Persist(ref)` is computed live from the CSV inside `build_reference()` —
not a separate model entry.

After scoring, each model gets a self-describing `metrics_test.json` written
**into its own results dir** so a single model can be inspected without
re-running the whole comparison.

### Predict-only (regenerate pred.npy from a checkpoint)

```bash
python <script> --predict_only --out_dir <existing-dir>
```

Reads `config.json`, rebuilds the model with saved hyperparameters, loads
`best_model.pt`, runs inference, writes `pred.npy`. Used automatically by
`compare_models.py` when `pred.npy` is missing. VAR1 has no checkpoint;
`--predict_only` re-fits OLS in ~2 seconds.

## I/O contract (every model script)

Result dir: `MODEL/results/{dataset}_SPX_IV_{seq_len}_{pred_len}_MODEL[_extras]/`

| File             | In git? | Purpose                                                  |
|------------------|---------|----------------------------------------------------------|
| `config.json`    | ✓       | Hyperparams + dataset + n_train/val/test + train_end_date + git_commit + seed |
| `best_model.pt`  | ✓       | Checkpoint of best validation epoch (neural only)        |
| `start_dates.npy`| ✓       | `[N_test]` window-start datetime64[D] for date alignment |
| `train_log.csv`  | ✓       | Per-epoch train/val loss                                 |
| `pred.npy`       | ✗       | `[N_test, pred_len, n_iv]` scaled-space predictions      |
| `metrics_test.json` | ✓    | Full test scoreboard for this one model (written by `compare_models`) |

Predictions are float32 in scaled space (StandardScaler fit on train).
HOT (when ported) saves `[N_test, H, W, pred_len]` with `H*W = n_iv`.

## Colab roundtrip

```bash
# On Colab
git pull
python <model_script> --dataset precovid --device cuda
git add VAR1/results/<dir>/{config.json,best_model.pt,start_dates.npy,train_log.csv}
git commit -m "trained" && git push

# Locally
git pull
python compare_models.py --dataset precovid    # auto-regens missing pred.npy
```

`pred.npy` (~20 MB per model) is gitignored; only the small artefacts move
through git, and predictions are reconstituted locally on demand.

## Implementation notes

- **No alphabetical sort of `iv_*` columns.** The CSV is sorted (maturities
  outer, Δ inner); the integer-encoded names (`iv_T30_D10`, …) avoid the
  alphabetical-sort bug that plagued the legacy moneyness-grid layout.
- **Train-only scaler:** every script fits `StandardScaler` on `iv[:n_train]`.
  Scaler params are not persisted — they are re-derived from the CSV using
  the dataset and the same split formula recorded in `config.json`.
- **Dataset slice happens before the split.** Precovid train/val/test sizes
  are computed on the 2768-row slice, not on a percentage of the full CSV.

## Editing rules

- Don't reintroduce shared model code across scripts. Each `_spx_iv.py`
  must be deletable without affecting the others.
- Keep the data pipeline in every script byte-identical (CSV column order,
  same border formula, same scaler). If you change one, change all.
- New result paths must match the registry glob in `compare_models.MODELS`.
- Always emit `config.json` alongside `best_model.pt` so `--predict_only`
  and `compare_models` auto-regen still work.
