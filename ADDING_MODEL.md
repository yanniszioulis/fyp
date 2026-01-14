## Adding a New Model

This guide describes how to add a new forecasting model so it integrates
with training, tuning, evaluation, and plotting in this repo.

### 1) Implement the Model

Create a new module under `models/<model_id>/` and implement `BaseModel`:

- File: `models/<model_id>/<model_file>.py`
- Class: `<ModelClass>` extends `models/base_model.py`
- Required methods:
  - `fit(X_train, y_train, context_length, horizon, **kwargs)`
  - `predict(X)` or `predict_horizon(X, horizon)`

Expect input shapes:
- `X`: `(n_samples, context_length, n_tau, n_logm)`
- `y`: `(n_samples, n_tau, n_logm)`

If you want checkpointing, implement:

- `save_checkpoint(path)` that writes a model file to disk

### 2) Wire Into Training CLI (`run_model.py`)

Add a new branch for `--model <model_id>`:

- Instantiate your model in the `model_factory` or direct call.
- Decide whether to use validation:
  - `train_on_val=False` and `use_val=True` for early stopping.
  - `train_on_val=True` and `use_val=False` if you want to train on train+val.
- Expose any hyperparameters as CLI args.
- If you want checkpoints:
  - Add a `--save-ckpt` flag and pass `save_checkpoints=True`.

### 3) Add Plotting Support (`plot_results.py`)

Update the maps so plots and labels are inferred:

- `label_map["<model_id>"] = "Readable Model Name"`
- `results_map["<model_id>"] = "results/metrics/<model_id>_results.json"`

### 4) Add Tuning Support (`tune_model.py`)

For tuning, add a grid spec and the model branch:

- Create `tuning/<model_id>_grid.json`
- Update `tune_model.py` to accept your model and map grid parameters
  into your model constructor.

The tuning pipeline:
- Uses windowed train/val splits (`use_val=True`)
- Evaluates on **validation** with `compute_all_metrics`
- Writes results to `results/tuning/`

### 5) Update Docs

Update `README.md`:
- Add the model to the “Models” section
- Add any run or tuning examples
- Note checkpoint location if applicable

Update `models/README.md` if you add new checkpoint folders.

### 6) Optional: Dependencies

If your model needs new packages (e.g., `torch`, `jax`), add them to
`requirements.txt`.

### 7) Expected Outputs

After running:
- `results/metrics/<model_id>_results.json`
- `results/forecasts/<model_id>_w*_c*_h*.npz` (if saving)
- `results/plots/<model_id>_*.png` from `plot_results.py`
- `models/<model_id>/checkpoints/` (if enabled)

### 8) Sanity Checklist

- Model accepts `(context, n_tau, n_logm)` inputs without reshaping errors
- Validation split is used when early stopping is enabled
- Metrics are saved and plots generate without code changes
- Checkpoints save/load cleanly
