## Volatility Surface Forecasting

This project forecasts SPX implied volatility (IV) surfaces on a fixed grid.
It provides a clean pipeline for training, testing, and evaluating multiple
models on rolling windows, with results saved for comparison and plotting.

### Data

Input data lives in `SPX_IV_fixed_grid.csv`, a fixed grid of implied volatility
surfaces. It is produced by the preprocessing scripts in `data_prep/`.
Each row is one surface point with fields like:

- `date`: trading date
- `tau`: time to maturity (years)
- `log_moneyness`: log(K / S)
- `implied_volatility`: IV value at that grid point

The loader reshapes these into a 3D tensor:
`(n_dates, n_tau, n_logm)`.

### Windows (Train / Val / Test)

We use rolling windows to evaluate stability across time:

- Window size: 10 years
- Split: 7 years train, 1 year val, 2 years test (7:1:2)
- Shift: 6 months forward per window

This yields multiple overlapping windows across the dataset.

### Metrics

Evaluation is in IV space (no conversion):

- `IV RMSE`: primary error metric
- `Relative RMSE`: RMSE normalized by true IV
- `MAE`: mean absolute error

We also compute metrics by maturity bucket and by moneyness region
(ATM vs OTM).

### Models

Each model is trained and evaluated on the same windows and horizons.
Add a new model by implementing `models/base_model.py` and using
`ForecastingPipeline.run_model`.

- `PersistenceModel` (`models/persistence/`):
  Baseline that predicts the last observed surface.
  Run via `run_model.py --model persistence`.

- `TransformerSurfaceModel` (`models/transformer/`):
  Temporal transformer that forecasts surfaces directly.
  Run via `run_model.py --model transformer`.

### Pipeline Overview

The pipeline lives in `forecasting/`:

- `data_loader.py`: loads `SPX_IV_fixed_grid.csv` and builds sequences
- `splits.py`: creates rolling train/val/test windows
- `pipeline.py`: model-agnostic training, testing, and evaluation runner

Core flow:

1. Load data to `(n_dates, n_tau, n_logm)`
2. Create rolling windows
3. For each window, context length, and horizon:
   - build sequences
   - train model
   - predict on test sequences
   - compute metrics
   - save forecasts and summary metrics

### Results and Outputs

Outputs are written to `results/`:

- `results/forecasts/`: per-configuration `.npz` predictions
- `results/metrics/`: aggregated JSON metrics
- `results/plots/`: visualizations (from plotting scripts)

### Visualization

Plotting is handled by `evaluation/visualizer.py` and is model-agnostic.
Use `visualize_results()` to point at any metrics JSON file:

```
from evaluation.visualizer import visualize_results

visualize_results(
    results_file='results/metrics/persistence_results.json',
    model_id='persistence',
    model_label='Persistence Model'
)
```

The function generates:
- Overall line and bar summaries
- Per-configuration plots by context length and horizon
- A combined grid summary across all configurations

`plot_results.py` is the generic CLI wrapper. It infers the results file
and label from `--model` and writes plots to `results/plots`:

```
python plot_results.py --model persistence
```

`run_model.py` is the generic training/evaluation CLI. It infers the model
from `--model` and writes outputs to `results/`:

```
python run_model.py --model persistence
python run_model.py --model transformer --save-ckpt --amp
```

`tune_model.py` is the tuning CLI. It runs a parameter grid on validation
splits for a given window/context/horizon and writes to `results/tuning/`:

```
python tune_model.py --model transformer --window-ids 0 --context 21 --horizon 21
python tune_model.py --model transformer --amp
```

### File Structure

```
data_prep/         Raw data prep and grid construction
forecasting/       Data loading, splits, main pipeline
models/            Model implementations
models/checkpoints/<model_id>/  Best checkpoints per window/config
evaluation/        Metrics and plotting utilities
results/           Forecasts, metrics, plots, tuning outputs
report/            Thesis report and references
plot_results.py    Generic plotting CLI
run_model.py       Generic model runner CLI
tune_model.py      Generic tuning CLI
ADDING_MODEL.md    Guide for adding new models
COLAB_GUIDE.md     Colab setup and tuning instructions
```
