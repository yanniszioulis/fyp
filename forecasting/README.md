# Forecasting Pipeline

## Structure

```
forecasting/
├── data_loader.py      # Load SPX_IV_fixed_grid.csv and reshape to 3D
├── splits.py          # Create rolling window splits (6-month shifts)
└── pipeline.py        # Main pipeline orchestrator

models/
├── base_model.py      # Base class for all models
└── persistence/
    └── persistence_model.py  # Persistence baseline

evaluation/
└── metrics.py         # IV RMSE, relative RMSE, MAE, etc.
```

## Usage

```python
from forecasting.pipeline import ForecastingPipeline

# Initialize pipeline
pipeline = ForecastingPipeline()

# Load data
pipeline.load_data()

# Create rolling windows (10 years, 7:1:2 split, 6-month shifts)
pipeline.create_windows(
    window_size_years=10.0,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    shift_months=6
)

# Run persistence model
results = pipeline.run_persistence(
    context_lengths=[5, 21, 63],  # 1 week, 1 month, 3 months
    horizons=[1, 5, 21],  # 1 day, 1 week, 1 month
    save_results=True
)
```

## Rolling Windows

With data from 2009 to mid-2023 and 6-month shifts, you'll get approximately 7 windows:

1. Window 1: 2009-01-02 to 2019-01-02
2. Window 2: 2009-07-02 to 2019-07-02
3. Window 3: 2010-01-02 to 2020-01-02
4. Window 4: 2010-07-02 to 2020-07-02
5. Window 5: 2011-01-02 to 2021-01-02
6. Window 6: 2011-07-02 to 2021-07-02
7. Window 7: 2012-01-02 to 2022-01-02

Each window:
- Train: 7 years
- Val: 1 year
- Test: 2 years

## Results

Results are saved to `results/` directory:
- `results/forecasts/`: Individual forecast files (.npz)
- `results/metrics/persistence_results.json`: Summary metrics
- `results/plots/`: Visualization plots (to be added)
