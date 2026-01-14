## Data Preparation

This folder contains the raw IvyDB CSVs and the script that builds the
fixed-grid implied volatility surface file used by the forecasting pipeline.

### Source CSVs (IvyDB US v6.0 Reference Manual)

The two inputs are IvyDB exports:

- `SPX_std_option.csv`: Standard option quotes for SPX.
  Fields used in the pipeline (per manual):
  `date`, `days`, `cp_flag`, `strike_price`, `forward_price`,
  `impl_volatility`.

- `SPX_surfaces.csv`: Preprocessed surface quotes for SPX.
  Fields used in the pipeline (per manual):
  `date`, `days`, `cp_flag`, `impl_strike`, `impl_volatility`.

The manual (`IvyDB_US_v6.0_Reference_Manual.pdf`) describes the full
schema and field definitions for these files.

### Output

`SPX_IV_fixed_grid.csv` (in the repo root) is the main dataset used by the
forecasting pipeline. Each row represents a single point on a fixed grid
with fields:

- `date`
- `tau` (years)
- `log_moneyness`
- `implied_volatility`

### Processing Steps

The script `create_vol_surface_grid.py` produces `SPX_IV_fixed_grid.csv`
from the two IvyDB CSVs:

1. Read both files in chunks.
2. Keep dates from 2009-01-01 onward and only dates common to both files.
3. Exclude 10-day maturities (`days` > 10).
4. Convert `days` to `tau` in years.
5. Compute `log_moneyness = log(K / F)` using:
   - `strike_price` and `forward_price` for `SPX_std_option.csv`
   - `impl_strike` and `forward_price` (merged from std options) for `SPX_surfaces.csv`
6. Filter to out-of-the-money options:
   - Calls: `log_moneyness > 0`
   - Puts: `log_moneyness < 0`
   - For `SPX_std_option.csv`, include ATM using a small tolerance.
7. Interpolate each day’s surface to a fixed grid using cubic spline
   (fallback to linear/nearest where needed).
8. Sample onto the fixed grid:
   - `tau`: 0.1 to 2.0 years (step 0.1, 20 points)
   - `log_moneyness`: -0.5 to 0.5 (step 0.05, 21 points)

Run the script from `data_prep/`:

```
python create_vol_surface_grid.py
```
