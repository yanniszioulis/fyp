# Diebold–Mariano significance analysis (pred_len = 21)

Predictions averaged across 6 seeds per deep model. Loss = per-window mean squared error across the 150 cells at a single horizon, in standardised log-IV space. Persistence = broadcast of the last input row. VAR = VAR(1) on first-differenced standardised log-IV, cumulated onto the last input row (`VAR/results/63_21/preds.npy`).

Two tests are reported per comparison:

- **Primary (concrete)**: sub-sample test windows at stride = P = 21 days so consecutive sampled forecast windows have *no overlap*; loss differentials are effectively non-autocorrelated and a plain one-sample, one-sided Student-t (H1: model loss < baseline loss) gives a HAC-free p-value. Trade-off: ≈21× fewer windows per regime, so power is low (~12 windows per regime, ~48 for Full).

- **Sensitivity (HAC)**: keep every window and use DM with Newey–West Bartlett HAC truncated at lag h-1 + Harvey–Leybourne–Newbold small-sample correction. Higher power; depends on the HAC kernel choice.

Star coding: `***` p<0.01, `**` p<0.05, `*` p<0.10.

## Seeds used

- DLinear: 6 seeds
- PatchTST: 6 seeds
- HOT (k-sum): 6 seeds
- HOT (k-prod): 6 seeds
- Tucker: 6 seeds
- GWN: 6 seeds
- iTransformer: 6 seeds
- PCAFormer: 6 seeds

## h = 21 by regime

### Full  (n_full=1007 windows, n_sub=48)

| model | MSE | gain% | t_prim | p_prim | p_HAC | vs_VAR_p_prim | vs_VAR_p_HAC |
|---|---|---|---|---|---|---|---|
| Tucker | 0.3892 | +10.7 | -0.45 | 0.327  | 0.020 ** | 0.328  | 0.017 ** |
| DLinear | 0.4124 | +5.4 | -0.38 | 0.351  | 0.101  | 0.354  | 0.092 * |
| iTransformer | 0.4332 | +0.7 | +0.41 | 0.658  | 0.452  | 0.668  | 0.451  |
| PCAFormer | 0.4351 | +0.2 | -0.76 | 0.224  | 0.486  | 0.223  | 0.486  |
| HOT (k-prod) | 0.4352 | +0.2 | -0.11 | 0.456  | 0.487  | 0.464  | 0.487  |
| PatchTST | 0.4365 | -0.1 | -0.26 | 0.399  | 0.508  | 0.410  | 0.508  |
| HOT (k-sum) | 0.4379 | -0.4 | +0.43 | 0.665  | 0.532  | 0.678  | 0.533  |
| GWN | 0.4390 | -0.7 | +0.43 | 0.667  | 0.543  | 0.676  | 0.544  |

### COVID  (n_full=254 windows, n_sub=13)

| model | MSE | gain% | t_prim | p_prim | p_HAC | vs_VAR_p_prim | vs_VAR_p_HAC |
|---|---|---|---|---|---|---|---|
| Tucker | 1.0350 | +10.8 | +0.08 | 0.531  | 0.062 * | 0.511  | 0.056 * |
| DLinear | 1.1123 | +4.2 | +0.22 | 0.587  | 0.259  | 0.563  | 0.246  |
| GWN | 1.1591 | +0.2 | +0.61 | 0.724  | 0.493  | 0.717  | 0.489  |
| iTransformer | 1.1762 | -1.3 | +0.60 | 0.721  | 0.562  | 0.709  | 0.561  |
| PCAFormer | 1.2080 | -4.1 | +0.06 | 0.525  | 0.681  | 0.500  | 0.683  |
| HOT (k-prod) | 1.2216 | -5.2 | +0.57 | 0.712  | 0.732  | 0.690  | 0.737  |
| PatchTST | 1.2448 | -7.2 | +0.78 | 0.774  | 0.862  | 0.747  | 0.864  |
| HOT (k-sum) | 1.2481 | -7.5 | +1.22 | 0.877  | 0.862  | 0.872  | 0.867  |

### Reflation  (n_full=252 windows, n_sub=12)

| model | MSE | gain% | t_prim | p_prim | p_HAC | vs_VAR_p_prim | vs_VAR_p_HAC |
|---|---|---|---|---|---|---|---|
| PCAFormer | 0.1146 | +24.3 | -1.62 | 0.067 * | 0.010 ** | 0.094 * | 0.012 ** |
| Tucker | 0.1219 | +19.5 | -1.55 | 0.074 * | 0.019 ** | 0.094 * | 0.017 ** |
| iTransformer | 0.1264 | +16.5 | -0.99 | 0.171  | 0.004 *** | 0.249  | 0.004 *** |
| DLinear | 0.1353 | +10.7 | -0.97 | 0.177  | 0.002 *** | 0.276  | 0.002 *** |
| HOT (k-sum) | 0.1370 | +9.5 | -0.94 | 0.185  | 0.220  | 0.240  | 0.240  |
| PatchTST | 0.1396 | +7.8 | -1.18 | 0.131  | 0.216  | 0.192  | 0.241  |
| HOT (k-prod) | 0.1522 | -0.5 | -0.53 | 0.302  | 0.513  | 0.356  | 0.526  |
| GWN | 0.1545 | -2.1 | -0.28 | 0.392  | 0.567  | 0.452  | 0.586  |

### Bear 2022  (n_full=251 windows, n_sub=12)

| model | MSE | gain% | t_prim | p_prim | p_HAC | vs_VAR_p_prim | vs_VAR_p_HAC |
|---|---|---|---|---|---|---|---|
| PatchTST | 0.1914 | +23.3 | -1.65 | 0.064 * | 0.031 ** | 0.057 * | 0.027 ** |
| HOT (k-sum) | 0.1959 | +21.5 | -1.62 | 0.066 * | 0.048 ** | 0.062 * | 0.044 ** |
| HOT (k-prod) | 0.2012 | +19.3 | -1.32 | 0.106  | 0.099 * | 0.101  | 0.093 * |
| DLinear | 0.2213 | +11.3 | -2.15 | 0.027 ** | 0.049 ** | 0.020 ** | 0.036 ** |
| PCAFormer | 0.2254 | +9.6 | -1.81 | 0.049 ** | 0.266  | 0.044 ** | 0.257  |
| Tucker | 0.2415 | +3.2 | -1.33 | 0.105  | 0.413  | 0.097 * | 0.398  |
| iTransformer | 0.2468 | +1.0 | -0.52 | 0.307  | 0.455  | 0.308  | 0.436  |
| GWN | 0.2774 | -11.3 | -0.39 | 0.352  | 0.806  | 0.350  | 0.806  |

### Normalisation  (n_full=250 windows, n_sub=12)

| model | MSE | gain% | t_prim | p_prim | p_HAC | vs_VAR_p_prim | vs_VAR_p_HAC |
|---|---|---|---|---|---|---|---|
| Tucker | 0.1509 | +13.2 | -0.86 | 0.204  | 0.039 ** | 0.239  | 0.047 ** |
| GWN | 0.1564 | +10.1 | -1.00 | 0.170  | 0.282  | 0.188  | 0.292  |
| HOT (k-prod) | 0.1564 | +10.1 | -0.73 | 0.240  | 0.274  | 0.285  | 0.283  |
| PatchTST | 0.1607 | +7.6 | -0.47 | 0.324  | 0.224  | 0.373  | 0.237  |
| HOT (k-sum) | 0.1610 | +7.4 | -0.88 | 0.199  | 0.298  | 0.237  | 0.311  |
| DLinear | 0.1726 | +0.8 | -0.26 | 0.401  | 0.434  | 0.513  | 0.477  |
| iTransformer | 0.1745 | -0.3 | +0.33 | 0.625  | 0.514  | 0.675  | 0.535  |
| PCAFormer | 0.1836 | -5.5 | +1.27 | 0.886  | 0.635  | 0.902  | 0.646  |

## Per-horizon view by regime (vs persistence)

### Full

**gain (% MSE reduction vs persistence)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | -2.4 | -4.0 | +0.8 | +2.9 | +5.4 |
| GWN | -84.9 | -22.2 | -12.2 | -5.2 | -0.7 |
| HOT (k-prod) | -57.9 | -20.0 | -11.1 | -5.8 | +0.2 |
| HOT (k-sum) | -44.1 | -17.8 | -11.2 | -4.7 | -0.4 |
| PCAFormer | -21.6 | -11.0 | -6.0 | -2.8 | +0.2 |
| PatchTST | -7.8 | -9.9 | -4.0 | -1.4 | -0.1 |
| Tucker | -67.9 | -9.4 | +2.2 | +7.0 | +10.7 |
| iTransformer | -61.2 | -18.7 | -9.0 | -3.4 | +0.7 |

**p_primary (stride=21, t-test)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.206  | 0.591  | 0.590  | 0.488  | 0.351  |
| GWN | 0.670  | 0.662  | 0.720  | 0.710  | 0.667  |
| HOT (k-prod) | 0.887  | 0.777  | 0.762  | 0.686  | 0.456  |
| HOT (k-sum) | 0.608  | 0.786  | 0.797  | 0.747  | 0.665  |
| PCAFormer | 0.633  | 0.477  | 0.700  | 0.457  | 0.224  |
| PatchTST | 0.323  | 0.719  | 0.675  | 0.581  | 0.399  |
| Tucker | 0.953  | 0.472  | 0.526  | 0.433  | 0.327  |
| iTransformer | 0.875  | 0.715  | 0.716  | 0.652  | 0.658  |

**p_HAC (all windows, DM-HLN, NW lag=h-1)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.820  | 0.777  | 0.447  | 0.308  | 0.101  |
| GWN | 1.000  | 0.966  | 0.842  | 0.703  | 0.543  |
| HOT (k-prod) | 1.000  | 0.932  | 0.822  | 0.714  | 0.487  |
| HOT (k-sum) | 1.000  | 0.932  | 0.824  | 0.687  | 0.532  |
| PCAFormer | 1.000  | 0.947  | 0.762  | 0.632  | 0.486  |
| PatchTST | 0.924  | 0.886  | 0.680  | 0.570  | 0.508  |
| Tucker | 1.000  | 0.855  | 0.377  | 0.128  | 0.020 ** |
| iTransformer | 1.000  | 0.982  | 0.842  | 0.672  | 0.452  |

### COVID

**gain (% MSE reduction vs persistence)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | -3.9 | -6.8 | -0.8 | +1.2 | +4.2 |
| GWN | -108.5 | -30.7 | -16.5 | -6.6 | +0.2 |
| HOT (k-prod) | -97.4 | -37.4 | -21.0 | -13.1 | -5.2 |
| HOT (k-sum) | -70.4 | -35.3 | -23.9 | -13.7 | -7.5 |
| PCAFormer | -22.6 | -18.5 | -14.1 | -9.3 | -4.1 |
| PatchTST | -14.7 | -21.2 | -12.1 | -9.0 | -7.2 |
| Tucker | -85.1 | -13.5 | +1.8 | +6.9 | +10.8 |
| iTransformer | -72.0 | -27.3 | -13.5 | -6.5 | -1.3 |

**p_primary (stride=21, t-test)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.171  | 0.662  | 0.599  | 0.540  | 0.587  |
| GWN | 0.250  | 0.628  | 0.757  | 0.742  | 0.724  |
| HOT (k-prod) | 0.708  | 0.800  | 0.718  | 0.694  | 0.712  |
| HOT (k-sum) | 0.251  | 0.808  | 0.771  | 0.772  | 0.877  |
| PCAFormer | 0.200  | 0.525  | 0.661  | 0.556  | 0.525  |
| PatchTST | 0.221  | 0.741  | 0.636  | 0.622  | 0.774  |
| Tucker | 0.859  | 0.455  | 0.577  | 0.524  | 0.531  |
| iTransformer | 0.560  | 0.709  | 0.668  | 0.631  | 0.721  |

**p_HAC (all windows, DM-HLN, NW lag=h-1)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.783  | 0.765  | 0.533  | 0.446  | 0.259  |
| GWN | 1.000  | 0.919  | 0.792  | 0.666  | 0.493  |
| HOT (k-prod) | 1.000  | 0.938  | 0.857  | 0.797  | 0.732  |
| HOT (k-sum) | 0.999  | 0.950  | 0.886  | 0.821  | 0.862  |
| PCAFormer | 0.987  | 0.941  | 0.847  | 0.765  | 0.681  |
| PatchTST | 0.918  | 0.925  | 0.806  | 0.763  | 0.862  |
| Tucker | 1.000  | 0.805  | 0.439  | 0.230  | 0.062 * |
| iTransformer | 1.000  | 0.958  | 0.819  | 0.705  | 0.562  |

### Reflation

**gain (% MSE reduction vs persistence)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | +1.5 | +5.1 | +9.1 | +8.4 | +10.7 |
| GWN | -45.1 | -9.0 | -1.3 | -2.0 | -2.1 |
| HOT (k-prod) | -13.7 | +4.2 | -0.6 | -4.9 | -0.5 |
| HOT (k-sum) | -13.9 | +8.3 | +7.1 | +5.3 | +9.5 |
| PCAFormer | -7.7 | +11.5 | +20.9 | +22.9 | +24.3 |
| PatchTST | +3.0 | +8.0 | +8.8 | +6.2 | +7.8 |
| Tucker | -19.8 | +6.6 | +14.0 | +14.9 | +19.5 |
| iTransformer | -19.0 | +4.6 | +11.4 | +13.5 | +16.5 |

**p_primary (stride=21, t-test)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.191  | 0.547  | 0.137  | 0.462  | 0.177  |
| GWN | 0.846  | 0.765  | 0.143  | 0.319  | 0.392  |
| HOT (k-prod) | 0.738  | 0.808  | 0.420  | 0.507  | 0.302  |
| HOT (k-sum) | 0.847  | 0.867  | 0.298  | 0.570  | 0.185  |
| PCAFormer | 0.712  | 0.628  | 0.216  | 0.185  | 0.067 * |
| PatchTST | 0.315  | 0.685  | 0.267  | 0.560  | 0.131  |
| Tucker | 0.679  | 0.268  | 0.116  | 0.144  | 0.074 * |
| iTransformer | 0.813  | 0.834  | 0.290  | 0.314  | 0.171  |

**p_HAC (all windows, DM-HLN, NW lag=h-1)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.294  | 0.148  | 0.019 ** | 0.023 ** | 0.002 *** |
| GWN | 1.000  | 0.818  | 0.580  | 0.583  | 0.567  |
| HOT (k-prod) | 0.988  | 0.302  | 0.520  | 0.625  | 0.513  |
| HOT (k-sum) | 0.986  | 0.135  | 0.193  | 0.295  | 0.220  |
| PCAFormer | 0.898  | 0.033 ** | 0.006 *** | 0.005 *** | 0.010 ** |
| PatchTST | 0.175  | 0.080 * | 0.095 * | 0.244  | 0.216  |
| Tucker | 0.995  | 0.194  | 0.022 ** | 0.022 ** | 0.019 ** |
| iTransformer | 0.994  | 0.189  | 0.045 ** | 0.013 ** | 0.004 *** |

### Bear 2022

**gain (% MSE reduction vs persistence)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | -1.1 | -1.1 | +3.2 | +8.2 | +11.3 |
| GWN | -66.7 | -21.4 | -17.1 | -10.0 | -11.3 |
| HOT (k-prod) | -17.8 | +3.2 | +10.2 | +15.9 | +19.3 |
| HOT (k-sum) | -13.2 | +4.3 | +13.6 | +19.1 | +21.5 |
| PCAFormer | -23.3 | -4.4 | +3.7 | +8.9 | +9.6 |
| PatchTST | -3.1 | +4.6 | +11.6 | +19.9 | +23.3 |
| Tucker | -88.9 | -15.5 | -6.9 | +1.0 | +3.2 |
| iTransformer | -61.4 | -8.3 | -4.3 | -1.0 | +1.0 |

**p_primary (stride=21, t-test)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.136  | 0.092 * | 0.380  | 0.007 *** | 0.027 ** |
| GWN | 0.695  | 0.369  | 0.318  | 0.213  | 0.352  |
| HOT (k-prod) | 0.527  | 0.187  | 0.081 * | 0.043 ** | 0.106  |
| HOT (k-sum) | 0.651  | 0.186  | 0.045 ** | 0.026 ** | 0.066 * |
| PCAFormer | 0.276  | 0.224  | 0.268  | 0.046 ** | 0.049 ** |
| PatchTST | 0.444  | 0.170  | 0.062 * | 0.014 ** | 0.064 * |
| Tucker | 0.987  | 0.789  | 0.492  | 0.184  | 0.105  |
| iTransformer | 0.570  | 0.013 ** | 0.447  | 0.281  | 0.307  |

**p_HAC (all windows, DM-HLN, NW lag=h-1)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.642  | 0.586  | 0.225  | 0.056 * | 0.049 ** |
| GWN | 1.000  | 0.988  | 0.962  | 0.863  | 0.806  |
| HOT (k-prod) | 0.995  | 0.340  | 0.156  | 0.127  | 0.099 * |
| HOT (k-sum) | 0.976  | 0.263  | 0.043 ** | 0.049 ** | 0.048 ** |
| PCAFormer | 1.000  | 0.707  | 0.372  | 0.261  | 0.266  |
| PatchTST | 0.781  | 0.244  | 0.097 * | 0.043 ** | 0.031 ** |
| Tucker | 1.000  | 0.903  | 0.707  | 0.469  | 0.413  |
| iTransformer | 1.000  | 0.778  | 0.662  | 0.545  | 0.455  |

### Normalisation

**gain (% MSE reduction vs persistence)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | -5.1 | -7.5 | -3.1 | -0.4 | +0.8 |
| GWN | -79.5 | -0.6 | +7.7 | +8.3 | +10.1 |
| HOT (k-prod) | -24.5 | -7.7 | -0.4 | +4.6 | +10.1 |
| HOT (k-sum) | -30.9 | -5.1 | +0.7 | +4.8 | +7.4 |
| PCAFormer | -39.9 | -15.4 | -6.3 | -4.2 | -5.5 |
| PatchTST | -3.5 | -3.4 | +2.5 | +5.1 | +7.6 |
| Tucker | -39.8 | -0.9 | +6.3 | +10.2 | +13.2 |
| iTransformer | -88.5 | -25.8 | -13.8 | -3.0 | -0.3 |

**p_primary (stride=21, t-test)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.372  | 0.327  | 0.814  | 0.737  | 0.401  |
| GWN | 0.685  | 0.383  | 0.280  | 0.391  | 0.170  |
| HOT (k-prod) | 0.761  | 0.048 ** | 0.775  | 0.821  | 0.240  |
| HOT (k-sum) | 0.618  | 0.028 ** | 0.490  | 0.750  | 0.199  |
| PCAFormer | 0.899  | 0.672  | 0.981  | 0.952  | 0.886  |
| PatchTST | 0.652  | 0.110  | 0.617  | 0.829  | 0.324  |
| Tucker | 0.848  | 0.461  | 0.418  | 0.498  | 0.204  |
| iTransformer | 0.934  | 0.382  | 0.897  | 0.884  | 0.625  |

**p_HAC (all windows, DM-HLN, NW lag=h-1)**

| model | h=1 | h=5 | h=10 | h=15 | h=21 |
|---|---|---|---|---|---|
| DLinear | 0.966  | 0.966  | 0.766  | 0.535  | 0.434  |
| GWN | 1.000  | 0.526  | 0.269  | 0.286  | 0.282  |
| HOT (k-prod) | 1.000  | 0.781  | 0.514  | 0.368  | 0.274  |
| HOT (k-sum) | 1.000  | 0.736  | 0.469  | 0.332  | 0.298  |
| PCAFormer | 1.000  | 0.962  | 0.730  | 0.622  | 0.635  |
| PatchTST | 0.847  | 0.720  | 0.372  | 0.299  | 0.224  |
| Tucker | 1.000  | 0.569  | 0.136  | 0.070 * | 0.039 ** |
| iTransformer | 1.000  | 0.998  | 0.967  | 0.654  | 0.514  |

## Caveats

- VAR is the differenced-VAR baseline (Medvedev–Wang §3.5.3 recipe); empirically it collapses to within ε of persistence, so vs-VAR and vs-persistence p-values track each other almost perfectly.
- The primary test uses a single greedy non-overlap sub-sample (offset=0). A small offset sweep would let you check stability; with ~12 windows per regime, individual p-values shift but the ranking of models is stable.
- No multiple-comparison correction is applied. With 8 models × 5 horizons × 5 regimes × 2 baselines = 400 tests, only the strongest individual results survive Bonferroni at α=0.05.
- All losses are in standardised log-IV space (the trainer's loss space). For thesis tables in raw-IV MAPE space the DM mechanics carry over after un-standardising and un-logging the predictions.