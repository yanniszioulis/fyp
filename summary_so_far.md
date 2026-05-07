# Summary so far

## Data preparation

Raw input is OptionMetrics SPX option quotes plus the IvyDB Forward_Price
table. For each trading day we:

1. Filter quotes to OTM options plus a tight ATM band ($|k| < 0.01$ with
   $k = \log(K/F)$, where $F$ is the same-expiry forward).
2. Drop quotes with $\text{vega} < 0.5$, $\sigma_{\text{IV}} \notin [0.01,\,3.0]$,
   or zero traded volume.
3. Fit a vega-weighted Nadaraya–Watson kernel smoother in $(\log\tau,\,k)$
   space, using the OptionMetrics manual form
   $$\hat\sigma(\tau_j, k_j) = \frac{\sum_i V_i\,\sigma_i\,\phi(x_{ij}, y_{ij})}
                                       {\sum_i V_i\,\phi(x_{ij}, y_{ij})},
     \qquad
     \phi(x, y) = \exp\!\left(-\frac{x^2}{2 h_\tau} - \frac{y^2}{2 h_m}\right),$$
   with $x_{ij} = \log(\tau_i / \tau_j)$, $y_{ij} = k_i - k_j$, and bandwidths
   (variances) $h_\tau = 5.00 \times 10^{-2}$, $h_m = 1.00 \times 10^{-3}$.
4. Evaluate $\hat\sigma$ on a fixed $20 \times 20$ grid: $k$ uniform over
   $[-0.10,\,0.10]$ and $\tau$ geometric over $[0.04,\,1.00]$ years.

Output is one row per date with $400$ flattened columns
`iv_{moneyness}_{tau}` (column order: $\tau$ outer, $k$ inner).
Sparse-data guards (effective sample size $> 2$, no single observation
holding $> 0.85$ of the kernel mass) blank out cells where the smoother
would be unstable.

## Task

The current evaluation task is **`level / precovid / c63 / h21`**:

- **`level`**: predict raw IV $\sigma$, not its log-difference.
- **`precovid`**: restrict the panel to dates $\leq$ `2019-12-31` to avoid
  the COVID-era regime break (so the model is judged on a stationary regime).
- **`c63`**: context length $L = 63$ trading days ($\approx 3$ months).
- **`h21`**: forecast horizon $H = 21$ trading days ($\approx 1$ month).

This gives $T = 2{,}768$ daily surfaces. The standard $70/10/20$ split puts
the train period ending `2016-09-12`, with $533$ rolling test windows
spanning `2017-10-19` $\rightarrow$ `2019-12-02`. All metrics below are
computed on this test set, in original-IV units (vol points).

## Best models (tuning winners)

For every architecture we ran a grid search and kept the combo with the
lowest validation loss (validation = the $10\%$ slice between train and
test). VAR is not tuned — its lag $p$ is picked by an information criterion
on the training set.

- **VAR(BIC)** — Vector autoregression on the $400$-dim flattened surface,
  fit by OLS with lag $p$ chosen to minimise BIC over $p \in \{1,\dots,5\}$.
  The BIC criterion selects $p = 1$ (so the model is effectively a
  $400 \times 400$ one-step transition matrix iterated forward). Strong,
  almost embarrassingly simple baseline that exploits the high
  cross-sectional and temporal correlations of the IV surface.
  Parameters: $1.60 \times 10^{5}$ ($K^{2} p + K$ with $K=400$, $p=1$).

- **DLinear** — Purely linear time-series baseline that decomposes each
  channel into a moving-average trend and a residual, then maps each via a
  single dense layer over the context window. Winner: `kernel_size = 13`,
  $\text{lr} = 4.00 \times 10^{-3}$, weight decay $= 0$, MSE loss
  (val $= 4.05 \times 10^{-2}$).
  Parameters: $1.08 \times 10^{6}$.

- **HOT(product)** — Tensorised transformer over the $20 \times 20$ surface.
  Patches the time axis, then applies multi-head attention with a Kronecker
  *product* factorisation across the moneyness and $\tau$ axes (a tensor
  attention that respects grid structure). Winner: $d_{\text{hidden}} = 64$,
  $d_{\text{mlp}} = 32$, $n_{\text{blocks}} = 2$, $n_{\text{head}} = 4$,
  $\text{patch}=3$, RoPE positional embedding, no spatial PE,
  $\text{lr} = 1.00 \times 10^{-3}$, weight decay $= 1.00 \times 10^{-3}$
  (val $= 3.29 \times 10^{-2}$).
  Parameters: $8.14 \times 10^{4}$.

- **HOT(sum)** — Same architecture as HOT(product) but with the Kronecker
  *sum* factorisation (separately attends along moneyness and $\tau$, then
  adds). Winner: $d_{\text{hidden}} = 64$, $d_{\text{mlp}} = 16$,
  $n_{\text{blocks}} = 4$, $n_{\text{head}} = 2$, $\text{patch}=9$,
  $\text{lr} = 1.00 \times 10^{-3}$, weight decay $= 1.00 \times 10^{-3}$
  (val $= 3.60 \times 10^{-2}$).
  Parameters: $1.49 \times 10^{5}$.

- **DynGWN** — Graph WaveNet variant where the $400$ surface cells form a
  graph that combines a fixed grid-adjacency component with a learned
  adaptive component. Winner (`graph_mode = grid_plus_adaptive`):
  $n_{\text{hid}} = 8$, $\text{blocks} = 2$, $\text{layers} = 2$,
  $\text{kernel} = 2$, $\text{dropout} = 0.3$,
  $\text{lr} = 1.00 \times 10^{-2}$, weight decay $= 1.00 \times 10^{-3}$
  (val $= 3.18 \times 10^{-2}$).
  Parameters: $3.67 \times 10^{4}$.

- **PatchTST** — Channel-independent transformer over time-axis patches,
  shared across the $400$ cells. Winner: $\text{patch\_len} = 3$,
  $\text{stride} = 3$, $d_{\text{model}} = 32$, $n_{\text{heads}} = 2$,
  $n_{\text{layers}} = 1$, $d_{\text{ff}} = 128$, $\text{dropout} = 0.3$,
  RevIN off, $\text{lr} = 3.00 \times 10^{-4}$, weight decay
  $= 1.00 \times 10^{-4}$ (val $= 3.12 \times 10^{-2}$).
  Parameters: $2.83 \times 10^{4}$.

## Metrics

Standard pointwise metrics: MSE, RMSE, MAE, $\text{RSE}$, bias, IC.
$\text{IC}$ is the per-window Spearman rank correlation between predicted
and true IV across the $400$ cells, then averaged over windows and horizons.

Surface-shape metrics (LSTS components, all squared and averaged over the
test windows, horizons, and the relevant axis):

$$\mathcal{L}_{\text{level}} = \mathbb{E}\!\left[\left(\overline{\hat\sigma} - \overline{\sigma}\right)^2\right]$$

$$\mathcal{L}_{\text{skew}}  = \mathbb{E}\!\left[\left(\Delta_k \hat\sigma - \Delta_k \sigma\right)^2\right]$$

$$\mathcal{L}_{\text{term}}  = \mathbb{E}\!\left[\left(\Delta_\tau \hat\sigma - \Delta_\tau \sigma\right)^2\right]$$

$$\mathcal{L}_{\text{curv}}  = \mathbb{E}\!\left[\left(\Delta^2_k \hat\sigma - \Delta^2_k \sigma\right)^2\right]$$

with $\Delta_k$ a finite difference along moneyness divided by the local
grid spacing, $\Delta_\tau$ likewise along $\tau$, and $\Delta^2_k$ the
uniform-grid second difference along moneyness. The overbar denotes the
mean across the $20 \times 20$ surface.

`Persist(ref)` is the naive last-value baseline: $\hat\sigma_{t+h} = \sigma_t$
for every horizon $h$.

In every table that follows, **bold** marks the best value in the row and
<u>underline</u> marks the second best. For bias the ranking is by absolute
value (closest to zero is best); for IC mean and per-horizon IC, larger
(more positive) is best; for every other metric, smaller is best.

## Overall test-set results

| Metric                       | Persist(ref)            | VAR(1)                              | DLinear                                   | HOT(product)            | HOT(sum)                | DynGWN                  | PatchTST                              |
|------------------------------|------------------------:|--------------------------------------:|------------------------------------------:|---------------------------:|---------------------------:|----------------------------:|--------------------------------------:|
| MSE                          | $7.29 \times 10^{-4}$   | $\mathbf{6.13 \times 10^{-4}}$             | <u>$6.33 \times 10^{-4}$</u>              | $9.73 \times 10^{-4}$   | $1.07 \times 10^{-3}$   | $8.50 \times 10^{-4}$   | $6.92 \times 10^{-4}$                 |
| RMSE                         | $2.70 \times 10^{-2}$   | $\mathbf{2.48 \times 10^{-2}}$             | <u>$2.52 \times 10^{-2}$</u>              | $3.12 \times 10^{-2}$   | $3.27 \times 10^{-2}$   | $2.92 \times 10^{-2}$   | $2.63 \times 10^{-2}$                 |
| MAE                          | <u>$1.72 \times 10^{-2}$</u> | $1.85 \times 10^{-2}$            | $\mathbf{1.70 \times 10^{-2}}$                 | $2.16 \times 10^{-2}$   | $2.18 \times 10^{-2}$   | $1.91 \times 10^{-2}$   | $1.72 \times 10^{-2}$                 |
| RSE                          | $6.68 \times 10^{-1}$   | $\mathbf{6.13 \times 10^{-1}}$             | <u>$6.23 \times 10^{-1}$</u>              | $7.72 \times 10^{-1}$   | $8.09 \times 10^{-1}$   | $7.22 \times 10^{-1}$   | $6.51 \times 10^{-1}$                 |
| Bias                         | <u>$-7.19 \times 10^{-4}$</u> | ${+}6.56 \times 10^{-3}$          | $\mathbf{{+}3.89 \times 10^{-4}}$                | $-7.98 \times 10^{-3}$  | $-1.48 \times 10^{-2}$  | $-6.67 \times 10^{-3}$  | $-4.15 \times 10^{-3}$                |
| IC mean                      | ${+}9.49 \times 10^{-1}$  | $\mathbf{{+}9.63 \times 10^{-1}}$            | ${+}9.55 \times 10^{-1}$                    | ${+}9.46 \times 10^{-1}$  | ${+}9.50 \times 10^{-1}$  | ${+}9.52 \times 10^{-1}$  | <u>${+}9.55 \times 10^{-1}$</u>         |
| IC std                       | $8.91 \times 10^{-2}$   | $\mathbf{7.02 \times 10^{-2}}$             | <u>$7.29 \times 10^{-2}$</u>              | $7.99 \times 10^{-2}$   | $8.43 \times 10^{-2}$   | $8.41 \times 10^{-2}$   | $8.24 \times 10^{-2}$                 |
| $\mathcal{L}_{\text{level}}$ | $6.12 \times 10^{-4}$   | $\mathbf{5.31 \times 10^{-4}}$             | <u>$5.37 \times 10^{-4}$</u>              | $8.57 \times 10^{-4}$   | $9.47 \times 10^{-4}$   | $7.45 \times 10^{-4}$   | $5.96 \times 10^{-4}$                 |
| $\mathcal{L}_{\text{skew}}$  | $1.12 \times 10^{-2}$   | $\mathbf{8.94 \times 10^{-3}}$             | $1.33 \times 10^{-2}$                     | $1.79 \times 10^{-2}$   | $2.14 \times 10^{-2}$   | $1.76 \times 10^{-2}$   | <u>$9.96 \times 10^{-3}$</u>          |
| $\mathcal{L}_{\text{term}}$  | $2.86 \times 10^{-2}$   | $\mathbf{1.80 \times 10^{-2}}$             | $2.46 \times 10^{-2}$                     | <u>$2.01 \times 10^{-2}$</u> | $2.14 \times 10^{-2}$ | $2.02 \times 10^{-2}$ | $2.11 \times 10^{-2}$                 |
| $\mathcal{L}_{\text{curv}}$  | $8.23 \times 10^{-8}$   | $\mathbf{6.14 \times 10^{-8}}$             | $1.62 \times 10^{-7}$                     | $2.20 \times 10^{-7}$   | $3.44 \times 10^{-7}$   | $4.69 \times 10^{-7}$   | <u>$6.27 \times 10^{-8}$</u>          |

## Per-horizon results

### Spearman IC

| $h$    | Persist(ref)                       | VAR(1)                            | DLinear                  | HOT(product)            | HOT(sum)                | DynGWN                  | PatchTST                            |
|--------|----------------------------------:|------------------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|------------------------------------:|
| $1$    | $\mathbf{{+}9.90 \times 10^{-1}}$        | <u>${+}9.90 \times 10^{-1}$</u>       | ${+}9.87 \times 10^{-1}$  | ${+}9.59 \times 10^{-1}$  | ${+}9.69 \times 10^{-1}$  | ${+}9.86 \times 10^{-1}$  | ${+}9.89 \times 10^{-1}$              |
| $5$    | ${+}9.66 \times 10^{-1}$            | $\mathbf{{+}9.70 \times 10^{-1}}$          | ${+}9.64 \times 10^{-1}$  | ${+}9.52 \times 10^{-1}$  | ${+}9.59 \times 10^{-1}$  | ${+}9.66 \times 10^{-1}$  | <u>${+}9.67 \times 10^{-1}$</u>       |
| $10$   | ${+}9.47 \times 10^{-1}$            | $\mathbf{{+}9.60 \times 10^{-1}}$          | ${+}9.49 \times 10^{-1}$  | ${+}9.47 \times 10^{-1}$  | ${+}9.50 \times 10^{-1}$  | ${+}9.51 \times 10^{-1}$  | <u>${+}9.54 \times 10^{-1}$</u>       |
| $21$   | ${+}9.25 \times 10^{-1}$            | $\mathbf{{+}9.52 \times 10^{-1}}$          | <u>${+}9.41 \times 10^{-1}$</u> | ${+}9.36 \times 10^{-1}$ | ${+}9.39 \times 10^{-1}$ | ${+}9.30 \times 10^{-1}$ | ${+}9.36 \times 10^{-1}$              |

### MSE

| $h$    | Persist(ref)                              | VAR(1)                              | DLinear                                | HOT(product)            | HOT(sum)                | DynGWN                  | PatchTST                |
|--------|-----------------------------------------:|--------------------------------------:|---------------------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|
| $1$    | $\mathbf{1.28 \times 10^{-4}}$                | <u>$1.30 \times 10^{-4}$</u>          | $1.41 \times 10^{-4}$                  | $4.23 \times 10^{-4}$   | $4.70 \times 10^{-4}$   | $2.08 \times 10^{-4}$   | $1.80 \times 10^{-4}$   |
| $5$    | $4.56 \times 10^{-4}$                    | $\mathbf{4.30 \times 10^{-4}}$             | <u>$4.34 \times 10^{-4}$</u>           | $6.99 \times 10^{-4}$   | $7.94 \times 10^{-4}$   | $5.27 \times 10^{-4}$   | $4.61 \times 10^{-4}$   |
| $10$   | $7.38 \times 10^{-4}$                    | $\mathbf{6.39 \times 10^{-4}}$             | <u>$6.43 \times 10^{-4}$</u>           | $9.52 \times 10^{-4}$   | $1.06 \times 10^{-3}$   | $8.23 \times 10^{-4}$   | $6.72 \times 10^{-4}$   |
| $21$   | $1.17 \times 10^{-3}$                    | $\mathbf{8.75 \times 10^{-4}}$             | <u>$9.40 \times 10^{-4}$</u>           | $1.40 \times 10^{-3}$   | $1.44 \times 10^{-3}$   | $1.39 \times 10^{-3}$   | $1.08 \times 10^{-3}$   |

### $\mathcal{L}_{\text{level}}$

| $h$    | Persist(ref)                          | VAR(1)                                | DLinear                              | HOT(product)            | HOT(sum)                | DynGWN                  | PatchTST                |
|--------|-------------------------------------:|----------------------------------------:|-------------------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|
| $1$    | $\mathbf{1.01 \times 10^{-4}}$            | <u>$1.03 \times 10^{-4}$</u>            | $1.09 \times 10^{-4}$                | $3.24 \times 10^{-4}$   | $3.77 \times 10^{-4}$   | $1.63 \times 10^{-4}$   | $1.51 \times 10^{-4}$   |
| $5$    | $3.69 \times 10^{-4}$                | <u>$3.60 \times 10^{-4}$</u>            | $\mathbf{3.53 \times 10^{-4}}$            | $5.92 \times 10^{-4}$   | $6.87 \times 10^{-4}$   | $4.45 \times 10^{-4}$   | $3.84 \times 10^{-4}$   |
| $10$   | $6.14 \times 10^{-4}$                | <u>$5.51 \times 10^{-4}$</u>            | $\mathbf{5.39 \times 10^{-4}}$            | $8.37 \times 10^{-4}$   | $9.38 \times 10^{-4}$   | $7.17 \times 10^{-4}$   | $5.73 \times 10^{-4}$   |
| $21$   | $9.98 \times 10^{-4}$                | $\mathbf{7.73 \times 10^{-4}}$               | <u>$8.23 \times 10^{-4}$</u>         | $1.27 \times 10^{-3}$   | $1.30 \times 10^{-3}$   | $1.25 \times 10^{-3}$   | $9.44 \times 10^{-4}$   |

### $\mathcal{L}_{\text{skew}}$

| $h$    | Persist(ref)                          | VAR(1)                              | DLinear                  | HOT(product)            | HOT(sum)                | DynGWN                  | PatchTST                              |
|--------|-------------------------------------:|--------------------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|--------------------------------------:|
| $1$    | <u>$3.76 \times 10^{-3}$</u>         | $\mathbf{3.46 \times 10^{-3}}$             | $7.44 \times 10^{-3}$   | $1.72 \times 10^{-2}$   | $1.76 \times 10^{-2}$   | $1.03 \times 10^{-2}$   | $4.10 \times 10^{-3}$                 |
| $5$    | $8.28 \times 10^{-3}$                | $\mathbf{7.26 \times 10^{-3}}$             | $1.03 \times 10^{-2}$   | $1.61 \times 10^{-2}$   | $1.85 \times 10^{-2}$   | $1.21 \times 10^{-2}$   | <u>$7.45 \times 10^{-3}$</u>          |
| $10$   | $1.16 \times 10^{-2}$                | $\mathbf{9.44 \times 10^{-3}}$             | $1.48 \times 10^{-2}$   | $1.71 \times 10^{-2}$   | $2.14 \times 10^{-2}$   | $1.65 \times 10^{-2}$   | <u>$9.91 \times 10^{-3}$</u>          |
| $21$   | $1.57 \times 10^{-2}$                | $\mathbf{1.13 \times 10^{-2}}$             | $1.48 \times 10^{-2}$   | $2.01 \times 10^{-2}$   | $2.47 \times 10^{-2}$   | $2.62 \times 10^{-2}$   | <u>$1.36 \times 10^{-2}$</u>          |

### $\mathcal{L}_{\text{term}}$

| $h$    | Persist(ref)                       | VAR(1)                              | DLinear                  | HOT(product)                              | HOT(sum)                | DynGWN                                   | PatchTST                              |
|--------|----------------------------------:|--------------------------------------:|------------------------:|------------------------------------------:|------------------------:|-----------------------------------------:|--------------------------------------:|
| $1$    | $\mathbf{7.72 \times 10^{-3}}$         | $8.90 \times 10^{-3}$                 | $1.42 \times 10^{-2}$   | $1.87 \times 10^{-2}$                     | $1.65 \times 10^{-2}$   | $1.01 \times 10^{-2}$                    | <u>$7.85 \times 10^{-3}$</u>          |
| $5$    | $2.34 \times 10^{-2}$             | $\mathbf{1.70 \times 10^{-2}}$             | $2.55 \times 10^{-2}$   | $1.95 \times 10^{-2}$                     | $1.97 \times 10^{-2}$   | <u>$1.80 \times 10^{-2}$</u>             | $1.92 \times 10^{-2}$                 |
| $10$   | $3.26 \times 10^{-2}$             | $\mathbf{1.92 \times 10^{-2}}$             | $2.68 \times 10^{-2}$   | <u>$2.02 \times 10^{-2}$</u>              | $2.19 \times 10^{-2}$   | $2.14 \times 10^{-2}$                    | $2.37 \times 10^{-2}$                 |
| $21$   | $3.39 \times 10^{-2}$             | $\mathbf{1.97 \times 10^{-2}}$             | $2.37 \times 10^{-2}$   | <u>$2.07 \times 10^{-2}$</u>              | $2.32 \times 10^{-2}$   | $2.32 \times 10^{-2}$                    | $2.39 \times 10^{-2}$                 |

### $\mathcal{L}_{\text{curv}}$

| $h$    | Persist(ref)            | VAR(1)                              | DLinear                  | HOT(product)            | HOT(sum)                | DynGWN                  | PatchTST                              |
|--------|------------------------:|--------------------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|--------------------------------------:|
| $1$    | $5.28 \times 10^{-8}$   | <u>$4.34 \times 10^{-8}$</u>          | $2.66 \times 10^{-7}$   | $2.39 \times 10^{-7}$   | $3.07 \times 10^{-7}$   | $3.92 \times 10^{-7}$   | $\mathbf{4.20 \times 10^{-8}}$             |
| $5$    | $7.00 \times 10^{-8}$   | <u>$5.42 \times 10^{-8}$</u>          | $1.64 \times 10^{-7}$   | $2.08 \times 10^{-7}$   | $3.04 \times 10^{-7}$   | $3.36 \times 10^{-7}$   | $\mathbf{5.15 \times 10^{-8}}$             |
| $10$   | $8.27 \times 10^{-8}$   | <u>$6.26 \times 10^{-8}$</u>          | $1.30 \times 10^{-7}$   | $2.12 \times 10^{-7}$   | $3.43 \times 10^{-7}$   | $4.34 \times 10^{-7}$   | $\mathbf{6.22 \times 10^{-8}}$             |
| $21$   | $9.95 \times 10^{-8}$   | $\mathbf{7.12 \times 10^{-8}}$             | $1.22 \times 10^{-7}$   | $2.33 \times 10^{-7}$   | $3.73 \times 10^{-7}$   | $5.90 \times 10^{-7}$   | <u>$7.80 \times 10^{-8}$</u>          |
