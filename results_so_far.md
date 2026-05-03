---
title: "SPX IV Surface Forecasting — Results So Far"
geometry: margin=2cm
---

# Setup

Forecasting the S&P 500 implied-volatility surface — **170 cells per day**
(10 maturities × 17 call-equivalent deltas), **21 days** of context, **63 days**
ahead. Canonical 70/10/20 train/val/test split.

Test period is **2017-10 → 2019-10** (491 windows), which spans Volmageddon
(Feb 2018) and the Q4 2018 selloff. All models trained with **MSE in scaled
space** for fair comparison.

\

# Headline scoreboard

Best variant of each model.

| Model                  |    MSE |    MAE |  IC mean | t+63 MSE |
|------------------------|-------:|-------:|---------:|---------:|
| **DynGWN (nh16)**      |  0.169 | **0.298** |  0.842 |    0.240 |
| HOT (product, no-norm) |  0.169 |  0.322 |    0.841 | **0.213** |
| DLinear                |  0.170 |  0.318 | **0.913** |    0.256 |
| HOT (sum, no-norm)     |  0.176 |  0.324 |    0.871 |    0.248 |
| VAR(BIC) p=1           |  0.180 |  0.324 |    0.849 |    0.236 |
| PatchTST (no-RevIN)    |  0.191 |  0.327 |    0.897 |    0.315 |
| Persistence (baseline) |  0.223 |  0.334 |    0.887 |    0.388 |

The top three (DynGWN, HOT-product, DLinear) are within **1%** of each other on
overall MSE — but each wins a *different* metric.

\

# What's statistically defensible

Diebold-Mariano vs Persistence with HAC variance and Holm-Bonferroni
correction. The picture below is the one that survives multiple-comparison
adjustment with the available 491 windows.

\

### 1.  DynGWN is the only model that significantly beats persistence overall

| Model              | DM stat | Holm p   | Significant? |
|--------------------|--------:|---------:|:------------:|
| **DynGWN**         |  **3.42** | **0.005** |  yes (p<0.01) |
| HOT (sum)          |    2.28 |   0.157  |       —      |
| HOT (product)      |    2.20 |   0.165  |       —      |
| DLinear            |    2.17 |   0.165  |       —      |
| VAR(BIC)           |    1.75 |   0.239  |       —      |

Mean improvements over persistence are similar across the top group ($\approx +0.05$).
What lets DynGWN cross the bar is **lower variance of the win**: its margin over
persistence is more consistent day-to-day. Graph message-passing produces
forecasts whose error differential against persistence has lower autocovariance.

\

### 2.  Five models significantly beat persistence at t+63

| Rank | Model              | DM stat |   Holm p |
|:----:|--------------------|--------:|---------:|
|  1   | **DynGWN**         |  **6.05** | $<10^{-7}$ |
|  2   | HOT (product)      |    5.20 | $<10^{-5}$ |
|  3   | VAR(BIC)           |    4.63 | $<10^{-4}$ |
|  4   | HOT (sum)          |    4.30 | $<10^{-4}$ |
|  5   | DLinear            |    3.57 |   0.001 |

Long-horizon mean reversion is real and reproducible across very different
architectures. **DynGWN** has the strongest signal here too.

\

### 3.  Persistence is the right baseline at t+1

Every non-persistence model is significantly *worse* at t+1.
Only DLinear is statistically tied. Worth caveating in the writeup: not every
architecture should be asked to beat persistence at every horizon.

\

# Per-model takeaways

Each model has emerged with a distinct character — these are the
single-sentence headlines worth quoting:

\

**DynGWN** — *Lowest MAE, lowest median MSE, biggest gap on trimmed MSE.*
The most precise per-cell predictor on a typical day. Significantly better
than persistence overall (Holm p = 0.005); strongest long-horizon DM stat.

**HOT (product)** — *Regime-shock specialist.* Tightest tail of any model
(max per-window MSE 0.64 vs persistence's 2.12). On the worst day in the test
set (Dec 26, 2018, post-Volmageddon recovery), HOT-product MSE = 0.23 vs every
other model > 0.7. Wins absolute t+63 MSE. Pays for it with collapse at t+1
and t+5.

**DLinear** — *Highest IC, balanced everywhere.* Best at preserving
cross-sectional ranking across all horizons (IC 0.913). Doesn't dominate any
single metric but is rarely worst either. Trains in 30 seconds on MPS.

**VAR(BIC)** — *Closed-form, competitive at long horizons.* BIC formally
selects p=1 (AIC overfits to p=6 with 30% worse t+63). Significantly beats
persistence at t+63. Two seconds to fit, no GPU needed.

**PatchTST** — *RevIN strips the level signal.* With per-window normalization
on (legacy default), bias = -0.13 and worst long-horizon MSE in the field.
Disabling RevIN fixes the bias but PatchTST still doesn't reach the top tier —
the seq_len = 21 / K = 170 / T = 1854-windows regime is too small for
transformer attention to add value over channel-independent linear maps.

\

# Robustness

**~10% of days drive most of the spread.** When the worst-decile windows are
removed (ranked by median MSE across models), DynGWN's lead opens up:

| Model         | full mean | common-trim 10% |
|---------------|----------:|----------------:|
| **DynGWN**    |     0.169 |       **0.126** |
| DLinear       |     0.170 |           0.137 |
| HOT (product) |     0.169 |           0.144 |
| Persistence   |     0.223 |           0.169 |

DynGWN's headline-MSE lead of 0.6% over DLinear becomes an **8% lead** when
outlier days are stripped.

\

**The worst common days cluster in Q4 2018.** Eight of the top-10 worst-MSE
windows are October 2018; the others are post-Christmas Dec 26 / Dec 31,
during the Volmageddon-recovery transition.

\

**Per-horizon claims are seed-sensitive.** Tested on DynGWN (nh32):
per-window correlation across seeds is 0.96 (predictions are consistent),
but per-horizon mean MSE swings 14–29% between seeds. **Aggregate metrics
(overall MSE, MAE) are stable across seeds; specific per-horizon numbers
should be reported with awareness of this variance.**

\

# Caveats

- **Effective sample size << 491.** Test windows share 62 of 63 forecast days
  with their neighbours, so DM tests need HAC variance with bandwidth equal
  to the forecast horizon. With this correction, only DynGWN crosses
  Holm-significance overall — even though three other models have positive
  mean differentials of similar magnitude.

- **Ranking among the top three (DynGWN, HOT-product, DLinear) is not
  statistically distinguishable.** All within 1% on headline MSE, all fail
  the pairwise DM test against each other under Holm. The substantive
  findings are the *per-model character* claims, not the leaderboard order.

\

# Next steps

Three things would strengthen the conclusions, in order of expected impact:

1. **Train on the full dataset** (~838 test windows including COVID and the
   2022 vol regime). The DM-significance gap is N-limited; full dataset
   would likely push DLinear and HOT across the bar too.

2. **Seed-sensitivity check on the shrunk DynGWN.** Confirms the headline
   significance claim doesn't disappear with a different initialization.

3. **HOT retune at uniform compute budget.** Current HOT (product, no-norm)
   trained for ~50 epochs vs DynGWN's 64 and DLinear's 31 — a controlled
   comparison would let us claim "best of each architecture" cleanly.
