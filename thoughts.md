# Thoughts — designing a model to beat the benchmark

Working brainstorm. Intentionally messy. Dissenting views, alternatives we
considered and rejected, and open questions. Trim or formalise later as
decisions get made.

---

## What the comparison is telling us (recap)

Five concrete signals from the benchmark, in rough order of how loud they shout:

1. **Channel-independent + near-linear wins.** DLinear (1st) and OLS-VAR1 (3rd)
   beat every transformer/graph model. PatchTST and HOT both mix channels
   non-trivially; both lose. → **Don't mix channels.**

2. **Loss-space is as important as architecture.** DynGWN went 6th → 2nd
   purely by switching from MSE-on-scaled to MAE-on-original-IV. Same weights,
   same architecture. → IV is amplitude-meaningful and strictly positive;
   treating it like generic standardised TS throws away that structure.

3. **The 20×20 grid is real prior information.** Local smoothness in
   moneyness/tau is genuinely there. DynGWN's `grid_plus_adaptive` outperforms
   `adaptive_only`. But the prior must be applied gently — heavy graph conv
   damages cross-sectional rank at long horizons.

4. **Decomposition helps at long horizons.** DLinear ≫ VAR1 mainly at t+42
   and t+63 (where MA-trend extraction starts pulling away). Slow component
   is predictable; residual is mostly noise.

5. **Long-horizon IC is the real differentiator.** At t+1, every reasonable
   model is within 2 percentage points of persistence on IC. The action is at
   t+21 to t+63. **Optimise for long-horizon IC, not for t+1 MSE.**

---

## Headline finding

`DLinear` trained with **Huber loss** (`--loss huber_scaled --huber_delta 1.0`)
is the dominant model on every metric we report.

```
Metric    DLinear  DLin(MAE)  DLin(MAE-s)  DLin(Hub-1)  DLin(Hub-0.3)
mse        0.148    0.159      0.159        0.147 ★      0.158
ic_mean   +0.571   +0.531     +0.531       +0.591 ★     +0.536
t+1 IC    +0.930   +0.839↓    +0.840       +0.933 ★     +0.842
t+1 MSE    0.016    0.053↓     0.052        0.014 ★      0.050
t+63 IC   +0.399   +0.382     +0.382       +0.433 ★     +0.389
bias      +0.027   -0.014     -0.015       -0.015       -0.011
```

DLinear-Huber-1 beats persistence by 25% on overall MSE, matches it at t+1,
and gives the best long-horizon Spearman IC of any model in the suite.

---

## How we got there — the loss-function story

### MAE-on-original DOES help DynGWN (huge), but HURTS DLinear (subtly)

DynGWN with `masked_mae(inverse_transform(pred), y_orig)` (legacy default):
mse 0.189 → 0.154; t+1 IC 0.776 → 0.919. Big win.

DLinear with the same loss:
- t+1 IC: 0.930 → 0.839 (collapse)
- t+1 MSE: 0.016 → 0.053 (3.3× worse)
- Long horizons unaffected

### Hypothesis 1 — std-weighting from inverse-transform → REJECTED

`mae_original` is mathematically equivalent to std-weighted MAE in scaled
space. Plausible the high-std channels were dominating. Tested by training
DLinear with plain `mae_scaled` (uniform weighting): result was identical
to `mae_original` (4th-decimal differences only). So it isn't channel
weighting.

### Hypothesis 2 — MAE's gradient magnitude → CONFIRMED

`∂|x|/∂x = sign(x)` is constant in magnitude regardless of error size.
`∂x²/∂x = 2x` is proportional to the error. At t+1, errors are small;
MSE provides fine-grained "make this small error smaller" gradient that
MAE cannot. Damage fingerprint matches:

```
Horizon  MSE-loss  MAE-loss  ratio
t+1      0.016     0.052     3.3×
t+5      0.060     0.082     1.4×
t+10     0.093     0.109     1.2×
t+21     0.131     0.149     1.1×
t+42     0.182     0.187     1.02×
t+63     0.200     0.200     1.00×
```

Concentrated at short horizons; fades to zero by t+63 where errors are large
for both losses.

### Verification — Huber should fix it

Huber is quadratic for `|err| < δ` and linear above. If gradient-magnitude
is the issue, Huber should recover MSE's t+1 performance while preserving
any long-horizon-rank benefit. Confirmed: DLinear-Huber-1.0 has best
t+1 (matches persistence) AND best t+63 IC across the suite. δ=0.3 is
too MSE-like (small errors fall in the linear regime); δ=1.0 is the right
magnitude on this dataset.

---

## Bias correction (post-hoc, naive offset) — FAILED

Tried: per-channel additive offset fit on val, applied at test. Compute
`bias = mean(y_val - pred_val)` per channel, then `pred_test += bias`.

Result on a representative MAE-trained model:
- val mean bias: +0.16 (scaled space)
- prior test bias: −0.077
- post-BC test bias: +0.083 (over-corrected by ~2×)
- mse worsened, IC worsened

**Cause: regime drift.** Val period (~2021-22) sits before the 2022-23 vol
regime change; test (2022-04 → 2025-06) is on the other side. Static offset
from val doesn't transfer.

**Lesson:** any future bias correction must be either time-adaptive
(rolling/online) or learned end-to-end (training distribution matches
inference). Naive offset rejected.

---

## Larger conv kernel (5×5) for the spatial-prior idea — also did not help

Tested as part of an earlier ablation: applying a 2D conv across the
moneyness×tau surface at the model's output. Identity-init so the layer
starts as a no-op. With kernel=3 it added a marginal benefit only when
paired with MAE loss; with kernel=5 it over-smoothed and *hurt* IC at
both losses. With Huber it became redundant — Huber alone gave the
short-horizon precision the conv was patching for.

Net: spatial smoothing at the output is not a load-bearing trick on this
dataset once the loss is right.

---

## Open questions resolved

- [x] **Q3a — does MAE-original help DLinear too?** No. It hurts at short horizons.
- [x] **Std-weighting hypothesis** — REJECTED. `mae_scaled` ≈ `mae_original`.
- [x] **Gradient-magnitude hypothesis** — CONFIRMED. Huber recovers t+1.
- [x] **Naive bias correction** — FAILED due to regime drift.
- [x] **Spatial smoothing of model output** — marginal at best, harmful with
  larger kernels. Not pursuing.
- [x] **`conv_kernel` ablation** — k=3 marginally beat k=5; both lose to Huber.
- [x] **DLinear-Huber as the recommended config** — confirmed across the
  full benchmark; matches/beats every alternative on every metric.

## Open questions remaining

- [ ] **Q2 — target space (log-IV, total-variance) ablation.**
  Cheap; should run.
- [ ] **Adaptive Huber threshold** (per-channel and/or per-horizon).
  Single global δ is a compromise. Per-channel δ ∝ that channel's std on
  training data would let the loss scale match each channel's natural error
  size. ~1-3% expected.
- [ ] **Ensemble of DLinear-Huber + VAR1 + DLinear-MSE.**
  VAR1 has slightly better long-horizon IC than MAE-DLinear at t+42;
  DLinear-MSE has marginally better short-horizon precision; DLinear-Huber
  dominates the average. A linear pool with per-horizon weights fit on
  validation almost always improves. Cheap, high-payoff.
- [ ] **Online / rolling bias correction.** The naive-BC failure pointed at
  regime drift. Take the most recent K days of residuals, add per-channel
  mean to predictions. K ∈ {21, 63, 126}. Worth one experiment.
- [ ] **Quantile / probabilistic forecasting.** For trading you usually want
  VaR-like quantiles. Train DLinear with quantile loss for τ ∈ {0.1, 0.5, 0.9}
  in parallel; same architecture, three output heads. Modest cost, big
  reporting upgrade.
- [ ] **Joint training across horizons (sequence head on the output).**
  Currently every horizon's prediction is independent. A small AR head
  could tighten consistency. Risk of error compounding.
- [ ] **Surface-shape regulariser.** Loss term penalising violations of
  cross-sectional curvature / no-arbitrage. Hard to debug; likely
  follow-up paper, not the right scope here.

---

## Pipeline state

- `train.py` accepts `--seq_len` as a list (e.g. `--seq_len 21 63 126`) and
  sweeps each value across all selected models. When `--loss` is unset, it
  also expands to **two loss variants per model** per `(seq_len, pred_len)`:
  `mse` + `huber_scaled` for DLinear/PatchTST/HOT; `mae_original` +
  `huber_original` for DynGWN; VAR1 has no loss option.
- `compare_models.py` parametrises seq/pred (registry built per-call). Output
  filenames carry an `_sl<sl>_pl<pl>` tag so different sweep configurations
  don't overwrite each other.
- Result-dir naming uniformly carries a `_lossX_d<delta>` suffix for
  non-default-loss runs; default-loss runs keep the historical no-suffix dir
  names so existing checkpoints continue to resolve unchanged.