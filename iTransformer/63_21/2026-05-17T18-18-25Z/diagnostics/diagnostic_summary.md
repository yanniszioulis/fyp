# iTransformer v2 per-cell head diagnostic

## Headline findings

### Probe 1 — cross-cell head similarity
- Mean off-diagonal cosine: **+0.987**
- Effective rank: **1.76** out of 150 possible
- Rank-1 explains 98.7%, rank-3 99.6%, rank-10 100.0% of variance.

### Probe 2 — spatial structure of head norms
- ‖W_n‖_F mean **3.110**, std/mean **0.039**, max/min **1.19**.
- Total variation along W (moneyness): 0.0628; along H (τ): 0.0885.
- v1 shared-head norm reference (different d_model): 2.726.

### Probe 3 — within-v2 shared/delta decomposition
- ‖v2_avg‖_F = 3.0893
- Mean ‖Δ_n‖_F / ‖v2_avg‖_F: **0.114**  (median 0.114, p95 0.182).
- v1 ↔ v2-average direct cosine: undefined (d_model mismatch: v1=16, v2=8); the within-v2 shared/delta decomposition above answers the same architectural question.

### Probe 4 — regime-conditional cell usage
- Mean cross-regime cosine of cell-contribution maps: **+0.971**.

## Architectural implication

**Heads are essentially identical.** Probe 1 reports mean off-diagonal cosine 0.987 and effective rank 1.76. v2 is functionally equivalent to v1 with ~27 k redundant parameters. The architectural recommendation is **drop v2 or apply heavy uniform weight decay** to recover v1's parameter efficiency without the overfitting tail.

