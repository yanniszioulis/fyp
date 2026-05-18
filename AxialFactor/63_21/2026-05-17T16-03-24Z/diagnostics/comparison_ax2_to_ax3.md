# AxialFactor Ax2 → Ax3 comparison

Ax2 (broken):   `AxialFactor/63_21/2026-05-17T14-50-25Z/` — trunc_normal query init, no penalty.
Ax3 (this run): `AxialFactor/63_21/2026-05-17T16-03-24Z/` — orthogonal query init (scale 0.1) + L_orth penalty (weight 1e-2).

Only the two changes above; same lr/wd/batch/epochs/patience/grad_clip; same seed=0; same data; same architecture and hyperparameters otherwise.

## Headline comparison

| Metric                                                | Ax2 (broken) | Ax3 (this run) | Δ        |
| ----------------------------------------------------- | ------------ | -------------- | -------- |
| **Test MSE** (1007 windows, standardised log-IV)      | 0.285651     | **0.251292**   | −12.0 %  |
| Test RMSE                                             | 0.534463     | 0.501290       | −6.2 %   |
| Test MAE                                              | 0.322417     | 0.308484       | −4.3 %   |
| Best val (epoch)                                      | 0.156 (e31)  | 0.166 (e6)     | val ↑    |
| Stop epoch                                            | 46           | 21             | faster   |
| factor_queries effective rank                         | 1.099        | **3.966**      | rank ↑   |
| factor_queries max abs off-diag cos-sim               | 0.999        | **0.003**      | ortho ✓  |
| Factor → regime LogReg accuracy (concat) vs baseline  | 27.1 / 25.2  | **34.3 / 25.2**| +9 pp    |
| Cross-factor top-1 horizon-shape similarity (max abs) | 0.999        | 0.998          | ≈ same   |
| Stacked spatial_loadings effective rank               | 1.806        | 1.720          | unchanged|
| Stacked loadings max abs off-diag cos-sim             | 0.987        | 0.966          | ≈ same   |
| Each factor's best-match training-PC                  | all → PC1    | all → PC1      | unchanged|

## Short summary (5 bullets)

- **The query fix worked exactly as designed at the parameter level.** factor_queries went from rank 1.099 / cos-sim 0.999 to rank **3.966 / cos-sim 0.003** — fully orthogonal. The penalty stayed in the 1e-3 range across training, confirming the orthogonal init alone was almost sufficient: the penalty had little corrective work to do.
- **Test MSE dropped by 12 % (0.286 → 0.251) — real, substantial, and from a single-line init change plus one penalty term.** That is the headline gain of breaking the symmetric basin.
- **Factor tokens now carry visibly more regime information.** Variance ratio 0.054 → 0.096 (+78 %); LogReg classifier 27.1 → 34.3 % against a 25.2 % baseline — the "above-chance" margin doubled from +2 pp to +9 pp. The factors are no longer regime-blind.
- **The downstream collapse persists, though.** Stacked spatial_loadings still have effective rank 1.72 (was 1.81); all four loadings still best-match PC1 with |cos-sim| 0.96–0.98; cross-factor top-1 horizon trajectories are still ±0.997. Even with orthogonal queries, the attention + temporal + loading pipeline funnels all four factors back onto the same dominant surface direction with sign flips. **This is the "Family 2 / 3" territory** the diagnosis flagged would matter if Family 1 wasn't enough alone — and it isn't.
- **Trajectory error is still mostly regime-blind** (probe 3): predicted-g varies far less across regimes than truth-g (factor 3 at h=21 in Normalisation: pred +0.28 vs truth **+1.99**; the model is using ~14 % of the amplitude needed). The 12 % MSE improvement comes mostly from the four factors finally being *distinct directions* in feature space, giving the residual path and out_bias more uncorrelated channels to ride on — not from the factor path learning regime-conditional dynamics.

## What the result tells us about next steps

The user's pre-stated framework was: **"If queries collapse → Family 1. If queries diverse but loadings collapse → Family 2 / 3."** Ax3 lands exactly in the second case. Probe 1.5's verdict in this run is explicit about it: queries have effective rank 3.97 (diverse), stacked loadings have effective rank 1.72 (collapsed). The next change to consider — *if* a further test-MSE drop matters more than the current price-of-admission complexity:

- Orthogonality penalty on the *stacked loadings* (mirror of the query penalty, weight tuned independently). Direct mechanism for the failure mode probe 4 keeps reporting.
- A residual-attention or value-projection regulariser inside `_FactorExtractor`. The two MultiheadAttention layers' value/output projections can map four orthogonal queries' outputs back onto one direction; a softer constraint there might keep the diversity intact through to the temporal map.
- Or, more radically, a non-axial reconstruction prior — the static-loading rank-F-per-horizon design is exactly the structure that funnels everything into a single rank-1 mode when the data has one dominant PC.

No fix is implemented here per the task spec ("If after retraining the queries still collapse, the penalty weight needs increasing. If they decorrelate but test MSE doesn't improve, the diagnosis was incomplete and we need to look further"). Queries decorrelated cleanly *and* test MSE did improve, but only partially — the diagnosis was correct as far as it went, and the next decision is yours.

---

Probes regenerated on 2026-05-17T16-04Z; checkpoint at
`AxialFactor/63_21/2026-05-17T16-03-24Z/best_model.pt` (saved by the updated `train.py`).
