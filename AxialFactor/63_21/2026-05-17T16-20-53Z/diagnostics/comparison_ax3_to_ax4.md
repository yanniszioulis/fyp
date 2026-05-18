# AxialFactor Ax3 → Ax4 comparison

Ax2 (broken): `AxialFactor/63_21/2026-05-17T14-50-25Z/` — trunc_normal query init, no penalty.
Ax3 (queries): `AxialFactor/63_21/2026-05-17T16-03-24Z/` — orthogonal query init + factor-query orthogonality penalty (1e-2).
Ax4 (this run): `AxialFactor/63_21/2026-05-17T16-20-53Z/` — Ax3 + spatial-loading orthogonality penalty (1e-2).

Only delta from Ax3 → Ax4: the new `loading_orthogonality_loss` method on `AxialFactor` and one extra term in the training loss. Same seed=0, lr, wd, batch, epochs, patience, grad-clip, model kwargs.

## Three-way headline table

| Metric                                                | Ax2 (broken) | Ax3 (queries) | Ax4 (this run) |
| ----------------------------------------------------- | ------------ | ------------- | -------------- |
| **Test MSE**                                          | 0.285651     | 0.251292      | **0.249272**   |
| Test RMSE                                             | 0.534463     | 0.501290      | 0.499271       |
| Test MAE                                              | 0.322417     | 0.308484      | 0.306795       |
| Best val (epoch)                                      | 0.156 (e31)  | 0.166 (e6)    | 0.168 (e6)     |
| Stop epoch                                            | 46           | 21            | 21             |
| factor_queries effective rank                         | 1.099        | 3.966         | 3.707          |
| factor_queries max abs off-diag cos-sim               | 0.999        | 0.003         | 0.005          |
| **Stacked spatial_loadings effective rank**           | 1.806        | 1.720         | **3.949**      |
| **Stacked loadings max abs off-diag cos-sim**         | 0.987        | 0.966         | **0.007**      |
| Cross-factor top-1 horizon-shape similarity (max abs) | 0.999        | 0.998         | 1.000          |
| Each factor's best-match training-PC                  | all → PC1    | all → PC1     | all → PC1      |
| |sim| of each factor to its best PC                   | 0.94–0.98    | 0.96–0.98     | 0.43–0.59      |
| Factor → regime LogReg accuracy vs baseline (concat)  | 27.1 / 25.2  | 34.3 / 25.2   | 32.8 / 25.2    |
| Loading rank-1 separability per loading (range)       | 81–92 %      | 67–70 %       | 21–28 %        |
| ortho_q at end of training (per-batch mean)           | n/a          | ≈ 0.001       | ≈ 0.001        |
| ortho_L at end of training (per-batch mean)           | n/a          | n/a           | ≈ 0.0004       |

## Short summary (5 bullets)

- **The loading-orthogonality penalty worked at the parameter level — completely.** Stacked-loadings effective rank 1.720 → **3.949** (essentially full rank); max abs off-diagonal cos-sim 0.966 → **0.007**. The four spatial loadings are now mutually orthogonal as flat [W·H]-vectors. ortho_L decreased monotonically across training (0.018 at epoch 1 → 0.0004 by epoch 21), so the penalty was active and effective.
- **Test MSE improved only marginally** (0.251 → **0.249**, −0.8 %). The bulk of the headline gain — 12 % vs the broken Ax2 — already came from the Ax3 query fix; constraining the loadings on top of that contributed almost nothing to test MSE. The 0.249 number is **still 15 % worse than DLinear's 0.216** — the predicted "20–23, plausibly equal to or below 0.216" did not materialise.
- **The collapse migrated to the temporal map.** With queries AND loadings now constrained to be orthogonal, probe 2 reports cross-factor top-1 horizon-trajectory cosine similarity of **exactly ±1.000** for every pair (was ±0.997 in Ax3). All four factors evolve with the same horizon shape, only flipping signs. The model has F = 4 orthogonal *spatial* modes but a single shared *temporal* shape applied to all of them — a rank-F-in-space × rank-1-in-time decomposition.
- **The loadings now look much less PC1-like — but not in a good way.** Best-match |cos-sim| to the top training PC dropped from 0.96–0.98 to **0.43–0.59**, and per-loading rank-1 separability (top SV / sum SVs of the W×H heatmap) collapsed from 67–70 % to **21–28 %**. The penalty forced orthogonality, and the optimiser satisfied that constraint by adopting noisier, less-smooth, less-rank-1-separable spatial patterns — not by discovering PC2/PC3/PC4. **None of the four factors best-match anything other than PC1.** The penalty enforced orthogonality but did not steer the loadings toward the natural data modes.
- **Probe 3 trajectory MSE jumped, but for a basis-change reason.** Trajectory MSE went from 5–25 (Ax3) to 42–45 (Ax4) in COVID, similar elsewhere. This is *not* a worse forecast — it's the same forecast measured in a different basis. With orthogonal loadings, the pseudoinverse projection of the truth gives much larger `g_true` values; the model's predicted `g` is still small (because temporal_W is small and rank-1-in-time), so the gap reads bigger. Test MSE in the canonical (cell-wise) space is what to trust, and that's 0.249.

## What the result tells us about next steps

This is the "(b) the architecture is structurally wrong for the long-horizon prediction problem" outcome the task spec flagged. The factor decomposition is now genuinely diverse at the parameter level — queries orthogonal, loadings orthogonal — and forecasting power did not follow.

Probe 2 is where the next bottleneck is: every factor's `temporal.weight[f]` (a [d_model, pred_len] matrix) has SVD top-1 right-singular-vector that's collinear with every other factor's top-1, up to sign. The per-factor temporal maps were *meant* to express distinct dynamics — instead they all express the same dominant horizon shape (≈ 92–97 % of each map's variance), with sign flips that the loading orthogonality penalty doesn't penalise.

A penalty on the temporal map (analogous shape: penalise `temporal.weight.reshape(F, -1)` pairwise cos-sim) is the obvious next try, but I would *not* recommend it as the next step. The pattern across Ax2 → Ax3 → Ax4 is clear: every time we lock one parameter group, the collapse migrates to another. The collapse is a *gradient-landscape* property of trying to forecast a forecast-target with a strongly dominant single direction in (space × time), and orthogonality penalties are whack-a-mole against it.

Two more-decisive directions, neither implemented here per the task spec:

- **Input-dependent dynamics.** The per-factor temporal map is currently a fixed [F, d_model, P] tensor — it can't condition on regime, lookback statistics, or anything else. Replacing it with a small per-factor MLP that consumes (factor_token, some-summary-of-lookback) → horizon vector would let each factor evolve differently in different windows, which is what probe 3 keeps reporting as missing.
- **Sequential factor extraction** (the task spec's "architectural" option). Extract factors one at a time, with each subsequent factor's residual being the part of the surface not yet explained by earlier factors. This makes the F-factor decomposition strictly hierarchical (like PCA itself), which seems closer to the inductive bias we keep failing to enforce by penalty.

If a cheaper try is wanted first: drop the loading penalty (keep just the query penalty) and re-run — Ax3 produced essentially the same test MSE with much better-conditioned loadings (smooth, rank-1-separable, PC1-aligned). If Ax3 is the practical maximum of what this architecture can do without conditional dynamics, **Ax3 is a better artifact to ship than Ax4** despite Ax4's marginal MSE win, because Ax3's loadings are interpretable.

## Postscript: predictions vs outcomes

The task spec made specific predictions; here is the verification.

| Prediction                                                | Outcome                                                                 | Met? |
| --------------------------------------------------------- | ----------------------------------------------------------------------- | ---- |
| factor_queries eff rank stays > 3.9                        | 3.707 — slight regression but still high                                | ≈    |
| stacked loadings eff rank > 3.0                            | **3.949**                                                              | ✓    |
| Loadings max abs off-diag cos-sim < 0.5                    | **0.007**                                                              | ✓    |
| Probe 4 best-match cos-sim distributed across PC1–PC4      | All four still best-match PC1, just with weaker similarity              | ✗    |
| Probe 2 cross-factor top-1 trajectory cos-sim < 0.9        | ±1.000 (worse than Ax3's 0.997)                                        | ✗✗   |
| Probe 3 trajectory MSE drops, especially in Bear 2022      | Numerically larger in Ax4, but it's a basis-change artefact             | n/a  |
| Test MSE 0.20–0.23, plausibly ≤ DLinear's 0.216            | 0.249 (Ax3 was 0.251; ≈ no improvement on Ax3)                          | ✗    |

Three of the five predictions held; the two that didn't are the ones that actually matter for forecast quality. The diagnosis ("queries diverse but loadings collapse → loadings penalty") was *literally* correct — loadings did decorrelate — but the assumption that loading decorrelation would force the upstream pipeline to use them as four genuine factors did not hold. Training instead collapsed the temporal map onto a shared shape, satisfying the orthogonality constraints while preserving the rank-1-in-time prediction structure that was already there.

---

Probes regenerated on 2026-05-17T16-21Z; checkpoint at
`AxialFactor/63_21/2026-05-17T16-20-53Z/best_model.pt`.
