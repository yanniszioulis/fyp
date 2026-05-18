# AxialFactor Ax5 → Ax6 comparison

Ax3 (queries):     `AxialFactor/63_21/2026-05-17T16-03-24Z/` — orthogonal query init + factor-query ortho penalty.
Ax5 (AR, no pens): `AxialFactor/63_21/2026-05-17T16-55-06Z/` — per-factor AR(1); ortho penalties removed.
Ax6 (this run):    `AxialFactor/63_21/2026-05-17T17-16-02Z/` — Ax5 + per-horizon sigmoid gate on the AR path (mirror of the residual path's gate).

Only delta from Ax5 → Ax6: the new `horizon_emb [P, 16]` + `gate_linear(16, 1)` inside `_PerFactorAR`, plus one extra line in `_PerFactorAR.forward` to apply the gate multiplicatively to `g`. +353 params total. Identical training hyperparameters and seed.

## Three-way headline table

| Metric                                                 | Ax3 (queries) | Ax5 (AR, no gate) | Ax6 (AR + gate) |
| ------------------------------------------------------ | ------------- | ----------------- | --------------- |
| **Test MSE**                                           | 0.2513        | **0.2390**        | 0.2478          |
| Test RMSE                                              | 0.5013        | 0.4889            | 0.4978          |
| Test MAE                                               | 0.3085        | 0.3081            | 0.3118          |
| Best val (epoch)                                       | 0.166 (e6)    | 0.181 (e15)       | **0.173 (e15)** |
| Stop epoch                                             | 21            | 30                | 30              |
| Total params                                           | 7,582         | 2,193             | 6,532           |
| Per-factor AR persistences ρ_f                         | n/a (linear)  | [0.85, 0.93, 0.90, 0.85] | [0.86, 0.85, 0.84, 0.89] |
| Per-factor AR long-run means μ_f                       | n/a           | [−0.009, −0.015, −0.011, −0.007] | [−0.008, −0.008, −0.006, −0.010] |
| AR gate @ h=1                                          | n/a           | **(1.00 implicit)** | **0.063**       |
| AR gate @ h=10                                         | n/a           | (1.00 implicit)    | **0.982**       |
| AR gate @ h=21                                         | n/a           | (1.00 implicit)    | **0.996**       |
| Horizon-shape cross-factor cos-sim (max abs)           | 0.998         | 1.000             | 0.999           |
| Horizon-shape participation-ratio rank (out of 4)      | n/a           | 1.32              | 1.16            |

## Short summary (5 bullets)

- **The gate prediction held at h=1 — completely. AR gate = 0.063 at h=1, 0.10 at h=2, 0.15 at h=3** — the model learned to suppress AR almost entirely at short horizons, exactly the targeted fix. Final 21-value gate vector ramps smoothly: `[0.063, 0.105, 0.151, 0.205, 0.270, 0.348, 0.429, 0.533, 0.677, 0.982, 0.997, ..., 0.996]`. The sharp transition between h=9 (0.68) and h=10 (0.98) is striking — the model treats h≤9 and h≥10 as nearly two different problems.
- **h=1 MSE ratio vs DLinear recovered from 2.14× (Ax5) to 1.19× (Ax6) — best result in any AxialFactor variant.** At h=1 Ax6 is essentially matching DLinear (0.0310 vs 0.0260; +19 %), versus Ax5's catastrophic +114 %. h=2 also recovered (1.20× vs 1.55×). The fix worked at the target.
- **But pooled MSE *regressed*** (0.2390 → 0.2478, **+3.7 %**). The gate-supplied gain at short horizons was eaten by mid-horizon (h=5–10) regression — Ax6 at h=10 is **1.26× DLinear**, vs Ax5's 1.07× (and Ax3's 1.18×). The gate that suppressed AR at h=1–9 also suppressed it at h=5–9, where Ax5 had been the closest variant to DLinear. Mid-horizon was Ax5's win zone; Ax6 traded it away.
- **The ρ-spread that made Ax5 interesting collapsed in Ax6.** Ax5 found three timescales: [0.85, 0.93, 0.90, 0.85] (range 0.08). Ax6 found essentially one: [0.86, 0.85, 0.84, 0.89] (range 0.05). The participation-ratio rank of the [F, P] horizon-shape matrix dropped from 1.32 (Ax5) to 1.16 (Ax6). The model decided it didn't need diverse AR timescales because the gate could provide the horizon-differentiation by itself — the gate is doing the work that ρ_f's spread used to do. A different (and IMO worse-conditioned) decomposition.
- **Val improved (0.181 → 0.173) but test got worse — a val/test mismatch.** The model genuinely fits the val period better with the gate, but the basin it found doesn't transfer. This is the val/test divergence the spec warned about ("if pooled MSE doesn't drop, the next step is the iTransformer pivot"). Looking at the per-regime numbers: Reflation calm went from a **0.98× win** for Ax5 to **1.17× loss** for Ax6. Normalisation went from 1.23× to 1.33×. Smoothly-evolving regimes were Ax5's strength, and the gate that helped h=1 hurt those regimes specifically.

## Verification of predictions from the task spec

| Prediction                                                | Outcome                                                   | Met? |
| --------------------------------------------------------- | --------------------------------------------------------- | ---- |
| AR gate shape: monotone increasing, ~0.2 at h=1 to ~0.9 at h=21 | Yes — actually sharper: 0.06 at h=1, near 1.0 by h=10     | **✓** |
| h=1 MSE ratio 1.4–1.7× DLinear (recovered)                | **1.19× — better than predicted** (almost matches DLinear) | **✓** |
| h=21 ratio 1.08–1.12× DLinear                             | 1.10× (matches Ax5)                                       | ✓    |
| Pooled MSE 0.225–0.232 (below 0.239)                      | **0.248 — REGRESSED above Ax5**                            | ✗    |

The shape prediction was perfect. The MSE prediction was wrong. The gate worked at its target (h=1) and at long horizons (h=21), but mid-horizon (h=5–10) got worse than the prediction allowed for. Net: the gate is the right *kind* of fix, but on this data the trade-off it implies (give up some mid-horizon AR signal in exchange for h=1) doesn't pay.

## Per-horizon and per-regime vs DLinear (seed=0)

### Per-horizon ratios

| h | DL MSE | Ax3/DL | Ax5/DL | Ax6/DL |
| --: | --: | --: | --: | --: |
| 1 | 0.0260 | 1.73 | 2.14 | **1.19** |
| 2 | 0.0463 | 1.45 | 1.55 | **1.20** |
| 3 | 0.0711 | 1.27 | 1.25 | **1.14** |
| 4 | 0.0873 | 1.28 | 1.22 | 1.21 |
| 5 | 0.1051 | 1.24 | **1.16** | 1.23 |
| 7 | 0.1447 | 1.17 | **1.09** | 1.21 |
| 10 | 0.1978 | 1.18 | **1.07** | 1.26 |
| 15 | 0.2932 | 1.14 | **1.10** | 1.13 |
| 21 | 0.3894 | 1.15 | 1.10 | **1.10** |

Cleanly: **Ax6 owns h=1–3; Ax5 owns h=5–10; both tie at h=15–21**. Ax6's gain at short horizons is real and substantial; its loss at mid horizons is also real and substantial.

### Per-regime pooled MSE + ratios vs DLinear

| Regime | n | DLinear | Ax3 | Ax5 | Ax6 | Ax3/DL | Ax5/DL | Ax6/DL |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| COVID | 254 | 0.5324 | 0.6509 | **0.5934** | 0.6048 | 1.22 | **1.12** | 1.14 |
| Reflation calm | 252 | 0.0833 | 0.0912 | **0.0813** | 0.0977 | 1.10 | **0.98** | 1.17 |
| Bear 2022 | 251 | 0.1553 | 0.1533 | 0.1662 | 0.1639 | 0.99 | 1.07 | 1.06 |
| Normalisation | 250 | 0.0906 | 0.1051 | 0.1110 | 0.1207 | 1.16 | 1.23 | **1.33** |

Ax5's win in Reflation calm is gone; Normalisation gets worse. The gate is hurting in the regimes where mid-horizon AR signal was most useful. Ax6 doesn't beat DLinear in any regime.

## What this tells us about next steps

Both the spec's success prediction *and* its failure prediction partially hold. The gate fixed h=1 cleanly (success); pooled MSE didn't drop (failure). The honest reading:

- **The AR path's mid-horizon contribution in Ax5 was load-bearing** — it was the *main reason* Ax5 beat the older variants. Gating it away forced the model to find a worse basin.
- **The gate is doing the differentiation that ρ_f used to do.** When the gate is free to vary smoothly across horizons, the optimiser doesn't need diverse AR timescales — the gate alone can shape the AR contribution. So ρ_f converged to one timescale and the gate carries the horizon-dependence. This is a *less interpretable* decomposition than Ax5's three-timescale story.
- **For test-set MSE, Ax5 is the operating model to ship.** Ax6 is technically interesting (it shows the gate mechanism works exactly as predicted at h=1) but on the test set we care about pooled MSE, and Ax5 wins by 0.009.

What the spec said next-if-this-fails: the iTransformer pivot. On the merit of these numbers (Ax3 → Ax5 → Ax6 pooled MSE: 0.251 → 0.239 → 0.248), the marginal value of further factor-architecture changes is exhausted. The model has been tuned to a local optimum at +10 % above DLinear, with the gap concentrated in volatile regimes (Bear 2022, Normalisation) where the smooth-AR inductive bias breaks. The factor framework as currently structured (4 PC1-anchored loadings, AR temporal dynamics, gated residual bypass) is near its ceiling on this data.

A different model class — cell-as-token attention without the factor bottleneck, like iTransformer — would lift exactly this ceiling: it lets each cell carry its own time-series state without forcing it through a 4-mode spatial bottleneck. The factor decomposition is interpretable but the data doesn't actually behave like a 4-factor surface; it behaves like 150 cells with mostly-shared but not-identical dynamics, which is what cell-as-token attention models natively. That's the next experiment.

For *this* line of work, **Ax5 stays the operating AxialFactor**.

## Postscript: final AR gate vector (Ax6)

```
Final AR persistences   ρ = [+0.8596  +0.8506  +0.8390  +0.8916]
Final AR long-run means μ = [-0.0075  -0.0079  -0.0059  -0.0103]
Final AR gate (h=1..21) =
  [0.063 0.105 0.151 0.205 0.270 0.348 0.429 0.533 0.677 0.982 0.997
   0.998 0.999 0.999 0.999 0.999 0.998 0.998 0.997 0.997 0.996]
```

The gate is a step function with a smooth ramp-up at h=1–9 and a sharp lock-in at h=10. It is exactly the kind of monotone-increasing horizon shape the residual path's gate would learn the mirror of. The two paths together approximate a clean horizon split: residual owns h=1–9, AR owns h=10–21. That this clean split *makes pooled MSE worse* is the substantive finding — the model thought it would help (val agreed), but on the test set the mid-horizon AR signal it gave up was worth more than the h=1 noise it cut.

---

Probes regenerated on 2026-05-17T17-16Z; checkpoint at
`AxialFactor/63_21/2026-05-17T17-16-02Z/best_model.pt`.
