# Ax5 → iTransformer comparison

DLinear (seed=0): `DLinear/eval/63_21/seed_0/preds.npy` — per-cell linear, no cross-cell sharing.
Ax5 (AR):         `AxialFactor/63_21/2026-05-17T16-55-06Z/` — factor decomposition with per-factor AR(1) and gated residual bypass.
iTransformer:     `iTransformer/63_21/2026-05-17T17-45-28Z/` — cell-as-token, 2 pre-norm transformer blocks (d_model=64, n_heads=4, ffn_ratio=2, dropout=0.1), no normalisation, fixed 2D sinusoidal PE, small-init head.

All three trained seed=0, same data split (70/10/20, data_end=2023-12-29), same trainer (AdamW lr=1e-3, wd=1e-4, batch=32, epochs≤100, patience=15, min_epochs=15, grad_clip=1.0).

## Headline

| Metric | DLinear | Ax5 | iTransformer |
| --- | --- | --- | --- |
| **Test MSE** | **0.2163** | 0.2390 | **0.2660** |
| Test RMSE | 0.4651 | 0.4889 | 0.5157 |
| Test MAE | 0.2847 | 0.3081 | 0.3220 |
| Best val (epoch) | n/a (single seed eval) | 0.181 (e15) | **0.184 (e4)** |
| Stop epoch | n/a | 30 | **19** |
| Final train MSE | n/a | 0.072 | **0.039** |
| Train/val gap at stop | n/a | ≈ 2.7× | **≈ 6.1×** |
| Total params | ~6.5k (DLinear had 2,360 trend+seasonal) | 2,193 | **72,661** |

## Per-horizon ratio vs DLinear (lower is better)

| h | DLinear MSE | Ax5/DL | iTransformer/DL |
| --: | --: | --: | --: |
| 1 | 0.0260 | 2.14 | **2.12** |
| 2 | 0.0463 | 1.55 | 1.61 |
| 3 | 0.0711 | 1.25 | 1.32 |
| 4 | 0.0873 | 1.22 | 1.30 |
| 5 | 0.1051 | **1.16** | 1.28 |
| 7 | 0.1447 | **1.09** | 1.24 |
| 10 | 0.1978 | **1.07** | 1.25 |
| 15 | 0.2932 | **1.10** | 1.20 |
| 21 | 0.3894 | **1.10** | 1.21 |

iTransformer's per-horizon profile is **worse than Ax5 at every horizon except h=1, where they tie**. At mid horizons (h=5–10) — Ax5's win zone vs DLinear — iTransformer is markedly worse than Ax5 (1.25–1.28× vs 1.07–1.16×). The cross-cell attention is **not** discovering a useful structure that DLinear misses; it's adding capacity that overfits and harms generalisation across the whole horizon range.

## Per-regime pooled MSE + ratios vs DLinear

| Regime | n | DLinear | Ax5 | iTransformer | Ax5/DL | iT/DL |
| --- | --: | --: | --: | --: | --: | --: |
| COVID | 254 | 0.5324 | 0.5934 | 0.6866 | **1.12** | 1.29 |
| Reflation calm | 252 | 0.0833 | 0.0813 | 0.0876 | **0.98** | 1.05 |
| Bear 2022 | 251 | 0.1553 | 0.1662 | 0.1657 | 1.07 | **1.07** |
| Normalisation | 250 | 0.0906 | 0.1110 | 0.1192 | 1.22 | 1.32 |

iTransformer doesn't beat DLinear in any regime. Closest in Reflation calm (1.05×); worst in Normalisation (1.32×) and COVID (1.29×). Ax5 still owns Reflation calm at 0.98×. iTransformer matches Ax5 only in Bear 2022 (both 1.07×).

## What happened — diagnosis

**The model overfit, hard.** The training curve tells the whole story:

| Epoch | train | val |
| --: | --: | --: |
| 1 | 0.249 | 0.193 |
| **4 [best]** | **0.103** | **0.184** |
| 7 | 0.080 | 0.234 |
| 10 | 0.060 | 0.252 |
| 15 | 0.045 | 0.222 |
| 19 (stop) | 0.039 | 0.238 |

After epoch 4, train MSE keeps falling (0.10 → 0.04) while val MSE *rises* (0.18 → 0.24). Best-val came at epoch 4. By stop, the train/val ratio is **6.1×** — the model had memorised the training windows and lost generalisation.

The prediction in the task spec called this risk explicitly: *"100k+ params on 1,200 windows is non-trivial. Watch the train-val gap."* The model has 72.7k params and the training set has 3,440 windows × 21-h forecasts. Even with that many target points, 72k params plus full cross-cell attention (which can route information non-locally in arbitrary ways) is too much capacity for this dataset. The model fits the train period (largely 2004–2018 IV regimes) and the 2019–2023 test period contains shifts it can't extrapolate to.

DLinear has roughly **~6.3k params** for a comparable forecast (per-cell linear seasonal + trend, n_channels=150, kernel=31, lookback=63, pred_len=21). Ax5 has **2.2k params**. Both stay well within the regime where capacity ≪ window count and don't overfit. iTransformer is an order of magnitude bigger.

## Per-prediction outcomes

| Prediction (from the task spec)                                | Outcome                                       | Met? |
| -------------------------------------------------------------- | --------------------------------------------- | ---- |
| Pooled MSE 0.20–0.23 (at or below DLinear's 0.216)             | **0.266 — far worse than 0.23**               | ✗    |
| Per-horizon shape: similar to DLinear, slightly better at long  | Worse than DLinear at every h ≥ 2             | ✗    |
| Per-regime: matches DLinear's regime profile                    | Roughly yes — same regime ranking, but uniformly +5–30 % | ≈ |
| Overfitting risk noted; watch train/val gap                     | **Realised — 6.1× train/val gap by stop**     | ⚠    |

The "iTransformer matches DLinear" outcome was the expected midpoint, "iTransformer beats DLinear cleanly" was the upside, and "iTransformer underperforms DLinear" was the informative-downside. We got the **informative-downside**.

## What the result tells us

The bet was that the IV surface has cross-cell structure that DLinear's per-cell-only design fails to exploit, and that cross-cell self-attention would discover and use that structure. On this dataset, with these training windows, that bet **did not pay**:

- **Cross-cell attention adds capacity faster than it adds inductive bias.** Full self-attention has ~17k params per block on a 150-token grid; two blocks (~34k attention params) plus the embed and head bring the total to 72k. That's an order of magnitude more capacity than DLinear or Ax5, and it fits the training distribution too well.
- **The factor-bottleneck designs (AxialFactor at 2.2k params) act as effective regularisation by construction**. Forcing cross-cell information through F=4 spatial loadings is exactly the inductive prior that limits capacity to "things that factor". The fact that Ax5 generalises better than iTransformer despite Ax5's collapsed factor decomposition (all loadings still aligning with PC1) is informative: **the bottleneck itself was doing useful work**, even when it wasn't using all its theoretical capacity. iTransformer has no equivalent constraint.
- **DLinear stays the operating baseline on this data.** No model in this lineage (DLinear, PatchTST, HOT, Tucker_DLinear, GWN, AxialFactor Ax2–Ax6, iTransformer) clearly beats it on pooled MSE. Several tie or beat it in specific regimes (Ax5 in Reflation calm, Ax3/Ax4 in Bear 2022) but none on pooled.

What would be needed to make iTransformer competitive here, in increasing order of intrusion:

1. **Aggressive weight decay.** Default wd=1e-4 was clearly insufficient. wd=1e-2 or higher might prevent the overfitting; worth trying as a one-line change.
2. **Reduce `n_blocks` to 1 and `d_model` to 32.** Cuts param count to ~20k. Closer to the dataset's effective complexity.
3. **Add input normalisation back** (RevIN-style or per-cell mean removal as the AxialFactor lineage used). This would simplify what the model has to learn — only the dynamics, not the levels — and likely close most of the overfit. But the task spec explicitly prohibited this.
4. **Increase training data**. The train period is 70 % of 5,033 rows — about 3,440 windows. Modern transformer-time-series papers typically train on tens of thousands. The dataset is just too small for a 72k-param attention model without regularisation.

If the iTransformer line is to be revisited, **option 1 (weight decay) is the cheapest test** of whether the architecture is fundamentally over-parameterised on this data or just under-regularised. If wd=1e-2 doesn't close the val/test gap, the conclusion is that without input normalisation, this model class is the wrong fit for IV surfaces with ~3.4k training windows.

For now: **the AxialFactor → iTransformer pivot did not deliver**. The most informative thing it told us is that the factor bottleneck wasn't a *fault* of the model — it was a *feature*, providing implicit regularisation that this dataset needs. The DLinear baseline remains undefeated.

---

Trained on 2026-05-17T17-45Z; checkpoint at
`iTransformer/63_21/2026-05-17T17-45-28Z/best_model.pt`.
