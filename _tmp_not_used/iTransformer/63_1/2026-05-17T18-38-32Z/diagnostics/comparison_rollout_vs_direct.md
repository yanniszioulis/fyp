# iTransformer: 1-step rollout vs 21-step direct

- 1-step model: `iTransformer/63_1/2026-05-17T18-38-32Z/best_model.pt` (pred_len=1, trained as a single-step forecaster).
- Rollout: recursive 21-step prediction in standardised log-IV space; at iteration t, the lookback buffer slides one step left and the latest prediction is appended.
- All test predictions are evaluated against the same Yte from the pred_len=21 split (1007 windows).

## Pooled (test MSE / RMSE / MAE)

| Model | Test MSE | RMSE | MAE | vs DLinear |
| --- | ---: | ---: | ---: | ---: |
| DLinear (seed=0) | 0.2163 | 0.4651 | 0.2847 | 1.000× |
| iT 1-step → 21-roll | **0.3395** | 0.5826 | 0.3861 | 1.569× |
| iT v2 direct (per-cell head) | 0.2242 | 0.4735 | 0.2934 | 1.036× |
| iT v1 direct (shared head) | 0.2253 | 0.4747 | 0.2943 | 1.041× |
| persistence | 0.2305 | 0.4801 | 0.3001 | 1.066× |

## Per-horizon MSE ratio vs DLinear

| h | DLinear MSE | persist | v2 direct | **rollout** |
| --: | ---: | ---: | ---: | ---: |
| 1 | 0.0260 | 0.946 | 1.537 | **1.097** |
| 2 | 0.0463 | 0.903 | 1.177 | **1.141** |
| 3 | 0.0711 | 0.891 | 1.000 | **1.122** |
| 4 | 0.0873 | 0.960 | 1.017 | **1.231** |
| 5 | 0.1051 | 0.976 | 1.013 | **1.290** |
| 7 | 0.1447 | 0.989 | 1.004 | **1.359** |
| 10 | 0.1978 | 1.044 | 1.048 | **1.482** |
| 15 | 0.2932 | 1.081 | 1.030 | **1.629** |
| 21 | 0.3894 | 1.124 | 1.047 | **1.720** |

## Per-regime pooled MSE (and ratios vs DLinear)

| Regime | n | DLinear | persistence | v2 direct | **rollout** | rollout/DL | rollout vs v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COVID | 254 | 0.5324 | 0.5744 | 0.5580 | **0.8010** | 1.504× | 1.435× |
| Reflation calm | 252 | 0.0833 | 0.0961 | 0.0883 | **0.1353** | 1.626× | 1.533× |
| Bear 2022 | 251 | 0.1553 | 0.1509 | 0.1615 | **0.3212** | 2.068× | 1.989× |
| Normalisation | 250 | 0.0906 | 0.0965 | 0.0850 | **0.0946** | 1.044× | 1.113× |

## Interpretation

- Rollout pooled MSE: **0.3395** vs v2-direct **0.2242** → rollout is worse by 115.3 milliMSE.
- Compare h=1 and h=21 ratios specifically: at h=1 the rollout uses its native target, at h=21 it has compounded 20 steps of error.
- If the rollout is uniformly worse than direct, error compounding ('exposure bias') dominates — the model never saw its own predictions during training.
- If the rollout beats direct at short horizons (h=1–3) but loses at long horizons, the 1-step model fits short-horizon dynamics better but compounds them too fast.
- If the rollout beats direct *everywhere*, the multi-step head was wasting capacity that the 1-step head spends on getting the near-term dynamics right.

