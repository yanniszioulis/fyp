# Thoughts — designing a model to beat the benchmark

Working brainstorm. This is intentionally messy, with dissenting views,
alternatives we considered and rejected, and open questions. Trim or
formalise later as decisions get made.

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

## Open question 1 — Is DCISM an existing model?

**No.** I made the name up in the previous reply. The pieces are all
established (DLinear's series decomposition + channel-indep linear maps;
MAE-on-original-IV from DynGWN's legacy loss; 2D conv smoothing for grid
priors), but the combination isn't a published model. We'd be **proposing
it as our contribution.**

Implications for the report:
- This is good — the project is supposed to produce a new model, not
  evaluate existing ones.
- Need to position carefully: "we propose DCISM, which combines [known
  component A] with [known component B] in light of empirical findings X,Y,Z
  on this dataset."
- Each component's inclusion is justified by a specific result in our
  comparison (signals 1-5 above) — that's a stronger story than picking
  components for theoretical reasons.

Alternative names to consider (cosmetic):
- **DCISM** — Decomposed Channel-Independent Surface Model
- **GraphLinear** — DLinear + 2D smoothing
- **D-Surface** — short, descriptive
- Probably want a name that signals "linear-first, surface-aware" not "yet
  another transformer".

---

## Open question 2 — Should the whole project shift to total-variance or log-IV space?

Three target spaces are on the table:

| Space | Quantity | Why it might help | Why it might hurt |
|---|---|---|---|
| **IV** (current) | σ_BS | What papers report; interpretable | Multiplicative dynamics → not stationary in IV |
| **log-IV** | log σ | Multiplicative shocks → additive; forces positivity at output | Slightly heavier tails; less interpretable; subtle bias when retransforming |
| **Total variance** | w = σ²·τ | Canonical pricing quantity (no-arb structure, monotone in τ); used in surface modelling | τ-scaling makes "fair" cross-cell comparison harder; literature for ML forecasting doesn't use it |

**Argument for shifting the whole project:**
- log-IV is a common choice in vol-forecasting literature; many "stylised
  facts" of vol (mean-reversion, clustering, leverage) are stated in log space.
- Total variance is the right financial primitive — if anyone uses these
  forecasts to price options or trade vol, total variance is what they want.
- If log-IV genuinely improves all models, it's "free" performance.

**Argument against shifting:**
- It's a 1-2 day refactor: every script, plus compare_models, plus the
  baseline ground truth. Risk of off-by-one bugs in the inverse-transform.
- The literature comparable to ours (PatchTST/HOT papers on TS forecasting,
  including IV extensions) reports in original space.
- Untested claim: most likely 0-5% gain, possibly negative for some models
  (e.g. PatchTST whose RevIN is tuned for symmetric standardised inputs).
- Adds an extra retransformation step that compounds prediction errors at
  long horizons.

**Recommendation: don't rebase the whole project. Run it as an ablation.**

Concretely:
- Add a `--target_transform {none,log,total_variance}` flag to ONE model
  (probably DCISM or DLinear for cheap iteration).
- Train all three variants with the same architecture and hyperparams.
- Report a small table: "for the same model, target = X gives MSE Y / IC Z".
- If log-IV uniformly wins by ≥ 3% across multiple horizons, then escalate
  and consider a project-wide switch.

This costs us 30 minutes of code and 30 minutes of training for a clean
empirical answer, vs. committing now to a refactor that may not help.

---

## Open question 3 — Should all models use MAE-on-original, and should we report MAE over MSE?

Two sub-questions, which deserve different answers.

### 3a. Should we switch all models' training loss to MAE-on-original-space?

The case for:
- DynGWN's 18% MSE improvement is the loudest signal in the benchmark.
- IV's amplitude is meaningful (vol of 0.5 vs 0.1 is a *real* difference,
  not a normalised feature). MAE preserves that; MSE-on-scaled doesn't.
- Vol spikes are real signal, not noise. MAE doesn't over-penalise them.

The case against:
- Sample of one. DynGWN may have been the only model whose architecture
  was hurt by scaled-MSE; switching DLinear could be neutral or harmful.
- DLinear's success suggests it's already robust. Forcing a different loss
  on a model that doesn't need it adds risk.
- Papers reporting these models use MSE; deviating muddies the comparison.\\

**Recommendation: empirical test on DLinear, then decide.**

Plan:
- Re-train DLinear with masked-MAE-on-original (the DynGWN loss) and
  everything else identical. Compare both checkpoints in our framework.
- If MAE-trained DLinear is ≥ same → switch DLinear (and probably the
  whole pipeline) to MAE.
- If meaningfully worse → keep MSE for the linear models, MAE for the
  graph/spatial models. Document the divergence as a finding ("loss
  selection is architecture-specific").

Quick to run: ~20 min on Colab.

### 3b. Should we report MAE-original over MSE?

Two distinct issues here:

**Reporting MAE in addition** (yes, easy):
- compare_models.py currently reports MAE *in scaled space*.
- Add MAE_original = mean |inverse_transform(pred) − raw_y|.
- This is more interpretable: "the model is on average X vol-points off".
- One extra column in the table; one extra dict key. ~10 lines of code.

**Reporting MAE *instead of* MSE** (no, keep both):
- MSE/RMSE is what TS forecasting papers report; we'd be hard to compare to.
- MSE penalises tail errors more — relevant for risk applications.
- Best practice: report MSE *and* MAE *and* IC. Pick the headline metric
  per claim ("DLinear has best long-horizon rank correlation" → IC chart;
  "DCISM has lowest squared error" → MSE).

**Recommendation:**
- Add MAE-original as an additional metric in compare_models.py.
- Keep MSE in scaled space as the headline number for cross-model comparison
  (consistent with prior published work).
- For the report: lead with IC (most relevant for trading); MSE/MAE as
  secondary.

---

## Model design — DCISM detailed proposal

Working name: **DCISM** = Decomposed Channel-Independent Surface Model.

### Architecture

```
input: [B, seq_len=21, 400]   (scaled-space surface history)
    │
    ├── Series decomposition (boundary-padded MA, kernel=13)
    │       trend [B, 21, 400]      ← slow component
    │       season = input − trend  ← residual
    │
    ├── Channel-independent linear maps (DLinear-style einsum)
    │       trend → trend_pred [B, 63, 400]
    │       season → season_pred [B, 63, 400]
    │       core_pred = trend_pred + season_pred
    │
    ├── Surface-graph polish     ← NEW
    │       reshape core_pred to [B, 63, 20, 20]   (H=mono, W=tau, F-order)
    │       per-horizon 2D conv, kernel 3×3, identity-init
    │       reshape back to [B, 63, 400]
    │
    └── output: residual_skip(core_pred) + polish_correction
```

Identity-init for the conv kernel: `[[0,0,0],[0,1,0],[0,0,0]]` so the layer
starts as the identity. Any deviation has to be paid for by reduced training
loss → conv only learns smoothing where it actively helps.

### Loss

```python
loss = mean | inverse_transform(model_output) − y_original |
```

That is, MAE-on-original-IV-space (the DynGWN insight). y stays in original
space throughout; x is scaled for input only.

### Why this should win

| Component | Picks up signal | Inspiration |
|---|---|---|
| Channel-indep linear maps | (1) | DLinear |
| Trend/season decomposition | (4) | DLinear |
| Direct multi-step output | (5) | DLinear |
| 2D conv "polish" (small) | (3) | DynGWN grid prior, but light |
| MAE on original IV space | (2) | DynGWN legacy loss |

Worst-case behaviour (if the conv learns nothing useful): output ≈ DLinear's
output. So DCISM ≥ DLinear in expectation, by construction.

Parameter count: ~1.1M from einsum maps + ~0.6K per horizon for 3×3 conv ≈
~1.13M total. Tiny compared to HOT (1.33M) and DynGWN (1.49M), comparable
to DLinear (1.11M).

### Risks

- Conv "polish" might overfit since channels are smoothed at every timestep.
  Mitigation: small kernel, identity init, possibly dropout on the conv
  output.
- Inverse-transform-then-MAE in the loss complicates the autograd graph
  slightly. Verified pattern from DynGWN works; should port cleanly.
- If MAE-on-original doesn't help DLinear-style architectures, DCISM might
  underperform DLinear with MSE-scaled. The MAE retraining test (3a)
  resolves this *before* we commit to DCISM's loss.

---

## Alternative model ideas (rejected/deferred but worth recording)

### A. Factor head (low-rank prior)

PCA on training data → top K=8 components → tiny VAR/DLinear on factors →
project back. Plus a per-cell residual model. Should help long-horizon IC
significantly because factors are smoother and more predictable.

**Why deferred**: more moving parts. Want DCISM's clean baseline first.
Could be a follow-up: "DCISM + factor head" if DCISM alone doesn't get us
to the top.

### B. Multi-resolution / wavelet decomposition

Decompose into multiple frequency bands; predict each separately; reconstruct.
Generalises DLinear's MA-decomposition.

**Why deferred**: more complex; benefit unclear; DLinear's two-band split
already captures most of the gain.

### C. Domain-specific normalisation

Inputs/targets transformed via log-IV or total-variance.

**Why deferred**: handled as orthogonal ablation (open question 2).

### D. Horizon-conditional ensemble

Linear pool of DLinear, DynGWN, VAR1 with weights learned on validation,
possibly per-horizon.

**Why deferred**: feels like a competition trick rather than a contribution.
The story would be weaker. But cheap; could be a "table stakes" baseline at
the end of the report.

### E. Surface-shape-preserving regulariser

Loss penalises deviations from observed cross-sectional curvature
(monotonicity in moneyness, no calendar-arb violation). Makes predictions
more "surface-like".

**Why deferred**: rich idea but expensive to debug. Could be a follow-up
paper, not the right scope here.

### F. RevIN-on-original-space variant

Rather than StandardScaler-then-RevIN, normalise per-instance directly on
raw IV. Tighter coupling to the data's actual scale.

**Why deferred**: orthogonal pipeline change. May or may not help; can be
swapped in DCISM as an experiment without restructuring.

---

## Experiments to run before committing to DCISM

In order of how cheap-and-informative they are:

1. **DLinear with MAE-on-original loss** (~20 min on Colab). Tests open
   question 3a. If MAE helps DLinear too, DCISM's loss choice is justified
   for the whole pipeline.

2. **DCISM v0** (just DLinear + 2D-conv polish, MSE-scaled loss) (~30 min).
   Tests whether the 2D conv adds value. If not, we drop it from DCISM.

3. **DCISM full** (decomposition + channel-indep linear + 2D conv +
   MAE-on-original) (~30 min). The proposed model.

4. **Target-transform ablation** on DLinear: IV vs log-IV vs total-variance
   (~1 hour, three runs). Tests open question 2.

5. **Factor-head extension** if DCISM clears the bar but isn't dominant
   (~1 hour). Adds factor structure on top of DCISM.

Total budget: ~3 hours of Colab time for definitive answers on all open
questions, plus a fully evaluated proposed model.

---

## Decisions needed (over to the user)

- [ ] **Naming**: DCISM, GraphLinear, D-Surface, something else?
- [ ] **Target space**: ablate (recommended) or rebase the whole project?
- [x] **Loss switch**: agreed — added MAE-original as an OPTION on DLinear
  (and DCISM) without replacing MSE. Result dirs use `_lossmae` suffix to
  avoid overwriting MSE runs.
- [ ] **MAE-original metric**: add to compare_models.py? (recommended yes)
- [x] **Implementation order**: DCISM v0 built first (DLinear core +
  identity-init 2D polish + optional MAE-original loss). Loss/target ablations
  on DLinear come next, results inform DCISM's loss default.

---

## Implementation log

**Built so far:**

1. **DLinear `--loss {mse,mae_original}`** (default mse) — adds masked-MAE
   loss in original IV space (port of DynGWN's legacy `masked_mae`). Result
   dir gets `_lossmae` suffix when loss=mae_original; MSE runs unchanged.

2. **DCISM v0** (`DCISM/dcism_spx_iv.py`):
   - DLinear core (boundary-padded MA decomp + channel-indep einsum maps)
   - 2D Conv polish on output reshaped as [H_mono=20, W_tau=20], with
     **identity initialisation** so it starts as a no-op (verified: at init,
     `max|DCISM_out − DLinear_core| = 0.0`).
   - Same `--loss {mse, mae_original}` flag as DLinear.
   - Default kernel: 3×3, no dropout. Total params: 1,144,584
     (~35K more than DLinear's 1,108,800; the polish conv).

3. **train.py**: dispatcher recognises `dcism`. `--loss` and `--conv_kernel`
   / `--conv_dropout` route correctly. `--models all` now runs 7 jobs
   (incl. DCISM at MSE default).

4. **compare_models.py**:
   - Added `name_filter` field to MODELS specs so MSE-suffixed and
     MAE-suffixed result dirs cluster into separate rows.
   - Registry now has `DLinear`, `DLinear(MAE)`, `DCISMv0`, `DCISMv0(MAE)`
     plus the existing models. Total 9 rows in the comparison table.

**Experiments to queue on Colab next:**

| # | Run | Tests |
|---|---|---|
| 1 | `DLinear --loss mae_original --epochs 100` | open question 3a (loss ablation on DLinear) |
| 2 | `DCISM --loss mse --epochs 100`            | does the conv polish add value over DLinear? |
| 3 | `DCISM --loss mae_original --epochs 100`   | does combining decomp + polish + MAE win? |

Three runs, ~1 hour total on a T4. After this round we'll have a clean
2×2 ablation: {architecture: DLinear vs DCISM} × {loss: MSE vs MAE-orig}.

**Still TBD** (deferred):
- Add `MAE_original` metric column to compare_models.py (just a few lines).
- log-IV / total-variance target ablation on DLinear or DCISM.
- Factor-head extension if DCISM clears the bar but isn't dominant.

---

## Findings from 2×2 ablation (post first Colab cycle)

```
Metric    DLinear  DLin(MAE) DCISMv0  DCISM(MAE)  | best
mse        0.148    0.159    0.150    0.153       | DLinear
ic_mean   +0.571   +0.531   +0.569   +0.576       | DCISM(MAE) ★
t+1 IC    +0.930   +0.839↓  +0.932   +0.928       | DCISM(MSE)
t+1 MSE    0.016    0.053↓   0.013★   0.014       | DCISM(MSE)
t+63 IC   +0.399   +0.382   +0.395   +0.415★      | DCISM(MAE) ★
bias      +0.027   -0.014   +0.034   -0.077       | DCISM(MSE) most-calibrated
```

**Three findings:**
1. **DCISM(MAE) wins on IC** — overall (0.576) and at every horizon, strongest at t+63 (0.415 vs DLinear 0.399).
2. **DCISM(MSE) wins on t+1 MSE** (0.0126, beats persistence 0.0127). Isolates the value of the conv polish.
3. **MAE-original helps DCISM but actively hurts DLinear** at short horizons. Initial hypothesis: per-channel std-weighting introduced by training in original space.

---

## Std-weighting hypothesis test — REJECTED

Plan: train DLinear with plain MAE in scaled space (`mae_scaled`), which is
equivalent to (1/std)-weighted MAE in original space — uniform per-channel
weighting. If std-weighting is the culprit, t+1 IC should recover.

Result: **identical to MAE-original**. t+1 IC 0.840 vs 0.839; t+1 MSE 0.052
vs 0.053; differences at the 4th decimal.

→ Std-weighting is NOT the cause.

## New hypothesis — MAE's gradient magnitude

`∂|x|/∂x = sign(x)` is constant in magnitude regardless of error size.
`∂x²/∂x = 2x` is proportional to the error. At t+1, errors are small;
MSE provides fine-grained "make this small error smaller" gradient that
MAE cannot. The damage fingerprint matches:

```
Horizon  MSE-loss  MAE-loss  ratio
t+1      0.016     0.052     3.3×
t+5      0.060     0.082     1.4×
t+10     0.093     0.109     1.2×
t+21     0.131     0.149     1.1×
t+42     0.182     0.187     1.02×
t+63     0.200     0.200     1.00×
```

Concentrated at short horizons, fades to zero by t+63. That's the gradient
property, not channel weighting. **DCISM(MAE) doesn't suffer this** because
the conv polish provides additional fine-grained spatial refinement that
fills the small-error gradient gap.

If correct, **Huber loss** (MSE for small errors, MAE for large) should
recover DLinear's t+1 performance while preserving long-horizon advantage.

## Bias correction for DCISM(MAE) — naive version FAILED

Implemented as `DCISM/apply_bias_correction.py` (post-hoc): per-channel
additive offset fit on val, applied to test predictions. Saves to
`<src>_bc/`. compare_models auto-regenerates if pred.npy is missing.

| Metric | DCISMv0(MAE) | DCISMv0(MAE+BC) |
|---|---|---|
| mse | 0.153 | **0.155** ↓ |
| ic_mean | +0.576 | **+0.482** ↓↓ |
| t+1 IC | +0.928 | +0.820 ↓↓ |
| bias (test) | −0.077 | **+0.083** (over-corrected) |

Cause: **regime drift.** Val mean bias = +0.16; test bias was −0.077.
Adding +0.16 over-corrects. Val (~2021-2022) sits before the 2022-23 vol
regime change; test (2022-04 → 2025-06) is on the other side. Static offset
from val doesn't transfer.

Lesson: any future bias correction must be either time-adaptive
(rolling/online) or learned end-to-end (so training distribution matches
inference). Naive offset rejected.

---

## Next experiments to queue on Colab

| # | Command | What it tests |
|---|---|---|
| 1 | `python train.py --models dlinear --loss huber_scaled --huber_delta 1.0` | gradient-magnitude hypothesis: does Huber recover DLinear's t+1 IC vs MAE? |
| 2 | `python train.py --models dlinear --loss huber_scaled --huber_delta 0.3` | smaller delta = more MSE-like; ablation along the MSE↔MAE axis |
| 3 | `python train.py --models dcism --loss huber_scaled --huber_delta 1.0` | does Huber-DCISM match (MAE)'s long-horizon IC win without sacrificing short-horizon? |
| 4 | `python train.py --models dcism --conv_kernel 5 --loss mae_original` | larger polish kernel — more spatial smoothing capacity |
| 5 | `python train.py --models dcism --conv_kernel 5` | same polish ablation under MSE loss |

Total ~1-1.5 hr on a T4. After this we'll have:
- Loss ablation on identical DLinear arch: MSE / MAE / Huber-1.0 / Huber-0.3
- Polish-kernel ablation on DCISM under both losses
- Enough to pick a headline DCISM config for the report

**BC line of investigation paused** — naive offset doesn't generalise; would
need rolling/learned correction. Not pursuing unless Huber results re-open
the question.

## Implementation log (newest)

- DLinear `--loss mae_scaled` — std-weighting test (rejected hypothesis).
  Suffix `_lossmaescaled`.
- DLinear & DCISM `--loss huber_scaled` with `--huber_delta` (default 1.0).
  Suffix `_losshuberscaled_d<delta>`. Tests gradient-magnitude hypothesis.
- `DCISM/apply_bias_correction.py` — standalone BC utility. Reads
  ckpt+config from src dir, fits per-channel additive bias on val, applies
  to test, writes `<src>_bc/`.
- `compare_models.py`:
  - new rows: `DLinear(MAE-s)`, `DLinear(Huber)`, `DCISMv0(MAE+BC)`,
    `DCISMv0(Huber)`.
  - `_maybe_regen` auto-detects `_bc` suffixed dirs: when pred.npy is
    missing AND the source dir (without `_bc`) has ckpt+config, runs
    `apply_bias_correction.py` instead of standard predict-only.
  - Tightened `name_filter` lambdas to use `"_loss" not in n` (handles all
    non-MSE suffixes) and `not n.endswith("_bc")` (separates BC from MAE).
