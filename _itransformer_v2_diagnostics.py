#!/usr/bin/env python3
"""
Diagnostic probes for the per-cell head of an iTransformer v2 checkpoint.

Run:
    python _itransformer_v2_diagnostics.py \
        --run_dir iTransformer/63_21/2026-05-17T18-18-25Z/

Four probes (all on saved weights / forward-pass intermediates; no training):
  1. Cross-cell head similarity   — pairwise cos-sim across the 150 per-cell
                                     heads + SVD effective rank.
  2. Spatial structure of norms   — Frobenius norm per cell, plotted on the
                                     (moneyness × τ) grid.
  3. v1 ↔ v2-average head         — does v2 decompose as shared baseline +
                                     per-cell deltas? Note: v1 and v2 have
                                     different d_model so a direct cos
                                     comparison is undefined; we compute the
                                     within-v2 shared/delta decomposition
                                     instead, which answers the same question.
  4. Per-cell head usage by regime — for each window, the magnitude of each
                                     cell's contribution (token @ head[cell])
                                     after the head_norm step; grouped by
                                     regime and averaged.

All outputs land in <run_dir>/diagnostics/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from train import (                                  # noqa: E402
    LOOKBACK, ITransformer, _ITransformerAdapter,
    load_dataset, pick_device,
)
from eval_full import (                              # noqa: E402
    REGIMES, assign_regime, test_target_dates,
)


# ─── Loading ──────────────────────────────────────────────────────────────

def load_v2(run_dir: str):
    """Reconstruct the v2 model and verify its checkpoint loads cleanly."""
    with open(os.path.join(run_dir, "hyperparams.json")) as f:
        hp = json.load(f)
    with open(os.path.join(run_dir, "metrics_test.json")) as f:
        metrics = json.load(f)
    model = ITransformer(**hp["model_kwargs"])
    adapter = _ITransformerAdapter(model, hp["grid"]["n_tau"],
                                   hp["grid"]["n_money"])
    state = torch.load(os.path.join(run_dir, "best_model.pt"),
                       map_location="cpu", weights_only=True)
    adapter.load_state_dict(state)
    adapter.eval()
    return hp, metrics, adapter, model


def load_v1_head(v1_ckpt_path: str):
    """Return v1's shared head weight + bias, transposed to [d_in, d_out].

    PyTorch's nn.Linear stores [out, in]; we transpose to [in, out] so the
    convention matches v2's head_weight: [d_model, pred_len] per cell.
    """
    state = torch.load(v1_ckpt_path, map_location="cpu", weights_only=True)
    w = state["model.head.weight"]                         # [P, d_model_v1]
    b = state["model.head.bias"]                           # [P]
    return w.T.contiguous(), b                             # [d_model_v1, P]


def recompute_test_mse(adapter, data, device, batch=64):
    Xte, Yte = data["test"]
    adapter.to(device)
    preds = np.empty_like(Yte)
    with torch.no_grad():
        for s in range(0, Xte.shape[0], batch):
            xb = torch.from_numpy(Xte[s:s+batch]).to(device)
            preds[s:s+batch] = adapter(xb).cpu().numpy()
    adapter.cpu()
    return preds, float(((preds - Yte) ** 2).mean())


# ─── Probe 1: cross-cell head similarity ──────────────────────────────────

def probe1(out_dir, model):
    W_flat = model.head_weight.detach().cpu().numpy()       # [W*H, d, P]
    n_cells, d_model, P = W_flat.shape
    H_flat = W_flat.reshape(n_cells, -1)                    # [N, d*P]

    # Pairwise cosine similarity.
    norms = np.linalg.norm(H_flat, axis=1, keepdims=True) + 1e-12
    unit = H_flat / norms
    cos = unit @ unit.T                                     # [N, N]
    off = cos[~np.eye(n_cells, dtype=bool)]
    perc = np.percentile(off, [5, 25, 50, 75, 95])

    # SVD of the stacked head matrix.
    s = np.linalg.svd(H_flat, compute_uv=False)
    eff = float(s.sum() ** 2 / max((s * s).sum(), 1e-12))
    cum = np.cumsum(s * s) / max((s * s).sum(), 1e-12)
    def ev(k): return float(cum[min(k - 1, len(cum) - 1)])

    lines = [
        "PROBE 1: Cross-cell head similarity",
        "=" * 36, "",
        f"head_weight shape: {W_flat.shape}",
        f"Flattened to: ({n_cells}, {d_model * P})",
        "",
        "Pairwise cosine similarity (off-diagonal):",
        f"  mean: {off.mean():+.3f}",
        f"  p5:   {perc[0]:+.3f}",
        f"  p25:  {perc[1]:+.3f}",
        f"  p50:  {perc[2]:+.3f}  (median)",
        f"  p75:  {perc[3]:+.3f}",
        f"  p95:  {perc[4]:+.3f}",
        "",
        "SVD analysis (of the stacked [N_cells, d*P] head matrix):",
        f"  effective rank (participation ratio): {eff:.2f}  "
        f"(out of {min(n_cells, d_model * P)})",
        "  cumulative variance explained:",
        f"    k= 1:  {ev(1)*100:5.1f}%",
        f"    k= 2:  {ev(2)*100:5.1f}%",
        f"    k= 3:  {ev(3)*100:5.1f}%",
        f"    k= 5:  {ev(5)*100:5.1f}%",
        f"    k=10:  {ev(10)*100:5.1f}%",
        "",
        "Interpretation hint:",
        "  - mean cos > 0.9 AND eff_rank < 3 ⇒ heads collapsed; v2 ≈ v1.",
        "  - mean cos < 0.5 AND eff_rank > 20 ⇒ heads genuinely diverse.",
        "  - intermediate ⇒ heads cluster into a few patterns, not 150 "
        "independent maps.",
    ]
    with open(os.path.join(out_dir, "probe1_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    # Plot 1: heatmap of the cos-sim matrix.
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto",
                   origin="lower")
    ax.set_xlabel("cell index (W-outer, H-inner)")
    ax.set_ylabel("cell index")
    ax.set_title("Per-cell head pairwise cosine similarity")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe1_head_similarity_matrix.png"),
                dpi=120)
    plt.close(fig)

    # Plot 2: singular value spectrum.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(np.arange(1, len(s) + 1), s, "o-", markersize=4)
    ax.set_xlabel("k")
    ax.set_ylabel("singular value (log)")
    ax.set_title(
        f"Stacked head SVD spectrum (eff. rank {eff:.2f} / {len(s)})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe1_head_singular_values.png"),
                dpi=120)
    plt.close(fig)

    return {"mean_off_cos": float(off.mean()),
            "eff_rank": eff, "ev1": ev(1), "ev3": ev(3), "ev10": ev(10)}


# ─── Probe 2: spatial structure of head magnitudes ────────────────────────

def probe2(out_dir, model, v1_head_w):
    W_flat = model.head_weight.detach().cpu().numpy()       # [W*H, d, P]
    n_cells = W_flat.shape[0]
    norms = np.linalg.norm(W_flat.reshape(n_cells, -1), axis=1)  # [N]
    W, H = model.W, model.H
    grid = norms.reshape(W, H)                              # [W, H]

    # v1 reference (single number, since v1's head is shared across cells).
    v1_norm = float(np.linalg.norm(v1_head_w.numpy()))

    # Spatial total variation.
    tv_W = float(np.mean(np.abs(np.diff(grid, axis=0))))
    tv_H = float(np.mean(np.abs(np.diff(grid, axis=1))))

    lines = [
        "PROBE 2: Spatial structure of per-cell head magnitudes",
        "=" * 55, "",
        "Per-cell head Frobenius norms:",
        f"  mean:    {norms.mean():.4f}",
        f"  std:     {norms.std():.4f}",
        f"  min:     {norms.min():.4f}",
        f"  max:     {norms.max():.4f}",
        f"  max/min: {norms.max() / max(norms.min(), 1e-12):.2f}",
        f"  std/mean (dispersion): {norms.std() / max(norms.mean(), 1e-12):.3f}",
        "",
        f"For reference, v1 shared head Frobenius norm: {v1_norm:.4f}",
        "(broadcast to every cell in v1's forward; per-cell norm there is "
        "the same scalar for all 150 cells. Note v1 has d_model=16 vs v2's "
        "d_model=8, so absolute norms are not directly comparable.)",
        "",
        "Spatial smoothness (total variation of per-cell norms):",
        f"  along W (moneyness): {tv_W:.4f}",
        f"  along H (τ):         {tv_H:.4f}",
        "",
        f"Plots: probe2_head_norm_heatmap.png",
        "",
        "Interpretation hint:",
        "  - std/mean < 0.2  ⇒ uniform per-cell capacity; uniform WD addresses it.",
        "  - std/mean > 0.5 + structured spatial pattern ⇒ per-cell capacity "
        "concentrated where it matters.",
        "  - std/mean > 0.5 + no structure (random hot/cold cells) ⇒ "
        "overfitting noise per cell.",
    ]
    with open(os.path.join(out_dir, "probe2_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    # Heatmap (H on y-axis, W on x-axis, with origin at the lower-left so
    # short-τ is at the bottom).
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("moneyness index (0=put wing → 14=call wing)")
    ax.set_ylabel("τ index (0=short → 9=long)")
    ax.set_title(f"Per-cell head ‖W_n‖_F   "
                  f"(mean={norms.mean():.3f}, std/mean="
                  f"{norms.std()/max(norms.mean(),1e-12):.2f})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe2_head_norm_heatmap.png"), dpi=120)
    plt.close(fig)

    return {"norms_mean": float(norms.mean()),
            "norms_std": float(norms.std()),
            "norms_max_over_min": float(norms.max() / max(norms.min(), 1e-12)),
            "tv_W": tv_W, "tv_H": tv_H,
            "v1_norm": v1_norm}


# ─── Probe 3: v1 vs v2-average (within-v2 shared/delta decomposition) ─────

def probe3(out_dir, model, v1_head_w):
    """v1 ↔ v2-average comparison.

    Direct cosine between v1 (d_model_v1=16) and v2's per-cell average
    (d_model_v2=8) is *undefined* — different d_model. We report this
    clearly and instead compute the *within-v2* shared/delta decomposition,
    which is what we actually need to decide whether shared+delta is the
    right next architecture.
    """
    W = model.head_weight.detach().cpu().numpy()           # [N, d, P]
    n_cells, d_v2, P = W.shape
    v2_avg = W.mean(axis=0)                                # [d, P]
    delta = W - v2_avg                                     # [N, d, P]

    avg_norm = float(np.linalg.norm(v2_avg))
    delta_norms = np.linalg.norm(delta.reshape(n_cells, -1), axis=1)
    rel = delta_norms / max(avg_norm, 1e-12)
    pct = np.percentile(rel, [50, 75, 95])

    v1_d = v1_head_w.shape[0]                              # d_model_v1

    lines = [
        "PROBE 3: v1 shared head vs v2 average head",
        "=" * 42, "",
        f"v1 head shape (transposed): {tuple(v1_head_w.shape)}  "
        f"(d_model_v1={v1_d}, P={P})",
        f"v2 mean head shape:         {v2_avg.shape}                  "
        f"(d_model_v2={d_v2}, P={P})",
        "",
        "** d_model mismatch (v1=16, v2=8): direct cosine between the two",
        "   shared heads is not defined. The within-v2 shared/delta",
        "   decomposition below answers the same architectural question",
        "   (is v2 effectively shared-baseline + per-cell deltas?). **",
        "",
        f"||v2_avg_head||_F: {avg_norm:.4f}",
        "",
        "v2 per-cell deltas relative to v2 average:",
        f"  mean ‖Δ_n‖_F / ‖v2_avg‖_F: {rel.mean():.3f}",
        f"  median:                    {pct[0]:.3f}",
        f"  p75:                       {pct[1]:.3f}",
        f"  p95:                       {pct[2]:.3f}",
        "",
        "Plots: probe3_v1_vs_v2avg.png  (v1 head [d_v1, P] vs v2_avg "
        "[d_v2, P], side-by-side with matched colour scales; not directly",
        " comparable in magnitude because of the d_model difference, but "
        "useful for shape inspection).",
        "",
        "Interpretation hint:",
        "  - mean Δ/avg < 0.3 ⇒ v2 is well-described as shared baseline + "
        "small deltas; shared+delta architecture (Move B) is natural.",
        "  - mean Δ/avg > 1.0 ⇒ v2's per-cell variation dominates the shared "
        "baseline; the per-cell capacity is doing real work.",
        "  - intermediate ⇒ both contribute; shared+delta with relatively "
        "low WD on deltas may help.",
    ]
    with open(os.path.join(out_dir, "probe3_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    # Side-by-side plot. Use independent colour scales because abs
    # magnitudes are not directly comparable (different d_model).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(v1_head_w.numpy(), aspect="auto", cmap="RdBu_r",
                          origin="lower",
                          vmin=-np.abs(v1_head_w.numpy()).max(),
                          vmax=+np.abs(v1_head_w.numpy()).max())
    axes[0].set_title(f"v1 shared head  [{v1_d}, {P}]")
    axes[0].set_xlabel("horizon p")
    axes[0].set_ylabel("d_model dim")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    vmax = float(np.abs(v2_avg).max())
    im1 = axes[1].imshow(v2_avg, aspect="auto", cmap="RdBu_r",
                          origin="lower", vmin=-vmax, vmax=+vmax)
    axes[1].set_title(f"v2 mean head  [{d_v2}, {P}]")
    axes[1].set_xlabel("horizon p")
    axes[1].set_ylabel("d_model dim")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe3_v1_vs_v2avg.png"), dpi=120)
    plt.close(fig)

    return {"v2_avg_norm": avg_norm,
            "mean_rel_delta": float(rel.mean()),
            "median_rel_delta": float(pct[0]),
            "p95_rel_delta": float(pct[2])}


# ─── Probe 4: per-cell head usage by regime ───────────────────────────────

def probe4(out_dir, model, adapter, data, regimes, device, batch=64):
    """For each test window, isolate the magnitude of each cell's
    contribution to the prediction: token_n @ head_weight[n] + head_bias[n].

    Captures the pre-head_norm-then-projection per-cell signal.
    """
    Xte, _ = data["test"]
    N = Xte.shape[0]
    W_grid = model.W
    H_grid = model.H
    n_cells = W_grid * H_grid
    P = model.pred_len

    # Hook the head_norm output (pre-head, post-norm tokens).
    box = {"tokens": None}
    h = model.head_norm.register_forward_hook(
        lambda mod, inp, out: box.__setitem__("tokens", out.detach().cpu())
    )
    contrib = np.empty((N, n_cells), dtype=np.float32)
    head_w = model.head_weight.detach().cpu().numpy()       # [N_cells, d, P]
    head_b = model.head_bias.detach().cpu().numpy()         # [N_cells, P]

    try:
        adapter.to(device)
        with torch.no_grad():
            for s in range(0, N, batch):
                xb = torch.from_numpy(Xte[s:s+batch]).to(device)
                _ = adapter(xb)
                tokens = box["tokens"].numpy()              # [B, N_cells, d]
                # cell_pred[b, n, p] = tokens[b, n, :] @ head_w[n, :, p]
                #                       + head_b[n, p]
                cell_pred = np.einsum("bnd,ndp->bnp", tokens, head_w) + head_b
                # Magnitude per (b, n): sum_p cell_pred^2.
                contrib[s:s+batch] = (cell_pred ** 2).sum(axis=2)
    finally:
        h.remove()
    adapter.cpu()

    # Mean per (regime, cell).
    regime_names = [r[0] for r in REGIMES]
    mean_contrib = np.full((len(regime_names), n_cells), np.nan,
                           dtype=np.float32)
    for ri, r in enumerate(regime_names):
        mask = regimes == r
        if mask.sum() == 0:
            continue
        mean_contrib[ri] = contrib[mask].mean(axis=0)

    # Cross-regime cosine similarity of the flattened maps.
    cross = np.full((len(regime_names), len(regime_names)), np.nan)
    for i in range(len(regime_names)):
        for j in range(len(regime_names)):
            vi = mean_contrib[i]; vj = mean_contrib[j]
            if np.any(np.isnan(vi)) or np.any(np.isnan(vj)):
                continue
            cross[i, j] = float(
                vi @ vj
                / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12)
            )
    off = cross.copy(); np.fill_diagonal(off, np.nan)
    mean_off = float(np.nanmean(off))

    # Top-5 cells per regime.
    top5 = {}
    for ri, r in enumerate(regime_names):
        if np.any(np.isnan(mean_contrib[ri])):
            continue
        idx = np.argsort(-mean_contrib[ri])[:5]
        coords = [(int(i // H_grid), int(i % H_grid)) for i in idx]
        top5[r] = coords

    lines = [
        "PROBE 4: Per-cell head usage by regime",
        "=" * 40, "",
        "For each test window, isolate each cell's contribution after",
        "head_norm:  cell_pred[b, n, p] = tokens[b, n, :] @ head_weight[n]",
        "                                  + head_bias[n].  Magnitude per",
        "(window, cell) is Σ_p cell_pred[b, n, p]².  Averaged within each",
        "regime to give a [4, 15, 10] grid map.",
        "",
        f"Test windows: {N}.  Cells: {n_cells}.",
        "",
        "Top-5 cells (w_index, h_index) by mean contribution magnitude:",
    ]
    for r in regime_names:
        if r in top5:
            cells = ", ".join(f"({w}, {h})" for w, h in top5[r])
            lines.append(f"  {r:<18}: [{cells}]")
        else:
            lines.append(f"  {r:<18}: (no windows)")
    lines += [
        "",
        "Cross-regime spatial correlation (cos of flattened contribution maps):",
    ]
    for i, ri in enumerate(regime_names):
        for j, rj in enumerate(regime_names):
            if i < j and not np.isnan(cross[i, j]):
                lines.append(f"  {ri:<14} ↔ {rj:<14}: {cross[i, j]:+.3f}")
    lines += [
        f"",
        f"Mean cross-regime cosine (off-diagonal): {mean_off:+.3f}",
        "",
        "Plots: probe4_cell_usage_by_regime.png",
        "",
        "Interpretation hint:",
        "  - mean cross-regime cos > 0.9 ⇒ per-cell head usage is regime-",
        "    invariant. Per-cell capacity has fit one average-case pattern.",
        "  - mean cross-regime cos < 0.5 ⇒ different cells matter in",
        "    different regimes — regime-conditional usage; per-cell capacity",
        "    doing genuinely useful work.",
        "  - intermediate ⇒ partial regime-conditionality.",
    ]
    with open(os.path.join(out_dir, "probe4_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    # Plot 4-panel heatmap.
    vmax = float(np.nanmax(mean_contrib))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ri, r in enumerate(regime_names):
        ax = axes[ri // 2, ri % 2]
        if np.any(np.isnan(mean_contrib[ri])):
            ax.set_visible(False)
            continue
        grid_map = mean_contrib[ri].reshape(W_grid, H_grid)
        im = ax.imshow(grid_map.T, origin="lower", aspect="auto",
                       cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{r} (n={int((regimes == r).sum())})")
        ax.set_xlabel("moneyness idx")
        ax.set_ylabel("τ idx")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Per-cell head contribution magnitude by regime")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe4_cell_usage_by_regime.png"),
                dpi=120)
    plt.close(fig)

    return {"mean_cross_regime_cos": mean_off,
            "top5_per_regime": top5}


# ─── Summary ──────────────────────────────────────────────────────────────

def write_summary(out_dir, p1, p2, p3, p4):
    # Pick the architectural verdict from the probe outputs.
    heads_collapsed = p1["mean_off_cos"] > 0.9 and p1["eff_rank"] < 3.0
    heads_diverse = p1["mean_off_cos"] < 0.5 and p1["eff_rank"] > 20
    norms_uniform = p2["norms_std"] / max(p2["norms_mean"], 1e-12) < 0.2
    norms_structured = (p2["norms_std"] / max(p2["norms_mean"], 1e-12) > 0.5
                        and (p2["tv_W"] > 0 or p2["tv_H"] > 0))
    shared_plus_small_delta = p3["mean_rel_delta"] < 0.3
    shared_plus_big_delta = p3["mean_rel_delta"] > 1.0
    regime_invariant = p4["mean_cross_regime_cos"] > 0.9
    regime_conditional = p4["mean_cross_regime_cos"] < 0.5

    if heads_collapsed:
        verdict = (
            "**Heads are essentially identical.** Probe 1 reports mean off-"
            "diagonal cosine "
            f"{p1['mean_off_cos']:.3f} and effective rank {p1['eff_rank']:.2f}. "
            "v2 is functionally equivalent to v1 with ~27 k redundant "
            "parameters. The architectural recommendation is **drop v2 or "
            "apply heavy uniform weight decay** to recover v1's parameter "
            "efficiency without the overfitting tail."
        )
    elif shared_plus_small_delta:
        verdict = (
            "**Heads are diverse but average to a clear shared baseline.** "
            f"Probe 3: mean per-cell delta is {p3['mean_rel_delta']:.3f} of "
            "the v2 average head's norm; "
            f"Probe 1 effective rank {p1['eff_rank']:.2f}. "
            "v2 has discovered something close to a shared+delta decomposition "
            "by gradient descent. **Shared+delta architecture is the natural "
            "fix**: keep one shared head, add small per-cell deltas, apply "
            "higher WD to the delta."
        )
    elif heads_diverse and norms_structured and regime_conditional:
        verdict = (
            "**Heads are diverse and structured spatially with "
            "regime-conditional usage.** Probe 1 mean off-diag cosine "
            f"{p1['mean_off_cos']:.3f}, eff. rank {p1['eff_rank']:.2f}; "
            f"Probe 2 std/mean {p2['norms_std']/max(p2['norms_mean'],1e-12):.2f} "
            f"with TV_W={p2['tv_W']:.4f}, TV_H={p2['tv_H']:.4f}; "
            f"Probe 4 mean cross-regime cosine {p4['mean_cross_regime_cos']:+.3f}. "
            "**Per-cell capacity is being used meaningfully.** The Reflation "
            "calm regression isn't from chaotic overfitting but from the "
            "per-cell heads actively making mistakes there. Targeted fix: "
            "**increase dropout, not weight decay** (dropout reduces what "
            "the heads can express without removing the per-cell structure)."
        )
    elif heads_diverse and not norms_structured and regime_invariant:
        verdict = (
            "**Heads are diverse but unstructured — looks like overfitting "
            "noise per cell.** Probe 1 mean cos "
            f"{p1['mean_off_cos']:.3f}, eff. rank {p1['eff_rank']:.2f}; "
            f"Probe 2 std/mean "
            f"{p2['norms_std']/max(p2['norms_mean'],1e-12):.2f} "
            f"(no spatial structure: TV_W={p2['tv_W']:.4f}, TV_H={p2['tv_H']:.4f}); "
            f"Probe 4 mean cross-regime cos {p4['mean_cross_regime_cos']:+.3f}. "
            "**Uniform WD bump (Move A) is the right fix.**"
        )
    else:
        # Mixed / borderline.
        verdict = (
            "**Mixed picture.** The probes don't fall cleanly into one of "
            "the four named buckets:\n\n"
            f"  - Probe 1 mean cos {p1['mean_off_cos']:.3f}, "
            f"eff. rank {p1['eff_rank']:.2f}\n"
            f"  - Probe 2 std/mean {p2['norms_std']/max(p2['norms_mean'],1e-12):.2f}, "
            f"max/min {p2['norms_max_over_min']:.2f}, "
            f"TV_W={p2['tv_W']:.4f}, TV_H={p2['tv_H']:.4f}\n"
            f"  - Probe 3 mean delta/avg {p3['mean_rel_delta']:.3f}\n"
            f"  - Probe 4 mean cross-regime cos {p4['mean_cross_regime_cos']:+.3f}\n\n"
            "Read the individual probe results to choose between "
            "shared+delta, higher dropout, and uniform WD."
        )

    md = [
        "# iTransformer v2 per-cell head diagnostic", "",
        "## Headline findings", "",
        "### Probe 1 — cross-cell head similarity",
        f"- Mean off-diagonal cosine: **{p1['mean_off_cos']:+.3f}**",
        f"- Effective rank: **{p1['eff_rank']:.2f}** out of "
        f"{min(150, 168)} possible",
        f"- Rank-1 explains {p1['ev1']*100:.1f}%, "
        f"rank-3 {p1['ev3']*100:.1f}%, rank-10 {p1['ev10']*100:.1f}% of variance.",
        "",
        "### Probe 2 — spatial structure of head norms",
        f"- ‖W_n‖_F mean **{p2['norms_mean']:.3f}**, "
        f"std/mean **{p2['norms_std']/max(p2['norms_mean'],1e-12):.3f}**, "
        f"max/min **{p2['norms_max_over_min']:.2f}**.",
        f"- Total variation along W (moneyness): {p2['tv_W']:.4f}; "
        f"along H (τ): {p2['tv_H']:.4f}.",
        f"- v1 shared-head norm reference (different d_model): {p2['v1_norm']:.3f}.",
        "",
        "### Probe 3 — within-v2 shared/delta decomposition",
        f"- ‖v2_avg‖_F = {p3['v2_avg_norm']:.4f}",
        f"- Mean ‖Δ_n‖_F / ‖v2_avg‖_F: **{p3['mean_rel_delta']:.3f}**  "
        f"(median {p3['median_rel_delta']:.3f}, p95 {p3['p95_rel_delta']:.3f}).",
        "- v1 ↔ v2-average direct cosine: undefined (d_model mismatch: "
        "v1=16, v2=8); the within-v2 shared/delta decomposition above "
        "answers the same architectural question.",
        "",
        "### Probe 4 — regime-conditional cell usage",
        f"- Mean cross-regime cosine of cell-contribution maps: "
        f"**{p4['mean_cross_regime_cos']:+.3f}**.",
        "",
        "## Architectural implication", "",
        verdict, "",
    ]
    with open(os.path.join(out_dir, "diagnostic_summary.md"), "w") as fp:
        fp.write("\n".join(md) + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="iTransformer v2 run dir "
                         "(must contain hyperparams.json + best_model.pt)")
    ap.add_argument("--v1_ckpt",
                    default="iTransformer/63_21/2026-05-17T17-59-05Z/"
                            "best_model.pt",
                    help="v1 (shared-head) checkpoint for Probe 2 / Probe 3.")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")
    hp, metrics, adapter, model = load_v2(run_dir)
    csv_path = os.path.join(ROOT, "SPX_surfaces.csv")
    data = load_dataset(
        csv_path, train_frac=0.7, val_frac=0.1, lookback=LOOKBACK,
        pred_len=hp["pred_len"], data_end=hp.get("data_end"),
    )
    n_test = data["test"][0].shape[0]
    print(f"  test windows: {n_test}")
    if n_test != metrics["n_test"]:
        raise SystemExit(
            f"n_test mismatch: data={n_test} vs metrics={metrics['n_test']}"
        )

    preds, recomputed = recompute_test_mse(adapter, data, device)
    drift = abs(recomputed - metrics["test_mse"])
    print(f"  test_mse: orig={metrics['test_mse']:.6f}  "
          f"recomputed={recomputed:.6f}  drift={drift:.2e}")
    if drift > 1e-3:
        raise SystemExit(f"checkpoint drift too large: {drift}")

    # Regime mask.
    end_dates, _, _ = test_target_dates(
        data, hp["pred_len"], csv_path, hp.get("data_end")
    )
    regimes = assign_regime(end_dates)
    regime_names = [r[0] for r in REGIMES]
    counts = {r: int((regimes == r).sum()) for r in regime_names}
    print(f"  regime counts: {counts}")
    expected = {"COVID": 254, "Reflation calm": 252,
                "Bear 2022": 251, "Normalisation": 250}
    if counts != expected:
        raise SystemExit(
            f"regime counts mismatch: got {counts}, expected {expected}"
        )

    # v1 head for probes 2/3.
    v1_head_w, _ = load_v1_head(args.v1_ckpt)
    print(f"  v1 head shape (transposed): {tuple(v1_head_w.shape)}; "
          f"v2 head_weight: {tuple(model.head_weight.shape)}")

    print("running probe 1 …"); p1 = probe1(out_dir, model)
    print("running probe 2 …"); p2 = probe2(out_dir, model, v1_head_w)
    print("running probe 3 …"); p3 = probe3(out_dir, model, v1_head_w)
    print("running probe 4 …"); p4 = probe4(out_dir, model, adapter, data,
                                            regimes, device)
    write_summary(out_dir, p1, p2, p3, p4)
    print(f"  diagnostics → {os.path.relpath(out_dir, ROOT)}")


if __name__ == "__main__":
    main()
