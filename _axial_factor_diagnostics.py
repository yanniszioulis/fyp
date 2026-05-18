#!/usr/bin/env python3
"""
Diagnostic probes for a trained AxialFactor checkpoint.

Usage
-----
    python _axial_factor_diagnostics.py --run_dir AxialFactor/63_21/<ts>/

Probes
------
1. Factor representation regime separability (extractor output).
2. Per-factor temporal map SVD (model.temporal.weight).
3. Per-regime factor amplitude trajectories (g = temporal output) vs.
   the loadings-pseudoinverse projection of the true future surface.
4. Spatial loading inspection (model.reconstructor.spatial_loadings).

Each probe is independent and writes to <run_dir>/diagnostics/. No
retraining, no model changes — only forward-pass hooks capture
intermediates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from train import (                              # noqa: E402
    LOOKBACK, AxialFactor, _AxialFactorAdapter,
    load_dataset, pick_device,
)
from eval_full import (                          # noqa: E402
    REGIMES, assign_regime, test_target_dates,
)


# ─── Loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: str):
    """Load model + checkpoint + dataset for a training-run directory."""
    with open(os.path.join(run_dir, "hyperparams.json")) as f:
        hp = json.load(f)
    ckpt = os.path.join(run_dir, "best_model.pt")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"Missing checkpoint: {ckpt}")
    with open(os.path.join(run_dir, "metrics_test.json")) as f:
        original_metrics = json.load(f)

    csv_path = os.path.join(ROOT, "SPX_surfaces.csv")
    # train.py defaults at the time the original run was produced.
    data = load_dataset(
        csv_path,
        train_frac=0.7, val_frac=0.1,
        lookback=LOOKBACK, pred_len=hp["pred_len"],
        data_end=hp.get("data_end"),
    )
    grid = data["grid"]
    model = AxialFactor(**hp["model_kwargs"])
    adapter = _AxialFactorAdapter(model, grid.n_tau, grid.n_money)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state)
    adapter.eval()
    return hp, data, adapter, model, original_metrics


# ─── Capture intermediates ────────────────────────────────────────────────

def capture_intermediates(adapter, model, data, device, batch=64):
    """Forward over all test windows; capture factor_tokens, g, preds."""
    Xte, Yte = data["test"]
    N, L, C = Xte.shape
    F = model.n_factors
    D = model.d_model
    P = model.pred_len

    factor_tokens = np.empty((N, F, D), dtype=np.float32)
    g_pred = np.empty((N, F, P), dtype=np.float32)
    preds = np.empty_like(Yte)

    box = {"ft": None, "g": None}
    h1 = model.extractor.register_forward_hook(
        lambda m, inp, out: box.__setitem__("ft", out.detach().cpu())
    )
    h2 = model.temporal.register_forward_hook(
        lambda m, inp, out: box.__setitem__("g", out.detach().cpu())
    )
    try:
        adapter.to(device)
        with torch.no_grad():
            for s in range(0, N, batch):
                xb = torch.from_numpy(Xte[s : s + batch]).to(device)
                yp = adapter(xb)
                preds[s : s + batch] = yp.cpu().numpy()
                factor_tokens[s : s + batch] = box["ft"].numpy()
                g_pred[s : s + batch] = box["g"].numpy()
    finally:
        h1.remove()
        h2.remove()
    adapter.cpu()

    test_mse = float(((preds - Yte) ** 2).mean())
    return factor_tokens, g_pred, preds, test_mse


# ─── Regime labels ────────────────────────────────────────────────────────

def get_regimes(data, hp):
    csv_path = os.path.join(ROOT, "SPX_surfaces.csv")
    end_dates, _, _ = test_target_dates(
        data, hp["pred_len"], csv_path, hp.get("data_end")
    )
    return end_dates, assign_regime(end_dates)


# ─── Probe 1: regime separability of factor tokens ────────────────────────

def probe1(out_dir, factor_tokens, regimes):
    """Per-factor between/within variance + linear regime classifier."""
    N, F, D = factor_tokens.shape
    regime_names = [r[0] for r in REGIMES]
    counts = {r: int((regimes == r).sum()) for r in regime_names}

    lines = [
        "PROBE 1: Factor representation regime separability",
        "=" * 51, "",
        "Method:",
        "  Variance ratio = (var across regime means, summed over d_model)",
        "                 / (mean-over-regimes of within-regime var,",
        "                    summed over d_model).",
        "  Classifier    = LogReg (multinomial), 5-fold CV, on standardised features.",
        "",
        f"Test windows: {N}.  Regimes: {counts}.",
        "",
        "Per-factor variance ratio (between / within):",
    ]
    ratios = []
    for f in range(F):
        ft = factor_tokens[:, f, :]                          # [N, D]
        within_vars, means = [], []
        for r in regime_names:
            mask = regimes == r
            if mask.sum() < 2:
                continue
            sub = ft[mask]
            within_vars.append(sub.var(axis=0, ddof=0).sum())
            means.append(sub.mean(axis=0))
        mw = float(np.mean(within_vars))
        between = float(np.stack(means).var(axis=0, ddof=0).sum())
        ratio = between / max(mw, 1e-12)
        ratios.append(ratio)
        lines.append(f"  factor {f}: {ratio:.3f}")
    lines.append("")

    # Classifier (per factor + concat).
    valid = np.isin(regimes, regime_names)
    y_text = regimes[valid]
    classes, y_enc = np.unique(y_text, return_inverse=True)
    baseline = float(np.bincount(y_enc).max() / len(y_enc))
    lines.append("Per-factor regime classification accuracy "
                 "(LogReg, 5-fold CV):")
    lines.append(f"  baseline (most-common regime): {baseline*100:.1f}%")
    accs = []
    for f in range(F):
        Xf = factor_tokens[valid, f, :]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=500))
        acc = float(cross_val_score(clf, Xf, y_enc, cv=5).mean())
        accs.append(acc)
        lines.append(f"  factor {f}: {acc*100:.1f}%")
    Xall = factor_tokens[valid].reshape(valid.sum(), -1)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=500))
    all_acc = float(cross_val_score(clf, Xall, y_enc, cv=5).mean())
    lines.append(f"  all factors concat: {all_acc*100:.1f}%")
    lines.append("")

    # 2-D embedding.
    try:
        import umap                                           # noqa: F401
        reducer = umap.UMAP(n_components=2, random_state=0)
        emb = reducer.fit_transform(Xall)
        red_name = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=0).fit_transform(Xall)
        red_name = "t-SNE (perplexity 30)"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in regime_names:
        m = y_text == r
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.6, label=r)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.set_title(f"Factor tokens by regime ({red_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe1_factor_umap.png"), dpi=120)
    plt.close(fig)
    lines.append(f"2-D embedding written to probe1_factor_umap.png "
                 f"(reduction: {red_name}).")

    lines += [
        "",
        "Interpretation hint:",
        "  - Ratios >> 1 AND accuracy >> baseline ⇒ factors carry regime info ⇒",
        "    temporal map is the bottleneck.",
        "  - Ratios ≈ 1 AND accuracy ≈ baseline ⇒ extractor collapses regime info ⇒",
        "    extractor is the bottleneck.",
    ]
    with open(os.path.join(out_dir, "probe1_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    return {"ratios": ratios, "accs": accs, "all_acc": all_acc,
            "baseline": baseline, "reduction": red_name}


# ─── Probe 1.5: factor_queries + stacked-loadings SVD ─────────────────────

def probe1_5(out_dir, model):
    """SVD on the raw parameter tensors that the probes 1 + 4 caveats
    point at — does the collapse already live in the query vectors and
    in the loading rows, or does it emerge through the pipeline?

    Looks at three things:
      - factor_queries.shape == [F, d_model].  SVD effective rank.
      - spatial_loadings stacked as [F, W*H].   SVD effective rank.
      - Pairwise cosine sim of the F query vectors.
    """
    fq = model.extractor.factor_queries.detach().cpu().numpy()   # [F, D]
    sl = model.reconstructor.spatial_loadings.detach().cpu().numpy()
    F = fq.shape[0]
    sl_flat = sl.reshape(F, -1)                                   # [F, W*H]

    def _eff_rank_and_ev(M):
        s = np.linalg.svd(M, compute_uv=False)
        eff = float(s.sum() ** 2 / max((s * s).sum(), 1e-12))
        ev = (s * s) / max((s * s).sum(), 1e-12)
        cum = np.cumsum(ev)
        return s, eff, cum

    s_q, eff_q, cum_q = _eff_rank_and_ev(fq)
    s_l, eff_l, cum_l = _eff_rank_and_ev(sl_flat)

    # Pairwise cosine similarity of query rows.
    q_norm = fq / (np.linalg.norm(fq, axis=1, keepdims=True) + 1e-12)
    cos_q = q_norm @ q_norm.T

    # Same for stacked loadings (we computed this in probe 4 too, but
    # repeat here so probe 1.5 is self-contained).
    l_norm = sl_flat / (np.linalg.norm(sl_flat, axis=1, keepdims=True) + 1e-12)
    cos_l = l_norm @ l_norm.T

    # Verdict logic.
    off_q = cos_q.copy(); np.fill_diagonal(off_q, 0.0)
    off_l = cos_l.copy(); np.fill_diagonal(off_l, 0.0)
    max_abs_off_q = float(np.max(np.abs(off_q)))
    max_abs_off_l = float(np.max(np.abs(off_l)))
    queries_collapsed = (eff_q < 1.5) and (max_abs_off_q > 0.7)
    loadings_collapsed = (eff_l < 1.5) and (max_abs_off_l > 0.7)

    if queries_collapsed:
        verdict = ("Collapse at the query level — the F query vectors are "
                   "near-collinear and the queries' SVD effective rank is "
                   "~1. Family 1 fix is on-target: an orthogonal init (or "
                   "an orthogonality penalty during training) for "
                   "factor_queries should be sufficient to break the "
                   "symmetry, since everything downstream is just feeding "
                   "off the same query direction.")
    elif loadings_collapsed and not queries_collapsed:
        verdict = ("Queries are diverse but the spatial loadings have "
                   "collapsed. The compression is happening *through* the "
                   "axial-attention + temporal-map + loading pipeline, "
                   "not at the input parameters. Family 2 / 3 fixes "
                   "(orthogonality penalty on loadings, attention "
                   "regularisation, or a different reconstruction prior) "
                   "are needed; reinitialising queries alone won't help.")
    elif queries_collapsed and loadings_collapsed:
        verdict = ("Both query vectors and stacked loadings have "
                   "collapsed. Family 1 (orthogonal query init) is "
                   "necessary but may not be sufficient — if the pipeline "
                   "tends to recollapse during training, an orthogonality "
                   "penalty on the loadings is also worth trying.")
    else:
        verdict = ("Neither queries nor stacked loadings show a clean "
                   "rank-1 collapse at the parameter level. The "
                   "near-collinearity reported in probe 4 may be "
                   "emerging from the attention dynamics on the test set "
                   "rather than being baked into the parameters. Family "
                   "2 / 3 fixes are indicated.")

    lines = [
        "PROBE 1.5: factor_queries + stacked-loadings SVD",
        "=" * 49, "",
        "Method:",
        "  - factor_queries (model.extractor.factor_queries) is [F, d_model].",
        "    SVD → singular values, effective rank ((Σs)²/Σs²),",
        "    cumulative explained variance.",
        "  - spatial_loadings stacked as [F, W*H]: same SVD treatment.",
        "  - Pairwise cosine similarity of the F query rows (and, for",
        "    cross-check, the F stacked-loading rows).",
        "",
        f"factor_queries shape: {fq.shape}",
        "  singular values:        " + ", ".join(f"{v:.4f}" for v in s_q),
        f"  effective rank:        {eff_q:.3f}  (out of {len(s_q)} possible)",
        "  cumulative ev:          "
        + ", ".join(f"k={i+1}:{cum_q[i]*100:.1f}%" for i in range(len(cum_q))),
        "",
        "  pairwise cosine similarity:",
    ]
    for row in cos_q:
        lines.append("    " + "  ".join(f"{x:+.3f}" for x in row))
    lines += [
        f"  max |off-diag|:        {max_abs_off_q:.3f}",
        "",
        f"stacked spatial_loadings shape: {sl_flat.shape}",
        "  singular values:        " + ", ".join(f"{v:.4f}" for v in s_l),
        f"  effective rank:        {eff_l:.3f}  (out of {len(s_l)} possible)",
        "  cumulative ev:          "
        + ", ".join(f"k={i+1}:{cum_l[i]*100:.1f}%" for i in range(len(cum_l))),
        "",
        "  pairwise cosine similarity:",
    ]
    for row in cos_l:
        lines.append("    " + "  ".join(f"{x:+.3f}" for x in row))
    lines += [
        f"  max |off-diag|:        {max_abs_off_l:.3f}",
        "",
        "Verdict:",
        "  " + verdict,
    ]
    with open(os.path.join(out_dir, "probe1_5_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    # Plots: 4×4 cosine-similarity heatmaps for both queries and loadings.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mat, title in (
        (axes[0], cos_q, "factor_queries cos-sim"),
        (axes[1], cos_l, "stacked loadings cos-sim"),
    ):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(F):
            for j in range(F):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        color="black", fontsize=9)
        ax.set_xticks(range(F)); ax.set_yticks(range(F))
        ax.set_xticklabels([f"f{i}" for i in range(F)])
        ax.set_yticklabels([f"f{i}" for i in range(F)])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe1_5_query_similarity.png"), dpi=120)
    plt.close(fig)

    # Singular-value plots side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, s, title in (
        (axes[0], s_q, "factor_queries singular values"),
        (axes[1], s_l, "stacked loadings singular values"),
    ):
        ax.semilogy(np.arange(1, len(s) + 1), s, "o-")
        ax.set_xlabel("k"); ax.set_ylabel("σ_k (log)")
        ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe1_5_singular_values.png"), dpi=120)
    plt.close(fig)

    return {
        "eff_rank_queries": eff_q,
        "eff_rank_loadings": eff_l,
        "cos_q": cos_q.tolist(),
        "cos_l": cos_l.tolist(),
        "queries_collapsed": queries_collapsed,
        "loadings_collapsed": loadings_collapsed,
        "verdict": verdict,
    }


# ─── Probe 2: temporal map SVD ────────────────────────────────────────────

def probe2(out_dir, model):
    """Per-factor temporal dynamics.

    Branches on the module type:
      - `_PerFactorTemporal` (free [F, d_model, P] map): SVD per factor.
      - `_PerFactorAR`       (AR(1) per factor):         ρ_f, μ_f and
                                                         cross-factor
                                                         horizon-shape
                                                         similarity.
    Returns a dict with a consistent shape: `eff_ranks` (list), `sim`
    (FxF matrix), `mode` ('linear' or 'ar'), plus mode-specific extras.
    """
    if hasattr(model.temporal, "rho_raw"):
        return _probe2_ar(out_dir, model)
    return _probe2_linear(out_dir, model)


def _probe2_linear(out_dir, model):
    W = model.temporal.weight.detach().cpu().numpy()         # [F, d_model, P]
    F, D, P = W.shape

    eff_ranks, ev_table, top1_dirs = [], [], []
    sv_per_factor = []
    for f in range(F):
        U, S, Vt = np.linalg.svd(W[f], full_matrices=False)  # S: [min(D,P)]
        sv_per_factor.append(S)
        # Participation ratio.
        eff = float(S.sum() ** 2 / np.maximum((S * S).sum(), 1e-12))
        eff_ranks.append(eff)
        cum = np.cumsum(S * S) / max((S * S).sum(), 1e-12)
        ev_table.append([float(cum[0]), float(cum[1]) if len(cum) > 1 else 1.0,
                         float(cum[2]) if len(cum) > 2 else 1.0])
        top1_dirs.append(Vt[0])                              # [P]

    top1 = np.stack(top1_dirs)                               # [F, P]
    top1 = top1 / (np.linalg.norm(top1, axis=1, keepdims=True) + 1e-12)
    sim = top1 @ top1.T                                      # [F, F]

    lines = [
        "PROBE 2: Per-factor temporal map analysis",
        "=" * 41, "",
        f"Weight tensor: model.temporal.weight, shape [F={F}, d_model={D}, P={P}].",
        "Per-factor SVD of W[f] = U diag(S) V^T (S ∈ R^{min(D,P)}).",
        "",
        "Per-factor effective rank (participation ratio  (Σs)² / Σs²):",
    ]
    for f, e in enumerate(eff_ranks):
        lines.append(f"  factor {f}: {e:.2f}  (out of {min(D, P)} possible)")
    lines += ["",
              "Per-factor rank-k explained variance:"]
    for f, ev in enumerate(ev_table):
        lines.append(f"  factor {f}: rank-1={ev[0]*100:5.1f}%  "
                     f"rank-2={ev[1]*100:5.1f}%  rank-3={ev[2]*100:5.1f}%")
    lines += ["",
              "Cross-factor top-trajectory cosine similarity matrix:"]
    for row in sim:
        lines.append("  " + "  ".join(f"{x:+.3f}" for x in row))
    lines += ["",
              "Interpretation hint:",
              "  - Low effective rank (<3) + high |off-diag sim| (>0.7) ⇒ "
              "map collapsed; capacity wasted.",
              "  - High effective rank (>5) + mixed-sign or near-zero off-diag ⇒ "
              "rich per-factor dynamics."]
    with open(os.path.join(out_dir, "probe2_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    # Singular-value plot (log y).
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=True)
    for f, ax in enumerate(axes.flat):
        ax.semilogy(np.arange(1, len(sv_per_factor[f]) + 1),
                    sv_per_factor[f], "o-")
        ax.set_title(f"factor {f}")
        ax.set_xlabel("k")
        if f % 2 == 0:
            ax.set_ylabel("singular value (log)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-factor temporal-map singular values")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe2_singular_values.png"), dpi=120)
    plt.close(fig)

    # Top-3 horizon trajectories per factor.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    horizons = np.arange(1, P + 1)
    for f, ax in enumerate(axes.flat):
        _, _, Vt = np.linalg.svd(W[f], full_matrices=False)
        for k in range(3):
            ax.plot(horizons, Vt[k], label=f"v{k+1}")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(f"factor {f}")
        ax.set_xlabel("horizon h")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Top-3 right singular vectors (horizon trajectories)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe2_horizon_shapes.png"), dpi=120)
    plt.close(fig)

    # Cross-factor top-1 similarity matrix.
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(F):
        for j in range(F):
            ax.text(j, i, f"{sim[i, j]:+.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_xticks(range(F)); ax.set_yticks(range(F))
    ax.set_xticklabels([f"f{i}" for i in range(F)])
    ax.set_yticklabels([f"f{i}" for i in range(F)])
    ax.set_title("Top-1 trajectory cosine similarity")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe2_factor_similarity.png"), dpi=120)
    plt.close(fig)
    return {"eff_ranks": eff_ranks, "ev_table": ev_table, "sim": sim.tolist(),
            "mode": "linear"}


def _probe2_ar(out_dir, model):
    """Per-factor AR(1) dynamics inspection.

    Reads ρ_f and μ_f directly, builds the implicit horizon-shape
    ρ_f^h for h=1..P, and computes the cross-factor cosine-similarity
    matrix of those horizon shapes.
    """
    ar = model.temporal
    F = ar.n_factors
    P = ar.pred_len
    rho_vec = ar.rho().detach().cpu().numpy()                # [F]
    mu_vec  = ar.mu.detach().cpu().numpy()                   # [F]
    horizons = np.arange(1, P + 1)

    # Per-factor horizon shapes ρ_f^h (NOT normalised; cosine sim will
    # rescale anyway). Use absolute value to avoid 0^0 numerics for
    # negative ρ at h=0 (we start at h=1 so no issue, but defensive).
    rho_pow = np.power(rho_vec[:, None], horizons[None, :])  # [F, P]

    # Cross-factor cosine similarity of horizon shapes.
    norms = np.linalg.norm(rho_pow, axis=1, keepdims=True) + 1e-12
    unit = rho_pow / norms
    sim = unit @ unit.T                                      # [F, F]

    # "Effective rank" of the [F, P] matrix of horizon shapes
    # (participation ratio of its singular values). Single scalar here,
    # returned as a length-1 list to keep a consistent type with the
    # linear branch.
    sv = np.linalg.svd(rho_pow, compute_uv=False)
    eff = float(sv.sum() ** 2 / max((sv * sv).sum(), 1e-12))

    lines = [
        "PROBE 2 (AR): Per-factor AR(1) dynamics",
        "=" * 39, "",
        "Method:",
        "  - Read model.temporal.rho() and model.temporal.mu.",
        "  - Build per-factor implicit horizon shape ρ_f^h, h=1..P.",
        "  - Cross-factor cosine similarity of those shapes.",
        "  - Participation-ratio rank of the [F, P] shape matrix.",
        "",
        f"Per-factor AR persistences ρ_f: [{', '.join(f'{r:+.4f}' for r in rho_vec)}]",
        f"Per-factor AR long-run means μ_f: [{', '.join(f'{m:+.4f}' for m in mu_vec)}]",
        "",
        "Implicit horizon shape  ρ_f^h  (h = 1, 5, 10, 15, 21):",
    ]
    snapshot_h = [0, 4, 9, 14, 20]
    header = ["h=" + str(h + 1).rjust(2) for h in snapshot_h]
    lines.append("  factor | " + "  ".join(f"{h:>10s}" for h in header))
    for f in range(F):
        cells = [f"{rho_pow[f, h]:+.4e}" for h in snapshot_h]
        lines.append(f"  f{f:>4d}  | " + "  ".join(f"{c:>10s}" for c in cells))
    lines += [
        "",
        f"Effective rank of [F, P] horizon-shape matrix: {eff:.3f}  "
        f"(out of {min(F, P)} possible)",
        "",
        "Cross-factor horizon-shape cosine similarity:",
    ]
    for row in sim:
        lines.append("  " + "  ".join(f"{x:+.3f}" for x in row))
    off = sim.copy(); np.fill_diagonal(off, 0.0)
    lines += [
        f"  max |off-diag|: {float(np.max(np.abs(off))):.3f}",
        "",
        "Interpretation hint:",
        "  - Well-separated ρ_f (e.g. one near 1, one near 0.5, etc.)",
        "    AND |off-diag| < 0.9 ⇒ AR fix delivered distinct per-factor",
        "    dynamics — collapse broken at the architectural level.",
        "  - ρ_f cluster at one value AND |off-diag| ≈ 1 ⇒ data does not",
        "    support multiple timescales; factor framework is the wrong",
        "    tool for this prediction problem (or AR(1) is too restrictive).",
    ]
    with open(os.path.join(out_dir, "probe2_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    # Plot 1: per-factor horizon shapes overlaid (probe2_horizon_shapes.png).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f in range(F):
        ax.plot(horizons, rho_pow[f], "o-",
                label=f"factor {f}: ρ={rho_vec[f]:+.3f}")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("horizon h")
    ax.set_ylabel(r"$\rho_f^h$  (implicit horizon shape)")
    ax.set_title("AR(1) per-factor horizon shapes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe2_horizon_shapes.png"), dpi=120)
    plt.close(fig)

    # Plot 2: ρ and μ as side-by-side bar chart
    # (probe2_singular_values.png — repurposed filename so the
    # diagnostics dir keeps a uniform set across model variants).
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(range(F), rho_vec, color="tab:blue")
    axes[0].set_xticks(range(F))
    axes[0].set_xticklabels([f"f{i}" for i in range(F)])
    axes[0].set_ylim(-1, 1); axes[0].axhline(0, color="0.6", lw=0.6)
    axes[0].set_title("AR persistence ρ_f")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(range(F), mu_vec, color="tab:orange")
    axes[1].set_xticks(range(F))
    axes[1].set_xticklabels([f"f{i}" for i in range(F)])
    axes[1].axhline(0, color="0.6", lw=0.6)
    axes[1].set_title("AR long-run mean μ_f")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe2_singular_values.png"), dpi=120)
    plt.close(fig)

    # Plot 3: cross-factor cosine similarity (probe2_factor_similarity.png).
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(F):
        for j in range(F):
            ax.text(j, i, f"{sim[i, j]:+.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_xticks(range(F)); ax.set_yticks(range(F))
    ax.set_xticklabels([f"f{i}" for i in range(F)])
    ax.set_yticklabels([f"f{i}" for i in range(F)])
    ax.set_title("AR horizon-shape cosine similarity")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe2_factor_similarity.png"), dpi=120)
    plt.close(fig)

    return {
        "mode": "ar",
        "rho": rho_vec.tolist(),
        "mu": mu_vec.tolist(),
        "eff_ranks": [eff],          # length-1 list for write_summary compat
        "sim": sim.tolist(),
        # For write_summary's max(p2['eff_ranks']) < 2.0 collapse check,
        # this single number is the AR analogue: rank of the horizon-
        # shape matrix.
        "ev_table": [[float(rho_pow[f, 0] ** 2 /
                            max((rho_pow[f] ** 2).sum(), 1e-12)),
                      0.0, 0.0]
                     for f in range(F)],
    }


# ─── Probe 3: per-regime factor trajectories ──────────────────────────────

def probe3(out_dir, model, data, g_pred, regimes):
    """Compare predicted factor amplitudes vs loadings-projected true ones."""
    Xte, Yte = data["test"]
    grid = data["grid"]
    n_tau, n_money = grid.n_tau, grid.n_money               # H, W
    N, L, C = Xte.shape
    P = model.pred_len
    F = model.n_factors

    loadings = model.reconstructor.spatial_loadings.detach().cpu().numpy()  # [F, W, H]
    A = loadings.reshape(F, -1).T                            # [W*H, F]
    A_pinv = np.linalg.pinv(A)                               # [F, W*H]

    # Reshape Xte/Yte from [N, L|P, C=H*W] to [N, L|P, W, H] using the
    # same convention the adapter uses: reshape -> [N, ., H, W] then
    # permute to [N, ., W, H].
    def to_WH(arr):
        N_, T_, C_ = arr.shape
        return arr.reshape(N_, T_, n_tau, n_money).transpose(0, 1, 3, 2)
    X_WH = to_WH(Xte)                                        # [N, L, W, H]
    Y_WH = to_WH(Yte)                                        # [N, P, W, H]

    mu = X_WH.mean(axis=1, keepdims=True)                    # [N, 1, W, H]
    Y_norm = Y_WH - mu                                       # [N, P, W, H]

    # Loadings projection of the truth at every (n, p).
    flat = Y_norm.reshape(N * P, -1)                         # [N*P, W*H]
    g_true_flat = flat @ A_pinv.T                            # [N*P, F]
    g_true = g_true_flat.reshape(N, P, F).transpose(0, 2, 1) # [N, F, P]

    regime_names = [r[0] for r in REGIMES]
    horizons = np.arange(1, P + 1)

    # Per (regime, factor) trajectory MSE.
    tmse = np.full((len(regime_names), F), np.nan)
    pred_means, true_means = {}, {}
    pred_stds, true_stds = {}, {}
    for ri, r in enumerate(regime_names):
        mask = regimes == r
        if mask.sum() < 2:
            continue
        gp = g_pred[mask]; gt = g_true[mask]                # [n, F, P]
        for f in range(F):
            tmse[ri, f] = float(((gp[:, f] - gt[:, f]) ** 2).mean())
        pred_means[r] = gp.mean(axis=0)                      # [F, P]
        true_means[r] = gt.mean(axis=0)
        pred_stds[r]  = gp.std(axis=0)
        true_stds[r]  = gt.std(axis=0)

    # Plot: 4 subplots (factors); solid=predicted, dashed=true; one
    # color per regime; shaded ±1 std for predicted.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for f, ax in enumerate(axes.flat):
        for ri, r in enumerate(regime_names):
            if r not in pred_means:
                continue
            c = color_cycle[ri % len(color_cycle)]
            mp = pred_means[r][f]; sp = pred_stds[r][f]
            mt = true_means[r][f]
            ax.fill_between(horizons, mp - sp, mp + sp, color=c, alpha=0.15)
            ax.plot(horizons, mp, color=c, lw=1.8, label=f"{r} pred")
            ax.plot(horizons, mt, color=c, lw=1.2, linestyle="--",
                    label=f"{r} true")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(f"factor {f}")
        ax.set_xlabel("horizon h")
        ax.set_ylabel("factor amplitude")
        ax.grid(True, alpha=0.3)
        if f == 0:
            ax.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Per-regime factor trajectories: model g (solid) vs "
                 "loadings-projected truth (dashed)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe3_factor_trajectories.png"), dpi=120)
    plt.close(fig)

    # Per-regime / per-factor amplitude snapshot at h=1, 10, 21.
    snap_h = [0, 9, 20]
    lines = [
        "PROBE 3: Per-regime factor amplitude trajectories",
        "=" * 50, "",
        "g_pred captured from model.temporal output (forward hook).",
        "g_true obtained by least-squares projection of the de-meaned",
        "true future surface onto the spatial loadings:",
        "    g_true[b, p, :] = pinv(loadings) @ vec(Y_true_norm[b, p]).",
        "",
        "Trajectory MSE per (regime × factor):",
        "  " + " " * 18 + " | " + "  ".join(f"factor{f}" for f in range(F)),
        "  " + "-" * 18 + "-+-" + "-" * (10 * F - 2),
    ]
    for ri, r in enumerate(regime_names):
        lines.append("  " + f"{r:<18}" + " | " +
                     "  ".join(f"{tmse[ri, f]:7.4f}" for f in range(F)))
    lines += ["",
              "Mean amplitudes at h ∈ {1, 10, 21} per (regime × factor):",
              "  (rows: regime;  cols: 'pred / true' at each h)"]
    for ri, r in enumerate(regime_names):
        if r not in pred_means:
            continue
        lines.append(f"  {r}:")
        for f in range(F):
            cells = []
            for h_idx in snap_h:
                cells.append(f"h{h_idx+1:>2}: "
                             f"{pred_means[r][f, h_idx]:+.3f} / "
                             f"{true_means[r][f, h_idx]:+.3f}")
            lines.append(f"    factor {f}:  " + " | ".join(cells))
    lines += ["",
              "Interpretation hint:",
              "  - Predicted ≈ true within each regime ⇒ factor path is correct;",
              "    look elsewhere for the regime gap.",
              "  - Predicted trajectories similar across regimes but true",
              "    trajectories differ ⇒ temporal map is regime-blind ⇒ need",
              "    conditional dynamics."]
    with open(os.path.join(out_dir, "probe3_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    return {"tmse": tmse.tolist(),
            "pred_means": {k: v.tolist() for k, v in pred_means.items()},
            "true_means": {k: v.tolist() for k, v in true_means.items()}}


# ─── Probe 4: spatial loading inspection ──────────────────────────────────

def probe4(out_dir, model, data, hp):
    """Heatmaps + smoothness + separability + PCA comparison."""
    loadings = model.reconstructor.spatial_loadings.detach().cpu().numpy()  # [F, W, H]
    F, W, H = loadings.shape
    grid = data["grid"]
    money_vals, tau_vals = grid.money_vals, grid.tau_vals    # W, H

    # ─ Smoothness (total variation along each axis, mean over the other).
    tv_W = [
        float(np.mean(np.abs(np.diff(loadings[f], axis=0))))
        for f in range(F)
    ]
    tv_H = [
        float(np.mean(np.abs(np.diff(loadings[f], axis=1))))
        for f in range(F)
    ]
    # ─ Rank-1 separability per loading (W×H matrix SVD).
    sep = []
    for f in range(F):
        s = np.linalg.svd(loadings[f], compute_uv=False)
        sep.append(float(s[0] / max(s.sum(), 1e-12)))

    # ─ Inter-factor inner-product (normalised).
    flat = loadings.reshape(F, -1)
    norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    inter = norm @ norm.T

    # ─ PCA of training surfaces (last day of each train window after
    #   per-channel train-period mean subtraction → same normalisation
    #   the scaler applied minus the std step).  Use the scaled (z-score)
    #   training rows: the surfaces the model trained on live in the same
    #   space the cell-mean subtraction operates over within a window.
    train_end = data["rows"]["train_end"]
    scaled = data["scaled_log_iv"]                            # [N_all, C]
    train_rows = scaled[:train_end]                           # [N_train, C]
    # Reshape rows: C → (H, W) → (W, H) to match the model's spatial axis order.
    n_tau, n_money = grid.n_tau, grid.n_money
    surfaces = (train_rows.reshape(-1, n_tau, n_money)
                          .transpose(0, 2, 1))                # [N_train, W, H]
    flat_s = surfaces.reshape(surfaces.shape[0], -1)
    flat_s = flat_s - flat_s.mean(axis=0, keepdims=True)
    # SVD-based PCA (avoid centring twice).
    _, _, Vt = np.linalg.svd(flat_s, full_matrices=False)
    pcs = Vt[:F].reshape(F, W, H)                             # [F, W, H]

    # Best-match cosine similarity per factor vs the top-F PCs.
    pcs_flat = pcs.reshape(F, -1)
    pcs_norm = pcs_flat / (np.linalg.norm(pcs_flat, axis=1, keepdims=True) + 1e-12)
    sims = norm @ pcs_norm.T                                  # [F, F]
    best_sims, best_pc = [], []
    for f in range(F):
        k = int(np.argmax(np.abs(sims[f])))
        best_pc.append(k)
        best_sims.append(float(sims[f, k]))

    # ─ Plots.
    def _heatmap(ax, data2d, title, xticks, yticks):
        v = float(np.max(np.abs(data2d)))
        im = ax.imshow(data2d.T, origin="lower", aspect="auto",
                       cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(title)
        ax.set_xlabel("moneyness idx")
        ax.set_ylabel("τ idx")
        return im

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for f, ax in enumerate(axes.flat):
        im = _heatmap(ax, loadings[f], f"factor {f}",
                       money_vals, tau_vals)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Spatial loadings (model.reconstructor.spatial_loadings)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe4_spatial_loadings.png"), dpi=120)
    plt.close(fig)

    # Side-by-side model loadings vs top-F training PCAs.
    fig, axes = plt.subplots(F, 2, figsize=(8, 2.4 * F))
    for f in range(F):
        ax_m, ax_p = axes[f]
        im = _heatmap(ax_m, loadings[f], f"factor {f} (model)",
                       money_vals, tau_vals)
        fig.colorbar(im, ax=ax_m, shrink=0.7)
        im = _heatmap(ax_p, pcs[f], f"PC {f+1} (train surfaces)",
                       money_vals, tau_vals)
        fig.colorbar(im, ax=ax_p, shrink=0.7)
    fig.suptitle("Model loadings vs. top training PCs")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe4_loadings_vs_pca.png"), dpi=120)
    plt.close(fig)

    lines = [
        "PROBE 4: Spatial loading inspection",
        "=" * 35, "",
        f"Loadings shape: {tuple(loadings.shape)}  "
        f"(F={F} factors, W={W} moneyness, H={H} tau).",
        "Total variation: mean |Δ| across the named axis "
        "(W-axis = adjacent moneyness, H-axis = adjacent τ).",
        "",
        "Loading smoothness (total variation):",
    ]
    for f in range(F):
        lines.append(f"  factor {f}: tv_W={tv_W[f]:.4f}  tv_H={tv_H[f]:.4f}")
    lines += ["",
              "Rank-1 separability (top SV / sum SVs of the W×H loading):"]
    for f, s in enumerate(sep):
        lines.append(f"  factor {f}: {s*100:5.1f}%")
    lines += ["",
              "Inter-factor inner-product matrix (normalised loadings):"]
    for row in inter:
        lines.append("  " + "  ".join(f"{x:+.3f}" for x in row))
    lines += ["",
              "Best-match cosine similarity to top-F train-surface PCAs:"]
    for f in range(F):
        lines.append(f"  factor {f}: |sim|={abs(best_sims[f]):.3f} "
                     f"(signed {best_sims[f]:+.3f}) with PC{best_pc[f]+1}")
    lines += ["",
              "Interpretation hint:",
              "  - Smooth, rank-1-separable, near-orthogonal loadings AND high",
              "    |sim| to PCs (>0.8) ⇒ bottleneck has learned the factor",
              "    structure correctly.",
              "  - Noisy loadings, low separability, low |sim| (<0.5) ⇒ the",
              "    static reconstruction is poorly learned."]
    with open(os.path.join(out_dir, "probe4_results.txt"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    return {"tv_W": tv_W, "tv_H": tv_H, "sep": sep,
            "inter": inter.tolist(),
            "best_sims": best_sims, "best_pc": best_pc}


# ─── Summary ──────────────────────────────────────────────────────────────

def write_summary(out_dir, run_dir, hp, original_metrics, recomputed_mse,
                  p1, p1_5, p2, p3, p4):
    F = hp["model_kwargs"]["n_factors"]
    regime_names = [r[0] for r in REGIMES]

    # Compose verdict from probe outcomes. Probe 1.5 is now the
    # primary localiser (parameter-level collapse vs pipeline-level
    # collapse); probes 1, 3 record downstream consequences.
    verdicts = []
    extractor_collapsed = (
        max(p1["ratios"]) < 0.5
        and p1["all_acc"] < p1["baseline"] + 0.05
    )
    temporal_collapsed = max(p2["eff_ranks"]) < 2.0
    loadings_smooth_ok = (np.mean(p4["sep"]) >= 0.55
                          and max(abs(s) for s in p4["best_sims"]) >= 0.5)
    factor_regime_blind = False
    pm = p3["pred_means"]; tm = p3["true_means"]
    if pm and tm:
        common = sorted(set(pm) & set(tm))
        pred_arr = np.array([pm[r] for r in common])         # [R, F, P]
        true_arr = np.array([tm[r] for r in common])
        pred_var = float(pred_arr.std(axis=0).mean())
        true_var = float(true_arr.std(axis=0).mean())
        ratio_var = (true_var + 1e-9) / (pred_var + 1e-9)
        factor_regime_blind = (ratio_var > 2.0)

    if p1_5["queries_collapsed"]:
        verdicts.append(
            "**Collapse at the query level** (probe 1.5): "
            f"factor_queries has effective rank {p1_5['eff_rank_queries']:.2f}, "
            f"all pairwise cos-sim near +1. "
            "Family 1 fix (orthogonal init or orthogonality penalty on "
            "factor_queries) is the right starting point."
        )
    elif p1_5["loadings_collapsed"]:
        verdicts.append(
            "**Pipeline-level collapse, not query-level** (probe 1.5): "
            f"queries have effective rank {p1_5['eff_rank_queries']:.2f} "
            "(diverse) but stacked spatial_loadings have effective rank "
            f"{p1_5['eff_rank_loadings']:.2f} (collapsed). "
            "Family 2/3 fixes (orthogonality penalty on loadings, attention "
            "regularisation, different reconstruction prior) are indicated."
        )
    else:
        verdicts.append(
            "Parameter-level SVD (probe 1.5) does not show a clean rank-1 "
            f"collapse: queries eff. rank {p1_5['eff_rank_queries']:.2f}, "
            f"loadings eff. rank {p1_5['eff_rank_loadings']:.2f}. "
            "Any near-collinearity downstream emerges from the attention + "
            "temporal pipeline rather than being baked into the parameters."
        )

    if extractor_collapsed:
        verdicts.append(
            "Factor tokens carry no regime information (probe 1): "
            f"between/within variance ratios ≈ {np.mean(p1['ratios']):.3f}, "
            f"regime classifier {p1['all_acc']*100:.1f}% vs "
            f"{p1['baseline']*100:.1f}% baseline."
        )
    if not extractor_collapsed and factor_regime_blind:
        verdicts.append(
            "Temporal map is regime-blind even though the extractor isn't "
            "(probe 3): predicted-g spread across regimes is much smaller "
            "than true-g spread."
        )
    if factor_regime_blind and extractor_collapsed:
        verdicts.append(
            "Regime-blind predicted trajectories (probe 3) are a "
            "*downstream consequence* of the extractor collapse — the "
            "temporal map is being fed regime-blind tokens, so its output "
            "cannot vary by regime."
        )
    if temporal_collapsed:
        verdicts.append(
            f"Temporal map effective rank < 2 (probe 2: "
            f"max {max(p2['eff_ranks']):.2f}). Temporal capacity collapsed."
        )
    cross_factor_sim = np.array(p2["sim"])
    np.fill_diagonal(cross_factor_sim, 0.0)
    if float(np.max(np.abs(cross_factor_sim))) > 0.9:
        verdicts.append(
            "Cross-factor top-1 horizon trajectories nearly identical "
            f"(probe 2: max |off-diag| {float(np.max(np.abs(cross_factor_sim))):.3f}). "
            "The four factors share one horizon shape, up to sign."
        )
    if loadings_smooth_ok:
        verdicts.append(
            "Individual spatial loadings are well-learned (probe 4: "
            "smooth, rank-1-separable, high cosine to top training PC) — "
            "the static-reconstruction *prior* is doing its job; it's the "
            "factor decomposition feeding it that has collapsed."
        )
    if not verdicts:
        verdicts.append("No single bottleneck identified by these probes.")
    verdict_label = (
        "**Multiple issues** (probe 1.5 localises the root cause; the "
        "rest are mostly downstream consequences):\n\n  - "
        + "\n\n  - ".join(verdicts)
        if len(verdicts) > 1 else verdicts[0]
    )

    md = []
    md.append("# AxialFactor diagnostic probes")
    md.append("")
    md.append(f"- Run directory: `{os.path.relpath(run_dir, ROOT)}`")
    md.append(f"- Checkpoint: `best_model.pt` (loaded via "
              f"`AxialFactor(**hyperparams_kw)`)")
    md.append(f"- Original test MSE: `{original_metrics['test_mse']:.6f}`")
    md.append(f"- Recomputed test MSE (this script's forward): "
              f"`{recomputed_mse:.6f}`")
    md.append(f"- Test windows: {original_metrics['n_test']}; regimes: "
              + ", ".join(regime_names))
    md.append("")
    md.append("## Probe 1 — factor representation regime separability")
    md.append(f"- Variance ratios (between / within) per factor: "
              + ", ".join(f"f{i}={r:.2f}" for i, r in enumerate(p1['ratios'])))
    md.append(f"- LogReg accuracy per factor: "
              + ", ".join(f"f{i}={a*100:.1f}%" for i, a in enumerate(p1['accs']))
              + f"; concat={p1['all_acc']*100:.1f}%; baseline={p1['baseline']*100:.1f}%.")
    md.append(f"- 2-D embedding: see `probe1_factor_umap.png` "
              f"({p1['reduction']}).")
    md.append("")
    md.append("## Probe 1.5 — factor_queries + stacked-loadings SVD")
    md.append(f"- factor_queries effective rank: "
              f"{p1_5['eff_rank_queries']:.3f}  (out of {F} possible)")
    md.append(f"- stacked spatial_loadings effective rank: "
              f"{p1_5['eff_rank_loadings']:.3f}  (out of {F} possible)")
    cos_q = np.array(p1_5["cos_q"]); off_q = cos_q.copy(); np.fill_diagonal(off_q, 0)
    cos_l = np.array(p1_5["cos_l"]); off_l = cos_l.copy(); np.fill_diagonal(off_l, 0)
    md.append(f"- factor_queries pairwise cos-sim: "
              f"max |off-diag| = {float(np.max(np.abs(off_q))):.3f}; "
              f"min off-diag = {float(off_q.min()):+.3f}; "
              f"max off-diag = {float(off_q.max()):+.3f}.")
    md.append(f"- stacked loadings pairwise cos-sim: "
              f"max |off-diag| = {float(np.max(np.abs(off_l))):.3f}; "
              f"min off-diag = {float(off_l.min()):+.3f}; "
              f"max off-diag = {float(off_l.max()):+.3f}.")
    md.append("- Plots: `probe1_5_query_similarity.png`, "
              "`probe1_5_singular_values.png`.")
    md.append("")
    md.append("## Probe 2 — per-factor temporal map")
    md.append(f"- Effective rank per factor: "
              + ", ".join(f"f{i}={r:.2f}" for i, r in enumerate(p2['eff_ranks'])))
    md.append(f"- Rank-1 / rank-3 explained variance per factor:")
    for i, ev in enumerate(p2['ev_table']):
        md.append(f"    - f{i}: rank-1 {ev[0]*100:.1f}%, rank-3 {ev[2]*100:.1f}%")
    md.append("- Top-1 trajectory cross-factor similarity matrix written to "
              "`probe2_factor_similarity.png`; singular values in "
              "`probe2_singular_values.png`; trajectory shapes in "
              "`probe2_horizon_shapes.png`.")
    md.append("")
    md.append("## Probe 3 — per-regime factor trajectories")
    md.append("- Trajectory MSE per (regime × factor) and pred vs. true "
              "amplitudes at h ∈ {1, 10, 21} in `probe3_results.txt`.")
    md.append("- Plot: `probe3_factor_trajectories.png` "
              "(solid = predicted g, dashed = loadings-projected truth, "
              "shaded = ±1 std of predicted).")
    md.append("")
    md.append("## Probe 4 — spatial loading inspection")
    md.append(f"- Rank-1 separability per factor (top SV / sum SVs): "
              + ", ".join(f"f{i}={s*100:.0f}%" for i, s in enumerate(p4['sep'])))
    md.append(f"- Best-match |cosine| to top-{F} train PCs: "
              + ", ".join(f"f{i}={abs(s):.2f}@PC{p4['best_pc'][i]+1}"
                          for i, s in enumerate(p4['best_sims'])))
    md.append("- Plots: `probe4_spatial_loadings.png` (4 heatmaps), "
              "`probe4_loadings_vs_pca.png` (model vs PCA side-by-side).")
    md.append("")
    md.append("## Architectural implications")
    md.append("")
    md.append(verdict_label)
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"_Generated by `_axial_factor_diagnostics.py` on "
              f"{datetime.now(timezone.utc).isoformat()} (UTC)._")
    with open(os.path.join(out_dir, "summary.md"), "w") as fp:
        fp.write("\n".join(md) + "\n")


# ─── Sanity / main ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Path to the AxialFactor training-run directory "
                         "(must contain hyperparams.json + best_model.pt).")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")
    hp, data, adapter, model, original_metrics = load_run(run_dir)
    n_test = data["test"][0].shape[0]
    print(f"  test windows: {n_test}")
    if n_test != original_metrics["n_test"]:
        raise SystemExit(
            f"n_test mismatch: data={n_test} vs metrics={original_metrics['n_test']}")

    factor_tokens, g_pred, preds, recomputed_mse = capture_intermediates(
        adapter, model, data, device,
    )
    expected_F = hp["model_kwargs"]["n_factors"]
    expected_D = hp["model_kwargs"]["d_model"]
    expected_P = hp["pred_len"]
    if factor_tokens.shape != (n_test, expected_F, expected_D):
        raise SystemExit(
            f"factor_tokens shape mismatch: {factor_tokens.shape} "
            f"vs expected {(n_test, expected_F, expected_D)}")
    if g_pred.shape != (n_test, expected_F, expected_P):
        raise SystemExit(
            f"g_pred shape mismatch: {g_pred.shape} "
            f"vs expected {(n_test, expected_F, expected_P)}")
    loadings_shape = tuple(model.reconstructor.spatial_loadings.shape)
    if loadings_shape != (expected_F,
                          hp["model_kwargs"]["W"],
                          hp["model_kwargs"]["H"]):
        raise SystemExit(f"loadings shape mismatch: {loadings_shape}")

    drift = abs(recomputed_mse - original_metrics["test_mse"])
    print(f"  test_mse: orig={original_metrics['test_mse']:.6f}, "
          f"recomputed={recomputed_mse:.6f}, drift={drift:.2e}")
    if drift > 0.01:
        raise SystemExit(
            f"Recomputed test_mse drifted by {drift:.4f} (>1e-2). "
            "Something is wrong with the checkpoint or data loader — "
            "investigate before trusting probe results.")

    end_dates, regimes = get_regimes(data, hp)
    regime_names = [r[0] for r in REGIMES]
    counts = {r: int((regimes == r).sum()) for r in regime_names}
    print(f"  regime counts: {counts}")

    print("running probe 1 …");   p1   = probe1(out_dir, factor_tokens, regimes)
    print("running probe 1.5 …"); p1_5 = probe1_5(out_dir, model)
    print("running probe 2 …");   p2   = probe2(out_dir, model)
    print("running probe 3 …"); p3 = probe3(out_dir, model, data, g_pred, regimes)
    print("running probe 4 …"); p4 = probe4(out_dir, model, data, hp)

    write_summary(out_dir, run_dir, hp, original_metrics, recomputed_mse,
                  p1, p1_5, p2, p3, p4)
    print(f"  diagnostics written to {os.path.relpath(out_dir, ROOT)}")


if __name__ == "__main__":
    main()
