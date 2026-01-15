#!/usr/bin/env python3
"""
Debug a delta_transformer checkpoint on a specific window/context/horizon.
"""

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np

from forecasting.pipeline import ForecastingPipeline
from evaluation.metrics import (
    compute_all_metrics,
    compute_metrics_by_maturity,
    compute_metrics_by_moneyness,
)
from models.delta_transformer.delta_transformer_model import DeltaTransformerSurfaceModel
from models.transformer.transformer_model import _SurfaceTransformer


def _summarize_array(name: str, arr: np.ndarray):
    if arr.size == 0:
        print(f"{name}: empty")
        return
    finite = np.isfinite(arr)
    if not finite.all():
        print(f"{name}: non-finite values present ({np.size(arr) - finite.sum()})")
    arr = arr[finite]
    print(
        f"{name}: mean={arr.mean():.6f} std={arr.std():.6f} "
        f"min={arr.min():.6f} max={arr.max():.6f}"
    )


def _load_transformer_checkpoint(path: str, device: str) -> Tuple[DeltaTransformerSurfaceModel, Dict[str, Any]]:
    import torch
    import sys
    import numpy.core as npcore

    # Shim for numpy 2.x where internal module path changed
    sys.modules.setdefault("numpy._core", npcore)
    sys.modules.setdefault("numpy._core.multiarray", npcore.multiarray)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DeltaTransformerSurfaceModel(
        name=os.path.basename(path),
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
        dropout=ckpt["dropout"],
        learning_rate=ckpt.get("learning_rate", 1e-3),
        weight_decay=ckpt.get("weight_decay", 1e-4),
        batch_size=ckpt.get("batch_size", 32),
        num_epochs=ckpt.get("num_epochs", 50),
        pool=ckpt.get("pool", "last"),
        normalize=ckpt.get("normalize", True),
        normalize_mode=ckpt.get("normalize_mode", "per_point"),
        use_amp=ckpt.get("use_amp", False),
        use_causal=ckpt.get("use_causal", True),
        input_delta=ckpt.get("input_delta", True),
        use_anchor_token=ckpt.get("use_anchor_token", False),
        scale_deltas=ckpt.get("scale_deltas", False),
        delta_scale_eps=ckpt.get("delta_scale_eps", 1e-6),
        delta_scale_factor=ckpt.get("delta_scale_factor", 10.0),
        delta_loss_weighting=ckpt.get("delta_loss_weighting", False),
        delta_loss_alpha=ckpt.get("delta_loss_alpha", 0.0),
        loss_scale=ckpt.get("loss_scale", 1.0),
        max_grad_norm=ckpt.get("max_grad_norm", 1.0),
        device=device,
    )

    model.n_tau = ckpt["n_tau"]
    model.n_logm = ckpt["n_logm"]
    model.n_features = ckpt["n_features"]
    model.context_length = ckpt["context_length"]
    model.mean = ckpt.get("mean")
    model.std = ckpt.get("std")
    model.delta_mean = ckpt.get("delta_mean")
    model.delta_std = ckpt.get("delta_std")
    model.delta_target_mean = ckpt.get("delta_target_mean")
    model.delta_target_std = ckpt.get("delta_target_std")
    model.anchor_mean = ckpt.get("anchor_mean")
    model.anchor_std = ckpt.get("anchor_std")

    model.model = _SurfaceTransformer(
        n_features=model.n_features,
        context_length=model.context_length,
        d_model=model.d_model,
        n_heads=model.n_heads,
        n_layers=model.n_layers,
        dropout=model.dropout,
        pool=model.pool,
        use_causal=model.use_causal,
    ).to(model.device)
    model.model.load_state_dict(ckpt["state_dict"])
    model.is_fitted = True
    return model, ckpt


def _print_top_point_errors(rmse_grid: np.ndarray, tau_grid: np.ndarray, logm_grid: np.ndarray, top_k: int = 10):
    flat = rmse_grid.reshape(-1)
    if flat.size == 0:
        return
    top_idx = np.argsort(flat)[-top_k:][::-1]
    print(f"\nTop {top_k} pointwise RMSEs:")
    for idx in top_idx:
        tau_idx, logm_idx = np.unravel_index(idx, rmse_grid.shape)
        print(
            f"  tau={tau_grid[tau_idx]:.4f} logm={logm_grid[logm_idx]:.4f} "
            f"rmse={rmse_grid[tau_idx, logm_idx]:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Debug a delta_transformer checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/delta_transformer/delta_transformer_w0_c21_h5.pt",
    )
    parser.add_argument("--window-id", type=int, default=0)
    parser.add_argument("--context", type=int, default=21)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit test samples (0 = all)")
    parser.add_argument("--train-samples", type=int, default=0, help="Limit train samples (0 = all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-debug", action="store_true", help="Write debug arrays to results/debug")
    parser.add_argument("--overfit-check", action="store_true", help="Train on a tiny subset to test fit")
    parser.add_argument("--overfit-samples", type=int, default=256)
    parser.add_argument("--overfit-epochs", type=int, default=200)
    parser.add_argument(
        "--overfit-no-regularization",
        action="store_true",
        help="Disable dropout and weight decay for overfit check",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    print("Loading checkpoint...")
    model, ckpt = _load_transformer_checkpoint(args.checkpoint, args.device)
    print(
        "Checkpoint config:",
        f"c{ckpt.get('context_length')} h{ckpt.get('horizon', 'n/a')} "
        f"d_model={ckpt.get('d_model')} layers={ckpt.get('n_layers')} "
        f"heads={ckpt.get('n_heads')} delta_mode={ckpt.get('delta_mode')} "
        f"input_delta={ckpt.get('input_delta')} scale_deltas={ckpt.get('scale_deltas')}"
    )

    pipeline = ForecastingPipeline()
    pipeline.load_data()
    pipeline.create_windows()

    if args.window_id >= len(pipeline.windows):
        raise SystemExit(f"window-id {args.window_id} out of range (0..{len(pipeline.windows)-1})")

    window = pipeline.windows[args.window_id]
    print(f"Using window {window.window_id} test range {window.test_start.date()} -> {window.test_end.date()}")

    X_train, y_train, X_val, y_val, X_test, y_test, sample_dates = pipeline._build_sequences_for_window(
        window=window,
        context_length=args.context,
        horizon=args.horizon,
        train_on_val=False,
        use_val=True,
    )

    if len(X_test) == 0:
        raise SystemExit("No test samples available for this window/context/horizon.")

    if args.max_samples and args.max_samples < len(X_test):
        X_test = X_test[:args.max_samples]
        y_test = y_test[:args.max_samples]
        sample_dates = sample_dates[:args.max_samples]

    if args.train_samples and args.train_samples < len(X_train):
        X_train = X_train[:args.train_samples]
        y_train = y_train[:args.train_samples]

    preds = model.predict_horizon(X_test, horizon=args.horizon)
    metrics = compute_all_metrics(y_test, preds)
    print("\nDelta transformer metrics:")
    print(metrics)

    baseline = X_test[:, -1, :, :]
    baseline_metrics = compute_all_metrics(y_test, baseline)
    print("\nPersistence baseline metrics:")
    print(baseline_metrics)

    deltas_true = y_test - baseline
    deltas_pred = preds - baseline
    _summarize_array("true_delta", deltas_true)
    _summarize_array("pred_delta", deltas_pred)
    _summarize_array("abs_error", np.abs(preds - y_test))

    if model.scale_deltas:
        delta_scale = model._compute_delta_scale(X_test)
        _summarize_array("delta_scale", delta_scale)
        print(f"delta_scale zeros: {(delta_scale <= model.delta_scale_eps).sum()}")

    print("\nTrain vs test delta std (checkpoint model):")
    train_preds = model.predict_horizon(X_train, horizon=args.horizon)
    train_delta_true = y_train - X_train[:, -1, :, :]
    train_delta_pred = train_preds - X_train[:, -1, :, :]
    _summarize_array("train_true_delta", train_delta_true)
    _summarize_array("train_pred_delta", train_delta_pred)
    _summarize_array("test_true_delta", deltas_true)
    _summarize_array("test_pred_delta", deltas_pred)

    mse_grid = np.mean((preds - y_test) ** 2, axis=0)
    rmse_grid = np.sqrt(mse_grid)
    _print_top_point_errors(rmse_grid, pipeline.tau_grid, pipeline.logm_grid)

    metrics_by_tau = compute_metrics_by_maturity(y_test, preds, pipeline.tau_grid, pipeline.tau_grid)
    metrics_by_moneyness = compute_metrics_by_moneyness(y_test, preds, pipeline.tau_grid, pipeline.logm_grid)
    print("\nMetrics by maturity (iv_rmse):")
    for tau, vals in metrics_by_tau.items():
        print(f"  tau={tau}: {vals['iv_rmse']:.6f}")
    print("\nMetrics by moneyness region (iv_rmse):")
    for region, vals in metrics_by_moneyness.items():
        print(f"  {region}: {vals['iv_rmse']:.6f}")

    if args.save_debug:
        out_dir = os.path.join("results", "debug")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "delta_transformer_rmse_grid.npy"), rmse_grid)
        np.save(os.path.join(out_dir, "delta_transformer_pred.npy"), preds)
        np.save(os.path.join(out_dir, "delta_transformer_true.npy"), y_test)
        np.save(os.path.join(out_dir, "delta_transformer_baseline.npy"), baseline)
        print(f"\nSaved debug arrays to {out_dir}")

    if args.overfit_check:
        print("\nOverfit check on a small subset...")
        n_overfit = min(args.overfit_samples, len(X_train))
        X_small = X_train[:n_overfit]
        y_small = y_train[:n_overfit]
        overfit_dropout = 0.0 if args.overfit_no_regularization else ckpt.get("dropout", 0.0)
        overfit_weight_decay = 0.0 if args.overfit_no_regularization else ckpt.get("weight_decay", 0.0)
        overfit_model = TransformerSurfaceModel(
            name="overfit_check",
            d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"],
            n_layers=ckpt["n_layers"],
            dropout=overfit_dropout,
            learning_rate=ckpt.get("learning_rate", 1e-3),
            weight_decay=overfit_weight_decay,
            batch_size=min(32, n_overfit),
            num_epochs=args.overfit_epochs,
            pool=ckpt.get("pool", "last"),
            normalize=ckpt.get("normalize", True),
            normalize_mode=ckpt.get("normalize_mode", "per_point"),
            patience=0,
            min_delta=0.0,
            use_amp=False,
            use_causal=ckpt.get("use_causal", True),
            delta_mode=ckpt.get("delta_mode", False),
            input_delta=ckpt.get("input_delta", False),
            use_anchor_token=ckpt.get("use_anchor_token", True),
            scale_deltas=ckpt.get("scale_deltas", False),
            delta_scale_eps=ckpt.get("delta_scale_eps", 1e-6),
            device=args.device,
        )
        overfit_model.fit(
            X_small,
            y_small,
            context_length=args.context,
            horizon=args.horizon,
        )
        overfit_preds = overfit_model.predict_horizon(X_small, horizon=args.horizon)
        overfit_metrics = compute_all_metrics(y_small, overfit_preds)
        print("Overfit metrics:")
        print(overfit_metrics)
        overfit_delta_true = y_small - X_small[:, -1, :, :]
        overfit_delta_pred = overfit_preds - X_small[:, -1, :, :]
        _summarize_array("overfit_true_delta", overfit_delta_true)
        _summarize_array("overfit_pred_delta", overfit_delta_pred)


if __name__ == "__main__":
    main()
