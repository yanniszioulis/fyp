#!/usr/bin/env python3
"""
Generic CLI for model tuning on validation splits.
"""

import os
import sys
import json
import argparse
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting.pipeline import ForecastingPipeline
from evaluation.metrics import compute_all_metrics
from models.transformer.transformer_model import TransformerSurfaceModel


def _parse_csv_list(value, cast_fn=int):
    return [cast_fn(x) for x in value.split(",") if x.strip()]


def _build_grid(grid_spec):
    keys = sorted(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _default_transformer_grid():
    return {
        "d_model": [128, 256],
        "n_heads": [4, 8],
        "n_layers": [2, 4],
        "dropout": [0.1],
        "learning_rate": [1e-3],
        "weight_decay": [1e-4],
        "batch_size": [32],
        "num_epochs": [30],
        "pool": ["last", "mean"],
        "normalize": [True],
        "normalize_mode": ["per_point", "global"],
        "use_causal": [True],
        "delta_mode": [False],
        "patience": [5],
        "min_delta": [0.0],
    }


def main():
    parser = argparse.ArgumentParser(description="Tune a model on validation splits.")
    parser.add_argument("--model", dest="model_id", required=True, help="Model id (e.g. transformer)")
    parser.add_argument(
        "--data",
        dest="data_file",
        default="SPX_IV_fixed_grid.csv",
        help="Path to fixed-grid IV data CSV (defaults to env FYP_DATA_PATH or ./SPX_IV_fixed_grid.csv)"
    )
    parser.add_argument("--results", dest="results_dir", default="results")
    parser.add_argument("--window-ids", dest="window_ids", default="0",
                        help="Comma-separated window ids (e.g. 0 or 0,1,2)")
    parser.add_argument("--context", dest="context_length", type=int, default=21)
    parser.add_argument("--horizon", dest="horizon", type=int, default=5)
    parser.add_argument("--grid-file", dest="grid_file", default=None,
                        help="Path to JSON grid spec (dict of param -> list)")
    parser.add_argument("--amp", dest="use_amp", action="store_true",
                        help="Enable automatic mixed precision (transformer)")
    parser.add_argument("--overfit", action="store_true",
                        help="Overfit mode: train on a small subset with no early stopping")
    parser.add_argument("--overfit-samples", type=int, default=0,
                        help="Number of training samples to use in overfit mode (0 = all)")
    parser.add_argument("--overfit-epochs", type=int, default=200,
                        help="Epochs to run in overfit mode (overrides grid)")
    parser.add_argument("--overfit-no-regularization", action="store_true",
                        help="Set dropout/weight decay to 0 in overfit mode")
    args = parser.parse_args()

    window_ids = _parse_csv_list(args.window_ids, int)

    if args.model_id not in {"transformer", "delta_transformer"}:
        raise SystemExit("Only transformer and delta_transformer tuning are supported right now.")

    grid_file = args.grid_file
    if grid_file is None:
        grid_file = os.path.join("tuning", f"{args.model_id}_grid.json")

    if os.path.exists(grid_file):
        with open(grid_file, "r") as f:
            grid_spec = json.load(f)
    else:
        grid_spec = _default_transformer_grid()

    data_file = args.data_file or os.environ.get("FYP_DATA_PATH") or "SPX_IV_fixed_grid.csv"
    pipeline = ForecastingPipeline(
        data_file=data_file,
        results_dir=args.results_dir
    )
    pipeline.load_data()
    pipeline.create_windows()

    all_results = []
    for window_id in window_ids:
        if window_id >= len(pipeline.windows):
            print(f"Skipping window {window_id}: out of range")
            continue
        window = pipeline.windows[window_id]

        X_train, y_train, X_val, y_val, _, _, _ = pipeline._build_sequences_for_window(
            window=window,
            context_length=args.context_length,
            horizon=args.horizon,
            train_on_val=False,
            use_val=True
        )

        if X_val is None or y_val is None or len(X_val) == 0:
            print(f"Skipping window {window_id}: no validation samples")
            continue

        grid_list = list(_build_grid(grid_spec))
        total_configs = len(grid_list)
        print(
            f"Tuning {args.model_id} | window={window_id} | "
            f"context={args.context_length} | horizon={args.horizon}"
        )
        print(f"Grid size: {total_configs} configs")

        best_val = float("inf")
        best_result = None

        for idx, config in enumerate(grid_list, start=1):
            if args.overfit:
                if args.overfit_samples and args.overfit_samples > 0:
                    n_overfit = min(args.overfit_samples, len(X_train))
                    X_train_use = X_train[:n_overfit]
                    y_train_use = y_train[:n_overfit]
                else:
                    X_train_use = X_train
                    y_train_use = y_train
                X_val_use = X_val
                y_val_use = y_val
                num_epochs = args.overfit_epochs
                dropout = 0.0 if args.overfit_no_regularization else config["dropout"]
                weight_decay = 0.0 if args.overfit_no_regularization else config["weight_decay"]
                normalize = False if args.model_id == "delta_transformer" else config["normalize"]
                delta_loss_weighting = args.model_id == "delta_transformer"
                delta_loss_alpha = 5.0 if args.model_id == "delta_transformer" else 0.0
                loss_scale = 420.0 if args.model_id == "delta_transformer" else 1.0
                patience = 0
                min_delta = 0.0
            else:
                X_train_use = X_train
                y_train_use = y_train
                X_val_use = X_val
                y_val_use = y_val
                num_epochs = config["num_epochs"]
                dropout = config["dropout"]
                weight_decay = config["weight_decay"]
                normalize = config["normalize"]
                delta_loss_weighting = False
                delta_loss_alpha = 0.0
                loss_scale = 1.0
                patience = config["patience"]
                min_delta = config["min_delta"]

            model = TransformerSurfaceModel(
                name=f"transformer_w{window_id}_c{args.context_length}_h{args.horizon}",
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_layers=config["n_layers"],
                dropout=dropout,
                learning_rate=config["learning_rate"],
                weight_decay=weight_decay,
                batch_size=config["batch_size"],
                num_epochs=num_epochs,
                pool=config["pool"],
                normalize=normalize,
                normalize_mode=config.get("normalize_mode", "per_point"),
                patience=patience,
                min_delta=min_delta,
                use_amp=args.use_amp,
                use_causal=config.get("use_causal", True),
                delta_mode=(args.model_id == "delta_transformer"),
                input_delta=(args.model_id == "delta_transformer"),
                use_anchor_token=not (args.model_id == "delta_transformer"),
                scale_deltas=False,
                delta_scale_factor=10.0 if args.model_id == "delta_transformer" else 1.0,
                delta_loss_weighting=delta_loss_weighting,
                delta_loss_alpha=delta_loss_alpha,
                loss_scale=loss_scale
            )

            fit_kwargs = dict(
                context_length=args.context_length,
                horizon=args.horizon,
            )
            if X_val_use is not None and y_val_use is not None:
                fit_kwargs["X_val"] = X_val_use
                fit_kwargs["y_val"] = y_val_use
            if args.overfit:
                fit_kwargs["log_interval"] = 5
                fit_kwargs["log_train_val"] = True
                fit_kwargs["use_val_for_early_stopping"] = False
            model.fit(
                X_train_use, y_train_use,
                **fit_kwargs
            )

            train_preds = model.predict_horizon(X_train_use, horizon=args.horizon)
            train_metrics = compute_all_metrics(y_train_use, train_preds)

            val_metrics = None
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_preds = model.predict_horizon(X_val, horizon=args.horizon)
                val_metrics = compute_all_metrics(y_val, val_preds)

            result = {
                "window_id": window_id,
                "context_length": args.context_length,
                "horizon": args.horizon,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "epochs_trained": model.epochs_trained,
                "config": config
            }
            all_results.append(result)

            print(f"[{idx}/{total_configs}] cfg={config}")
            print(
                f"  train_iv_rmse={train_metrics['iv_rmse']:.6f}  "
                f"val_iv_rmse={(val_metrics or {}).get('iv_rmse', float('nan')):.6f}"
            )
            print(f"  epochs_trained={model.epochs_trained}")

            if val_metrics is not None and val_metrics["iv_rmse"] < best_val:
                best_val = val_metrics["iv_rmse"]
                best_result = result
                print(f"  best_so_far: val_iv_rmse={best_val:.6f}")

        if best_result:
            print("\nBest config:")
            print(f"  val_iv_rmse={best_result['val_metrics']['iv_rmse']:.6f}")
            print(f"  train_iv_rmse={best_result['train_metrics']['iv_rmse']:.6f}")
            print(f"  epochs_trained={best_result['epochs_trained']}")
            print(f"  cfg={best_result['config']}")

    os.makedirs(os.path.join(args.results_dir, "tuning"), exist_ok=True)
    out_file = os.path.join(
        args.results_dir,
        "tuning",
        f"{args.model_id}_tuning_w{','.join(map(str, window_ids))}_c{args.context_length}_h{args.horizon}.json"
    )
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved tuning results to {out_file}")


if __name__ == "__main__":
    main()
