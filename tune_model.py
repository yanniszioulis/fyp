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
        "pool": ["last"],
        "normalize": [True],
        "patience": [5],
        "min_delta": [0.0],
    }


def main():
    parser = argparse.ArgumentParser(description="Tune a model on validation splits.")
    parser.add_argument("--model", dest="model_id", required=True, help="Model id (e.g. transformer)")
    parser.add_argument("--data", dest="data_file", default="SPX_IV_fixed_grid.csv")
    parser.add_argument("--results", dest="results_dir", default="results")
    parser.add_argument("--window-ids", dest="window_ids", default="0",
                        help="Comma-separated window ids (e.g. 0 or 0,1,2)")
    parser.add_argument("--context", dest="context_length", type=int, default=21)
    parser.add_argument("--horizon", dest="horizon", type=int, default=5)
    parser.add_argument("--grid-file", dest="grid_file", default=None,
                        help="Path to JSON grid spec (dict of param -> list)")
    parser.add_argument("--amp", dest="use_amp", action="store_true",
                        help="Enable automatic mixed precision (transformer)")
    args = parser.parse_args()

    window_ids = _parse_csv_list(args.window_ids, int)

    if args.model_id != "transformer":
        raise SystemExit("Only transformer tuning is supported right now.")

    grid_file = args.grid_file
    if grid_file is None:
        grid_file = os.path.join("tuning", f"{args.model_id}_grid.json")

    if os.path.exists(grid_file):
        with open(grid_file, "r") as f:
            grid_spec = json.load(f)
    else:
        grid_spec = _default_transformer_grid()

    pipeline = ForecastingPipeline(
        data_file=args.data_file,
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

        for config in _build_grid(grid_spec):
            model = TransformerSurfaceModel(
                name=f"transformer_w{window_id}_c{args.context_length}_h{args.horizon}",
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_layers=config["n_layers"],
                dropout=config["dropout"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                batch_size=config["batch_size"],
                num_epochs=config["num_epochs"],
                pool=config["pool"],
                normalize=config["normalize"],
                patience=config["patience"],
                min_delta=config["min_delta"],
                use_amp=args.use_amp
            )

            model.fit(
                X_train, y_train,
                context_length=args.context_length,
                horizon=args.horizon,
                X_val=X_val,
                y_val=y_val
            )

            preds = model.predict_horizon(X_val, horizon=args.horizon)
            metrics = compute_all_metrics(y_val, preds)

            result = {
                "window_id": window_id,
                "context_length": args.context_length,
                "horizon": args.horizon,
                "metrics": metrics,
                "config": config
            }
            all_results.append(result)
            print(
                f"W{window_id} cfg={config} iv_rmse={metrics['iv_rmse']:.6f}"
            )

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
