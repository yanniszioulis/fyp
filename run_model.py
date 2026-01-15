#!/usr/bin/env python3
"""
Generic CLI for running models through the forecasting pipeline.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting.pipeline import ForecastingPipeline
from models.transformer.transformer_model import TransformerSurfaceModel


def main():
    parser = argparse.ArgumentParser(description="Run a model via the forecasting pipeline.")
    parser.add_argument(
        "--model",
        dest="model_id",
        required=True,
        help="Model id, e.g. persistence"
    )
    parser.add_argument(
        "--data",
        dest="data_file",
        default="SPX_IV_fixed_grid.csv",
        help="Path to fixed-grid IV data CSV (defaults to env FYP_DATA_PATH or ./SPX_IV_fixed_grid.csv)"
    )
    parser.add_argument(
        "--results",
        dest="results_dir",
        default="results",
        help="Results output directory"
    )
    parser.add_argument(
        "--contexts",
        dest="context_lengths",
        default="5,21,63",
        help="Comma-separated context lengths, e.g. 5,21,63"
    )
    parser.add_argument(
        "--horizons",
        dest="horizons",
        default="1,5,21",
        help="Comma-separated horizons, e.g. 1,5,21"
    )
    parser.add_argument(
        "--no-save",
        dest="save_results",
        action="store_false",
        help="Disable saving forecasts/metrics"
    )
    parser.add_argument(
        "--d-model",
        dest="d_model",
        type=int,
        default=256,
        help="Transformer d_model"
    )
    parser.add_argument(
        "--n-heads",
        dest="n_heads",
        type=int,
        default=8,
        help="Transformer number of heads"
    )
    parser.add_argument(
        "--n-layers",
        dest="n_layers",
        type=int,
        default=4,
        help="Transformer number of layers"
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        default=0.1,
        help="Transformer dropout"
    )
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-3,
        help="Transformer learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-4,
        help="Transformer weight decay"
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
        help="Transformer batch size"
    )
    parser.add_argument(
        "--epochs",
        dest="num_epochs",
        type=int,
        default=50,
        help="Transformer epochs"
    )
    parser.add_argument(
        "--pool",
        dest="pool",
        default="last",
        choices=["last", "mean"],
        help="Transformer pooling: last or mean"
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable transformer normalization"
    )
    parser.add_argument(
        "--normalize-mode",
        dest="normalize_mode",
        default="per_point",
        choices=["per_point", "global"],
        help="Normalization mode for transformer"
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=10,
        help="Early stopping patience (transformer)"
    )
    parser.add_argument(
        "--min-delta",
        dest="min_delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement for early stopping"
    )
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        help="Enable automatic mixed precision (transformer)"
    )
    parser.add_argument(
        "--causal",
        dest="use_causal",
        action="store_true",
        help="Enable causal attention mask (transformer)"
    )
    parser.add_argument(
        "--save-ckpt",
        dest="save_checkpoint",
        action="store_true",
        help="Save best checkpoint per window/context/horizon (transformer)"
    )
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    data_file = args.data_file or os.environ.get("FYP_DATA_PATH") or "SPX_IV_fixed_grid.csv"
    pipeline = ForecastingPipeline(
        data_file=data_file,
        results_dir=args.results_dir
    )

    print("=" * 60)
    print(f"Model Run: {args.model_id}")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    pipeline.load_data()

    print("\n[2/3] Creating rolling windows...")
    pipeline.create_windows(
        window_size_years=10.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        shift_months=6
    )

    print("\n[3/3] Running model...")
    if args.model_id == "persistence":
        results = pipeline.run_persistence(
            context_lengths=context_lengths,
            horizons=horizons,
            save_results=args.save_results
        )
    elif args.model_id == "transformer":
        results = pipeline.run_model(
            model_factory=lambda name, **kwargs: TransformerSurfaceModel(
                name=name,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                pool=args.pool,
                normalize=args.normalize,
                normalize_mode=args.normalize_mode,
                patience=args.patience,
                min_delta=args.min_delta,
                use_amp=args.use_amp,
                use_causal=args.use_causal
            ),
            model_id="transformer",
            context_lengths=context_lengths,
            horizons=horizons,
            train_on_val=False,
            use_val=True,
            save_checkpoints=args.save_checkpoint,
            min_train_samples=1,
            save_results=args.save_results
        )
    else:
        raise SystemExit(
            f"Unknown model '{args.model_id}'. "
            "Supported: persistence, transformer."
        )

    print("\n" + "=" * 60)
    print(f"Total configurations tested: {len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
