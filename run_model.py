#!/usr/bin/env python3
"""
Generic CLI for running models through the forecasting pipeline.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting.pipeline import ForecastingPipeline


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
        help="Path to fixed-grid IV data CSV"
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
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    pipeline = ForecastingPipeline(
        data_file=args.data_file,
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
    else:
        raise SystemExit(
            f"Unknown model '{args.model_id}'. "
            "Supported: persistence."
        )

    print("\n" + "=" * 60)
    print(f"Total configurations tested: {len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
