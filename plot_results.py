#!/usr/bin/env python3
"""
Generic CLI for plotting model results.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.visualizer import visualize_results


def main():
    parser = argparse.ArgumentParser(description="Plot model results from metrics JSON.")
    parser.add_argument(
        "--model",
        dest="model_id",
        required=True,
        help="Model id used for plot filenames, e.g. persistence"
    )
    parser.add_argument(
        "--file",
        dest="results_file",
        default=None,
        help="Path to results JSON (optional if model id is known)"
    )
    parser.add_argument(
        "--out",
        dest="save_dir",
        default="results/plots",
        help="Output directory for plots (default: results/plots)"
    )
    args = parser.parse_args()

    label_map = {
        "persistence": "Persistence Model",
    }
    model_label = label_map.get(args.model_id, args.model_id.replace("_", " ").title())

    results_map = {
        "persistence": "results/metrics/persistence_results.json",
    }
    results_file = args.results_file or results_map.get(args.model_id)
    if not results_file:
        raise SystemExit(
            f"Unknown model '{args.model_id}' and no --file provided. "
            "Provide --file or add to results_map."
        )

    visualize_results(
        results_file=results_file,
        model_id=args.model_id,
        model_label=model_label,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
