#!/usr/bin/env python3
"""
Generic CLI for plotting model results.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.visualizer import visualize_results, load_results, plot_model_comparison_summary


def main():
    parser = argparse.ArgumentParser(description="Plot model results from metrics JSON.")
    parser.add_argument(
        "--model",
        dest="model_id",
        required=False,
        help="Model id used for plot filenames, e.g. persistence"
    )
    parser.add_argument(
        "--compare",
        dest="compare_models",
        default=None,
        help="Comma-separated model ids to compare (e.g. persistence,transformer)"
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
        default=None,
        help="Output directory for plots (default: results/plots/<model_id>)"
    )
    args = parser.parse_args()

    label_map = {
        "persistence": "Persistence",
        "transformer": "Transformer",
        "delta_transformer": "Delta Transformer",
    }
    model_label = None
    if args.model_id:
        model_label = label_map.get(args.model_id, args.model_id.replace("_", " ").title())

    results_map = {
        "persistence": "results/metrics/persistence_results.json",
        "transformer": "results/metrics/transformer_results.json",
        "delta_transformer": "results/metrics/delta_transformer_results.json",
    }

    if args.compare_models:
        model_ids = [m.strip() for m in args.compare_models.split(",") if m.strip()]
        if len(model_ids) < 2:
            raise SystemExit("--compare requires at least two model ids")
        model_dfs = {}
        for mid in model_ids:
            results_file = results_map.get(mid)
            if results_file is None:
                raise SystemExit(f"Unknown model '{mid}' in --compare")
            model_dfs[mid] = load_results(results_file)
        compare_dir = args.save_dir or os.path.join(
            "results", "plots", f"comparing_{'_'.join(model_ids)}"
        )
        os.makedirs(compare_dir, exist_ok=True)
        save_path = os.path.join(
            compare_dir,
            f"summary_all_configs_{'_'.join(model_ids)}.png"
        )
        plot_model_comparison_summary(model_dfs, label_map, save_path=save_path)
        return

    if not args.model_id:
        raise SystemExit("--model is required unless --compare is used")

    results_file = args.results_file or results_map.get(args.model_id)
    if not results_file:
        raise SystemExit(
            f"Unknown model '{args.model_id}' and no --file provided. "
            "Provide --file or add to results_map."
        )

    save_dir = args.save_dir or os.path.join("results", "plots", args.model_id)
    visualize_results(
        results_file=results_file,
        model_id=args.model_id,
        model_label=model_label,
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()
