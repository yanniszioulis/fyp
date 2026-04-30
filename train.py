#!/usr/bin/env python3
"""
Master training dispatcher for SPX IV surface forecasting.

Runs one or more model training scripts in sequence, then optionally
compares all results with compare_models.py.

Usage:
    python train.py --models all
    python train.py --models var1 dlinear patchtst
    python train.py --models hot --attention_type kronecker_sum
    python train.py --models dyngwn --epochs 500 --compare

Each model script lives in its own subfolder and is always invoked with
python <subfolder>/<script>.py [model-specific args passed through].
"""

import argparse
import subprocess
import sys


MODELS = {
    "var1":     "VAR1/var1_spx_iv.py",
    "dlinear":  "DLinear/dlinear_spx_iv.py",
    "patchtst": "PatchTST/patchtst_spx_iv.py",
    "hot":      "HOT/hot_spx_iv.py",
    "dyngwn":   "DynGWN/dyngwn_spx_iv.py",
}

# Args passed to every model script (if the script accepts them)
COMMON_ARGS = ["csv_path", "seq_len", "pred_len", "device", "seed"]

# Per-model extra args (model_name → list of (--arg, attr_name) tuples)
MODEL_ARGS = {
    "var1":     [("--tune_ridge", "tune_ridge"), ("--ridge_lambda", "ridge_lambda")],
    "dlinear":  [("--epochs", "epochs"), ("--batch_size", "batch_size"),
                 ("--lr", "lr"), ("--patience", "patience"),
                 ("--kernel_size", "kernel_size")],
    "patchtst": [("--epochs", "epochs"), ("--batch_size", "batch_size"),
                 ("--lr", "lr"), ("--patience", "patience"),
                 ("--patch_len", "patch_len"), ("--stride", "stride"),
                 ("--d_model", "d_model"), ("--n_heads", "n_heads"),
                 ("--n_layers", "n_layers"), ("--d_ff", "d_ff"),
                 ("--dropout", "dropout")],
    "hot":      [("--epochs", "epochs"), ("--batch_size", "batch_size"),
                 ("--lr", "lr"), ("--patience", "patience"),
                 ("--d_hidden", "d_hidden"), ("--d_mlp", "d_mlp"),
                 ("--n_blocks", "n_blocks"), ("--n_head", "n_head"),
                 ("--patch_size", "patch_size"),
                 ("--attention_type", "attention_type"),
                 ("--pe", "pe"),
                 ("--dropout", "dropout")],
    "dyngwn":   [("--epochs", "epochs"), ("--batch_size", "batch_size"),
                 ("--lr", "lr"), ("--patience", "patience"),
                 ("--nhid", "nhid"), ("--blocks", "blocks"),
                 ("--layers", "layers"),
                 ("--graph_mode", "graph_mode"),
                 ("--dropout", "dropout")],
}


def _run(script: str, extra_args: list[str]):
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[WARN] {script} exited with code {result.returncode}")
    return result.returncode


def _build_args(args, model_name: str) -> list[str]:
    out = []
    # Common args
    if args.csv_path != "SPX_surfaces.csv":
        out += ["--csv_path", args.csv_path]
    if args.seq_len != 21:
        out += ["--seq_len", str(args.seq_len)]
    if args.pred_len != 63:
        out += ["--pred_len", str(args.pred_len)]
    if args.device != "auto":
        out += ["--device", args.device]
    if args.seed != 42:
        out += ["--seed", str(args.seed)]

    # Model-specific args
    for flag, attr in MODEL_ARGS.get(model_name, []):
        val = getattr(args, attr, None)
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                out.append(flag)
        else:
            out += [flag, str(val)]

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Master training dispatcher for SPX IV forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available models: " + ", ".join(MODELS.keys()) + ", all"
    )
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="Which models to train (space-separated, or 'all')")
    ap.add_argument("--compare", action="store_true",
                    help="Run compare_models.py after training")

    # Common args
    ap.add_argument("--csv_path", default="SPX_surfaces.csv")
    ap.add_argument("--seq_len",  type=int, default=21)
    ap.add_argument("--pred_len", type=int, default=63)
    ap.add_argument("--device",   default="auto")
    ap.add_argument("--seed",     type=int, default=42)

    # VAR1
    ap.add_argument("--tune_ridge",   action="store_true")
    ap.add_argument("--ridge_lambda", type=float, default=None)

    # DLinear
    ap.add_argument("--kernel_size",  type=int, default=None)

    # PatchTST
    ap.add_argument("--patch_len",    type=int,   default=None)
    ap.add_argument("--stride",       type=int,   default=None)
    ap.add_argument("--d_model",      type=int,   default=None)
    ap.add_argument("--n_heads",      type=int,   default=None)
    ap.add_argument("--n_layers",     type=int,   default=None)
    ap.add_argument("--d_ff",         type=int,   default=None)

    # HOT
    ap.add_argument("--d_hidden",     type=int,   default=None)
    ap.add_argument("--d_mlp",        type=int,   default=None)
    ap.add_argument("--n_blocks",     type=int,   default=None)
    ap.add_argument("--n_head",       type=int,   default=None)
    ap.add_argument("--patch_size",   type=int,   default=None)
    ap.add_argument("--attention_type", default=None,
                    choices=["kronecker_product", "kronecker_sum"])
    ap.add_argument("--pe",           default=None,
                    choices=["rope", "nope"])

    # DynGWN
    ap.add_argument("--nhid",         type=int,   default=None)
    ap.add_argument("--blocks",       type=int,   default=None)
    ap.add_argument("--layers",       type=int,   default=None)
    ap.add_argument("--graph_mode",   default=None,
                    choices=["grid_plus_adaptive", "adaptive_only"])

    # Shared training
    ap.add_argument("--epochs",       type=int,   default=None)
    ap.add_argument("--batch_size",   type=int,   default=None)
    ap.add_argument("--lr",           type=float, default=None)
    ap.add_argument("--patience",     type=int,   default=None)
    ap.add_argument("--dropout",      type=float, default=None)

    args = ap.parse_args()

    # Resolve model list
    selected = args.models
    if "all" in selected:
        selected = list(MODELS.keys())
    else:
        unknown = [m for m in selected if m not in MODELS]
        if unknown:
            ap.error(f"Unknown model(s): {unknown}. Choose from: {list(MODELS.keys())} or 'all'")

    print(f"Models to train: {selected}")

    failed = []
    for model_name in selected:
        script     = MODELS[model_name]
        extra_args = _build_args(args, model_name)
        rc = _run(script, extra_args)
        if rc != 0:
            failed.append(model_name)

    if args.compare:
        print(f"\n{'='*70}")
        print("Running compare_models.py ...")
        print("=" * 70)
        compare_args = [sys.executable, "compare_models.py",
                        "--csv_path", args.csv_path]
        subprocess.run(compare_args)

    print(f"\n{'='*70}")
    if failed:
        print(f"Completed with errors in: {failed}")
    else:
        print(f"All models trained successfully: {selected}")
    if args.compare:
        print("Comparison results saved to comparison_results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
