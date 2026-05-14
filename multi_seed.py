#!/usr/bin/env python3
"""
multi_seed.py — run train.py for a list of (model, seed) pairs and report
seed-to-seed noise in best_val_mse / test_mse. Used to decide whether the
val-loss gap between two models is real or within run-to-run variability.

By default each model uses whatever defaults are currently set in
train.py (LR_*, WD_*, build_model kwargs, grad_clip, per-group LR for
tucker, etc.). Pass --from_winner to instead drive every run from the
tuning winner's config.json for (model, pred_len) — i.e. reproduce the
hyperparameter_tuning.py result across seeds, using its model_kwargs,
optimizer choice, lr/wd/lr_g/wd_g/grad_clip/min_epochs/batch_size.

Usage:
    python multi_seed.py --pred_len 21
    python multi_seed.py --pred_len 21 --seeds 0 1 2 3 4
    python multi_seed.py --pred_len 21 --models dlinear tucker_dlinear
    python multi_seed.py --pred_len 21 --from_winner
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from statistics import mean, stdev


MODEL_DIR = {
    "dlinear":        "DLinear",
    "patchtst":       "PatchTST",
    "hot":            "HOT",
    "tucker_dlinear": "Tucker_DLinear",
    "gwn":            "GWN",
}


def latest_run_dir(model: str, pred_len: int) -> str:
    """Newest timestamped subdir of <ModelDir>/63_<pred_len>/."""
    base = os.path.join(MODEL_DIR[model], f"63_{pred_len}")
    subs = [d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))]
    if not subs:
        raise RuntimeError(f"No run dirs under {base}")
    return os.path.join(
        base, max(subs, key=lambda d: os.path.getmtime(os.path.join(base, d))),
    )


def run_one(model: str, pred_len: int, seed: int,
            batch_size: int | None = None,
            from_winner: bool = False) -> dict:
    bar = "=" * 72
    bs_str = f"  batch_size={batch_size}" if batch_size is not None else ""
    fw_str = "  from_winner" if from_winner else ""
    print(f"\n{bar}\n  RUNNING  model={model}  seed={seed}"
          f"{bs_str}{fw_str}\n{bar}", flush=True)
    cmd = [sys.executable, "train.py", "--model", model,
           "--pred_len", str(pred_len), "--seed", str(seed)]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]
    if from_winner:
        cmd += ["--from_winner"]
    subprocess.run(cmd, check=True)
    run_dir = latest_run_dir(model, pred_len)
    with open(os.path.join(run_dir, "metrics_test.json")) as f:
        m = json.load(f)
    with open(os.path.join(run_dir, "hyperparams.json")) as f:
        h = json.load(f)
    return {
        "model":      model,
        "seed":       seed,
        "run_dir":    run_dir,
        "best_val":   float(m["best_val_mse"]),
        "best_epoch": int(m["best_epoch"]),
        "stop_epoch": int(m["stop_epoch"]),
        "test_mse":   float(m["test_mse"]),
        "test_rmse":  float(m["test_rmse"]),
        "test_mae":   float(m["test_mae"]),
        "n_params":   int(h["n_params"]),
        "lr":         float(h["lr"]),
        "weight_decay": float(h["weight_decay"]),
    }


def print_summary(results: list[dict], seeds: list[int]):
    bar = "=" * 72
    print(f"\n\n{bar}\n  PER-RUN RESULTS\n{bar}")
    print(f"{'model':<18} {'seed':<5} {'best_val':<10} {'epoch':<6} "
          f"{'stop':<6} {'test_mse':<10} {'test_mae':<10}")
    for r in results:
        print(f"{r['model']:<18} {r['seed']:<5} {r['best_val']:<10.5f} "
              f"{r['best_epoch']:<6} {r['stop_epoch']:<6} "
              f"{r['test_mse']:<10.5f} {r['test_mae']:<10.5f}")

    print(f"\n{bar}\n  PER-MODEL NOISE  (n={len(seeds)} seeds)\n{bar}")
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    for model, rs in by_model.items():
        v = [r["best_val"] for r in rs]
        t = [r["test_mse"] for r in rs]
        n = rs[0]["n_params"]
        lr = rs[0]["lr"]
        wd = rs[0]["weight_decay"]
        print(f"\n{model}   params={n:,}   lr={lr}   wd={wd}")
        v_std = stdev(v) if len(v) > 1 else 0.0
        t_std = stdev(t) if len(t) > 1 else 0.0
        print(f"  best_val_mse  mean={mean(v):.5f}  std={v_std:.5f}  "
              f"range=[{min(v):.5f}, {max(v):.5f}]")
        print(f"  test_mse      mean={mean(t):.5f}  std={t_std:.5f}  "
              f"range=[{min(t):.5f}, {max(t):.5f}]")

    if len(by_model) >= 2 and len(seeds) > 1:
        names = list(by_model.keys())
        print(f"\n{bar}\n  PAIRWISE GAPS  ({len(names)} models, {len(seeds)} seeds each)\n{bar}")
        print(f"  Δ = (B - A) means model B's metric is larger than model A's.")
        print(f"  |Δ| / pooled-std interprets gap significance against seed noise.\n")
        header = f"  {'A':<18} {'B':<18} {'metric':<10} {'Δ':>10} {'pooled_std':>12} {'|Δ|/std':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, a_name in enumerate(names):
            for b_name in names[i + 1:]:
                a, b = by_model[a_name], by_model[b_name]
                for metric in ("best_val", "test_mse"):
                    va = [r[metric] for r in a]
                    vb = [r[metric] for r in b]
                    gap = mean(vb) - mean(va)
                    pooled = (stdev(va) + stdev(vb)) / 2
                    sig = abs(gap) / pooled if pooled > 0 else float("inf")
                    print(f"  {a_name:<18} {b_name:<18} {metric:<10} "
                          f"{gap:>+10.5f} {pooled:>12.5f} {sig:>10.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_len", type=int, default=21)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+",
                    default=["dlinear", "patchtst", "tucker_dlinear"])
    ap.add_argument("--out", default=None,
                    help="Optional JSON path to dump all per-run results.")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Override train.py's BATCH_SIZE for every run. "
                         "Ignored under --from_winner (winner's batch "
                         "size wins).")
    ap.add_argument("--from_winner", action="store_true",
                    help="For each model, load the tuning winner's "
                         "config.json from <ModelDir>/tuning_results/"
                         "63_<pred_len>/summary.json and use its "
                         "model_kwargs + optimizer settings instead of "
                         "train.py's defaults.")
    args = ap.parse_args()

    bad = [m for m in args.models if m not in MODEL_DIR]
    if bad:
        raise SystemExit(f"Unknown models: {bad}. Known: {list(MODEL_DIR)}")

    results: list[dict] = []
    for model in args.models:
        for seed in args.seeds:
            results.append(run_one(model, args.pred_len, seed,
                                   batch_size=args.batch_size,
                                   from_winner=args.from_winner))

    print_summary(results, args.seeds)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
