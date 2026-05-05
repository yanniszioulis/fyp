#!/usr/bin/env python3
"""
hyperparameter_tuning.py — orchestrate a sweep over a model's hyperparameter grid.

Reads `<MODEL>/tuning_grid.json` (scalars = fixed, lists = swept via cartesian
product). Results are organised by *task* (dataset × target_space × seq_len ×
pred_len) and optionally by *variant* (per-grid `_variants` block):

    <MODEL>/tuning_results/
      {dataset}_{target_space}_SPX_IV_{sl}_{pl}/        ← task tag
        [<variant_name>/]                                 ← only if grid has _variants
          combo_0001/{config.json, train_log.csv}
          combo_0002/{config.json, train_log.csv}
          ...
          combo_NNNN/{config.json, train_log.csv,
                      best_model.pt, start_dates.npy}    ← winner (lowest val_loss)
          summary.json   leaderboard.csv   manifest.json

For every combo, a running winner is maintained by min `val_loss` from each
combo's `train_log.csv`. Only the running winner retains its `best_model.pt`;
losing combos are pruned to `{config.json, train_log.csv}`.

Variants (optional, declared in the grid as `_variants`):
    Each entry runs the full grid independently with its own scalar
    overrides. Used to treat e.g. HOT(kronecker_product) and
    HOT(kronecker_sum) as separate sweeps.
    [
      {"name": "kronecker_product", "fix": {"attention_type": "kronecker_product"}},
      {"name": "kronecker_sum",     "fix": {"attention_type": "kronecker_sum"}}
    ]

Resume semantics: a combo whose `train_log.csv` has a parseable `val_loss`
and whose `config.json` matches the requested params is skipped (its score
is read from the log). Use `--overwrite` to wipe.

Usage
-----
    python hyperparameter_tuning.py --model hot --dataset precovid \\
        --seq_len 21 --pred_len 63 --grid HOT/tuning_grid.json
    python hyperparameter_tuning.py --model hot --grid HOT/tuning_grid.json --dry_run
"""
import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys

import pandas as pd


MODEL_SCRIPTS = {
    "hot":      "HOT/hot.py",
    "dlinear":  "DLinear/dlinear.py",
    "patchtst": "PatchTST/patchtst.py",
    "dyngwn":   "DynGWN/dyngwn.py",
}

KEEP_ALWAYS = {"config.json", "train_log.csv"}
KEEP_WINNER = KEEP_ALWAYS | {"best_model.pt", "start_dates.npy"}


# ─── Grid ─────────────────────────────────────────────────────────────────────

def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of list-valued keys; scalars held fixed. Keys
    starting with '_' are ignored (free-form metadata)."""
    keys, vals = [], []
    for k, v in grid.items():
        if k.startswith("_"):
            continue
        keys.append(k)
        vals.append(v if isinstance(v, list) else [v])
    return [dict(zip(keys, t)) for t in itertools.product(*vals)]


def combo_to_args(params: dict) -> list[str]:
    out = []
    for k, v in params.items():
        out += [f"--{k}", str(v)]
    return out


def build_common_args(args) -> list[str]:
    return [
        "--csv_path",     args.csv_path,
        "--dataset",      args.dataset,
        "--target_space", args.target_space,
        "--seq_len",      str(args.seq_len),
        "--pred_len",     str(args.pred_len),
        "--device",       args.device,
        "--seed",         str(args.seed),
    ]


# ─── Combo state ──────────────────────────────────────────────────────────────

def read_min_val_loss(log_path: str):
    if not os.path.isfile(log_path):
        return None
    df = pd.read_csv(log_path)
    if df.empty or "val_loss" not in df.columns:
        return None
    return float(df["val_loss"].min())


def combo_is_complete(combo_dir: str, params: dict) -> bool:
    """Combo is complete if `train_log.csv` has a parseable val_loss and
    `config.json`'s recorded params match the requested combo."""
    log = os.path.join(combo_dir, "train_log.csv")
    cfg = os.path.join(combo_dir, "config.json")
    if not (os.path.isfile(log) and os.path.isfile(cfg)):
        return False
    if read_min_val_loss(log) is None:
        return False
    with open(cfg) as f:
        saved = json.load(f)
    for k, v in params.items():
        if str(saved.get(k)) != str(v):
            return False
    return True


def cleanup(combo_dir: str, keep_checkpoint: bool):
    """Remove everything in `combo_dir` except {config.json, train_log.csv};
    optionally also keep best_model.pt and start_dates.npy (running winner)."""
    if not os.path.isdir(combo_dir):
        return
    keep = KEEP_WINNER if keep_checkpoint else KEEP_ALWAYS
    for entry in os.listdir(combo_dir):
        if entry in keep:
            continue
        path = os.path.join(combo_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def write_leaderboard(path: str, rows: list, fieldnames: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(grid_resolved: dict, base_dir: str, script: str,
              common_cli: list[str], dry_run: bool):
    """Run a single grid sweep into `base_dir` (must already exist)."""
    combos = expand_grid(grid_resolved)
    if not combos:
        print(f"  Grid expanded to 0 combinations; skipping.")
        return

    swept_keys = sorted({k for p in combos for k in p})
    fieldnames = ["combo_id", "status", "best_val_loss"] + swept_keys

    manifest = {
        "n_combos": len(combos),
        "grid":     grid_resolved,
        "combos": [{"combo_id": f"combo_{i:04d}", "params": p}
                   for i, p in enumerate(combos)],
    }
    with open(os.path.join(base_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if dry_run:
        print(f"  Dry run — {len(combos)} combos:")
        for i, p in enumerate(combos):
            ps = ", ".join(f"{k}={v}" for k, v in p.items())
            print(f"    combo_{i:04d}: {ps}")
        return

    leader_path  = os.path.join(base_dir, "leaderboard.csv")
    leaderboard  = []
    running_best = (float("inf"), None)  # (score, combo_dir)

    for i, params in enumerate(combos):
        combo_id  = f"combo_{i:04d}"
        combo_dir = os.path.join(base_dir, combo_id)
        os.makedirs(combo_dir, exist_ok=True)

        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n  [{i + 1:>3}/{len(combos)}] {combo_id}  {ps}")

        if combo_is_complete(combo_dir, params):
            score = read_min_val_loss(os.path.join(combo_dir, "train_log.csv"))
            print(f"     resume → val={score:.6f}")
            status = "resumed"
        else:
            cmd = ([sys.executable, "-u", script, "--out_dir", combo_dir]
                   + common_cli + combo_to_args(params))
            rc = subprocess.run(cmd).returncode
            score = read_min_val_loss(os.path.join(combo_dir, "train_log.csv"))
            if rc != 0 or score is None:
                print(f"     FAILED (rc={rc})")
                row = {"combo_id": combo_id, "status": "failed",
                       "best_val_loss": None, **params}
                leaderboard.append(row)
                write_leaderboard(leader_path, leaderboard, fieldnames)
                continue
            status = "completed"
            print(f"     val={score:.6f}")

        if score < running_best[0]:
            if running_best[1] is not None:
                cleanup(running_best[1], keep_checkpoint=False)
            running_best = (score, combo_dir)
            cleanup(combo_dir, keep_checkpoint=True)
        else:
            cleanup(combo_dir, keep_checkpoint=False)

        row = {"combo_id": combo_id, "status": status,
               "best_val_loss": score, **params}
        leaderboard.append(row)
        write_leaderboard(leader_path, leaderboard, fieldnames)

    completed = [r for r in leaderboard if r["best_val_loss"] is not None]
    if not completed:
        print(f"\n  All combos failed in {base_dir}; no winner.")
        return

    completed_sorted = sorted(completed, key=lambda r: r["best_val_loss"])
    winner     = completed_sorted[0]
    winner_dir = os.path.join(base_dir, winner["combo_id"])
    winner_pt  = os.path.join(winner_dir, "best_model.pt")

    summary = {
        "n_combos":    len(combos),
        "n_completed": len(completed),
        "n_failed":    len(combos) - len(completed),
        "winner": {
            "combo_id":      winner["combo_id"],
            "best_val_loss": winner["best_val_loss"],
            "params":        {k: winner[k] for k in swept_keys},
            "result_dir":    winner_dir,
            "checkpoint":    winner_pt,
        },
        "ranking": [
            {"combo_id":      r["combo_id"],
             "best_val_loss": r["best_val_loss"]}
            for r in completed_sorted
        ],
    }
    with open(os.path.join(base_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Winner: {winner['combo_id']}  val={winner['best_val_loss']:.6f}")
    print(f"    params:  {', '.join(f'{k}={winner[k]}' for k in swept_keys)}")
    print(f"    ckpt:    {winner_pt}")
    if not os.path.isfile(winner_pt):
        print(f"    WARNING: winner .pt missing — re-run with --overwrite to "
              f"regenerate cleanly.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model",        required=True, choices=list(MODEL_SCRIPTS.keys()))
    ap.add_argument("--grid",         required=True,
                    help="Path to tuning_grid.json (scalars=fixed, lists=swept).")
    ap.add_argument("--dataset",      default="full",  choices=["full", "precovid"])
    ap.add_argument("--target_space", default="level", choices=["level", "logdiff"])
    ap.add_argument("--seq_len",      type=int, default=63)
    ap.add_argument("--pred_len",     type=int, default=21)
    ap.add_argument("--csv_path",     default="SPX_surfaces.csv")
    ap.add_argument("--device",       default="auto")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--overwrite",    action="store_true",
                    help="Wipe the task subfolder (and all variants) before starting.")
    ap.add_argument("--dry_run",      action="store_true",
                    help="Print the planned sweep and exit without training.")
    args = ap.parse_args()

    script = MODEL_SCRIPTS[args.model]
    if not os.path.isfile(script):
        raise SystemExit(f"Model script not found: {script}")
    if not os.path.isfile(args.grid):
        raise SystemExit(f"Grid file not found: {args.grid}")

    with open(args.grid) as f:
        grid = json.load(f)

    task_tag = (f"{args.dataset}_{args.target_space}_SPX_IV_"
                f"{args.seq_len}_{args.pred_len}")
    task_dir = os.path.join(os.path.dirname(script), "tuning_results", task_tag)

    if args.overwrite and os.path.isdir(task_dir):
        print(f"Wiping {task_dir}/ ...")
        shutil.rmtree(task_dir)
    os.makedirs(task_dir, exist_ok=True)

    base_grid = {k: v for k, v in grid.items() if not k.startswith("_")}
    variants  = grid.get("_variants") or [{"name": None, "fix": {}}]

    common_cli = build_common_args(args)

    print(f"\nTuning {args.model.upper()}  task={task_tag}")
    print(f"Output: {task_dir}/")
    print(f"Variants: {len(variants)}  "
          f"({sum(1 for v in base_grid.values() if isinstance(v, list))} "
          f"swept axes per variant)")

    for v in variants:
        v_name = v.get("name")
        v_fix  = v.get("fix") or {}

        if v_name is None:
            v_dir   = task_dir
            v_label = "(default)"
        else:
            v_dir   = os.path.join(task_dir, v_name)
            v_label = v_name
            os.makedirs(v_dir, exist_ok=True)

        v_grid = {**base_grid, **v_fix}

        print(f"\n{'─' * 70}\nVariant: {v_label}\n{'─' * 70}")
        run_sweep(v_grid, v_dir, script, common_cli, args.dry_run)

    print(f"\n{'=' * 70}\nDone. Results: {task_dir}/")


if __name__ == "__main__":
    main()
