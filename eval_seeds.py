#!/usr/bin/env python3
"""
eval_seeds.py — retrain each tuning winner at multiple seeds so the
evaluation report can average across initialization noise.

Two modes:

  --list                       Print every (model, variant, seed) command
                               needed to populate seed slots. Pipe to a
                               shell or run them one at a time.

  --run-one                    Train one (model, [variant], seed) using
                               the tuning winner's saved config and write
                               preds + metadata to the canonical seed
                               slot.

Output layout (deep models with a tuning winner):
  <ModelDir>/eval/63_<P>/[<variant>/]seed_<S>/
      preds.npy
      metrics_test.json
      hyperparams.json
      source.json

VAR is deterministic — leave it at <ModelDir>/eval/63_<P>/preds.npy (one
copy, no seed dim).

Usage
-----
    python eval_seeds.py --list --pred_len 21
    python eval_seeds.py --run-one --model dlinear --seed 0 --pred_len 21
    python eval_seeds.py --run-one --model hot --variant kronecker_sum \\
                          --seed 42 --pred_len 21
"""
from __future__ import annotations

import argparse
import json
import os

from train import (
    LOOKBACK,
    MODEL_DIR,
    ROOT,
    load_dataset,
    pick_device,
    train_deep_model,
)


DEEP_NAMES     = ("dlinear", "patchtst", "hot", "tucker_dlinear", "gwn",
                  "itransformer", "pcaformer")
VARIANT_MODELS = {
    "hot": ("kronecker_product", "kronecker_sum"),
}
DEFAULT_SEEDS  = (0, 1, 2, 3, 4, 42)


# ─── Locations ────────────────────────────────────────────────────────────

def _summary_path(name: str, pred_len: int, variant: str | None) -> str:
    base = os.path.join(ROOT, MODEL_DIR[name], "tuning_results",
                        f"{LOOKBACK}_{pred_len}")
    if variant is None:
        return os.path.join(base, "summary.json")
    return os.path.join(base, variant, "summary.json")


def _canonical_seed_dir(name: str, pred_len: int, seed: int,
                        variant: str | None) -> str:
    base = os.path.join(ROOT, MODEL_DIR[name], "eval",
                        f"{LOOKBACK}_{pred_len}")
    parts = ([variant] if variant else []) + [f"seed_{seed}"]
    return os.path.join(base, *parts)


def _load_winner_cfg(name: str, pred_len: int, variant: str | None) -> dict:
    sp = _summary_path(name, pred_len, variant)
    if not os.path.isfile(sp):
        raise SystemExit(f"missing tuning summary: {os.path.relpath(sp, ROOT)}")
    with open(sp) as f:
        summary = json.load(f)
    winner = summary.get("winner")
    if not winner:
        raise SystemExit(f"no winner in {os.path.relpath(sp, ROOT)}")
    cfg_path = os.path.join(ROOT, winner["config_path"])
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["_winner_combo"]  = winner["combo_id"]
    cfg["_winner_source"] = os.path.relpath(cfg_path, ROOT)
    return cfg


# ─── --list ───────────────────────────────────────────────────────────────

def list_commands(pred_len: int, seeds: tuple[int, ...]):
    cmds = []
    for name in DEEP_NAMES:
        for v in VARIANT_MODELS.get(name, (None,)):
            for s in seeds:
                v_arg = f" --variant {v}" if v else ""
                cmds.append(
                    f"python eval_seeds.py --run-one --model {name}"
                    f"{v_arg} --seed {s} --pred_len {pred_len}"
                )
    print(f"# {len(cmds)} commands  (seeds={list(seeds)}, pred_len={pred_len})")
    print()
    for c in cmds:
        print(c)


# ─── --run-one ────────────────────────────────────────────────────────────

def run_one(name: str, variant: str | None, seed: int, pred_len: int,
            csv_path: str, data_end: str | None, force: bool):
    if name not in DEEP_NAMES:
        raise SystemExit(f"unknown model {name!r}; pick from {DEEP_NAMES}")
    valid_variants = VARIANT_MODELS.get(name, (None,))
    if variant not in valid_variants:
        raise SystemExit(f"{name} expects variant in {valid_variants}, got {variant!r}")

    out_dir = _canonical_seed_dir(name, pred_len, seed, variant)
    preds_path = os.path.join(out_dir, "preds.npy")
    if os.path.isfile(preds_path) and not force:
        print(f"skip (exists): {os.path.relpath(out_dir, ROOT)}")
        return

    cfg    = _load_winner_cfg(name, pred_len, variant)
    device = pick_device()
    print(f"loading dataset ...")
    data   = load_dataset(csv_path, 0.7, 0.1, LOOKBACK, pred_len,
                          data_end=data_end)

    print(f"\n=== {name}{('/' + variant) if variant else ''}  "
          f"seed={seed}  combo={cfg['_winner_combo']} ===")
    os.makedirs(out_dir, exist_ok=True)
    train_deep_model(name, data, pred_len, device, seed,
                     winner_cfg=cfg, out_dir=out_dir)

    # Augment with a source.json that records what produced this slot.
    src = {
        "model":          name,
        "variant":        variant,
        "seed":           seed,
        "winner_combo":   cfg["_winner_combo"],
        "winner_source":  cfg["_winner_source"],
        "pred_len":       pred_len,
        "data_end":       data_end,
    }
    with open(os.path.join(out_dir, "source.json"), "w") as f:
        json.dump(src, f, indent=2)
    print(f"  written → {os.path.relpath(out_dir, ROOT)}")


# ─── Entry ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list",    action="store_true")
    mode.add_argument("--run-one", action="store_true")

    ap.add_argument("--model",    default=None)
    ap.add_argument("--variant",  default=None)
    ap.add_argument("--seed",     type=int, default=None)
    ap.add_argument("--pred_len", type=int, default=21, choices=(5, 21, 63))
    ap.add_argument("--seeds",    type=int, nargs="+", default=list(DEFAULT_SEEDS),
                    help="(for --list) seed values to enumerate. "
                         f"Default {list(DEFAULT_SEEDS)}.")
    ap.add_argument("--csv_path", default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--data_end", default="2023-12-29")
    ap.add_argument("--force",    action="store_true",
                    help="(for --run-one) retrain even if preds.npy exists.")
    args = ap.parse_args()

    if args.list:
        list_commands(args.pred_len, tuple(args.seeds))
        return

    # --run-one
    if args.model is None or args.seed is None:
        raise SystemExit("--run-one requires --model and --seed")
    data_end = None if args.data_end.lower() == "none" else args.data_end
    run_one(args.model, args.variant, args.seed, args.pred_len,
            args.csv_path, data_end, args.force)


if __name__ == "__main__":
    main()
