#!/usr/bin/env python3
"""
evaluate.py — benchmark tuned models on the held-out test set.

For each selected model:
  - reads <ModelDir>/tuning_results/63_<pred_len>/[<variant>/]summary.json
  - loads the winner's config.json + best_model.pt
  - rebuilds the model, runs inference on the test set
  - saves per-model preds + metrics
  - aggregates results into a top-level comparison

VAR has no tuning sweep; it is re-fitted here (lag=1, OLS) on the same
train rows the tuning sweep used so its predictions align with the
same test windows.

Inputs MUST match the tuning script's split. Defaults are 70/10/20 with
data_end=2023-12-29 — override only if you also overrode them in tuning.

Outputs
-------
<ModelDir>/test_results/63_<pred_len>/[<variant>/]
    metrics_test.json    # mse, rmse, mae in standardized log-IV space
    preds.npy            # (N_test, pred_len, n_channels)
    source.json          # which tuning combo this was loaded from

test_results/63_<pred_len>/
    comparison.json
    comparison.csv

A model with `_variants` (HOT: kronecker_product, kronecker_sum) emits
one entry per variant in the comparison.

Usage
-----
    python evaluate.py --model all --pred_len 21
    python evaluate.py --model dlinear,patchtst,var --pred_len 5
    python evaluate.py --model hot --pred_len 63
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from train import (
    DLinear,
    HOT,
    LOOKBACK,
    MODEL_DIR,
    PatchTST,
    ROOT,
    TuckerDLinear,
    _DLinearAdapter,
    _HOTAdapter,
    _PatchTSTAdapter,
    _TuckerAdapter,
    _predict,
    fit_var_p,
    load_dataset,
    make_step_window_fn,
    pick_device,
)


DEEP_NAMES = ("dlinear", "patchtst", "hot", "tucker_dlinear")
ALL_NAMES  = (*DEEP_NAMES, "var")

MODEL_CLASS_BY_NAME = {
    "dlinear":        DLinear,
    "patchtst":       PatchTST,
    "hot":            HOT,
    "tucker_dlinear": TuckerDLinear,
}
ADAPTER_BY_NAME = {
    "dlinear":        _DLinearAdapter,
    "patchtst":       _PatchTSTAdapter,
    "hot":            _HOTAdapter,
    "tucker_dlinear": _TuckerAdapter,
}
EVAL_BATCH = 64


# ─── Discovery ────────────────────────────────────────────────────────────

def find_variants(name: str, pred_len: int) -> list[tuple[str | None, str]]:
    """Return [(variant_name | None, variant_dir), ...] for a tunable model."""
    task_dir = os.path.join(ROOT, MODEL_DIR[name], "tuning_results",
                            f"{LOOKBACK}_{pred_len}")
    if not os.path.isdir(task_dir):
        raise SystemExit(f"Missing tuning results: {task_dir}")
    if os.path.isfile(os.path.join(task_dir, "summary.json")):
        return [(None, task_dir)]
    out = []
    for entry in sorted(os.listdir(task_dir)):
        sub = os.path.join(task_dir, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "summary.json")):
            out.append((entry, sub))
    if not out:
        raise SystemExit(
            f"No summary.json found under {task_dir} (run tuning first).")
    return out


# ─── Model rebuild ────────────────────────────────────────────────────────

def rebuild_adapter(name: str, model_kwargs: dict,
                    n_tau: int, n_money: int) -> torch.nn.Module:
    """Reconstruct an adapter-wrapped model from a saved config's
    `model_kwargs`. Grid kwargs were saved as-is by the tuner, including
    the structural args (seq_len, pred_len, c_in/n_channels/W/H, ...),
    so the model is fully described by `model_kwargs`."""
    cls = MODEL_CLASS_BY_NAME[name]
    model = cls(**model_kwargs)
    adapter_cls = ADAPTER_BY_NAME[name]
    if name in ("hot", "tucker_dlinear"):
        return adapter_cls(model, n_tau, n_money)
    return adapter_cls(model)


# ─── Per-model evaluation ─────────────────────────────────────────────────

def evaluate_tuned(name: str, variant_name: str | None, variant_dir: str,
                   data: dict, device: torch.device):
    """Load the variant's winner and predict on the test set.
    Returns (preds, source_record)."""
    with open(os.path.join(variant_dir, "summary.json")) as f:
        summary = json.load(f)
    winner = summary.get("winner")
    if winner is None:
        raise SystemExit(
            f"{os.path.relpath(variant_dir, ROOT)}/summary.json has no winner")
    combo_id  = winner["combo_id"]
    combo_dir = os.path.join(variant_dir, combo_id)

    cfg_path  = os.path.join(combo_dir, "config.json")
    ckpt_path = os.path.join(combo_dir, "best_model.pt")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"Missing config.json: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    grid    = data["grid"]
    adapter = rebuild_adapter(name, cfg["model_kwargs"], grid.n_tau, grid.n_money)
    state   = torch.load(ckpt_path, map_location=device)
    adapter.load_state_dict(state)
    adapter.to(device)
    adapter.eval()

    Xte, _ = data["test"]
    preds = _predict(adapter, Xte, EVAL_BATCH, device)

    source = {
        "model":            name,
        "variant":          variant_name,
        "tuning_dir":       os.path.relpath(variant_dir, ROOT),
        "winner_combo":     combo_id,
        "val_loss":         winner.get("best_val_loss"),
        "best_epoch":       winner.get("best_epoch"),
        "model_kwargs":     cfg["model_kwargs"],
        "n_params":         cfg.get("n_params"),
        "tuning_seed":      cfg.get("seed"),
        "checkpoint":       os.path.relpath(ckpt_path, ROOT),
    }
    return preds, source


def evaluate_var(data: dict, pred_len: int):
    """Fit VAR(1) on the tuning train rows; iterative rollout per test
    window from the last input row. Returns (preds, source)."""
    train_end = data["rows"]["train_end"]
    X_fit     = data["scaled_log_iv"][:train_end].astype(np.float64)
    c, A_list, _ = fit_var_p(X_fit, p=1)
    step = make_step_window_fn(c, A_list)

    Xte, _ = data["test"]
    N, _, C = Xte.shape
    preds = np.empty((N, pred_len, C), dtype=np.float32)
    seed_rows = Xte[:, -1, :].astype(np.float64)
    for i in range(N):
        cur = seed_rows[i].reshape(1, C)
        out = np.empty((pred_len, C), dtype=np.float64)
        for h in range(pred_len):
            nxt = step(cur)
            out[h] = nxt
            cur = nxt.reshape(1, C)
        preds[i] = out

    source = {
        "model":       "var",
        "variant":     None,
        "kind":        "refit",
        "lag":         1,
        "n_train_rows": int(train_end),
        "fit_space":   "standardized_log_iv",
    }
    return preds, source


def compute_metrics(preds: np.ndarray, Yte: np.ndarray) -> dict:
    diff = preds - Yte
    mse  = float(np.mean(diff ** 2))
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {
        "test_mse":  mse,
        "test_rmse": rmse,
        "test_mae":  mae,
        "n_test":    int(Yte.shape[0]),
        "space":     "standardized_log_iv",
    }


# ─── Persistence ──────────────────────────────────────────────────────────

def save_per_model(out_dir: str, preds: np.ndarray, metrics: dict,
                   source: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "preds.npy"), preds.astype(np.float32))
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "source.json"), "w") as f:
        json.dump(source, f, indent=2, default=str)


# ─── Entry point ──────────────────────────────────────────────────────────

def parse_models(arg: str) -> list[str]:
    if arg == "all":
        return list(ALL_NAMES)
    names = [s.strip() for s in arg.split(",") if s.strip()]
    bad = [n for n in names if n not in ALL_NAMES]
    if bad:
        raise SystemExit(
            f"Unknown / unsupported model(s): {bad}. "
            f"Pick from {ALL_NAMES} or 'all'.")
    return names


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate tuned models on the same held-out test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help=f"Comma-separated names from {ALL_NAMES} or 'all'.")
    ap.add_argument("--pred_len", required=True, type=int, choices=(5, 21, 63))
    ap.add_argument("--csv_path",   default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--data_end",   type=str,   default="2023-12-29")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must be < 1.")

    names  = parse_models(args.model)
    device = pick_device()

    print(f"device: {device}")
    print(f"loading {os.path.relpath(args.csv_path, ROOT)} ...")
    data_end = None if args.data_end.lower() == "none" else args.data_end
    data = load_dataset(args.csv_path, args.train_frac, args.val_frac,
                        LOOKBACK, args.pred_len, data_end=data_end)
    g, r = data["grid"], data["rows"]
    Xte, Yte = data["test"]
    print(f"  rows: N={r['N']}  train_end={r['train_end']}  "
          f"val_end={r['val_end']}  C={r['n_channels']}  "
          f"grid={g.n_tau}×{g.n_money} (tau×moneyness)")
    print(f"  test windows: {Xte.shape[0]}   "
          f"starts predicting at: {data['test_first_target_date']}")

    rows: list[dict] = []

    for name in names:
        if name == "var":
            print(f"\n[var]  refitting VAR(1) on {r['train_end']} train rows ...")
            preds, source = evaluate_var(data, args.pred_len)
            metrics = compute_metrics(preds, Yte)
            out_dir = os.path.join(ROOT, MODEL_DIR["var"], "test_results",
                                   f"{LOOKBACK}_{args.pred_len}")
            save_per_model(out_dir, preds, metrics, source)
            print(f"  test: mse={metrics['test_mse']:.6f}  "
                  f"rmse={metrics['test_rmse']:.6f}  "
                  f"mae={metrics['test_mae']:.6f}")
            print(f"  saved → {os.path.relpath(out_dir, ROOT)}")
            rows.append({
                "display":   "var",
                "model":     "var",
                "variant":   None,
                "val_loss":  None,
                "n_params":  None,
                **metrics,
            })
            continue

        for v_name, v_dir in find_variants(name, args.pred_len):
            display = name + (f"/{v_name}" if v_name else "")
            print(f"\n[{display}]  loading winner from "
                  f"{os.path.relpath(v_dir, ROOT)} ...")
            preds, source = evaluate_tuned(name, v_name, v_dir, data, device)
            metrics = compute_metrics(preds, Yte)
            out_subdir = os.path.join(
                ROOT, MODEL_DIR[name], "test_results",
                f"{LOOKBACK}_{args.pred_len}",
                *([v_name] if v_name else []),
            )
            save_per_model(out_subdir, preds, metrics, source)
            print(f"  combo={source['winner_combo']}  "
                  f"val={source['val_loss']:.6f}  "
                  f"params={source['n_params']:,}")
            print(f"  test: mse={metrics['test_mse']:.6f}  "
                  f"rmse={metrics['test_rmse']:.6f}  "
                  f"mae={metrics['test_mae']:.6f}")
            print(f"  saved → {os.path.relpath(out_subdir, ROOT)}")
            rows.append({
                "display":   display,
                "model":     name,
                "variant":   v_name,
                "val_loss":  source["val_loss"],
                "n_params":  source["n_params"],
                **metrics,
            })

    # Comparison artefacts.
    rows_sorted = sorted(rows, key=lambda r: r["test_mse"])
    comp_dir = os.path.join(ROOT, "test_results", f"{LOOKBACK}_{args.pred_len}")
    os.makedirs(comp_dir, exist_ok=True)

    comparison = {
        "pred_len":   args.pred_len,
        "lookback":   LOOKBACK,
        "data_end":   data.get("data_end"),
        "first_date": data.get("first_date"),
        "last_date":  data.get("last_date"),
        "test_first_target_date": data.get("test_first_target_date"),
        "n_test":     int(Yte.shape[0]),
        "ranking":    rows_sorted,
    }
    with open(os.path.join(comp_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    csv_fields = ["display", "model", "variant",
                  "test_mse", "test_rmse", "test_mae",
                  "val_loss", "n_params", "n_test"]
    with open(os.path.join(comp_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_sorted)

    print(f"\n{'=' * 84}")
    print(f"Test ranking  (pred_len={args.pred_len}, n_test={Yte.shape[0]})")
    print(f"{'=' * 84}")
    print(f"{'model':30s} {'test_mse':>11s} {'test_rmse':>11s} "
          f"{'test_mae':>11s} {'val_loss':>11s} {'params':>12s}")
    for row in rows_sorted:
        vl     = (f"{row['val_loss']:.6f}"
                  if row["val_loss"] is not None else "—")
        params = (f"{row['n_params']:,}"
                  if row["n_params"] is not None else "—")
        print(f"{row['display']:30s} "
              f"{row['test_mse']:>11.6f} "
              f"{row['test_rmse']:>11.6f} "
              f"{row['test_mae']:>11.6f} "
              f"{vl:>11s} "
              f"{params:>12s}")
    print(f"\nsaved → {os.path.relpath(comp_dir, ROOT)}/")


if __name__ == "__main__":
    main()
