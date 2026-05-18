#!/usr/bin/env python3
"""
hyperparameter_tuning.py — sweep a model's tuning_grid.json on SPX_surfaces.csv.

Reuses train.py's data pipeline (log-IV + per-channel scaler, chronological
split, optional `--data_end` cutoff) but with tuning defaults: train/val/
test = 70/10/20 and `data_end=2023-12-29`. Test data is held out and never
touched during tuning — winners are chosen on validation loss.

What it does
------------
For each model in `--model`, reads `<ModelDir>/tuning_grid.json` and runs
the cartesian product of its list-valued keys (scalars held fixed). Grid
fields override `train.py`'s trainer constants (`epochs`, `patience`,
`min_epochs`, `batch_size`, `lr`, `weight_decay`); model-architecture keys
go to the model's `__init__`.

Supported bundle keys (mirrored from existing grids):
  `_variants`                 → run the whole grid once per variant;
                                each variant's `fix` dict overlays the grid.
                                Output goes under `<variant_name>/`.
  `_spatial_temporal_pairs`   → TuckerDLinear-only. Joint sweep over
                                `(rank_W, rank_H, rank_L, rank_P_max)`;
                                `rank_P = min(pred_len, rank_P_max)`.

Layout
------
<ModelDir>/tuning_results/63_<pred_len>/[<variant_name>/]
    combo_0000/{config.json, train_log.csv}
    combo_0001/{config.json, train_log.csv}
    ...
    combo_NNNN/{config.json, train_log.csv, best_model.pt}   ← running winner
    summary.json

Only the running winner of each variant keeps `best_model.pt`. When a
later combo beats the current winner's val loss, the new combo's `.pt`
is written and the old winner's `.pt` is deleted. `summary.json` lists
every combo's score, the ranking, and the winner.

Notes
-----
* Supported models: dlinear, patchtst, hot, tucker_dlinear, gwn.
  GWN's grid uses `nhid` as a unified channel knob — the build branch
  expands it into residual/dilation/skip/end channels.
* String knobs in grids are coerced: "on" → True, "off" → False.
* Resume by default: combos whose params already match a saved
  `config.json` are reused (their existing combo_id is kept, no retrain).
  New combos in the grid are trained with fresh combo_ids. Pass
  `--overwrite` to wipe and start from scratch.

Usage
-----
    python hyperparameter_tuning.py --model dlinear --pred_len 21
    python hyperparameter_tuning.py --model dlinear,patchtst --pred_len 5
    python hyperparameter_tuning.py --model all --pred_len 63 --overwrite
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch

from train import (
    BATCH_SIZE,
    DLinear,
    EPOCHS,
    GWN,
    HOT,
    LOOKBACK,
    MIN_EPOCHS,
    MODEL_DIR,
    PATIENCE,
    PatchTST,
    ROOT,
    TuckerDLinear,
    _DLinearAdapter,
    _GWNAdapter,
    _HOTAdapter,
    _PatchTSTAdapter,
    _TuckerAdapter,
    _epoch,
    _predict,
    load_dataset,
    pick_device,
)


TUNABLE = ("dlinear", "patchtst", "hot", "tucker_dlinear", "gwn")

# Grid-key buckets.
TRAINER_KEYS    = {"epochs", "patience", "min_epochs", "batch_size"}
OPTIMIZER_KEYS  = {"lr", "weight_decay", "lr_g_mult", "wd_g_mult"}
META_KEYS       = {"loss"}            # currently always MSE; ignored


# ─── Grid expansion ───────────────────────────────────────────────────────

def _coerce(v):
    """Map common string knobs in the grids to bools."""
    if isinstance(v, str):
        if v.lower() == "on":  return True
        if v.lower() == "off": return False
    return v


def _signature(combo: dict) -> str:
    """Canonical JSON signature of a combo's params (for resume matching).

    Values are coerced so semantically equivalent encodings match
    (e.g. "off"/"on" ↔ False/True), which keeps resume robust to grid
    edits that only change a value's *encoding*.
    """
    norm = {k: _coerce(combo[k]) for k in sorted(combo)}
    return json.dumps(norm, default=str, sort_keys=True)


def discover_existing(v_dir: str) -> dict[str, tuple[str, float, dict]]:
    """Scan `v_dir` for completed combos. Returns
        {param_signature: (combo_id, best_val_loss, cfg_dict)}.
    A combo is "complete" if its config.json has both `combo` and a
    `best_val_loss`. Combos without those keys are ignored."""
    if not os.path.isdir(v_dir):
        return {}
    out: dict[str, tuple[str, float, dict]] = {}
    for entry in sorted(os.listdir(v_dir)):
        if not entry.startswith("combo_"):
            continue
        sub      = os.path.join(v_dir, entry)
        cfg_path = os.path.join(sub, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            continue
        params = cfg.get("combo")
        score  = cfg.get("best_val_loss")
        if params is None or score is None:
            continue
        out[_signature(params)] = (entry, float(score), cfg)
    return out


def find_existing_winner(v_dir: str,
                         existing: dict[str, tuple[str, float, dict]]
                         ) -> tuple[str, float, int] | None:
    """Return (combo_id, val_loss, best_epoch) for the existing combo that
    owns `best_model.pt`. If multiple combos somehow have a .pt, pick the
    one with the lowest stored val_loss. Returns None if no .pt exists."""
    candidates = []
    for combo_id, score, cfg in existing.values():
        pt = os.path.join(v_dir, combo_id, "best_model.pt")
        if os.path.isfile(pt):
            candidates.append((combo_id, score, int(cfg.get("best_epoch", 0))))
    if not candidates:
        return None
    return min(candidates, key=lambda r: r[1])


def split_combo(combo: dict) -> tuple[dict, dict, dict]:
    """Split a flat combo dict into (model_kwargs, optimizer, trainer)."""
    trainer = {k: combo[k] for k in TRAINER_KEYS  if k in combo}
    opt     = {k: combo[k] for k in OPTIMIZER_KEYS if k in combo}
    model   = {k: _coerce(v) for k, v in combo.items()
               if k not in TRAINER_KEYS
               and k not in OPTIMIZER_KEYS
               and k not in META_KEYS}
    return model, opt, trainer


def expand_grid(grid: dict, model_name: str,
                pred_len: int) -> list[tuple[str | None, dict]]:
    """Return [(variant_name | None, flat_combo_dict), ...].

    Grid structure:
      - Top-level keys (no leading underscore) are normal axes — list
        values cross-product, scalar values stay fixed.
      - Underscore-prefixed keys ending in '_pairs' (with a list value)
        are "pair axes" — bundles of correlated kwargs that must vary
        together. Each entry in such a list is a dict of kwargs merged
        into the combo verbatim. Multiple pair axes are supported and
        cross-product with each other and with the top-level axes.
      - Any key (top-level or pair-sourced) ending in '_max' is clamped
        to pred_len and the suffix is stripped — used for rank_P_*
        entries that must not exceed the forecast horizon.
      - '_variants' (optional): list of named variants, each with a
        'fix' dict that overrides base axes.
    """
    variants = grid.get("_variants") or [{"name": None, "fix": {}}]
    base     = {k: v for k, v in grid.items() if not k.startswith("_")}

    pair_axis_keys = [k for k, v in grid.items()
                      if k.startswith("_") and k.endswith("_pairs")
                      and isinstance(v, list)]

    out: list[tuple[str | None, dict]] = []
    for v in variants:
        v_name = v.get("name")
        v_fix  = v.get("fix") or {}
        merged = {**base, **v_fix}

        axes_keys, axes_vals = [], []
        for k, vv in merged.items():
            axes_keys.append(k)
            axes_vals.append(vv if isinstance(vv, list) else [vv])

        # Append each pair axis as a single cartesian-product dimension.
        for pair_key in pair_axis_keys:
            axes_keys.append(pair_key)
            axes_vals.append(grid[pair_key])

        for tup in itertools.product(*axes_vals):
            combo = dict(zip(axes_keys, tup))
            # Merge each chosen pair's kwargs into the combo.
            for pair_key in pair_axis_keys:
                combo.update(combo.pop(pair_key))
            # Apply '_max' clamp-to-pred_len rule.
            for k in list(combo.keys()):
                if k.endswith("_max"):
                    combo[k[:-4]] = min(pred_len, int(combo.pop(k)))
            out.append((v_name, combo))
    return out


# ─── Model construction ───────────────────────────────────────────────────

def build_model_for_tuning(name: str, model_kw: dict, pred_len: int,
                           n_channels: int, n_tau: int, n_money: int):
    L, P, C = LOOKBACK, pred_len, n_channels
    if name == "dlinear":
        kw = dict(seq_len=L, pred_len=P, n_channels=C, **model_kw)
        return _DLinearAdapter(DLinear(**kw)), kw
    if name == "patchtst":
        kw = dict(c_in=C, seq_len=L, pred_len=P, **model_kw)
        return _PatchTSTAdapter(PatchTST(**kw)), kw
    if name == "hot":
        kw = dict(context_length=L, prediction_length=P, **model_kw)
        return _HOTAdapter(HOT(**kw), n_tau, n_money), kw
    if name == "tucker_dlinear":
        kw = dict(seq_len=L, pred_len=P, W=n_money, H=n_tau, **model_kw)
        return _TuckerAdapter(TuckerDLinear(**kw), n_tau, n_money), kw
    if name == "gwn":
        # `nhid` collapses the four GWN channel widths: residual = dilation
        # = skip = nhid, end = 2 * nhid. Other GWN init args are fixed to
        # mirror train.py's small starting point (in_dim=1, supports=None,
        # gcn_bool=True, addaptadj=True).
        mk   = dict(model_kw)
        nhid = int(mk.pop("nhid"))
        kw = dict(
            num_nodes=C, seq_len=L, pred_len=P,
            in_dim=1, supports=None,
            gcn_bool=True, addaptadj=True, aptinit=None,
            residual_channels=nhid, dilation_channels=nhid,
            skip_channels=nhid,     end_channels=2 * nhid,
            **mk,
        )
        return _GWNAdapter(GWN(**kw)), kw
    raise ValueError(f"Unsupported model: {name!r}")


# ─── One combo: train and persist ─────────────────────────────────────────

def run_one_combo(name: str, pred_len: int, combo: dict, data: dict,
                  device: torch.device, seed: int,
                  combo_dir: str) -> tuple[dict, dict | None]:
    """Train one combo to completion (or early stop). Returns the combo's
    `config.json` payload and its best-epoch state_dict (CPU tensors)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model_kw, opt_kw, trainer_kw = split_combo(combo)
    grid = data["grid"]
    C    = data["rows"]["n_channels"]
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]

    adapter, resolved = build_model_for_tuning(
        name, model_kw, pred_len, C, grid.n_tau, grid.n_money,
    )
    adapter.to(device)
    n_params = sum(p.numel() for p in adapter.parameters())

    lr = float(opt_kw["lr"])
    wd = float(opt_kw.get("weight_decay", 0.0))
    if name == "tucker_dlinear":
        # Mirror train.py: AdamW with per-group LR / WD for the G core vs
        # the factor matrices, and grad_clip=1.0. lr_g_mult / wd_g_mult
        # are taken from the grid (default 1.0 = uniform).
        g_params, other_params = [], []
        for pname, p in adapter.named_parameters():
            (g_params if pname.endswith(".G") else other_params).append(p)
        lr_g = lr * float(opt_kw.get("lr_g_mult", 1.0))
        wd_g = wd * float(opt_kw.get("wd_g_mult", 1.0))
        optimizer = torch.optim.AdamW(
            [{"params": other_params, "lr": lr,   "weight_decay": wd},
             {"params": g_params,     "lr": lr_g, "weight_decay": wd_g}],
        )
        grad_clip = 1.0
    elif name in ("hot", "patchtst"):
        # Mirror train.py: AdamW + grad_clip=1.0. AdamW's decoupled weight
        # decay matters for transformers (Adam couples WD through the
        # second-moment normalisation in ways that destabilise attention),
        # and grad_clip is the standard transformer init stabiliser.
        lr_g = None
        wd_g = None
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=wd)
        grad_clip = 1.0
    else:
        lr_g = None
        wd_g = None
        optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=wd)
        grad_clip = None

    epochs     = int(trainer_kw.get("epochs",     EPOCHS))
    patience   = int(trainer_kw.get("patience",   PATIENCE))
    min_epochs = int(trainer_kw.get("min_epochs", MIN_EPOCHS))
    batch_size = int(trainer_kw.get("batch_size", BATCH_SIZE))

    log_path = os.path.join(combo_dir, "train_log.csv")
    log_f    = open(log_path, "w", newline="")
    log_w    = csv.writer(log_f)
    log_w.writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_time_s"])

    best_val   = float("inf")
    best_train = float("nan")
    best_epoch = 0
    best_state: dict | None = None
    last_epoch = 0

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        tr_loss, _, _ = _epoch(adapter, Xtr, Ytr, batch_size, device,
                               optimizer=optimizer, generator=gen,
                               grad_clip=grad_clip)
        va_loss, _, _ = _epoch(adapter, Xva, Yva, batch_size, device, optimizer=None)
        dt = time.time() - t0

        improved = va_loss < best_val
        if improved:
            best_val   = va_loss
            best_train = tr_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in adapter.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"     epoch {epoch:3d}/{epochs}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  "
                  f"best_val={best_val:.6f} (@{best_epoch})  ({dt:.1f}s)")
        log_w.writerow([epoch, f"{tr_loss:.8f}", f"{va_loss:.8f}",
                        f"{lr:.8g}", f"{dt:.3f}"])
        log_f.flush()

        if epoch >= min_epochs and (epoch - best_epoch) >= patience:
            break

    log_f.close()

    config = {
        "model":           name,
        "pred_len":        pred_len,
        "lookback":        LOOKBACK,
        "combo":           combo,
        "model_kwargs":    resolved,
        "optimizer":       "AdamW" if name in ("tucker_dlinear", "hot", "patchtst") else "Adam",
        "lr":              lr,
        "weight_decay":    wd,
        "lr_g":            lr_g,
        "wd_g":            wd_g,
        "grad_clip":       grad_clip,
        "epochs":          epochs,
        "patience":        patience,
        "min_epochs":      min_epochs,
        "batch_size":      batch_size,
        "n_params":        n_params,
        "seed":            seed,
        "device":          str(device),
        "stop_epoch":      int(last_epoch),
        "best_epoch":      int(best_epoch),
        "best_val_loss":   float(best_val),
        "best_train_loss": float(best_train),
    }
    with open(os.path.join(combo_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    return config, best_state


# ─── Test-set evaluation for a newly-promoted winner ─────────────────────

def evaluate_winner_on_test(name: str, model_kwargs: dict,
                            best_state: dict, data: dict, pred_len: int,
                            device: torch.device) -> dict:
    """Re-build the model from the saved kwargs, load the best-val state,
    predict on the test set, and break the test MSE down by calendar year
    of each window's last-target date. Returns
        {"overall": float, "per_year": {year: float, ...}, "n_test": int}
    in the same standardised-log-IV space the loss is computed in.
    """
    grid = data["grid"]
    C    = data["rows"]["n_channels"]
    # build_model_for_tuning re-adds shape kwargs (seq_len, pred_len, W,
    # H, c_in, n_channels, etc.) from its arguments — the saved
    # model_kwargs already contains them after run_one_combo, so strip
    # them here to avoid duplicate-keyword errors.
    _SHAPE_KEYS = {
        "seq_len", "pred_len", "W", "H",
        "c_in", "n_channels", "num_nodes",
        "context_length", "prediction_length",
    }
    clean_mk = {k: v for k, v in model_kwargs.items()
                if k not in _SHAPE_KEYS}
    adapter, _ = build_model_for_tuning(
        name, clean_mk, pred_len, C, grid.n_tau, grid.n_money,
    )
    adapter.load_state_dict(best_state)
    adapter.to(device)
    adapter.eval()

    Xte, Yte = data["test"]
    preds = _predict(adapter, Xte, BATCH_SIZE, device)
    overall = float(((preds - Yte) ** 2).mean())

    dates = data.get("test_target_last_dates")
    per_year: dict[int, float] = {}
    n_per_year: dict[int, int]   = {}
    if dates is not None and len(dates) == len(Yte):
        years = pd.DatetimeIndex(dates).year.to_numpy()
        for yr in sorted(set(int(y) for y in years)):
            mask = years == yr
            if mask.sum() == 0:
                continue
            per_year[yr]   = float(((preds[mask] - Yte[mask]) ** 2).mean())
            n_per_year[yr] = int(mask.sum())

    return {
        "overall":    overall,
        "per_year":   per_year,
        "n_per_year": n_per_year,
        "n_test":     int(len(Yte)),
    }


# ─── Per-model sweep ──────────────────────────────────────────────────────

def run_sweep_for_model(name: str, pred_len: int, data: dict,
                        device: torch.device, seed: int,
                        overwrite: bool):
    grid_path = os.path.join(ROOT, MODEL_DIR[name], "tuning_grid.json")
    if not os.path.isfile(grid_path):
        raise SystemExit(f"Missing tuning grid: {grid_path}")
    with open(grid_path) as f:
        grid = json.load(f)

    task_dir = os.path.join(ROOT, MODEL_DIR[name], "tuning_results",
                            f"{LOOKBACK}_{pred_len}")
    if os.path.isdir(task_dir) and overwrite:
        shutil.rmtree(task_dir)
    os.makedirs(task_dir, exist_ok=True)

    pairs = expand_grid(grid, name, pred_len)
    by_variant: dict[str | None, list[dict]] = {}
    for v_name, combo in pairs:
        by_variant.setdefault(v_name, []).append(combo)

    n_variants = len(by_variant)
    print(f"\n{'=' * 72}")
    print(f"  {name.upper()}   pred_len={pred_len}   "
          f"variants={n_variants}   total combos={len(pairs)}")
    print(f"  out: {os.path.relpath(task_dir, ROOT)}/")
    print(f"{'=' * 72}")

    for v_name, combos in by_variant.items():
        if v_name is None:
            v_dir = task_dir
            label = "(no variant)"
        else:
            v_dir = os.path.join(task_dir, v_name)
            os.makedirs(v_dir, exist_ok=True)
            label = v_name

        # Resume support: keep already-trained combos whose params match
        # the new grid expansion. Only run combos that are genuinely new.
        existing = discover_existing(v_dir)
        existing_ids = sorted(
            int(cid.split("_")[1]) for cid, _, _ in existing.values()
        )
        next_id = (existing_ids[-1] + 1) if existing_ids else 0

        ew = find_existing_winner(v_dir, existing)
        if ew is None:
            winner_id, winner_val, winner_epoch = None, float("inf"), 0
        else:
            winner_id, winner_val, winner_epoch = ew

        n_reuse = sum(1 for c in combos if _signature(c) in existing)
        n_new   = len(combos) - n_reuse
        print(f"\n  ── variant: {label}   {len(combos)} combos "
              f"(reuse {n_reuse}, new {n_new}) ──")
        if winner_id is not None:
            print(f"     existing winner: {winner_id}  "
                  f"val={winner_val:.6f}  @ epoch {winner_epoch}")

        leaderboard: list[dict] = []

        for i, combo in enumerate(combos):
            sig = _signature(combo)
            cs  = ", ".join(f"{k}={v}" for k, v in combo.items())

            # Reuse a previously trained combo with matching params.
            if sig in existing:
                combo_id, score, cfg = existing[sig]
                print(f"\n  [{i + 1:>3}/{len(combos)}] {combo_id}  {cs}")
                print(f"     skipped (val={score:.6f}  "
                      f"@ epoch {cfg.get('best_epoch')})")
                leaderboard.append({
                    "combo_id":        combo_id,
                    "best_val_loss":   score,
                    "best_train_loss": cfg.get("best_train_loss"),
                    "best_epoch":      cfg.get("best_epoch"),
                    "stop_epoch":      cfg.get("stop_epoch"),
                    "n_params":        cfg.get("n_params"),
                    "status":          "reused",
                    **combo,
                })
                continue

            # Fresh combo: allocate next free id and train.
            combo_id  = f"combo_{next_id:04d}"
            next_id  += 1
            combo_dir = os.path.join(v_dir, combo_id)
            os.makedirs(combo_dir, exist_ok=True)
            print(f"\n  [{i + 1:>3}/{len(combos)}] {combo_id}  {cs}")

            try:
                config, best_state = run_one_combo(
                    name, pred_len, combo, data, device, seed, combo_dir,
                )
            except Exception as e:
                print(f"     FAILED: {type(e).__name__}: {e}")
                leaderboard.append({
                    "combo_id":        combo_id,
                    "best_val_loss":   None,
                    "best_train_loss": None,
                    "best_epoch":      None,
                    "n_params":        None,
                    "status":          "failed",
                    **combo,
                })
                continue

            print(f"     best val: {config['best_val_loss']:.6f}  "
                  f"@ epoch {config['best_epoch']}  "
                  f"(stopped at {config['stop_epoch']})")
            leaderboard.append({
                "combo_id":        combo_id,
                "best_val_loss":   config["best_val_loss"],
                "best_train_loss": config["best_train_loss"],
                "best_epoch":      config["best_epoch"],
                "stop_epoch":      config["stop_epoch"],
                "n_params":        config["n_params"],
                "status":          "trained",
                **combo,
            })

            # Promote running winner (only new combos can win — existing
            # non-winners don't have a .pt to fall back on, but by
            # construction their val_loss is ≥ the existing winner's, so
            # they can't beat the running winner anyway).
            if config["best_val_loss"] < winner_val:
                if winner_id is not None:
                    old_pt = os.path.join(v_dir, winner_id, "best_model.pt")
                    if os.path.isfile(old_pt):
                        os.remove(old_pt)
                if best_state is not None:
                    torch.save(best_state,
                               os.path.join(combo_dir, "best_model.pt"))
                winner_id    = combo_id
                winner_val   = config["best_val_loss"]
                winner_epoch = config["best_epoch"]
                print(f"     ** new running winner ({combo_id}, "
                      f"val={winner_val:.6f}) **")

                # Test-set evaluation: overall + per-year. Print only
                # (does not feed back into selection — winners are still
                # chosen on val loss). Failures here must not crash the
                # sweep; they're logged loudly and we continue.
                if best_state is not None:
                    try:
                        tm = evaluate_winner_on_test(
                            name, config["model_kwargs"], best_state,
                            data, pred_len, device,
                        )
                        per_y_str = "  ".join(
                            f"{y}: {mse:.6f} (n={tm['n_per_year'].get(y, 0)})"
                            for y, mse in sorted(tm["per_year"].items())
                        ) or "(no date info)"
                        print(f"     test MSE overall: {tm['overall']:.6f}  "
                              f"(n={tm['n_test']})")
                        print(f"     test MSE per year:  {per_y_str}")
                        # Persist alongside the winner's config.json.
                        config["test_metrics"] = tm
                        with open(os.path.join(combo_dir,
                                               "config.json"), "w") as f:
                            json.dump(config, f, indent=2, default=str)
                        # Also record on the leaderboard row.
                        leaderboard[-1]["test_mse"] = tm["overall"]
                        leaderboard[-1]["test_mse_per_year"] = tm["per_year"]
                    except Exception as e:
                        print(f"     test eval FAILED: "
                              f"{type(e).__name__}: {e}")

        # Variant summary.
        scored = [r for r in leaderboard if r["best_val_loss"] is not None]
        ranked = sorted(scored, key=lambda r: r["best_val_loss"])
        summary = {
            "model":        name,
            "pred_len":     pred_len,
            "lookback":     LOOKBACK,
            "variant":      v_name,
            "n_combos":     len(combos),
            "n_succeeded":  len(scored),
            "n_failed":     len(combos) - len(scored),
            "data_end":     data.get("data_end"),
            "first_date":   data.get("first_date"),
            "last_date":    data.get("last_date"),
            "test_first_target_date": data.get("test_first_target_date"),
            "winner": (None if winner_id is None else {
                "combo_id":      winner_id,
                "best_val_loss": winner_val,
                "best_epoch":    winner_epoch,
                "config_path":   os.path.relpath(
                    os.path.join(v_dir, winner_id, "config.json"), ROOT),
                "checkpoint":    os.path.relpath(
                    os.path.join(v_dir, winner_id, "best_model.pt"), ROOT),
            }),
            "ranking":      ranked,
            "leaderboard":  leaderboard,
        }
        with open(os.path.join(v_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if winner_id is None:
            print(f"\n  variant {label}: every combo failed.")
        else:
            print(f"\n  variant {label} winner: {winner_id}  "
                  f"val={winner_val:.6f}  @ epoch {winner_epoch}")


# ─── Entry point ──────────────────────────────────────────────────────────

def parse_models(arg: str) -> list[str]:
    if arg == "all":
        return list(TUNABLE)
    names = [s.strip() for s in arg.split(",") if s.strip()]
    bad = [n for n in names if n not in TUNABLE]
    if bad:
        raise SystemExit(
            f"Unknown / untunable model(s): {bad}. "
            f"Pick from {TUNABLE} or 'all'.")
    return names


def main():
    ap = argparse.ArgumentParser(
        description="Sweep <ModelDir>/tuning_grid.json on SPX_surfaces.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help='Comma-separated names from '
                         f'{TUNABLE} or "all".')
    ap.add_argument("--pred_len", required=True, type=int, choices=(5, 21, 63))
    ap.add_argument("--csv_path",   default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--data_end",   type=str,   default="2023-12-29",
                    help="Drop CSV rows with date > this. "
                         "Pass 'none' to keep all rows.")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--overwrite",  action="store_true",
                    help="Wipe each model's tuning_results/63_<pred_len>/ "
                         "before starting. Without this flag, combos whose "
                         "params already exist on disk are reused and only "
                         "new grid additions are trained.")
    args = ap.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must be < 1.")

    names = parse_models(args.model)

    device = pick_device()
    print(f"device: {device}")
    print(f"loading {os.path.relpath(args.csv_path, ROOT)} ...")
    data_end = None if args.data_end.lower() == "none" else args.data_end
    data = load_dataset(args.csv_path, args.train_frac, args.val_frac,
                        LOOKBACK, args.pred_len, data_end=data_end)
    g, r = data["grid"], data["rows"]
    print(f"  rows: N={r['N']}  train_end={r['train_end']}  "
          f"val_end={r['val_end']}  C={r['n_channels']}  "
          f"grid={g.n_tau}×{g.n_money} (tau×moneyness)")
    print(f"  windows: train={data['train'][0].shape[0]}  "
          f"val={data['val'][0].shape[0]}  test={data['test'][0].shape[0]}  "
          f"(test held out; never used in tuning)")
    print(f"  test starts predicting at: {data['test_first_target_date']}")

    # Compute per-window last-target date for the test set so each
    # newly-promoted winner can be evaluated overall + per-year.
    df = pd.read_csv(args.csv_path)
    if data_end is not None:
        df = df[df["date"] <= data_end].reset_index(drop=True)
    all_dates = pd.to_datetime(df["date"].to_numpy())
    L, P = LOOKBACK, args.pred_len
    starts = np.arange(r["N"] - L - P + 1)
    target_end = starts + L + P
    test_starts = starts[target_end > r["val_end"]]
    data["test_target_last_dates"] = all_dates[test_starts + L + P - 1]

    for name in names:
        run_sweep_for_model(name, args.pred_len, data, device,
                            args.seed, args.overwrite)


if __name__ == "__main__":
    main()
