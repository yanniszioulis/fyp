#!/usr/bin/env python3
"""
train.py — train one (or all) IV-surface forecasters on SPX_surfaces.csv.

Models
------
Deep:  dlinear, patchtst, hot, tucker_dlinear, gwn
       Each uses its own __init__ defaults; lr is read from the LR_*
       constants below. Shared trainer: Adam, MSE on standardized log-IV,
       max EPOCHS, early stop with PATIENCE (suppressed until MIN_EPOCHS).
Stat:  var (lag=1, OLS, no training loop).

`--model all` runs the deep models sequentially. VAR must be invoked
explicitly with `--model var`.

Data
----
Columns of SPX_surfaces.csv shaped iv_{moneyness}_{tau} are parsed into a
[H=tau × W=moneyness] grid (here 10 × 15 = 150 cells). All IV values are
taken in log space. A per-channel StandardScaler is fit on the training
rows; the same transform is applied to val/test. Train/val/test split is
chronological by window-end position (default 80/10/10).

Output
------
Deep models:  <ModelDir>/63_<pred_len>/<UTC-timestamp>/
                  hyperparams.json
                  metrics_test.json
                  train_log.csv
                  preds.npy            (N_test, pred_len, n_channels)
VAR:          VAR/results/63_<pred_len>/
                  hyperparams.json
                  metrics_test.json
                  preds.npy

Preds and stats are in standardized log-IV space (the training space).
hyperparams.json stores the scaler mean/scale so preds can be inverted
back to IV later.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Make the per-model packages importable as top-level modules.
ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("DLinear", "PatchTST", "HOT", "Tucker_DLinear", "GWN", "VAR"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from dlinear import DLinear                # noqa: E402
from patchtst import PatchTST              # noqa: E402
from hot import HOT                        # noqa: E402
from tucker_dlinear import TuckerDLinear   # noqa: E402
from gwn import GWN                        # noqa: E402
from var import fit_var_p, make_step_window_fn  # noqa: E402


# ─── User-editable per-model optimiser settings ───────────────────────────
# Edit these to set the lr / weight_decay used at training time. Adam
# optimiser; all other training knobs (epochs, patience, min_epochs) are
# shared, see below.
LR_DLINEAR        = 4e-3
LR_PATCHTST       = 1e-3
LR_HOT            = 1e-3
LR_TUCKER_DLINEAR = 1e-3
LR_GWN            = 1e-3

WD_DLINEAR        = 0.0
WD_PATCHTST       = 1e-4
WD_HOT            = 0.1
WD_TUCKER_DLINEAR = 1e-1
WD_GWN            = 1e-3

# Tucker-only (with AdamW): the G core gets its own multipliers on top of
# LR_TUCKER_DLINEAR / WD_TUCKER_DLINEAR. The factor matrices stay at the
# base values. Lower G LR dampens gauge-direction noise; higher G WD
# regularises the dominant (overfit-prone) parameter group.
# Set both to 1.0 to apply the base values uniformly.
LR_TUCKER_DLINEAR_G_MULT = 0.25
WD_TUCKER_DLINEAR_G_MULT = 10

# Shared trainer settings (same for every deep model).
EPOCHS     = 100
PATIENCE   = 15
MIN_EPOCHS = 15
BATCH_SIZE = 32

LOOKBACK   = 63   # fixed across the project
VALID_PRED_LEN = (5, 21, 63)
DEEP_MODELS    = ("dlinear", "patchtst", "hot", "tucker_dlinear", "gwn")

# Folder names per model (where outputs land relative to repo root).
MODEL_DIR = {
    "dlinear":        "DLinear",
    "patchtst":       "PatchTST",
    "hot":            "HOT",
    "tucker_dlinear": "Tucker_DLinear",
    "gwn":            "GWN",
    "var":            "VAR",
}


# ─── Device pick: cuda → mps → cpu ────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Data loading ─────────────────────────────────────────────────────────

@dataclass
class GridSpec:
    n_tau: int          # H
    n_money: int        # W
    iv_cols: list       # ordered column names matching reshape [H, W]
    tau_vals: list
    money_vals: list


def parse_grid(columns: list[str]) -> GridSpec:
    """Parse `iv_{moneyness}_{tau}` columns into a (tau × moneyness) grid.

    Column order in the reshape is set to (tau outer, moneyness inner) so
    that `data[:, 150].reshape(N, H=n_tau, W=n_money)` matches the CSV
    column ordering when iv_cols is laid out in that order.
    """
    parsed = []
    for c in columns:
        if not c.startswith("iv_"):
            continue
        _, m, t = c.split("_", 2)
        parsed.append((c, float(m), float(t)))
    money_vals = sorted({m for _, m, _ in parsed})
    tau_vals   = sorted({t for _, _, t in parsed})
    # Layout iv_cols as tau-outer, moneyness-inner.
    lookup = {(m, t): name for name, m, t in parsed}
    iv_cols = [lookup[(m, t)] for t in tau_vals for m in money_vals]
    return GridSpec(
        n_tau=len(tau_vals), n_money=len(money_vals),
        iv_cols=iv_cols, tau_vals=tau_vals, money_vals=money_vals,
    )


def load_dataset(csv_path: str, train_frac: float, val_frac: float,
                 lookback: int, pred_len: int, data_end: str | None = None):
    """Return windowed train/val/test tensors and the per-channel scaler.

    Steps
    -----
    1. Read CSV, optionally truncate to rows with date <= data_end,
       parse the (tau × moneyness) grid, take log of IV values.
    2. Row-level split: first train_frac rows define the scaler-fit domain.
    3. StandardScaler per channel, fit on train rows, applied globally.
    4. Build sliding windows (input=L, target=P) and split each window
       into train/val/test by the position of its *target end*. Train
       windows therefore have every value within the scaler's fit domain
       (no leakage).
    """
    df = pd.read_csv(csv_path)
    if data_end is not None:
        n_before = len(df)
        df = df[df["date"] <= data_end].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows with date <= {data_end}.")
        print(f"  truncated to date <= {data_end}: "
              f"{n_before} → {len(df)} rows  "
              f"(last kept: {df['date'].iloc[-1]})")
    grid = parse_grid(df.columns.tolist())
    raw = df[grid.iv_cols].to_numpy(dtype=np.float64)  # [N, C]
    if not np.all(raw > 0):
        raise ValueError("Non-positive IV values; log is undefined.")
    log_iv = np.log(raw).astype(np.float32)

    N, C = log_iv.shape
    train_end = int(N * train_frac)
    val_end   = int(N * (train_frac + val_frac))

    mean  = log_iv[:train_end].mean(axis=0)            # [C]
    std   = log_iv[:train_end].std(axis=0) + 1e-12     # [C]
    scaled = (log_iv - mean) / std                     # [N, C]

    L, P = lookback, pred_len
    n_win = N - L - P + 1
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({N}) for L={L}, P={P}.")
    starts = np.arange(n_win)
    target_end = starts + L + P    # exclusive

    train_mask = target_end <= train_end
    val_mask   = (target_end > train_end) & (target_end <= val_end)
    test_mask  = target_end > val_end

    def stack(idx):
        X = np.stack([scaled[s : s + L]         for s in idx], axis=0)
        Y = np.stack([scaled[s + L : s + L + P] for s in idx], axis=0)
        return X, Y

    Xtr, Ytr = stack(starts[train_mask])
    Xva, Yva = stack(starts[val_mask])
    Xte, Yte = stack(starts[test_mask])

    dates = df["date"].tolist()
    test_starts = starts[test_mask]
    test_first_target_date = (dates[int(test_starts[0]) + L]
                              if test_starts.size else None)

    return {
        "grid":   grid,
        "scaler": {"mean": mean.astype(np.float32),
                   "std":  std.astype(np.float32)},
        "rows":   {"N": N, "train_end": train_end, "val_end": val_end,
                   "n_channels": C},
        "data_end": data_end,
        "first_date": str(df["date"].iloc[0]),
        "last_date":  str(df["date"].iloc[-1]),
        "test_first_target_date": test_first_target_date,
        "train":  (Xtr, Ytr),
        "val":    (Xva, Yva),
        "test":   (Xte, Yte),
        "scaled_log_iv": scaled,  # full series for VAR
    }


# ─── Model adapters ───────────────────────────────────────────────────────
# Each adapter wraps a model so that its forward sees its native input
# shape, and outputs are reshaped back to [B, P, C] (the trainer's space).

class _Adapter(nn.Module):
    """Shared interface: input/output in [B, T, C] (the canonical space)."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _DLinearAdapter(_Adapter):
    def forward(self, x):           # [B, L, C] → [B, P, C]
        return self.model(x)


class _PatchTSTAdapter(_Adapter):
    def forward(self, x):           # [B, L, C] → [B, P, C]
        return self.model(x)


class _HOTAdapter(_Adapter):
    def __init__(self, model, n_tau, n_money):
        super().__init__(model)
        self.H, self.W = n_tau, n_money

    def forward(self, x):
        # x: [B, L, C=H*W] → [B, H, W, L] → model → [B, H, W, P] → [B, P, C]
        B, L, C = x.shape
        z = x.reshape(B, L, self.H, self.W).permute(0, 2, 3, 1).contiguous()
        z = self.model(z)
        return z.permute(0, 3, 1, 2).reshape(B, -1, C)


class _TuckerAdapter(_Adapter):
    def __init__(self, model, n_tau, n_money):
        super().__init__(model)
        # Tucker takes [B, L, W, H]; we have [B, L, H, W] from the reshape.
        self.H, self.W = n_tau, n_money

    def forward(self, x):
        # x: [B, L, C=H*W] → [B, L, H, W] → permute to [B, L, W, H]
        B, L, C = x.shape
        z = x.reshape(B, L, self.H, self.W).permute(0, 1, 3, 2).contiguous()
        z = self.model(z)        # [B, P, W, H]
        return z.permute(0, 1, 3, 2).reshape(B, -1, C)


class _GWNAdapter(_Adapter):
    def forward(self, x):
        # x: [B, L, C] → [B, 1, C, L] → model → [B, P, C, 1] → [B, P, C]
        B, L, C = x.shape
        z = x.permute(0, 2, 1).unsqueeze(1)
        z = self.model(z)
        return z.squeeze(-1)


def build_adapter_from_kwargs(name: str, model_kwargs: dict,
                              n_tau: int, n_money: int) -> nn.Module:
    """Construct an adapter-wrapped model from explicit model_kwargs
    (no defaults applied). Used by --from_winner to reproduce a tuning
    combo exactly. Mirrors evaluate.rebuild_adapter."""
    if name == "dlinear":
        return _DLinearAdapter(DLinear(**model_kwargs))
    if name == "patchtst":
        return _PatchTSTAdapter(PatchTST(**model_kwargs))
    if name == "hot":
        return _HOTAdapter(HOT(**model_kwargs), n_tau, n_money)
    if name == "tucker_dlinear":
        return _TuckerAdapter(TuckerDLinear(**model_kwargs), n_tau, n_money)
    if name == "gwn":
        return _GWNAdapter(GWN(**model_kwargs))
    raise ValueError(f"Unknown model: {name}")


def load_winner_config(name: str, pred_len: int) -> dict:
    """Read <ModelDir>/tuning_results/<lookback>_<pred_len>/summary.json,
    locate the winner combo, and return its config.json dict augmented
    with `_winner_combo` and `_winner_source` (path relative to ROOT)."""
    summary_path = os.path.join(
        ROOT, MODEL_DIR[name], "tuning_results",
        f"{LOOKBACK}_{pred_len}", "summary.json",
    )
    if not os.path.isfile(summary_path):
        raise SystemExit(
            f"--from_winner: no tuning summary at "
            f"{os.path.relpath(summary_path, ROOT)} "
            f"(run hyperparameter_tuning.py first).")
    with open(summary_path) as f:
        summary = json.load(f)
    winner = summary.get("winner")
    if not winner:
        raise SystemExit(
            f"--from_winner: no winner in {os.path.relpath(summary_path, ROOT)}")
    cfg_path = os.path.join(ROOT, winner["config_path"])
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["_winner_combo"]  = winner["combo_id"]
    cfg["_winner_source"] = os.path.relpath(cfg_path, ROOT)
    return cfg


def build_model(name: str, pred_len: int, n_channels: int,
                n_tau: int, n_money: int) -> tuple[nn.Module, dict]:
    """Construct a model using its own __init__ defaults. Returns
    (adapter, resolved_kwargs)."""
    L, P, C = LOOKBACK, pred_len, n_channels
    if name == "dlinear":
        # Matches the prior tuning winner (combo_0044): kernel_size=31.
        kw = dict(seq_len=L, pred_len=P, n_channels=C, kernel_size=31)
        m = DLinear(**kw)
        return _DLinearAdapter(m), {**kw, "revin": False,
                                    "revin_affine": False, "revin_eps": 1e-5}
    if name == "patchtst":
        # Matches the prior tuning winner (combo_0001): patch_len=stride=7,
        # n_heads=2, d_ff=64, dropout=0.1, head_dropout=0.01, revin=False.
        kw = dict(
            c_in=C, seq_len=L, pred_len=P,
            patch_len=7, stride=7, d_model=32, n_heads=2,
            n_layers=2, d_ff=64, attn_dropout=0.0, dropout=0.1,
            head_dropout=0.01, revin=False, padding_patch="end",
        )
        m = PatchTST(**kw)
        return _PatchTSTAdapter(m), kw
    if name == "hot":
        # Typical small HOT that lives in the current tuning grid (one
        # representative point per axis: middle-of-grid). Lets us smoke-
        # test HOT locally on MPS at a fast size before kicking off the
        # full sweep.
        kw = dict(
            context_length=L, prediction_length=P,
            d_hidden=8, n_blocks=1, n_head=1, patch_size=7,
            attention_type="kronecker_product",
            dropout=0.1, attn_dropout=0.0, head_dropout=0.1,
            pe="rope", norm=False, head_type="mean",
        )
        m = HOT(**kw)
        return _HOTAdapter(m, n_tau, n_money), kw
    if name == "tucker_dlinear":
        # Current best config, validated via multi_seed (94,660 params,
        # val mean 0.165, test mean 0.226, ~tied with DLinear within seed
        # noise). Full spatial rank with very low temporal rank (rL=1,
        # rP=2) — partial-rank-aware orthonormal init + AdamW with
        # per-group LR/WD (lr_g_mult=0.25, wd_g_mult=10) + grad_clip=1.0
        # are wired up automatically in train_deep_model.
        kw = dict(
            seq_len=L, pred_len=P, W=n_money, H=n_tau,
            rank_L=1, rank_P=min(2, P), rank_W=n_money, rank_H=n_tau,
            kernel_size=31, norm=False, tie_spatial=False,
        )
        m = TuckerDLinear(**kw)
        return _TuckerAdapter(m, n_tau, n_money), kw
    if name == "gwn":
        # Small GWN starting point (~40K params, vs ~6M with the paper's
        # defaults). The reference defaults assume traffic-prediction-
        # scale graphs; for our 150-node IV surface, end_conv_1 =
        # Conv2d(skip, end, (1, end_kernel)) at (256, 512, 51) was the
        # entire 6M parameter count by itself. Shrinking skip→16, end→32
        # and collapsing to blocks=2, layers=1 puts the model in the same
        # ballpark as the small DLinear / Tucker / HOT candidates.
        kw = dict(
            num_nodes=C, seq_len=L, pred_len=P,
            in_dim=1, supports=None,
            gcn_bool=True, addaptadj=True, aptinit=None,
            residual_channels=16, dilation_channels=16,
            skip_channels=16, end_channels=32,
            kernel_size=2, blocks=2, layers=1,
            dropout=0.3,
        )
        m = GWN(**kw)
        return _GWNAdapter(m), kw
    raise ValueError(f"Unknown model: {name}")


LR_BY_MODEL = {
    "dlinear":        LR_DLINEAR,
    "patchtst":       LR_PATCHTST,
    "hot":            LR_HOT,
    "tucker_dlinear": LR_TUCKER_DLINEAR,
    "gwn":            LR_GWN,
}

WD_BY_MODEL = {
    "dlinear":        WD_DLINEAR,
    "patchtst":       WD_PATCHTST,
    "hot":            WD_HOT,
    "tucker_dlinear": WD_TUCKER_DLINEAR,
    "gwn":            WD_GWN,
}


# ─── Training loop ────────────────────────────────────────────────────────

def _iter_batches(X: np.ndarray, Y: np.ndarray, batch: int,
                  shuffle: bool, device: torch.device, generator=None):
    n = X.shape[0]
    idx = (torch.randperm(n, generator=generator).numpy()
           if shuffle else np.arange(n))
    for s in range(0, n, batch):
        sel = idx[s : s + batch]
        xb = torch.from_numpy(X[sel]).to(device)
        yb = torch.from_numpy(Y[sel]).to(device)
        yield xb, yb


def _epoch(model, X, Y, batch, device, optimizer=None, generator=None,
           grad_clip=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in _iter_batches(X, Y, batch, shuffle=train,
                                    device=device, generator=generator):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = xb.shape[0]
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)


def _predict(model, X, batch, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[s : s + batch]).to(device)
            out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0,))


def train_deep_model(name: str, data: dict, pred_len: int,
                     device: torch.device, seed: int,
                     batch_size: int = BATCH_SIZE,
                     winner_cfg: dict | None = None):
    """Train one deep model with the shared trainer; save artefacts.

    If `winner_cfg` is provided (the saved config.json from a tuning
    winner), its `model_kwargs`, optimizer choice, `lr`, `weight_decay`,
    `lr_g`, `wd_g`, `grad_clip`, `min_epochs`, and `batch_size` override
    the train.py defaults. Used by multi_seed.py --from_winner.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    grid = data["grid"]
    C    = data["rows"]["n_channels"]
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]
    Xte, Yte = data["test"]

    # Resolve model + trainer knobs. winner_cfg, if given, fully drives
    # them so we reproduce the tuning combo exactly (same optimizer,
    # same grad_clip policy, same min_epochs).
    if winner_cfg is None:
        adapter, resolved = build_model(
            name, pred_len, C, grid.n_tau, grid.n_money,
        )
        lr = LR_BY_MODEL[name]
        wd = WD_BY_MODEL[name]
        min_epochs = MIN_EPOCHS
        # AdamW for the channel-independent / transformer / Tucker
        # families; Adam for GWN. Grad-clip on the same set.
        use_adamw = name in ("tucker_dlinear", "hot", "patchtst", "dlinear")
        grad_clip = 1.0 if use_adamw else None
        if name == "tucker_dlinear":
            # G core gets its own multipliers; factor matrices stay at
            # base lr / wd.
            lr_g = lr * LR_TUCKER_DLINEAR_G_MULT
            wd_g = wd * WD_TUCKER_DLINEAR_G_MULT
        else:
            lr_g = wd_g = None
    else:
        resolved = winner_cfg["model_kwargs"]
        adapter  = build_adapter_from_kwargs(
            name, resolved, grid.n_tau, grid.n_money,
        )
        lr  = float(winner_cfg["lr"])
        wd  = float(winner_cfg["weight_decay"])
        min_epochs = int(winner_cfg["min_epochs"])
        batch_size = int(winner_cfg["batch_size"])
        use_adamw  = (winner_cfg.get("optimizer") == "AdamW")
        gc = winner_cfg.get("grad_clip")
        grad_clip = float(gc) if gc is not None else None
        lr_g = winner_cfg.get("lr_g")
        wd_g = winner_cfg.get("wd_g")
        if lr_g is not None:
            lr_g = float(lr_g)
        if wd_g is not None:
            wd_g = float(wd_g)

    adapter.to(device)
    n_params = sum(p.numel() for p in adapter.parameters())

    if name == "tucker_dlinear" and lr_g is not None:
        g_params, other_params = [], []
        for pname, p in adapter.named_parameters():
            (g_params if pname.endswith(".G") else other_params).append(p)
        optimizer = torch.optim.AdamW(
            [{"params": other_params, "lr": lr,   "weight_decay": wd},
             {"params": g_params,     "lr": lr_g, "weight_decay": wd_g}],
        )
        opt_name = "AdamW"
    else:
        opt_cls  = torch.optim.AdamW if use_adamw else torch.optim.Adam
        optimizer = opt_cls(adapter.parameters(), lr=lr, weight_decay=wd)
        opt_name = "AdamW" if use_adamw else "Adam"

    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = os.path.join(ROOT, MODEL_DIR[name],
                           f"{LOOKBACK}_{pred_len}", ts)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{name}  pred_len={pred_len}  device={device}]")
    print(f"  windows: train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")
    print(f"  params:  {n_params:,}")
    lr_str = f"lr={lr}" + (f" (lr_G={lr_g:.4g})" if lr_g is not None else "")
    wd_str = f"wd={wd}" + (f" (wd_G={wd_g:.4g})" if wd_g is not None else "")
    print(f"  {opt_name}  {lr_str}  {wd_str}  batch={batch_size}  "
          f"epochs<={EPOCHS}  patience={PATIENCE} (min_epochs={min_epochs})"
          f"{f'  grad_clip={grad_clip}' if grad_clip is not None else ''}")
    if winner_cfg is not None:
        print(f"  from_winner: {winner_cfg['_winner_source']} "
              f"(combo={winner_cfg['_winner_combo']})")
    print(f"  out:     {os.path.relpath(out_dir, ROOT)}")

    log_path = os.path.join(out_dir, "train_log.csv")
    log_f    = open(log_path, "w", newline="")
    log_w    = csv.writer(log_f)
    log_w.writerow(["epoch", "train_loss", "val_loss", "lr",
                    "epoch_time_s"])

    best_val   = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone()
                  for k, v in adapter.state_dict().items()}
    stop_epoch = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = _epoch(adapter, Xtr, Ytr, batch_size, device,
                         optimizer=optimizer, generator=gen,
                         grad_clip=grad_clip)
        va_loss = _epoch(adapter, Xva, Yva, batch_size, device,
                         optimizer=None)
        dt = time.time() - t0
        improved = va_loss < best_val
        if improved:
            best_val   = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in adapter.state_dict().items()}

        marker = "  [best]" if improved else ""
        print(f"  epoch {epoch:3d}/{EPOCHS}  "
              f"train={tr_loss:.6f}  val={va_loss:.6f}  "
              f"({dt:.1f}s){marker}")
        log_w.writerow([epoch, f"{tr_loss:.8f}", f"{va_loss:.8f}",
                        f"{lr:.8g}", f"{dt:.3f}"])
        log_f.flush()

        # Early stop only after min_epochs.
        if epoch >= min_epochs and (epoch - best_epoch) >= PATIENCE:
            stop_epoch = epoch
            print(f"  early stop at epoch {epoch} "
                  f"(best val={best_val:.6f} @ epoch {best_epoch})")
            break

    log_f.close()
    if stop_epoch is None:
        stop_epoch = EPOCHS
        print(f"  finished {EPOCHS} epochs (best val={best_val:.6f} "
              f"@ epoch {best_epoch})")

    # Restore best weights for the test pass.
    adapter.load_state_dict(best_state)
    preds_te = _predict(adapter, Xte, batch_size, device)   # [N, P, C]
    np.save(os.path.join(out_dir, "preds.npy"), preds_te.astype(np.float32))

    mse = float(np.mean((preds_te - Yte) ** 2))
    mae = float(np.mean(np.abs(preds_te - Yte)))
    rmse = float(np.sqrt(mse))
    stats = {
        "model":        name,
        "pred_len":     pred_len,
        "lookback":     LOOKBACK,
        "n_test":       int(Xte.shape[0]),
        "best_val_mse": float(best_val),
        "best_epoch":   best_epoch,
        "stop_epoch":   stop_epoch,
        "test_mse":     mse,
        "test_rmse":    rmse,
        "test_mae":     mae,
        "space":        "standardized_log_iv",
    }
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(stats, f, indent=2)

    hyper = {
        "model":          name,
        "model_kwargs":   resolved,
        "optimizer":      opt_name,
        "lr":             lr,
        "weight_decay":   wd,
        "lr_g":           lr_g,
        "wd_g":           wd_g,
        "grad_clip":      grad_clip,
        "epochs":         EPOCHS,
        "patience":       PATIENCE,
        "min_epochs":     min_epochs,
        "batch_size":     batch_size,
        "lookback":       LOOKBACK,
        "pred_len":       pred_len,
        "seed":           seed,
        "device":         str(device),
        "n_params":       n_params,
        "winner_source":  (winner_cfg["_winner_source"]
                           if winner_cfg is not None else None),
        "winner_combo":   (winner_cfg["_winner_combo"]
                           if winner_cfg is not None else None),
        "data_end":       data.get("data_end"),
        "first_date":     data.get("first_date"),
        "last_date":      data.get("last_date"),
        "test_first_target_date": data.get("test_first_target_date"),
        "grid": {
            "n_tau":      grid.n_tau,
            "n_money":    grid.n_money,
            "tau_vals":   grid.tau_vals,
            "money_vals": grid.money_vals,
        },
        "scaler": {
            "space":      "log_iv",
            "mean":       data["scaler"]["mean"].tolist(),
            "std":        data["scaler"]["std"].tolist(),
            "channel_order": grid.iv_cols,
        },
    }
    with open(os.path.join(out_dir, "hyperparams.json"), "w") as f:
        json.dump(hyper, f, indent=2)

    print(f"  test: mse={mse:.6f}  rmse={rmse:.6f}  mae={mae:.6f}")
    print(f"  saved → {os.path.relpath(out_dir, ROOT)}\n")


# ─── VAR (lag=1) ──────────────────────────────────────────────────────────

def train_var(data: dict, pred_len: int, seed: int):
    """Fit VAR(1) on standardized log-IV train series; forecast pred_len
    steps from each test window's last input row, save preds + stats."""
    np.random.seed(seed)

    grid = data["grid"]
    Xte, Yte = data["test"]   # Xte: [N, L, C], Yte: [N, P, C]

    # Fit on train rows only — same rows used to fit the scaler.
    train_end = data["rows"]["train_end"]
    full      = data["scaled_log_iv"]
    X_fit     = full[:train_end].astype(np.float64)

    c, A_list, _ = fit_var_p(X_fit, p=1)
    step = make_step_window_fn(c, A_list)

    # For each test window, use the last input row as the seed for an
    # iterative lag-1 rollout of length pred_len. With lag=1 the rest of
    # the lookback is unused but window alignment matches the deep models.
    N = Xte.shape[0]
    P = pred_len
    C = data["rows"]["n_channels"]
    preds = np.empty((N, P, C), dtype=np.float32)
    seed_rows = Xte[:, -1, :].astype(np.float64)     # [N, C]
    for i in range(N):
        cur = seed_rows[i].reshape(1, C)
        out = np.empty((P, C), dtype=np.float64)
        for h in range(P):
            nxt = step(cur)                          # [C]
            out[h] = nxt
            cur = nxt.reshape(1, C)
        preds[i] = out

    out_dir = os.path.join(ROOT, MODEL_DIR["var"], "results",
                           f"{LOOKBACK}_{pred_len}")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "preds.npy"), preds)

    mse  = float(np.mean((preds - Yte) ** 2))
    mae  = float(np.mean(np.abs(preds - Yte)))
    rmse = float(np.sqrt(mse))
    stats = {
        "model":     "var",
        "lag":       1,
        "pred_len":  pred_len,
        "lookback":  LOOKBACK,
        "n_test":    int(N),
        "test_mse":  mse,
        "test_rmse": rmse,
        "test_mae":  mae,
        "space":     "standardized_log_iv",
    }
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(stats, f, indent=2)

    hyper = {
        "model":     "var",
        "lag":       1,
        "fit_space": "standardized_log_iv",
        "lookback":  LOOKBACK,
        "pred_len":  pred_len,
        "seed":      seed,
        "n_train_rows": int(train_end),
        "data_end":     data.get("data_end"),
        "first_date":   data.get("first_date"),
        "last_date":    data.get("last_date"),
        "test_first_target_date": data.get("test_first_target_date"),
        "grid": {
            "n_tau":      grid.n_tau,
            "n_money":    grid.n_money,
            "tau_vals":   grid.tau_vals,
            "money_vals": grid.money_vals,
        },
        "scaler": {
            "space":         "log_iv",
            "mean":          data["scaler"]["mean"].tolist(),
            "std":           data["scaler"]["std"].tolist(),
            "channel_order": grid.iv_cols,
        },
    }
    with open(os.path.join(out_dir, "hyperparams.json"), "w") as f:
        json.dump(hyper, f, indent=2)

    print(f"[var  pred_len={pred_len}]")
    print(f"  fit on {train_end} train rows, K={C}")
    print(f"  test: mse={mse:.6f}  rmse={rmse:.6f}  mae={mae:.6f}")
    print(f"  saved → {os.path.relpath(out_dir, ROOT)}\n")


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train an IV-surface forecaster on SPX_surfaces.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    choices=(*DEEP_MODELS, "var", "all"))
    ap.add_argument("--pred_len", required=True, type=int,
                    choices=VALID_PRED_LEN)
    ap.add_argument("--csv_path", default=os.path.join(ROOT, "SPX_surfaces.csv"))
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--data_end",   type=str,   default="2023-12-29",
                    help="Drop CSV rows with date > this (YYYY-MM-DD). "
                         "Pass 'none' to keep all rows.")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--batch_size", type=int,   default=BATCH_SIZE,
                    help=f"Mini-batch size. Default: {BATCH_SIZE} "
                         "(the BATCH_SIZE constant in train.py). "
                         "Ignored when --from_winner is set.")
    ap.add_argument("--from_winner", action="store_true",
                    help="Override train.py defaults with the tuning "
                         "winner's config.json for (--model, --pred_len). "
                         "Reads <ModelDir>/tuning_results/63_<pred_len>/"
                         "summary.json. Not valid with --model var/all.")
    args = ap.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must be < 1 to leave a test split.")

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
          f"val={data['val'][0].shape[0]}  test={data['test'][0].shape[0]}")
    print(f"  test starts predicting at: {data['test_first_target_date']}\n")

    if args.from_winner and args.model in ("var", "all"):
        raise SystemExit("--from_winner requires a single deep model "
                         "(not 'var' or 'all').")

    if args.model == "var":
        train_var(data, args.pred_len, args.seed)
        return
    if args.model == "all":
        for name in DEEP_MODELS:
            train_deep_model(name, data, args.pred_len, device, args.seed,
                             batch_size=args.batch_size)
        return

    winner_cfg = (load_winner_config(args.model, args.pred_len)
                  if args.from_winner else None)
    train_deep_model(args.model, data, args.pred_len, device, args.seed,
                     batch_size=args.batch_size, winner_cfg=winner_cfg)


if __name__ == "__main__":
    main()
