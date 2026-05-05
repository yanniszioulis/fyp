#!/usr/bin/env python3
"""
DLinear training script for SPX IV surface forecasting.

Channel-independent DLinear (trend + seasonality decomposition), trained on
the SPX surface CSV using the canonical 70/10/20 split. The number of IV
cells (`n_iv`) is derived at runtime from columns matching `iv_*`. Inputs
are standardised with a single global (mean, std) fitted on the training
portion of the target-space data — pooled across time and all IV cells.
The model trains in scaled space; predictions are inverse-transformed and
saved in original target-space units.

Dataset selection:
    --dataset full      use the full CSV (default).
    --dataset precovid  slice to date <= 2019-12-31 before splitting.

Outputs (in --out_dir; default
`DLinear/results/{dataset}_SPX_IV_{seq_len}_{pred_len}_DLinear_individual_k{ks}_ep{ep}{loss_suffix}`):
    pred.npy          [N_test, pred_len, n_iv]   original-space predictions  (gitignored)
    start_dates.npy   [N_test]                   datetime64[D] start of each window
    train_log.csv     epoch, train_loss, val_loss
    best_model.pt     checkpoint of best validation weights
    config.json       full hyperparam + split + git record
"""

import argparse
import csv
import json
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FRAC           = 0.70
TEST_FRAC            = 0.20
PRECOVID_END         = "2019-12-31"
DATASET_CHOICES      = ["full", "precovid"]
TARGET_SPACE_CHOICES = ["level", "logdiff"]


# ─── Model ────────────────────────────────────────────────────────────────────

class _MovingAvg(nn.Module):
    """Boundary-padded 1-D moving average preserving sequence length."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, C]
        x = torch.cat([
            x[:, :1].expand(-1, self.pad, -1),
            x,
            x[:, -1:].expand(-1, self.pad, -1),
        ], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """
    Channel-independent DLinear.

    Each IV feature gets its own pair of linear maps (seasonal and trend).
    Vectorised batched matmul over channels — equivalent to n_channels
    independent nn.Linear layers but ~100× faster than a ModuleList loop.

    Input:  [B, seq_len, C]
    Output: [B, pred_len, C]
    """
    def __init__(self, seq_len: int, pred_len: int, n_channels: int, kernel_size: int = 13):
        super().__init__()
        self.decomp = _MovingAvg(kernel_size)
        w0 = (1.0 / seq_len) * torch.ones(n_channels, pred_len, seq_len)
        self.W_s = nn.Parameter(w0.clone())
        self.W_t = nn.Parameter(w0.clone())
        self.b_s = nn.Parameter(torch.zeros(n_channels, pred_len))
        self.b_t = nn.Parameter(torch.zeros(n_channels, pred_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, S, C]
        trend = self.decomp(x)
        seas  = x - trend
        s = seas.permute(0, 2, 1)
        t = trend.permute(0, 2, 1)
        out = (torch.einsum('bcs,cps->bcp', s, self.W_s) + self.b_s +
               torch.einsum('bcs,cps->bcp', t, self.W_t) + self.b_t)
        return out.permute(0, 2, 1)


# ─── Loss helpers ────────────────────────────────────────────────────────────

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val=float("nan")) -> torch.Tensor:
    """Masked-MAE (DynGWN legacy). For SPX (no NaNs) reduces to plain MAE."""
    if null_val != null_val:
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


# ─── Data ─────────────────────────────────────────────────────────────────────

def _slice_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "full":
        return df
    if dataset == "precovid":
        end = pd.Timestamp(PRECOVID_END)
        return df[df["date"] <= end].reset_index(drop=True)
    raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASET_CHOICES}")


def _apply_target_space(iv_raw: np.ndarray, dates_full: np.ndarray, target_space: str):
    """level → unchanged; logdiff → log(IV)[1:]-log(IV)[:-1], dates trimmed by 1."""
    if target_space == "level":
        return iv_raw, dates_full
    if target_space == "logdiff":
        if (iv_raw <= 0).any():
            raise ValueError("logdiff target_space requires all IV > 0")
        log_iv = np.log(iv_raw)
        return (log_iv[1:] - log_iv[:-1]).astype(iv_raw.dtype), dates_full[1:]
    raise ValueError(f"Unknown target_space {target_space!r}; choose from {TARGET_SPACE_CHOICES}")


def load_splits(csv_path: str, dataset: str, target_space: str,
                seq_len: int, pred_len: int):
    """
    Returns scaled-space windows and the (mean, std) used to scale them.
    A single global mean/std is fit on the training portion of the
    target-space data (pooled across time and all IV cells, not
    per-column). The same scalars are used to transform the entire series.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df = _slice_dataset(df, dataset)

    iv_cols = [c for c in df.columns if c.startswith('iv_')]
    if not iv_cols:
        raise ValueError(f"No iv_* columns found in {csv_path}")
    n_iv = len(iv_cols)

    iv_raw     = df[iv_cols].to_numpy(dtype=np.float32)
    dates_full = df['date'].to_numpy(dtype='datetime64[D]')
    data, dates = _apply_target_space(iv_raw, dates_full, target_space)

    T       = len(data)
    n_train = int(T * TRAIN_FRAC)
    n_test  = int(T * TEST_FRAC)
    n_val   = T - n_train - n_test

    b1 = [0,           n_train - seq_len,  T - n_test - seq_len]
    b2 = [n_train,     n_train + n_val,    T]

    train_data = data[b1[0]:b2[0]]
    mean = float(train_data.mean())
    std  = float(train_data.std())
    iv   = ((data - mean) / std).astype(np.float32)

    def _windows(start, end):
        sl = iv[start:end]
        n  = len(sl) - seq_len - pred_len + 1
        X  = np.stack([sl[i        : i + seq_len]           for i in range(n)])
        y  = np.stack([sl[i+seq_len : i + seq_len + pred_len] for i in range(n)])
        return X, y

    X_tr, y_tr = _windows(b1[0], b2[0])
    X_va, y_va = _windows(b1[1], b2[1])
    X_te, y_te = _windows(b1[2], b2[2])

    test_slice_dates = dates[b1[2]:b2[2]]
    test_start_dates = np.array([test_slice_dates[i + seq_len] for i in range(len(X_te))])

    split_info = dict(
        T=T, n_iv=n_iv, n_train=n_train, n_val=n_val, n_test=n_test,
        train_end_date=str(dates[n_train - 1]),
        train_windows=len(X_tr), val_windows=len(X_va), test_windows=len(X_te),
    )
    return X_tr, y_tr, X_va, y_va, X_te, y_te, test_start_dates, split_info, mean, std


# ─── Training ─────────────────────────────────────────────────────────────────

def _compute_loss(pred_scaled: torch.Tensor, y_scaled: torch.Tensor,
                  loss_kind: str, mean: torch.Tensor = None,
                  std: torch.Tensor = None, huber_delta: float = 1.0) -> torch.Tensor:
    if loss_kind == 'mse':
        return nn.functional.mse_loss(pred_scaled, y_scaled)
    if loss_kind == 'mae_scaled':
        return nn.functional.l1_loss(pred_scaled, y_scaled)
    if loss_kind == 'huber_scaled':
        return nn.functional.smooth_l1_loss(pred_scaled, y_scaled, beta=huber_delta)
    if loss_kind == 'mae_original':
        pred_orig = pred_scaled * std + mean
        y_orig    = y_scaled    * std + mean
        return masked_mae(pred_orig, y_orig, null_val=float('nan'))
    raise ValueError(f"unknown loss_kind: {loss_kind!r}")


def _epoch(model, loader, opt, device, loss_kind, mean=None, std=None, huber_delta=1.0):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = _compute_loss(model(xb), yb, loss_kind, mean, std, huber_delta)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(xb)
        n     += len(xb)
    return total / n


@torch.no_grad()
def _val_loss(model, loader, device, loss_kind, mean=None, std=None, huber_delta=1.0):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += _compute_loss(model(xb), yb, loss_kind, mean, std, huber_delta).item() * len(xb)
        n     += len(xb)
    return total / n


def train(model, X_tr, y_tr, X_va, y_va, mean, std, args, out_dir, device):
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=args.batch_size, num_workers=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device)

    best_val, wait = float('inf'), 0
    ckpt = os.path.join(out_dir, 'best_model.pt')
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = _epoch(model, tr_loader, opt, device, args.loss, mean_t, std_t, args.huber_delta)
        va_loss = _val_loss(model, va_loader, device, args.loss, mean_t, std_t, args.huber_delta)
        log_rows.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss})

        if va_loss < best_val:
            best_val = va_loss
            wait     = 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1

        if epoch % 1 == 0 or epoch == 1:
            print(f'  epoch {epoch:4}/{args.epochs}  '
                  f'train={tr_loss:.6f}  val={va_loss:.6f}  best={best_val:.6f}')

        if wait >= args.patience:
            print(f'  early stop at epoch {epoch}  (patience={args.patience})')
            break

    with open(os.path.join(out_dir, 'train_log.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        w.writeheader(); w.writerows(log_rows)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    label = {
        'mse':           'MSE (scaled)',
        'mae_original':  'MAE (original IV)',
        'mae_scaled':    'MAE (scaled)',
        'huber_scaled':  f'Huber (scaled, δ={args.huber_delta})',
    }[args.loss]
    print(f'  best val {label}: {best_val:.6f}')
    return model


# ─── Misc ────────────────────────────────────────────────────────────────────

def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _build_config(args, info: dict) -> dict:
    return {
        'model':          'DLinear',
        'dataset':        args.dataset,
        'target_space':   args.target_space,
        'csv_path':       args.csv_path,
        'seq_len':        args.seq_len,
        'pred_len':       args.pred_len,
        'kernel_size':    args.kernel_size,
        'epochs':         args.epochs,
        'batch_size':     args.batch_size,
        'lr':             args.lr,
        'patience':       args.patience,
        'seed':           args.seed,
        'loss':           args.loss,
        'huber_delta':    args.huber_delta,
        'n_iv':           info['n_iv'],
        'n_train':        info['n_train'],
        'n_val':          info['n_val'],
        'n_test':         info['n_test'],
        'train_end_date': info['train_end_date'],
        'git_commit':     _git_commit(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Train DLinear on SPX IV surface')
    ap.add_argument('--csv_path',    default='SPX_surfaces.csv')
    ap.add_argument('--dataset',     default='full', choices=DATASET_CHOICES,
                    help='full = entire CSV; precovid = dates <= 2019-12-31')
    ap.add_argument('--target_space', default='level', choices=TARGET_SPACE_CHOICES,
                    help='level = train on raw IV (default); logdiff = train on '
                         'log(IV)[1:]-log(IV)[:-1]. logdiff loses one day at the front.')
    ap.add_argument('--seq_len',     type=int,   default=63)
    ap.add_argument('--pred_len',    type=int,   default=21)
    ap.add_argument('--kernel_size', type=int,   default=13,
                    help='Moving-avg kernel for decomposition; must be odd and <= seq_len')
    ap.add_argument('--epochs',      type=int,   default=100)
    ap.add_argument('--batch_size',  type=int,   default=64)
    ap.add_argument('--lr',          type=float, default=1e-4)
    ap.add_argument('--patience',    type=int,   default=15,
                    help='Early stopping patience (val loss epochs without improvement)')
    ap.add_argument('--device',      default='auto',
                    help='"auto" | "cuda" | "mps" | "cpu"')
    ap.add_argument('--seed',        type=int,   default=42)
    ap.add_argument('--out_dir',     default=None,
                    help='Override output directory (auto-named from config by default)')
    ap.add_argument('--predict_only', action='store_true',
                    help='Skip training; load best_model.pt + config.json from --out_dir, '
                         'run inference, write pred.npy.')
    ap.add_argument('--loss', default='mse',
                    choices=['mse', 'mae_original', 'mae_scaled', 'huber_scaled'])
    ap.add_argument('--huber_delta', type=float, default=1.0)
    args = ap.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    if args.predict_only:
        if args.out_dir is None:
            raise SystemExit('--predict_only requires --out_dir <dir with config.json + best_model.pt>')
        cfg_path  = os.path.join(args.out_dir, 'config.json')
        ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
        if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
            raise SystemExit(f'--predict_only: need config.json and best_model.pt in {args.out_dir}')
        with open(cfg_path) as f:
            cfg = json.load(f)
        args.csv_path     = cfg.get('csv_path',     args.csv_path)
        args.dataset      = cfg.get('dataset',      args.dataset)
        args.target_space = cfg.get('target_space', args.target_space)
        args.seq_len      = cfg.get('seq_len',      args.seq_len)
        args.pred_len    = cfg.get('pred_len',    args.pred_len)
        args.kernel_size = cfg.get('kernel_size', args.kernel_size)
        args.batch_size  = cfg.get('batch_size',  args.batch_size)
        args.loss        = cfg.get('loss',        args.loss)
        args.huber_delta = cfg.get('huber_delta', args.huber_delta)
        print(f'[predict_only] {cfg_path}')

    if args.out_dir is None:
        loss_suffix = {
            'mse':           '',
            'mae_original':  '_lossmae',
            'mae_scaled':    '_lossmaescaled',
            'huber_scaled':  f'_losshuberscaled_d{args.huber_delta:g}',
        }[args.loss]
        args.out_dir = (
            f'DLinear/results/'
            f'{args.dataset}_{args.target_space}_SPX_IV_'
            f'{args.seq_len}_{args.pred_len}'
            f'_DLinear_individual'
            f'_k{args.kernel_size}'
            f'_ep{args.epochs}'
            f'{loss_suffix}'
        )
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Dataset    : {args.dataset}')
    print(f'Target     : {args.target_space}')
    print(f'Device     : {device}')
    print(f'Output dir : {args.out_dir}')

    print('\nLoading data...')
    X_tr, y_tr, X_va, y_va, X_te, _y_te, test_dates, info, mean, std = load_splits(
        args.csv_path, args.dataset, args.target_space, args.seq_len, args.pred_len,
    )
    print(f'  T={info["T"]}  n_iv={info["n_iv"]}  '
          f'train={info["train_windows"]}  '
          f'val={info["val_windows"]}  '
          f'test={info["test_windows"]} windows')
    print(f'  Train ends {info["train_end_date"]}')

    n_iv = info['n_iv']
    model = DLinear(args.seq_len, args.pred_len, n_iv, args.kernel_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    if args.predict_only:
        ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f'  Loaded checkpoint from {ckpt_path}')
    else:
        config = _build_config(args, info)
        with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f'\nTraining (loss={args.loss}) ...')
        model = train(model, X_tr, y_tr, X_va, y_va, mean, std, args, args.out_dir, device)

    print('\nPredicting on test set...')
    te_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te)),
        batch_size=args.batch_size, num_workers=0,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = (preds * std + mean).astype(np.float32)

    np.save(os.path.join(args.out_dir, 'pred.npy'),        preds)
    np.save(os.path.join(args.out_dir, 'start_dates.npy'), test_dates)
    print(f'  pred.npy        shape={preds.shape}')
    print(f'  start_dates.npy shape={test_dates.shape}  '
          f'range={test_dates[0]} → {test_dates[-1]}')
    print(f'\nDone. Results in {args.out_dir}/')


if __name__ == '__main__':
    main()
