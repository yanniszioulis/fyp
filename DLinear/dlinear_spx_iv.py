#!/usr/bin/env python3
"""
DLinear training script for SPX IV surface forecasting.

Trains a channel-independent DLinear (trend + seasonality decomposition) on
SPX_surfaces.csv using the canonical 70/10/20 split and StandardScaler.
Saves predictions in scaled space to match compare_models.py expectations.

Quick start (Colab):
    python dlinear_spx_iv.py --device cuda

Outputs (in --out_dir):
    pred.npy          [N_test, pred_len, 400]  scaled-space predictions
    start_dates.npy   [N_test]                  datetime64[D] start of each window
    train_log.csv     epoch, train_loss, val_loss
    best_model.pt     checkpoint of best validation weights
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
N_IV       = 400


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
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, C]


class DLinear(nn.Module):
    """
    Channel-independent DLinear.

    Each of the 400 IV features gets its own pair of linear maps (seasonal and
    trend). Implemented as a vectorised batched matmul over channels — equivalent
    to 400 independent nn.Linear layers but ~100× faster than a ModuleList loop.

    Input:  [B, seq_len, C]
    Output: [B, pred_len, C]
    """
    def __init__(self, seq_len: int, pred_len: int, n_channels: int, kernel_size: int = 13):
        super().__init__()
        self.decomp = _MovingAvg(kernel_size)
        # Initialise to "predict the mean" (same as the original paper)
        w0 = (1.0 / seq_len) * torch.ones(n_channels, pred_len, seq_len)
        self.W_s = nn.Parameter(w0.clone())   # seasonal weights [C, P, S]
        self.W_t = nn.Parameter(w0.clone())   # trend weights    [C, P, S]
        self.b_s = nn.Parameter(torch.zeros(n_channels, pred_len))
        self.b_t = nn.Parameter(torch.zeros(n_channels, pred_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, S, C]
        trend = self.decomp(x)
        seas  = x - trend
        # [B, C, S] batched linear: out[b,c,p] = sum_s( x[b,c,s] * W[c,p,s] ) + b[c,p]
        s = seas.permute(0, 2, 1)
        t = trend.permute(0, 2, 1)
        out = (torch.einsum('bcs,cps->bcp', s, self.W_s) + self.b_s +
               torch.einsum('bcs,cps->bcp', t, self.W_t) + self.b_t)
        return out.permute(0, 2, 1)  # [B, P, C]


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_splits(csv_path: str, seq_len: int, pred_len: int):
    df = pd.read_csv(csv_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    iv_cols = [c for c in df.columns if c.startswith('iv_')]
    assert len(iv_cols) == N_IV, f'Expected {N_IV} iv_ columns, found {len(iv_cols)}'

    T       = len(df)
    n_train = int(T * TRAIN_FRAC)
    n_test  = int(T * TEST_FRAC)
    n_val   = T - n_train - n_test

    b1 = [0,           n_train - seq_len,  T - n_test - seq_len]
    b2 = [n_train,     n_train + n_val,    T]

    iv = df[iv_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(iv[b1[0]:b2[0]])
    iv = scaler.transform(iv).astype(np.float32)

    dates = df['date'].to_numpy(dtype='datetime64[D]')

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

    split_info = dict(T=T, n_train=n_train, n_val=n_val, n_test=n_test,
                      train_windows=len(X_tr), val_windows=len(X_va),
                      test_windows=len(X_te))
    return X_tr, y_tr, X_va, y_va, X_te, y_te, test_start_dates, split_info


# ─── Training ─────────────────────────────────────────────────────────────────

def _epoch(model, loader, opt, device):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = nn.functional.mse_loss(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(xb)
        n     += len(xb)
    return total / n


@torch.no_grad()
def _val_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += nn.functional.mse_loss(model(xb), yb).item() * len(xb)
        n     += len(xb)
    return total / n


def train(model, X_tr, y_tr, X_va, y_va, args, out_dir, device):
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=args.batch_size, num_workers=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, wait = float('inf'), 0
    ckpt = os.path.join(out_dir, 'best_model.pt')
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = _epoch(model, tr_loader, opt, device)
        va_loss = _val_loss(model, va_loader, device)
        log_rows.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss})

        if va_loss < best_val:
            best_val = va_loss
            wait     = 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'  epoch {epoch:4}/{args.epochs}  '
                  f'train={tr_loss:.6f}  val={va_loss:.6f}  best={best_val:.6f}')

        if wait >= args.patience:
            print(f'  early stop at epoch {epoch}  (patience={args.patience})')
            break

    with open(os.path.join(out_dir, 'train_log.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        w.writeheader(); w.writerows(log_rows)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f'  best val MSE: {best_val:.6f}')
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Train DLinear on SPX IV surface')
    ap.add_argument('--csv_path',    default='SPX_surfaces.csv')
    ap.add_argument('--seq_len',     type=int,   default=21)
    ap.add_argument('--pred_len',    type=int,   default=63)
    ap.add_argument('--kernel_size', type=int,   default=13,
                    help='Moving-avg kernel for decomposition; must be odd and <= seq_len')
    ap.add_argument('--epochs',      type=int,   default=100)
    ap.add_argument('--batch_size',  type=int,   default=64)
    ap.add_argument('--lr',          type=float, default=1e-3)
    ap.add_argument('--patience',    type=int,   default=15,
                    help='Early stopping patience (val MSE epochs without improvement)')
    ap.add_argument('--device',      default='auto',
                    help='"auto" | "cuda" | "mps" | "cpu"')
    ap.add_argument('--seed',        type=int,   default=42)
    ap.add_argument('--out_dir',     default=None,
                    help='Override output directory (auto-named from config by default)')
    ap.add_argument('--predict_only', action='store_true',
                    help='Skip training; load best_model.pt + config.json from --out_dir, '
                         'run inference, write pred.npy.')
    args = ap.parse_args()

    # Device
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

    # ── Predict-only: load config, rebuild model, load checkpoint, predict. ──
    if args.predict_only:
        if args.out_dir is None:
            raise SystemExit('--predict_only requires --out_dir <dir with config.json + best_model.pt>')
        cfg_path  = os.path.join(args.out_dir, 'config.json')
        ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
        if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
            raise SystemExit(f'--predict_only: need config.json and best_model.pt in {args.out_dir}')
        with open(cfg_path) as f:
            cfg = json.load(f)
        args.csv_path    = cfg.get('csv_path',    args.csv_path)
        args.seq_len     = cfg.get('seq_len',     args.seq_len)
        args.pred_len    = cfg.get('pred_len',    args.pred_len)
        args.kernel_size = cfg.get('kernel_size', args.kernel_size)
        args.batch_size  = cfg.get('batch_size',  args.batch_size)
        print(f'[predict_only] {cfg_path}')

    if args.out_dir is None:
        args.out_dir = (
            f'DLinear/results/'
            f'SPX_IV_{args.seq_len}_{args.pred_len}'
            f'_DLinear_individual'
            f'_k{args.kernel_size}'
            f'_ep{args.epochs}'
        )
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Device     : {device}')
    print(f'Output dir : {args.out_dir}')

    print('\nLoading data...')
    X_tr, y_tr, X_va, y_va, X_te, _y_te, test_dates, info = load_splits(
        args.csv_path, args.seq_len, args.pred_len,
    )
    print(f'  T={info["T"]}  '
          f'train={info["train_windows"]}  '
          f'val={info["val_windows"]}  '
          f'test={info["test_windows"]} windows')

    model = DLinear(args.seq_len, args.pred_len, N_IV, args.kernel_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    if args.predict_only:
        ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f'  Loaded checkpoint from {ckpt_path}')
    else:
        # Save reproducibility config alongside the checkpoint.
        config = {
            'model':       'DLinear',
            'csv_path':    args.csv_path,
            'seq_len':     args.seq_len,
            'pred_len':    args.pred_len,
            'kernel_size': args.kernel_size,
            'epochs':      args.epochs,
            'batch_size':  args.batch_size,
            'lr':          args.lr,
            'patience':    args.patience,
            'seed':        args.seed,
        }
        with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print('\nTraining...')
        model = train(model, X_tr, y_tr, X_va, y_va, args, args.out_dir, device)

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
    preds = np.concatenate(preds, axis=0).astype(np.float32)  # [N_test, pred_len, 400]

    np.save(os.path.join(args.out_dir, 'pred.npy'),        preds)
    np.save(os.path.join(args.out_dir, 'start_dates.npy'), test_dates)
    print(f'  pred.npy        shape={preds.shape}')
    print(f'  start_dates.npy shape={test_dates.shape}  '
          f'range={test_dates[0]} → {test_dates[-1]}')
    print(f'\nDone. Results in {args.out_dir}/')


if __name__ == '__main__':
    main()
