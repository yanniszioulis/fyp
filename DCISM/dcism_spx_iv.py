#!/usr/bin/env python3
"""
DCISM v0 — Decomposed Channel-Independent Surface Model.

Proposed in this project. Combines, in light of the benchmark's findings:

  (1) DLinear's series decomposition (boundary-padded MA) + channel-independent
      linear maps over trend and seasonality components.
  (2) A small 2D convolution applied to the predicted surface, reshaped as
      [H_mono=20, W_tau=20]. Identity-initialised so the layer starts as a
      no-op; only learns to smooth across moneyness/tau where it actively helps.
  (3) Optional masked-MAE-on-original-IV loss (matches DynGWN's legacy training
      objective; the highest-impact intervention observed in our benchmark).

Architecture sketch:

  x [B,seq,400]
    ├─ MA decomp ─→ trend, season
    ├─ einsum trend → trend_pred [B,pred,400]
    ├─ einsum season → season_pred [B,pred,400]
    ├─ core = trend_pred + season_pred
    ├─ reshape core → [B, pred, 20, 20] (F-order: H=mono, W=tau)
    ├─ 2D conv k=3, identity-init, per-horizon shared weights
    └─ output = core + conv_correction (residual skip)

Outputs (in --out_dir):
    pred.npy          [N_test, pred_len, 400]  scaled-space predictions
    start_dates.npy   [N_test]                  datetime64[D] start of each window
    config.json       full hyperparam record
    best_model.pt     checkpoint of best validation epoch
    train_log.csv     per-epoch losses
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
H_MONO     = 20   # moneyness axis (rows after F-order reshape)
W_TAU      = 20   # tau axis       (cols after F-order reshape)


# ─── Loss helpers (identical pattern to DLinear) ─────────────────────────────

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val=float("nan")) -> torch.Tensor:
    if null_val != null_val:  # NaN
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


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


class DCISM(nn.Module):
    """
    Decomposed channel-independent linear forecaster + 2D surface polish.

    Input:  [B, seq_len, 400]   scaled-space
    Output: [B, pred_len, 400]  scaled-space
    """
    def __init__(self, seq_len: int, pred_len: int, n_channels: int = N_IV,
                 kernel_size: int = 13, conv_kernel: int = 3,
                 conv_dropout: float = 0.0):
        super().__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.n_channels = n_channels
        assert n_channels == H_MONO * W_TAU, \
            f"DCISM expects 400-cell surface; got {n_channels}"

        # ── DLinear core ──────────────────────────────────────────────
        self.decomp = _MovingAvg(kernel_size)
        # Init "predict the mean" — same as DLinear / paper.
        w0 = (1.0 / seq_len) * torch.ones(n_channels, pred_len, seq_len)
        self.W_s = nn.Parameter(w0.clone())   # seasonal weights [C, P, S]
        self.W_t = nn.Parameter(w0.clone())   # trend weights    [C, P, S]
        self.b_s = nn.Parameter(torch.zeros(n_channels, pred_len))
        self.b_t = nn.Parameter(torch.zeros(n_channels, pred_len))

        # ── 2D conv polish (identity-init, residual skip) ─────────────
        # Shared across horizons: input/output channels = pred_len; treats
        # the [H, W] surface at each horizon as one "image" with pred_len
        # channels. Identity-init = each output channel is the corresponding
        # input channel; deviations have to be learned.
        pad = conv_kernel // 2
        self.polish = nn.Conv2d(
            in_channels=pred_len, out_channels=pred_len,
            kernel_size=conv_kernel, padding=pad, bias=True,
        )
        with torch.no_grad():
            self.polish.weight.zero_()
            self.polish.bias.zero_()
            # Identity kernel: weight[c, c, center, center] = 1 for each c.
            for c in range(pred_len):
                self.polish.weight[c, c, pad, pad] = 1.0
        self.conv_dropout = nn.Dropout2d(conv_dropout) if conv_dropout > 0 else None

    def _dlinear_core(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, C] → [B, P, C], DLinear-style einsum."""
        trend = self.decomp(x)
        seas  = x - trend
        s = seas.permute(0, 2, 1)
        t = trend.permute(0, 2, 1)
        out = (torch.einsum('bcs,cps->bcp', s, self.W_s) + self.b_s +
               torch.einsum('bcs,cps->bcp', t, self.W_t) + self.b_t)
        return out.permute(0, 2, 1)  # [B, P, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, S, C]
        core = self._dlinear_core(x)                         # [B, P, 400]

        # Reshape to [B, P, H, W] using F-order (matches HOT/compare_models loader):
        # column k of CSV = i_tau*20 + i_mono → element [i_mono, i_tau].
        B, P, _ = core.shape
        # F-order reshape via permute: split 400 = i_mono + 20*i_tau, so
        # core_grid[b, p, i_mono, i_tau] = core[b, p, i_mono + 20*i_tau].
        core_grid = core.view(B, P, W_TAU, H_MONO).transpose(2, 3)  # [B, P, H, W]

        # 2D polish: identity-init Conv2d treating P as channel axis.
        polished = self.polish(core_grid)
        if self.conv_dropout is not None:
            polished = self.conv_dropout(polished)

        # Flatten back to [B, P, 400] in CSV column order (inverse F-order).
        out_grid  = polished.transpose(2, 3).contiguous()         # [B, P, W_TAU, H_MONO]
        out       = out_grid.view(B, P, self.n_channels)          # [B, P, 400]
        return out


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
        X  = np.stack([sl[i        : i + seq_len]            for i in range(n)])
        y  = np.stack([sl[i+seq_len : i + seq_len + pred_len] for i in range(n)])
        return X, y

    X_tr, y_tr = _windows(b1[0], b2[0])
    X_va, y_va = _windows(b1[1], b2[1])
    X_te, _y   = _windows(b1[2], b2[2])

    test_slice_dates = dates[b1[2]:b2[2]]
    test_start_dates = np.array([test_slice_dates[i + seq_len] for i in range(len(X_te))])

    info = dict(T=T, n_train=n_train, n_val=n_val, n_test=n_test,
                train_windows=len(X_tr), val_windows=len(X_va),
                test_windows=len(X_te))
    return X_tr, y_tr, X_va, y_va, X_te, test_start_dates, info, scaler


# ─── Training ─────────────────────────────────────────────────────────────────

def _compute_loss(pred_scaled, y_scaled, loss_kind, mean=None, std=None, huber_delta=1.0):
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
        total += loss.item() * len(xb); n += len(xb)
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


def train(model, X_tr, y_tr, X_va, y_va, scaler, args, out_dir, device):
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=args.batch_size, num_workers=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    mean = torch.tensor(scaler.mean_,  dtype=torch.float32, device=device)
    std  = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    best_val, wait = float('inf'), 0
    ckpt = os.path.join(out_dir, 'best_model.pt')
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = _epoch(model, tr_loader, opt, device, args.loss, mean, std, args.huber_delta)
        va_loss = _val_loss(model, va_loader, device, args.loss, mean, std, args.huber_delta)
        log_rows.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss})

        if va_loss < best_val:
            best_val, wait = va_loss, 0
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
    label = {
        'mse':           'MSE (scaled)',
        'mae_original':  'MAE (original IV)',
        'mae_scaled':    'MAE (scaled)',
        'huber_scaled':  f'Huber (scaled, δ={args.huber_delta})',
    }[args.loss]
    print(f'  best val {label}: {best_val:.6f}')
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Train DCISM v0 on SPX IV surface')
    ap.add_argument('--csv_path',     default='SPX_surfaces.csv')
    ap.add_argument('--seq_len',      type=int,   default=21)
    ap.add_argument('--pred_len',     type=int,   default=63)
    ap.add_argument('--kernel_size',  type=int,   default=13,
                    help='Moving-avg kernel for series decomposition (must be odd).')
    ap.add_argument('--conv_kernel',  type=int,   default=3,
                    help='2D polish conv kernel size (must be odd).')
    ap.add_argument('--conv_dropout', type=float, default=0.0,
                    help='Dropout on the polish conv output.')
    ap.add_argument('--epochs',       type=int,   default=100)
    ap.add_argument('--batch_size',   type=int,   default=64)
    ap.add_argument('--lr',           type=float, default=1e-3)
    ap.add_argument('--patience',     type=int,   default=15)
    ap.add_argument('--device',       default='auto')
    ap.add_argument('--seed',         type=int,   default=42)
    ap.add_argument('--out_dir',      default=None)
    ap.add_argument('--predict_only', action='store_true',
                    help='Skip training; load best_model.pt + config.json from --out_dir, '
                         'run inference, write pred.npy.')
    ap.add_argument('--loss',         default='mse',
                    choices=['mse', 'mae_original', 'mae_scaled', 'huber_scaled'])
    ap.add_argument('--huber_delta',  type=float, default=1.0,
                    help='Threshold for huber_scaled (default 1.0 = ~1 stdev).')
    args = ap.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():           device = torch.device('cuda')
        elif torch.backends.mps.is_available(): device = torch.device('mps')
        else:                                   device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)

    if args.predict_only:
        if args.out_dir is None:
            raise SystemExit('--predict_only requires --out_dir <dir with config.json + best_model.pt>')
        cfg_path  = os.path.join(args.out_dir, 'config.json')
        ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
        if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
            raise SystemExit(f'--predict_only: need config.json and best_model.pt in {args.out_dir}')
        with open(cfg_path) as f:
            cfg = json.load(f)
        for k in ('csv_path', 'seq_len', 'pred_len', 'kernel_size',
                  'conv_kernel', 'conv_dropout', 'batch_size', 'loss',
                  'huber_delta'):
            if k in cfg:
                setattr(args, k, cfg[k])
        print(f'[predict_only] {cfg_path}')

    if args.out_dir is None:
        loss_suffix = {
            'mse':           '',
            'mae_original':  '_lossmae',
            'mae_scaled':    '_lossmaescaled',
            'huber_scaled':  f'_losshuberscaled_d{args.huber_delta:g}',
        }[args.loss]
        args.out_dir = (
            f'DCISM/results/'
            f'SPX_IV_{args.seq_len}_{args.pred_len}'
            f'_DCISMv0'
            f'_k{args.kernel_size}_ck{args.conv_kernel}'
            f'_ep{args.epochs}'
            f'{loss_suffix}'
        )
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Device     : {device}')
    print(f'Output dir : {args.out_dir}')

    print('\nLoading data...')
    X_tr, y_tr, X_va, y_va, X_te, test_dates, info, scaler = load_splits(
        args.csv_path, args.seq_len, args.pred_len,
    )
    print(f"  T={info['T']}  train={info['train_windows']}  "
          f"val={info['val_windows']}  test={info['test_windows']} windows")

    model = DCISM(
        seq_len=args.seq_len, pred_len=args.pred_len, n_channels=N_IV,
        kernel_size=args.kernel_size, conv_kernel=args.conv_kernel,
        conv_dropout=args.conv_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    if args.predict_only:
        ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f'  Loaded checkpoint from {ckpt_path}')
    else:
        config = {
            'model':         'DCISMv0',
            'csv_path':      args.csv_path,
            'seq_len':       args.seq_len,
            'pred_len':      args.pred_len,
            'kernel_size':   args.kernel_size,
            'conv_kernel':   args.conv_kernel,
            'conv_dropout':  args.conv_dropout,
            'epochs':        args.epochs,
            'batch_size':    args.batch_size,
            'lr':            args.lr,
            'patience':      args.patience,
            'seed':          args.seed,
            'loss':          args.loss,
            'huber_delta':   args.huber_delta,
        }
        with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f'\nTraining (loss={args.loss}) ...')
        model = train(model, X_tr, y_tr, X_va, y_va, scaler, args, args.out_dir, device)

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
    print(f'  start_dates.npy range={test_dates[0]} → {test_dates[-1]}')
    print(f'\nDone. Results in {args.out_dir}/')


if __name__ == '__main__':
    main()
