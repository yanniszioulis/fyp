#!/usr/bin/env python3
"""
Test script to overfit the new correction-based transformer on W0, c5, h1.
Custom training loop to track per-epoch losses and see if it can memorize one sample.
"""

import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
except ImportError:
    torch = None
    nn = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting.pipeline import ForecastingPipeline
from models.transformer.transformer_model import TransformerSurfaceModel
from models.persistence.persistence_model import PersistenceModel
from evaluation.metrics import compute_all_metrics

def main():
    print("=" * 70)
    print("Testing Correction-Based Transformer - Memorization Test")
    print("W0, c5, h1 - Custom training loop with per-epoch stats")
    print("=" * 70)
    
    data_file = os.environ.get("FYP_DATA_PATH") or "SPX_IV_fixed_grid.csv"
    pipeline = ForecastingPipeline(data_file=data_file, results_dir="results")
    
    print("\n[1/4] Loading data...")
    pipeline.load_data()
    print(f"Data shape: {pipeline.data.shape}")
    
    print("\n[2/4] Creating rolling windows...")
    pipeline.create_windows(
        window_size_years=10.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        shift_months=6
    )
    
    if len(pipeline.windows) == 0:
        print("ERROR: No windows created!")
        return
    
    window = pipeline.windows[0]  # Window 0
    context_length = 21
    horizon = 21
    
    print(f"\n[3/4] Selecting consecutive days from Window {window.window_id}...")
    print(f"  Context: {context_length} days")
    print(f"  Horizon: {horizon} days")
    print(f"  Window range: {window.start_date.date()} to {window.end_date.date()}")
    
    # Get all available dates in window 0
    all_window_indices = np.concatenate([
        window.train_indices,
        window.val_indices,
        window.test_indices
    ])
    all_window_indices = np.sort(all_window_indices)
    
    # Sample size requirements
    max_train_samples = 700
    max_val_samples = 100
    max_test_samples = 200
    
    # Calculate days needed for each split
    train_days_needed = max_train_samples + context_length + horizon - 1
    val_days_needed = max_val_samples + context_length + horizon - 1
    test_days_needed = max_test_samples + context_length + horizon - 1
    
    total_days_needed = train_days_needed + val_days_needed + test_days_needed
    
    if len(all_window_indices) < total_days_needed:
        print(f"ERROR: Window has {len(all_window_indices)} days, need at least {total_days_needed}")
        return
    
    # Pick random starting point
    max_start_idx = len(all_window_indices) - total_days_needed
    start_idx = np.random.randint(0, max_start_idx + 1)
    
    # Take consecutive days
    selected_indices = all_window_indices[start_idx:start_idx + total_days_needed]
    selected_dates = pipeline.dates[selected_indices]
    
    print(f"  Selected consecutive days: {selected_dates[0].date()} to {selected_dates[-1].date()}")
    
    # Split into train/val/test
    train_end = train_days_needed
    val_start = train_end
    val_end = val_start + val_days_needed
    test_start = val_end
    
    train_indices = selected_indices[:train_end]
    val_indices = selected_indices[val_start:val_end]
    test_indices = selected_indices[test_start:]
    
    print(f"\n  Split:")
    print(f"    Train days: {pipeline.dates[train_indices[0]].date()} to {pipeline.dates[train_indices[-1]].date()} ({len(train_indices)} days)")
    print(f"    Val days: {pipeline.dates[val_indices[0]].date()} to {pipeline.dates[val_indices[-1]].date()} ({len(val_indices)} days)")
    print(f"    Test days: {pipeline.dates[test_indices[0]].date()} to {pipeline.dates[test_indices[-1]].date()} ({len(test_indices)} days)")
    
    # Build sequences
    from forecasting.data_loader import create_sequences
    
    train_data = pipeline.data[train_indices]
    train_dates = pipeline.dates[train_indices]
    X_train, y_train, train_sample_dates = create_sequences(
        train_data, train_dates,
        context_length=context_length,
        horizon=horizon,
        verbose=False
    )
    
    val_data = pipeline.data[val_indices]
    val_dates = pipeline.dates[val_indices]
    X_val, y_val, val_sample_dates = create_sequences(
        val_data, val_dates,
        context_length=context_length,
        horizon=horizon,
        verbose=False
    )
    
    test_data = pipeline.data[test_indices]
    test_dates = pipeline.dates[test_indices]
    X_test, y_test, test_sample_dates = create_sequences(
        test_data, test_dates,
        context_length=context_length,
        horizon=horizon,
        verbose=False
    )
    
    # Limit to exact sample sizes
    X_train = X_train[:max_train_samples]
    y_train = y_train[:max_train_samples]
    X_val = X_val[:max_val_samples]
    y_val = y_val[:max_val_samples]
    X_test = X_test[:max_test_samples]
    y_test = y_test[:max_test_samples]
    
    print(f"\n  Final sample sizes:")
    print(f"    Train: {len(X_train)}")
    print(f"    Val: {len(X_val)}")
    print(f"    Test: {len(X_test)}")
    
    if len(X_train) == 0:
        print("ERROR: No training samples!")
        return
    
    # Setup device
    if torch is not None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"\n  Using MPS device (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"\n  Using CUDA device")
        else:
            device = torch.device("cpu")
            print(f"\n  Using CPU device")
    else:
        print("ERROR: PyTorch not available!")
        return
    
    # Initialize model (we'll use the transformer architecture)
    n_samples, _, n_tau, n_logm = X_train.shape
    n_features = n_tau * n_logm
    
    # Create model instance to get baseline computation
    model_wrapper = TransformerSurfaceModel(
        name="test_memorization",
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        learning_rate=1e-1,
        weight_decay=0.0,
        batch_size=1,
        num_epochs=1,  # Not used, we'll train manually
        pool="last",
        normalize=True,
        baseline_decay=-1,  # -1 means use persistence (last surface) as baseline
        device=str(device)
    )
    
    # Compute baselines
    baseline_train = model_wrapper._compute_baseline(X_train)
    baseline_val = model_wrapper._compute_baseline(X_val)
    baseline_test = model_wrapper._compute_baseline(X_test)
    
    # Compute corrections
    y_correction_train = y_train - baseline_train
    y_correction_val = y_val - baseline_val
    y_correction_test = y_test - baseline_test
    
    # Flatten and normalize
    X_train_flat = X_train.reshape(n_samples, context_length, n_features).astype(np.float32)
    y_correction_train_flat = y_correction_train.reshape(n_samples, n_features).astype(np.float32)
    
    # Normalization stats from training data
    # X uses its own stats, corrections use their own stats
    flat_for_stats = X_train_flat.reshape(-1, n_features)
    mean_X = flat_for_stats.mean(axis=0, keepdims=True)
    std_X = flat_for_stats.std(axis=0, keepdims=True)
    std_X = np.maximum(std_X, 1e-8)
    
    # Normalize corrections with their own statistics (like MLP_HAR)
    # This helps the model learn corrections in a normalized space, then we denormalize before adding to baseline
    mean_corr = y_correction_train_flat.mean(axis=0, keepdims=True)
    std_corr = y_correction_train_flat.std(axis=0, keepdims=True)
    std_corr = np.maximum(std_corr, 1e-8)
    
    X_train_flat_norm = (X_train_flat - mean_X) / std_X
    # Normalize corrections with their own statistics (like MLP_HAR)
    y_correction_train_flat_norm = (y_correction_train_flat - mean_corr) / std_corr
    
    X_val_flat = X_val.reshape(len(X_val), context_length, n_features).astype(np.float32)
    y_correction_val_flat = y_correction_val.reshape(len(X_val), n_features).astype(np.float32)
    X_val_flat_norm = (X_val_flat - mean_X) / std_X
    # Normalize corrections with training correction stats
    y_correction_val_flat_norm = (y_correction_val_flat - mean_corr) / std_corr
    
    X_test_flat = X_test.reshape(len(X_test), context_length, n_features).astype(np.float32)
    y_correction_test_flat = y_correction_test.reshape(len(X_test), n_features).astype(np.float32)
    X_test_flat_norm = (X_test_flat - mean_X) / std_X
    # Normalize corrections with training correction stats
    y_correction_test_flat_norm = (y_correction_test_flat - mean_corr) / std_corr
    
    # Create the transformer model
    from models.transformer.transformer_model import _SurfaceTransformer
    
    transformer = _SurfaceTransformer(
        n_features=n_features,
        context_length=context_length,
        d_model=8,  # Increased from 512
        n_heads=4,    # Increased from 16
        n_layers=2,   # Increased from 6
        dropout=0.2,
        pool="last"
    ).to(device)
    
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-2, weight_decay=0.0)  # Lower LR for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    # Note: loss is computed as RMSE on surfaces (not MSE on corrections)
    scaler = GradScaler(enabled=False)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_flat_norm, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_correction_train_flat_norm, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_flat_norm, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_correction_val_flat_norm, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_flat_norm, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_correction_test_flat_norm, dtype=torch.float32).to(device)
    
    mean_X_t = torch.tensor(mean_X, dtype=torch.float32).to(device)
    std_X_t = torch.tensor(std_X, dtype=torch.float32).to(device)
    mean_corr_t = torch.tensor(mean_corr, dtype=torch.float32).to(device)
    std_corr_t = torch.tensor(std_corr, dtype=torch.float32).to(device)
    
    baseline_train_t = torch.tensor(baseline_train, dtype=torch.float32).to(device)
    baseline_val_t = torch.tensor(baseline_val, dtype=torch.float32).to(device)
    baseline_test_t = torch.tensor(baseline_test, dtype=torch.float32).to(device)
    
    y_train_true_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_true_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    y_test_true_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    print("\n[4/4] Training transformer with custom loop...")
    print(f"  Model: d_model=1024, n_heads=32, n_layers=12 (LARGE MODEL)")
    print(f"  Training on {len(X_train)} sample(s)")
    print(f"  Learning rate: 1e-3 (with ReduceLROnPlateau scheduler)")
    print(f"  No gradient clipping (free learning)")
    print(f"  Epochs: 50")
    
    # Debug: Check correction magnitude
    print(f"\n  Debug info:")
    print(f"    Correction magnitude: mean={np.abs(y_correction_train_flat).mean():.6f}, max={np.abs(y_correction_train_flat).max():.6f}")
    print(f"    Normalized correction magnitude: mean={np.abs(y_correction_train_flat_norm).mean():.6f}, max={np.abs(y_correction_train_flat_norm).max():.6f}")
    print(f"    Baseline magnitude: mean={np.abs(baseline_train).mean():.6f}")
    print(f"    Target magnitude: mean={np.abs(y_train).mean():.6f}")
    print(f"    Correction std: {std_corr.mean():.6f} (corrections normalized with their own stats, like MLP_HAR)")
    
    print("\n" + "=" * 70)
    print("EPOCH | TRAIN LOSS | VAL LOSS | TEST LOSS | TRAIN RMSE | VAL RMSE | TEST RMSE")
    print("=" * 70)
    
    num_epochs = 60  # More epochs for overfitting
    for epoch in range(1, num_epochs + 1):
        # Training
        transformer.train()
        optimizer.zero_grad(set_to_none=True)
        
        pred_correction_norm = transformer(X_train_t)  # Predicts normalized corrections
        # Denormalize corrections before adding to baseline
        pred_correction = pred_correction_norm * std_corr_t + mean_corr_t
        # Compute loss as RMSE on surfaces (not corrections)
        pred_train_flat = pred_correction + baseline_train_t.reshape(len(X_train), n_features)
        pred_train = pred_train_flat.reshape(len(X_train), n_tau, n_logm)
        loss = torch.sqrt(torch.mean((pred_train - y_train_true_t) ** 2))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # No gradient clipping for memorization test - let it learn freely
        # torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Debug: print gradient norms occasionally
        if epoch <= 5 or epoch % 50 == 0:
            total_norm = 0.0
            for p in transformer.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"      [grad_norm={total_norm:.2f}]")
        
        train_loss = loss.item()
        
        # Learning rate scheduling
        scheduler.step(train_loss)
        
        # Evaluation (compute RMSE in original units)
        transformer.eval()
        with torch.no_grad():
            # Train
            pred_corr_train_norm = transformer(X_train_t)  # (n_train, 420) - normalized corrections
            pred_corr_train = pred_corr_train_norm * std_corr_t + mean_corr_t  # Denormalize
            pred_train_flat = pred_corr_train + baseline_train_t.reshape(len(X_train), n_features)  # (n_train, 420)
            pred_train = pred_train_flat.reshape(len(X_train), n_tau, n_logm)  # (n_train, 20, 21)
            train_rmse = train_loss  # Same as loss now (RMSE on surfaces)
            
            # Val
            pred_corr_val_norm = transformer(X_val_t)  # (val_samples, 420) - normalized corrections
            pred_corr_val = pred_corr_val_norm * std_corr_t + mean_corr_t  # Denormalize
            pred_val_flat = pred_corr_val + baseline_val_t.reshape(len(X_val), n_features)  # (val_samples, 420)
            pred_val = pred_val_flat.reshape(len(X_val), n_tau, n_logm)  # (val_samples, 20, 21)
            val_loss = torch.sqrt(torch.mean((pred_val - y_val_true_t) ** 2)).item()  # RMSE on surfaces
            val_rmse = val_loss  # Same as loss now
            
            # Test
            pred_corr_test_norm = transformer(X_test_t)  # (test_samples, 420) - normalized corrections
            pred_corr_test = pred_corr_test_norm * std_corr_t + mean_corr_t  # Denormalize
            pred_test_flat = pred_corr_test + baseline_test_t.reshape(len(X_test), n_features)  # (test_samples, 420)
            pred_test = pred_test_flat.reshape(len(X_test), n_tau, n_logm)  # (test_samples, 20, 21)
            test_loss = torch.sqrt(torch.mean((pred_test - y_test_true_t) ** 2)).item()  # RMSE on surfaces
            test_rmse = test_loss  # Same as loss now
        
        # Print every epoch (or every 10 for less spam)
        if epoch % 1 == 0 or epoch == num_epochs:
            print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:8.6f} | {test_loss:9.6f} | "
                  f"{train_rmse:10.6f} | {val_rmse:8.6f} | {test_rmse:9.6f}")
    
    print("=" * 70)
    
    # Final evaluation with persistence comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON WITH PERSISTENCE")
    print("=" * 70)
    
    # Get final predictions
    transformer.eval()
    with torch.no_grad():
        pred_corr_train_norm = transformer(X_train_t)  # (n_train, 420) - normalized corrections
        pred_corr_train = pred_corr_train_norm * std_corr_t + mean_corr_t  # Denormalize
        pred_train_flat = pred_corr_train + baseline_train_t.reshape(len(X_train), n_features)  # (n_train, 420)
        pred_train_final = pred_train_flat.reshape(len(X_train), n_tau, n_logm).cpu().numpy()  # (n_train, 20, 21)
        
        pred_corr_val_norm = transformer(X_val_t)  # (val_samples, 420) - normalized corrections
        pred_corr_val = pred_corr_val_norm * std_corr_t + mean_corr_t  # Denormalize
        pred_val_flat = pred_corr_val + baseline_val_t.reshape(len(X_val), n_features)  # (val_samples, 420)
        pred_val_final = pred_val_flat.reshape(len(X_val), n_tau, n_logm).cpu().numpy()  # (val_samples, 20, 21)
        
        pred_corr_test_norm = transformer(X_test_t)  # (test_samples, 420) - normalized corrections
        pred_corr_test = pred_corr_test_norm * std_corr_t + mean_corr_t  # Denormalize
        pred_test_flat = pred_corr_test + baseline_test_t.reshape(len(X_test), n_features)  # (test_samples, 420)
        pred_test_final = pred_test_flat.reshape(len(X_test), n_tau, n_logm).cpu().numpy()  # (test_samples, 20, 21)
    
    # Persistence predictions
    persistence_model = PersistenceModel(name="persistence_baseline")
    persistence_model.fit(X_train, y_train)
    pred_train_pers = persistence_model.predict_horizon(X_train, horizon=horizon)
    pred_val_pers = persistence_model.predict_horizon(X_val, horizon=horizon)
    pred_test_pers = persistence_model.predict_horizon(X_test, horizon=horizon)
    
    # Compute metrics
    train_metrics_trans = compute_all_metrics(y_train, pred_train_final)
    train_metrics_pers = compute_all_metrics(y_train, pred_train_pers)
    
    val_metrics_trans = compute_all_metrics(y_val, pred_val_final)
    val_metrics_pers = compute_all_metrics(y_val, pred_val_pers)
    
    test_metrics_trans = compute_all_metrics(y_test, pred_test_final)
    test_metrics_pers = compute_all_metrics(y_test, pred_test_pers)
    
    print("\n[Train Set]")
    print(f"  Transformer RMSE: {train_metrics_trans['iv_rmse']:.6f}")
    print(f"  Persistence RMSE: {train_metrics_pers['iv_rmse']:.6f}")
    print(f"  Improvement: {(train_metrics_pers['iv_rmse'] - train_metrics_trans['iv_rmse']) / train_metrics_pers['iv_rmse'] * 100:+.2f}%")
    
    print("\n[Validation Set]")
    print(f"  Transformer RMSE: {val_metrics_trans['iv_rmse']:.6f}")
    print(f"  Persistence RMSE: {val_metrics_pers['iv_rmse']:.6f}")
    print(f"  Improvement: {(val_metrics_pers['iv_rmse'] - val_metrics_trans['iv_rmse']) / val_metrics_pers['iv_rmse'] * 100:+.2f}%")
    
    print("\n[Test Set]")
    print(f"  Transformer RMSE: {test_metrics_trans['iv_rmse']:.6f}")
    print(f"  Persistence RMSE: {test_metrics_pers['iv_rmse']:.6f}")
    print(f"  Improvement: {(test_metrics_pers['iv_rmse'] - test_metrics_trans['iv_rmse']) / test_metrics_pers['iv_rmse'] * 100:+.2f}%")
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
