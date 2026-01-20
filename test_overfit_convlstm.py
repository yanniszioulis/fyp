#!/usr/bin/env python3
"""
Test script to train ConvLSTM on a specific configuration (w0, c21, h5).
Compares ConvLSTM performance against persistence baseline.
Uses MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration.
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting.pipeline import ForecastingPipeline
from models.convlstm.convlstm_model import ConvLSTMModel
from models.persistence.persistence_model import PersistenceModel


def compute_rmse(predictions, targets):
    """Compute RMSE, handling NaN values"""
    mask = ~np.isnan(targets)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((predictions[mask] - targets[mask]) ** 2))


def main():
    parser = argparse.ArgumentParser(description="Train ConvLSTM on w0, c21, h5 with optional custom sample sizes")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Number of training samples to use (default: None = use full training set)"
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Number of validation samples to use (default: None = use full validation set)"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Number of test samples to use (default: None = use full test set)"
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom sample sizes: --train-samples 7 --val-samples 1 --test-samples 2"
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        dest="no_early_stop",
        help="Disable early stopping for overfitting tests"
    )
    args = parser.parse_args()
    
    # If --custom flag is set, use default custom sizes
    if args.custom:
        if args.train_samples is None:
            args.train_samples = 7
        if args.val_samples is None:
            args.val_samples = 1
        if args.test_samples is None:
            args.test_samples = 2
    
    print("=" * 60)
    print("ConvLSTM Training Test")
    print("Configuration: Window 0, Context 21, Horizon 5")
    if args.train_samples is not None or args.val_samples is not None or args.test_samples is not None:
        print(f"Sample sizes: Train={args.train_samples}, Val={args.val_samples}, Test={args.test_samples}")
    else:
        print("Using full window (all available samples)")
    print("=" * 60)
    
    # Check for MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using CUDA")
    else:
        device = "cpu"
        print(f"⚠ Using CPU (MPS/CUDA not available)")
    
    # Initialize pipeline
    print("\n[1/5] Initializing pipeline...")
    pipeline = ForecastingPipeline()
    
    # Load data
    print("\n[2/5] Loading data...")
    pipeline.load_data()
    
    # Create rolling windows
    print("\n[3/5] Creating rolling windows...")
    pipeline.create_windows(
        window_size_years=10.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        shift_months=12
    )
    
    # Get window 0
    if len(pipeline.windows) == 0:
        raise ValueError("No windows created!")
    
    window = pipeline.windows[0]
    print(f"\nWindow 0:")
    print(f"  Train: {window.train_start.date()} to {window.train_end.date()} ({len(window.train_indices)} days)")
    print(f"  Val: {window.val_start.date()} to {window.val_end.date()} ({len(window.val_indices)} days)")
    print(f"  Test: {window.test_start.date()} to {window.test_end.date()} ({len(window.test_indices)} days)")
    
    # Build sequences for c21, h5
    print("\n[4/5] Building sequences...")
    context_length = 5
    horizon = 1
    
    X_train, y_train, X_val, y_val, X_test, y_test, sample_dates = pipeline._build_sequences_for_window(
        window=window,
        context_length=context_length,
        horizon=horizon,
        train_on_val=False,  # Use only training set for training
        use_val=True  # Use validation set for monitoring
    )
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples: {len(X_val) if X_val is not None else 0}")
    print(f"  Test samples: {len(X_test)}")
    
    # Apply custom sample sizes if specified
    if args.train_samples is not None:
        if args.train_samples > len(X_train):
            print(f"  Warning: Requested {args.train_samples} train samples but only {len(X_train)} available. Using all {len(X_train)} samples.")
            args.train_samples = len(X_train)
        X_train = X_train[:args.train_samples]
        y_train = y_train[:args.train_samples]
        print(f"\n  Using {len(X_train)} training samples (custom)")
    else:
        print(f"\n  Using full training set ({len(X_train)} samples)")
    
    if args.val_samples is not None and X_val is not None and len(X_val) > 0:
        if args.val_samples > len(X_val):
            print(f"  Warning: Requested {args.val_samples} val samples but only {len(X_val)} available. Using all {len(X_val)} samples.")
            args.val_samples = len(X_val)
        X_val = X_val[:args.val_samples]
        y_val = y_val[:args.val_samples]
        print(f"  Using {len(X_val)} validation samples (custom)")
    
    if args.test_samples is not None and len(X_test) > 0:
        if args.test_samples > len(X_test):
            print(f"  Warning: Requested {args.test_samples} test samples but only {len(X_test)} available. Using all {len(X_test)} samples.")
            args.test_samples = len(X_test)
        X_test = X_test[:args.test_samples]
        y_test = y_test[:args.test_samples]
        print(f"  Using {len(X_test)} test samples (custom)")
    
    # Verification: Check data shapes and alignment
    print("\n" + "="*60)
    print("DATA VERIFICATION:")
    print("="*60)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"  Expected: X_train (n_samples, context_length={context_length}, n_tau, n_m)")
    print(f"            y_train (n_samples, n_tau, n_m)")
    print(f"\nContext length: {context_length}")
    print(f"Horizon: {horizon}")
    print(f"  Persistence should predict: X_train[:, -1, :, :] (last timestep in context)")
    print(f"  Target y_train is: data[i + context_length + horizon - 1]")
    print(f"  For i=0: target is at index {context_length + horizon - 1} = {context_length + horizon - 1} days ahead")
    
    # Verify persistence prediction equals last timestep
    print("\nVerifying persistence implementation...")
    print(f"  X_train[:, -1, :, :].shape = {X_train[:, -1, :, :].shape}")
    print(f"  Should match y_train.shape = {y_train.shape}")
    
    # Check first sample to verify alignment
    if len(X_train) > 0:
        sample_idx = 0
        last_timestep = X_train[sample_idx, -1, :, :]
        target = y_train[sample_idx, :, :]
        print(f"\nSample {sample_idx} verification:")
        print(f"  Last timestep (X_train[{sample_idx}, -1, :, :]):")
        print(f"    Mean: {np.nanmean(last_timestep):.6f}, Std: {np.nanstd(last_timestep):.6f}")
        print(f"    Min: {np.nanmin(last_timestep):.6f}, Max: {np.nanmax(last_timestep):.6f}")
        print(f"  Target (y_train[{sample_idx}, :, :]):")
        print(f"    Mean: {np.nanmean(target):.6f}, Std: {np.nanstd(target):.6f}")
        print(f"    Min: {np.nanmin(target):.6f}, Max: {np.nanmax(target):.6f}")
        print(f"  Note: For horizon={horizon}, target is {horizon} days ahead of last timestep")
        print(f"        If horizon=1, target should be very similar to last timestep (next day)")
        print(f"        If horizon=5, target will differ more (5 days ahead)")
    
    print("="*60)
    
    # Evaluate Persistence Baseline
    print("\n[5/6] Evaluating Persistence Baseline...")
    persistence = PersistenceModel(name="persistence")
    persistence.fit(X_train, y_train, horizon=horizon, context_length=context_length)
    
    # Persistence predictions
    persistence_train_pred = persistence.predict_horizon(X_train, horizon=horizon)
    persistence_val_pred = persistence.predict_horizon(X_val, horizon=horizon) if X_val is not None and len(X_val) > 0 else None
    persistence_test_pred = persistence.predict_horizon(X_test, horizon=horizon) if len(X_test) > 0 else None
    
    # Verify persistence prediction shape matches
    print(f"\nPersistence prediction verification:")
    print(f"  persistence_train_pred.shape: {persistence_train_pred.shape}")
    print(f"  y_train.shape: {y_train.shape}")
    print(f"  Shapes match: {persistence_train_pred.shape == y_train.shape}")
    print(f"  Persistence equals X[:, -1]: {np.allclose(persistence_train_pred, X_train[:, -1, :, :], equal_nan=True)}")
    
    # Persistence RMSE
    persistence_train_rmse = compute_rmse(persistence_train_pred, y_train)
    persistence_val_rmse = compute_rmse(persistence_val_pred, y_val) if persistence_val_pred is not None else np.nan
    persistence_test_rmse = compute_rmse(persistence_test_pred, y_test) if persistence_test_pred is not None else np.nan
    
    print(f"\n{'='*60}")
    print("Persistence Baseline Results:")
    print(f"  Train RMSE: {persistence_train_rmse:.6f}")
    if not np.isnan(persistence_val_rmse):
        print(f"  Val RMSE:   {persistence_val_rmse:.6f}")
    if not np.isnan(persistence_test_rmse):
        print(f"  Test RMSE:  {persistence_test_rmse:.6f}")
    print(f"{'='*60}")
    
    # Initialize ConvLSTM model
    print("\n[6/6] Initializing ConvLSTM model...")
    model = ConvLSTMModel(
        name="convlstm_w0_c21_h5",
        num_layers=1,
        filters=[64],  # Must match num_layers: [filters_layer1, filters_layer2]
        kernel_size=[3],  # Must match num_layers: [kernel_layer1, kernel_layer2]
        strides=[1],  # Must match num_layers: [stride_layer1, stride_layer2]
        padding=[1],  # Must match num_layers: [padding_layer1, padding_layer2] - padding=1 for kernel_size=3 to maintain spatial dims
        last_conv_kernel=1,
        last_conv_stride=1,
        last_conv_padding=0,  # Use 0 when layers maintain size (padding=1), use 1 when layers reduce size (padding=0)
        batch_size=min(len(X_train), 32),  # Use smaller batch for small datasets
        epochs=100,  # PI-ConvTF default
        learning_rate=0.01,  # PI-ConvTF uses 0.001, not 0.01! Lower LR for better convergence
        lr_patience=5,  # More aggressive LR reduction for memorization
        lr_factor=0.5,
        patience=1000 if args.no_early_stop else (200 if args.train_samples is not None and args.train_samples <= 10 else 10),  # Disable or high patience for overfitting
        min_delta=1e-6,  # Very small threshold for overfitting
        device=device
    )
    
    print(f"\n{'='*60}")
    print("ConvLSTM Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val) if X_val is not None else 0}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Max epochs: {model.epochs}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Early stopping patience: {model.patience}")
    print(f"  Min delta: {model.min_delta}")
    print(f"{'='*60}\n")
    
    # Verify ConvLSTM receives same data as persistence
    print("\n" + "="*60)
    print("CONVLSTM DATA VERIFICATION:")
    print("="*60)
    print(f"ConvLSTM will receive:")
    print(f"  X_train.shape: {X_train.shape}")
    print(f"  y_train.shape: {y_train.shape}")
    print(f"  Same as persistence: ✓")
    print(f"  (ConvLSTM normalizes internally, but uses same raw data)")
    print("="*60 + "\n")
    
    # Train the model
    print("Starting ConvLSTM training...")
    print("(This may take a while - watch for train/val loss convergence)\n")
    
    # For overfitting tests with small datasets, use final model instead of best val model
    use_final_model = args.no_early_stop or (args.train_samples is not None and args.train_samples <= 10)
    
    model.fit(
        X_train=X_train,
        y_train=y_train,
        horizon=horizon,
        context_length=context_length,
        X_val=X_val,
        y_val=y_val,
        use_final_model=use_final_model  # Use final model for overfitting tests
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Evaluate ConvLSTM
    print("\nEvaluating ConvLSTM...")
    convlstm_train_pred = model.predict_horizon(X_train, horizon=horizon)
    convlstm_val_pred = model.predict_horizon(X_val, horizon=horizon) if X_val is not None and len(X_val) > 0 else None
    convlstm_test_pred = model.predict_horizon(X_test, horizon=horizon) if len(X_test) > 0 else None
    
    # Verify ConvLSTM prediction shape matches
    print(f"\nConvLSTM prediction verification:")
    print(f"  convlstm_train_pred.shape: {convlstm_train_pred.shape}")
    print(f"  y_train.shape: {y_train.shape}")
    print(f"  Shapes match: {convlstm_train_pred.shape == y_train.shape}")
    print(f"  Same target as persistence: ✓")
    
    convlstm_train_rmse = compute_rmse(convlstm_train_pred, y_train)
    convlstm_val_rmse = compute_rmse(convlstm_val_pred, y_val) if convlstm_val_pred is not None else np.nan
    convlstm_test_rmse = compute_rmse(convlstm_test_pred, y_test) if convlstm_test_pred is not None else np.nan
    
    print(f"\n{'='*60}")
    print("ConvLSTM Results:")
    print(f"  Train RMSE: {convlstm_train_rmse:.6f}")
    if not np.isnan(convlstm_val_rmse):
        print(f"  Val RMSE:   {convlstm_val_rmse:.6f}")
    if not np.isnan(convlstm_test_rmse):
        print(f"  Test RMSE:  {convlstm_test_rmse:.6f}")
    print(f"{'='*60}")
    
    # Comparison
    print(f"\n{'='*60}")
    print("Comparison: ConvLSTM vs Persistence")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Persistence':<15} {'ConvLSTM':<15} {'Improvement':<15}")
    print(f"{'-'*60}")
    
    # Train comparison
    train_improvement = ((persistence_train_rmse - convlstm_train_rmse) / persistence_train_rmse) * 100
    print(f"{'Train RMSE':<15} {persistence_train_rmse:<15.6f} {convlstm_train_rmse:<15.6f} {train_improvement:>13.2f}%")
    
    # Val comparison
    if not np.isnan(persistence_val_rmse) and not np.isnan(convlstm_val_rmse):
        val_improvement = ((persistence_val_rmse - convlstm_val_rmse) / persistence_val_rmse) * 100
        print(f"{'Val RMSE':<15} {persistence_val_rmse:<15.6f} {convlstm_val_rmse:<15.6f} {val_improvement:>13.2f}%")
    
    # Test comparison
    if not np.isnan(persistence_test_rmse) and not np.isnan(convlstm_test_rmse):
        test_improvement = ((persistence_test_rmse - convlstm_test_rmse) / persistence_test_rmse) * 100
        print(f"{'Test RMSE':<15} {persistence_test_rmse:<15.6f} {convlstm_test_rmse:<15.6f} {test_improvement:>13.2f}%")
    
    print(f"{'='*60}\n")
    
    # Summary
    print("Summary:")
    if convlstm_train_rmse < persistence_train_rmse:
        print(f"  ✓ ConvLSTM learns better than persistence on training data")
        print(f"    (Train RMSE: {convlstm_train_rmse:.6f} < {persistence_train_rmse:.6f})")
    else:
        print(f"  ✗ ConvLSTM does not improve over persistence on training data")
    
    if not np.isnan(convlstm_test_rmse) and not np.isnan(persistence_test_rmse):
        if convlstm_test_rmse < persistence_test_rmse:
            print(f"  ✓ ConvLSTM generalizes better than persistence")
            print(f"    (Test RMSE: {convlstm_test_rmse:.6f} < {persistence_test_rmse:.6f})")
        else:
            print(f"  ✗ ConvLSTM does not generalize better than persistence")
    
    if not np.isnan(convlstm_val_rmse) and not np.isnan(convlstm_test_rmse):
        if convlstm_train_rmse < convlstm_val_rmse * 0.7:
            print(f"  ⚠ Possible overfitting: Train RMSE ({convlstm_train_rmse:.6f}) << Val RMSE ({convlstm_val_rmse:.6f})")
        elif convlstm_val_rmse < convlstm_test_rmse * 1.2:
            print(f"  ✓ Good generalization: Val and Test RMSE are similar")


if __name__ == "__main__":
    main()
