#!/usr/bin/env python3
"""
Test GNN Transformer model's ability to overfit on a small sample.

Custom training loop to track per-epoch losses and see if it can memorize samples.
Compares against persistence baseline.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting.data_loader import load_data
from forecasting.splits import create_rolling_windows
from forecasting.pipeline import ForecastingPipeline
from models.gnn.gnn_model import GNNTransformerModel
from models.persistence.persistence_model import PersistenceModel
from evaluation.metrics import compute_all_metrics

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")
    sys.exit(1)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")


def main():
    print("=" * 70)
    print("Testing GNN Transformer - Memorization Test")
    print("W0, c5, h1 - Custom training loop with per-epoch stats")
    print("=" * 70)
    
    # Configuration
    window_id = 0
    context_length = 21
    horizon = 5
    
    max_train_samples = 10
    max_val_samples = 10
    max_test_samples = 30
    
    print(f"\n[1/4] Loading data...")
    data, tau_grid, logm_grid, dates = load_data('SPX_IV_fixed_grid.csv')
    print(f"Data shape: {data.shape}")
    
    print(f"\n[2/4] Creating rolling windows...")
    pipeline = ForecastingPipeline()
    pipeline.data = data
    pipeline.tau_grid = tau_grid
    pipeline.logm_grid = logm_grid
    pipeline.dates = dates
    pipeline.create_windows(
        window_size_years=10.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        shift_months=6
    )
    print(f"Created {len(pipeline.windows)} rolling windows")
    
    print(f"\n[3/4] Selecting consecutive days from Window {window_id}...")
    window = pipeline.windows[window_id]
    print(f"  Context: {context_length} days")
    print(f"  Horizon: {horizon} days")
    print(f"  Window range: {window.train_start.date()} to {window.test_end.date()}")
    
    # Select consecutive days from all window indices
    all_window_indices = np.concatenate([
        window.train_indices,
        window.val_indices,
        window.test_indices
    ])
    all_window_indices = np.sort(all_window_indices)
    
    # Calculate days needed for each split (accounting for sequence creation)
    train_days_needed = max_train_samples + context_length + horizon - 1
    val_days_needed = max_val_samples + context_length + horizon - 1
    test_days_needed = max_test_samples + context_length + horizon - 1
    total_days_needed = train_days_needed + val_days_needed + test_days_needed
    
    if len(all_window_indices) < total_days_needed:
        print(f"Warning: Not enough days. Need {total_days_needed}, have {len(all_window_indices)}")
        # Adjust sample sizes
        available = len(all_window_indices) - (context_length + horizon - 1) * 3
        max_train_samples = min(max_train_samples, available // 3)
        max_val_samples = min(max_val_samples, available // 3)
        max_test_samples = min(max_test_samples, available // 3)
        train_days_needed = max_train_samples + context_length + horizon - 1
        val_days_needed = max_val_samples + context_length + horizon - 1
        test_days_needed = max_test_samples + context_length + horizon - 1
        total_days_needed = train_days_needed + val_days_needed + test_days_needed
    
    # Pick random consecutive block
    max_start_idx = len(all_window_indices) - total_days_needed
    if max_start_idx < 0:
        print(f"ERROR: Cannot create sequences. Need {total_days_needed} days, have {len(all_window_indices)}")
        return
    start_idx = np.random.randint(0, max_start_idx + 1)
    selected_indices = all_window_indices[start_idx:start_idx + total_days_needed]
    selected_dates = dates[selected_indices]
    
    print(f"  Selected consecutive days: {selected_dates[0].date()} to {selected_dates[-1].date()}")
    
    # Split into train/val/test (consecutive blocks)
    train_end = train_days_needed
    val_start = train_end
    val_end = val_start + val_days_needed
    test_start = val_end
    
    train_indices = selected_indices[:train_end]
    val_indices = selected_indices[val_start:val_end]
    test_indices = selected_indices[test_start:]
    
    print(f"\n  Split:")
    print(f"    Train days: {dates[train_indices[0]].date()} to {dates[train_indices[-1]].date()} ({len(train_indices)} days)")
    print(f"    Val days: {dates[val_indices[0]].date()} to {dates[val_indices[-1]].date()} ({len(val_indices)} days)")
    print(f"    Test days: {dates[test_indices[0]].date()} to {dates[test_indices[-1]].date()} ({len(test_indices)} days)")
    
    # Build sequences
    from forecasting.data_loader import create_sequences
    
    train_data = data[train_indices]
    train_dates = dates[train_indices]
    X_train, y_train, _ = create_sequences(train_data, train_dates, context_length, horizon, verbose=False)
    
    val_data = data[val_indices]
    val_dates = dates[val_indices]
    X_val, y_val, _ = create_sequences(val_data, val_dates, context_length, horizon, verbose=False)
    
    test_data = data[test_indices]
    test_dates = dates[test_indices]
    X_test, y_test, _ = create_sequences(test_data, test_dates, context_length, horizon, verbose=False)
    
    # Limit sample sizes
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
    
    print(f"\n  Using {device} device")
    
    # Create model instance to get baseline computation
    model_wrapper = GNNTransformerModel(
        name="test_gnn_memorization",
        d_latent=128,
        d_model=256,
        n_heads=8,
        n_layers_gnn=3,
        n_layers_temporal=4,
        dropout=0.0,
        learning_rate=1e-1,
        weight_decay=0.0,
        batch_size=1,
        num_epochs=1,  # Not used, we'll train manually
        baseline_decay=-1,  # -1 means use persistence (last surface) as baseline
        use_gat=False,
        k_neighbors=8,
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
    
    n_samples, _, n_tau, n_logm = X_train.shape
    n_nodes = n_tau * n_logm
    
    # Normalization stats from training data
    X_train_flat = X_train.reshape(-1, n_tau, n_logm)
    mean_X = X_train_flat.mean(axis=0, keepdims=True)
    std_X = X_train_flat.std(axis=0, keepdims=True)
    std_X = np.maximum(std_X, 1e-8)
    
    # Normalize corrections with their own statistics
    y_correction_train_flat = y_correction_train.reshape(-1, n_tau, n_logm)
    mean_corr = y_correction_train_flat.mean(axis=0, keepdims=True)
    std_corr = y_correction_train_flat.std(axis=0, keepdims=True)
    std_corr = np.maximum(std_corr, 1e-8)
    
    # Build graph structure
    from models.gnn.gnn_model import build_surface_graph
    edge_index, _ = build_surface_graph(n_tau, n_logm, tau_grid, logm_grid, k_neighbors=8)
    edge_index = edge_index.to(device)
    
    # Create the GNN model components
    from models.gnn.gnn_model import SpatialGNNEncoder, SpatialGNNDecoder, TemporalTransformer
    
    encoder = SpatialGNNEncoder(n_nodes, d_latent=128, n_layers=3, dropout=0.0, use_gat=False).to(device)
    decoder = SpatialGNNDecoder(n_nodes, d_latent=128, n_layers=3, dropout=0.0, use_gat=False).to(device)
    temporal = TemporalTransformer(d_latent=128, context_length=context_length, d_model=64, 
                                   n_heads=4, n_layers=4, dropout=0.0).to(device)
    
    # Initialize weights
    for module in [encoder, decoder, temporal]:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(temporal.parameters()),
        lr=1e-1,
        weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Note: loss is computed as RMSE on surfaces (not MSE on corrections)
    scaler = GradScaler(enabled=False)
    
    # Convert to tensors
    X_train_t = torch.tensor((X_train - mean_X) / std_X, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    baseline_train_t = torch.tensor(baseline_train, dtype=torch.float32).to(device)
    
    X_val_t = torch.tensor((X_val - mean_X) / std_X, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    baseline_val_t = torch.tensor(baseline_val, dtype=torch.float32).to(device)
    
    X_test_t = torch.tensor((X_test - mean_X) / std_X, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    baseline_test_t = torch.tensor(baseline_test, dtype=torch.float32).to(device)
    
    mean_corr_t = torch.tensor(mean_corr, dtype=torch.float32).to(device)
    std_corr_t = torch.tensor(std_corr, dtype=torch.float32).to(device)
    
    print("\n[4/4] Training GNN with custom loop...")
    print(f"  Model: d_latent=128, d_model=256, n_heads=8, n_layers_gnn=3, n_layers_temporal=4")
    print(f"  Training on {len(X_train)} sample(s)")
    print(f"  Learning rate: 1e-1 (with ReduceLROnPlateau scheduler)")
    print(f"  No gradient clipping (free learning)")
    print(f"  Epochs: 50")
    
    # Debug: Check correction magnitude
    print(f"\n  Debug info:")
    print(f"    Correction magnitude: mean={np.abs(y_correction_train).mean():.6f}, max={np.abs(y_correction_train).max():.6f}")
    print(f"    Baseline magnitude: mean={np.abs(baseline_train).mean():.6f}")
    print(f"    Target magnitude: mean={np.abs(y_train).mean():.6f}")
    print(f"    Graph edges: {edge_index.shape[1] // 2} (undirected)")
    
    print("\n" + "=" * 70)
    print("EPOCH | TRAIN LOSS | VAL LOSS | TEST LOSS | TRAIN RMSE | VAL RMSE | TEST RMSE")
    print("=" * 70)
    
    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        # Training
        encoder.train()
        decoder.train()
        temporal.train()
        optimizer.zero_grad(set_to_none=True)
        
        # Encode context surfaces
        context_latents = []
        for t in range(context_length):
            latent_t = encoder(X_train_t[:, t, :, :], edge_index)
            context_latents.append(latent_t)
        context_latents = torch.stack(context_latents, dim=1)  # (batch, context_length, d_latent)
        
        # Temporal prediction
        future_latent = temporal(context_latents)  # (batch, d_latent)
        
        # Decode to correction
        pred_correction_norm = decoder(future_latent, edge_index, n_tau, n_logm)  # (batch, n_tau, n_logm)
        
        # Denormalize corrections
        pred_correction = pred_correction_norm * std_corr_t + mean_corr_t
        
        # Final prediction
        pred_train = baseline_train_t + pred_correction
        
        # Compute loss as RMSE on surfaces (not corrections)
        loss = torch.sqrt(torch.mean((pred_train - y_train_t) ** 2))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss = loss.item()
        scheduler.step(train_loss)
        
        # Evaluation (compute RMSE in original units)
        if epoch % 1 == 0 or epoch == num_epochs:
            encoder.eval()
            decoder.eval()
            temporal.eval()
            
            with torch.no_grad():
                # Train RMSE
                train_rmse = train_loss  # Same as loss now (RMSE on surfaces)
                
                # Val RMSE
                if len(X_val) > 0:
                    val_context_latents = []
                    for t in range(context_length):
                        latent_t = encoder(X_val_t[:, t, :, :], edge_index)
                        val_context_latents.append(latent_t)
                    val_context_latents = torch.stack(val_context_latents, dim=1)
                    val_future_latent = temporal(val_context_latents)
                    pred_val_correction_norm = decoder(val_future_latent, edge_index, n_tau, n_logm)
                    pred_val_correction = pred_val_correction_norm * std_corr_t + mean_corr_t
                    pred_val = baseline_val_t + pred_val_correction
                    val_loss = torch.sqrt(torch.mean((pred_val - y_val_t) ** 2)).item()
                    val_rmse = val_loss
                else:
                    val_loss = float('nan')
                    val_rmse = float('nan')
                
                # Test RMSE
                if len(X_test) > 0:
                    test_context_latents = []
                    for t in range(context_length):
                        latent_t = encoder(X_test_t[:, t, :, :], edge_index)
                        test_context_latents.append(latent_t)
                    test_context_latents = torch.stack(test_context_latents, dim=1)
                    test_future_latent = temporal(test_context_latents)
                    pred_test_correction_norm = decoder(test_future_latent, edge_index, n_tau, n_logm)
                    pred_test_correction = pred_test_correction_norm * std_corr_t + mean_corr_t
                    pred_test = baseline_test_t + pred_test_correction
                    test_loss = torch.sqrt(torch.mean((pred_test - y_test_t) ** 2)).item()
                    test_rmse = test_loss
                else:
                    test_loss = float('nan')
                    test_rmse = float('nan')
            
            val_str = f"{val_loss:8.6f}" if not np.isnan(val_loss) else "     N/A"
            test_str = f"{test_loss:9.6f}" if not np.isnan(test_loss) else "      N/A"
            val_rmse_str = f"{val_rmse:8.6f}" if not np.isnan(val_rmse) else "     N/A"
            test_rmse_str = f"{test_rmse:9.6f}" if not np.isnan(test_rmse) else "      N/A"
            print(f"{epoch:5d} | {train_loss:10.6f} | {val_str} | {test_str} | "
                  f"{train_rmse:10.6f} | {val_rmse_str} | {test_rmse_str}")
    
    print("=" * 70)
    
    # Final evaluation with metrics
    print("\n" + "=" * 70)
    print("FINAL COMPARISON WITH PERSISTENCE")
    print("=" * 70)
    
    # Get final predictions
    encoder.eval()
    decoder.eval()
    temporal.eval()
    
    with torch.no_grad():
        # Train predictions
        train_context_latents = []
        for t in range(context_length):
            latent_t = encoder(X_train_t[:, t, :, :], edge_index)
            train_context_latents.append(latent_t)
        train_context_latents = torch.stack(train_context_latents, dim=1)
        train_future_latent = temporal(train_context_latents)
        train_pred_correction_norm = decoder(train_future_latent, edge_index, n_tau, n_logm)
        train_pred_correction = train_pred_correction_norm * std_corr_t + mean_corr_t
        train_preds = (baseline_train_t + train_pred_correction).cpu().numpy()
        
        # Val predictions
        val_context_latents = []
        for t in range(context_length):
            latent_t = encoder(X_val_t[:, t, :, :], edge_index)
            val_context_latents.append(latent_t)
        val_context_latents = torch.stack(val_context_latents, dim=1)
        val_future_latent = temporal(val_context_latents)
        val_pred_correction_norm = decoder(val_future_latent, edge_index, n_tau, n_logm)
        val_pred_correction = val_pred_correction_norm * std_corr_t + mean_corr_t
        val_preds = (baseline_val_t + val_pred_correction).cpu().numpy()
        
        # Test predictions
        if len(X_test) > 0:
            test_context_latents = []
            for t in range(context_length):
                latent_t = encoder(X_test_t[:, t, :, :], edge_index)
                test_context_latents.append(latent_t)
            test_context_latents = torch.stack(test_context_latents, dim=1)
            test_future_latent = temporal(test_context_latents)
            test_pred_correction_norm = decoder(test_future_latent, edge_index, n_tau, n_logm)
            test_pred_correction = test_pred_correction_norm * std_corr_t + mean_corr_t
            test_preds = (baseline_test_t + test_pred_correction).cpu().numpy()
        else:
            test_preds = np.array([])
    
    # Persistence model
    persistence_model = PersistenceModel()
    persistence_model.fit(X_train, y_train)
    
    pers_train_preds = persistence_model.predict_horizon(X_train, horizon=horizon)
    pers_val_preds = persistence_model.predict_horizon(X_val, horizon=horizon)
    if len(X_test) > 0:
        pers_test_preds = persistence_model.predict_horizon(X_test, horizon=horizon)
    else:
        pers_test_preds = np.array([])
    
    # Compute metrics
    train_metrics_gnn = compute_all_metrics(y_train, train_preds)
    val_metrics_gnn = compute_all_metrics(y_val, val_preds)
    if len(X_test) > 0:
        test_metrics_gnn = compute_all_metrics(y_test, test_preds)
    else:
        test_metrics_gnn = None
    
    train_metrics_pers = compute_all_metrics(y_train, pers_train_preds)
    val_metrics_pers = compute_all_metrics(y_val, pers_val_preds)
    if len(X_test) > 0:
        test_metrics_pers = compute_all_metrics(y_test, pers_test_preds)
    else:
        test_metrics_pers = None
    
    print("\n[Train Set]")
    print(f"  GNN RMSE: {train_metrics_gnn['iv_rmse']:.6f}")
    print(f"  Persistence RMSE: {train_metrics_pers['iv_rmse']:.6f}")
    improvement = (train_metrics_pers['iv_rmse'] - train_metrics_gnn['iv_rmse']) / train_metrics_pers['iv_rmse'] * 100
    print(f"  Improvement: {improvement:+.2f}%")
    
    print("\n[Validation Set]")
    print(f"  GNN RMSE: {val_metrics_gnn['iv_rmse']:.6f}")
    print(f"  Persistence RMSE: {val_metrics_pers['iv_rmse']:.6f}")
    improvement = (val_metrics_pers['iv_rmse'] - val_metrics_gnn['iv_rmse']) / val_metrics_pers['iv_rmse'] * 100
    print(f"  Improvement: {improvement:+.2f}%")
    
    if len(X_test) > 0:
        print("\n[Test Set]")
        print(f"  GNN RMSE: {test_metrics_gnn['iv_rmse']:.6f}")
        print(f"  Persistence RMSE: {test_metrics_pers['iv_rmse']:.6f}")
        improvement = (test_metrics_pers['iv_rmse'] - test_metrics_gnn['iv_rmse']) / test_metrics_pers['iv_rmse'] * 100
        print(f"  Improvement: {improvement:+.2f}%")
    else:
        print("\n[Test Set]")
        print("  No test samples available")
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
