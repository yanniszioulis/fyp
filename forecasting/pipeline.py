"""
Main forecasting pipeline orchestrator.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Optional, Any
import os
import json
from datetime import datetime

from forecasting.data_loader import load_data, create_sequences
from forecasting.splits import create_rolling_windows, WindowSplit
from models.persistence.persistence_model import PersistenceModel
from evaluation.metrics import (compute_all_metrics, compute_metrics_by_maturity,
                               compute_metrics_by_moneyness)


class ForecastingPipeline:
    """Main pipeline for forecasting volatility surfaces"""
    
    def __init__(self, data_file: str = 'SPX_IV_fixed_grid.csv',
                 results_dir: str = 'results'):
        self.data_file = data_file
        self.results_dir = results_dir
        self.data = None
        self.tau_grid = None
        self.logm_grid = None
        self.dates = None
        self.windows = None
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'forecasts'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    def load_data(self):
        """Load and prepare data"""
        self.data, self.tau_grid, self.logm_grid, self.dates = load_data(self.data_file)
        print(f"Loaded data: shape {self.data.shape}")
    
    def create_windows(self, window_size_years: float = 10.0,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.2,
                      shift_months: int = 6):
        """Create rolling window splits"""
        self.windows = create_rolling_windows(
            self.dates,
            window_size_years=window_size_years,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            shift_months=shift_months
        )
        print(f"Created {len(self.windows)} rolling windows")
    
    def _ensure_loaded(self):
        if self.data is None or self.dates is None:
            raise ValueError("Must load data first (call load_data())")
    
    def _ensure_windows(self):
        if self.windows is None:
            raise ValueError("Must create windows first (call create_windows())")
    
    def _print_window_header(self, window: WindowSplit):
        print(f"\n{'='*60}")
        print(f"Window {window.window_id + 1}/{len(self.windows)}")
        print(f"Train: {window.train_start.date()} to {window.train_end.date()}")
        print(f"Val: {window.val_start.date()} to {window.val_end.date()}")
        print(f"Test: {window.test_start.date()} to {window.test_end.date()}")
        print(f"{'='*60}")
    
    def _build_sequences_for_window(self, window: WindowSplit,
                                    context_length: int,
                                    horizon: int,
                                    train_on_val: bool = True,
                                    use_val: bool = False
                                    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
                                               np.ndarray, np.ndarray, np.ndarray]:
        if train_on_val:
            train_indices = np.concatenate([window.train_indices, window.val_indices])
        else:
            train_indices = window.train_indices
        
        train_data = self.data[train_indices]
        train_dates = self.dates[train_indices]
        
        X_train, y_train, _ = create_sequences(
            train_data, train_dates,
            context_length=context_length,
            horizon=horizon,
            verbose=False
        )
        
        X_val = None
        y_val = None
        if use_val:
            train_val_indices = np.concatenate([window.train_indices, window.val_indices])
            train_val_data = self.data[train_val_indices]
            train_val_dates = self.dates[train_val_indices]
            X_val_all, y_val_all, val_dates = create_sequences(
                train_val_data, train_val_dates,
                context_length=context_length,
                horizon=horizon,
                verbose=False
            )
            if len(val_dates) > 0:
                val_mask = (val_dates >= window.val_start) & (val_dates <= window.val_end)
                X_val = X_val_all[val_mask]
                y_val = y_val_all[val_mask]
        
        combined_indices = np.concatenate([window.val_indices, window.test_indices])
        combined_data = self.data[combined_indices]
        combined_dates = self.dates[combined_indices]
        
        X_test, y_test, sample_dates = create_sequences(
            combined_data, combined_dates,
            context_length=context_length,
            horizon=horizon,
            verbose=False
        )
        
        if len(sample_dates) == 0:
            return X_train, y_train, X_val, y_val, X_test, y_test, sample_dates
        
        test_mask = (sample_dates >= window.test_start) & (sample_dates <= window.test_end)
        return X_train, y_train, X_val, y_val, X_test[test_mask], y_test[test_mask], sample_dates[test_mask]
    
    def _evaluate_and_package(self, window: WindowSplit, context_length: int, horizon: int,
                              predictions: np.ndarray, y_test: np.ndarray,
                              extra_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metrics = compute_all_metrics(y_test, predictions)
        metrics_by_tau = compute_metrics_by_maturity(
            y_test, predictions, self.tau_grid, self.tau_grid
        )
        metrics_by_moneyness = compute_metrics_by_moneyness(
            y_test, predictions, self.tau_grid, self.logm_grid
        )
        
        print(f"    IV RMSE: {metrics['iv_rmse']:.4f}")
        print(f"    Relative RMSE: {metrics['relative_rmse']:.4f}")
        print(f"    MAE: {metrics['mae']:.4f}")
        
        result = {
            'window_id': window.window_id,
            'context_length': context_length,
            'horizon': horizon,
            'metrics': metrics,
            'metrics_by_tau': {str(k): v for k, v in metrics_by_tau.items()},
            'metrics_by_moneyness': metrics_by_moneyness,
            'n_samples': len(predictions),
            'test_start': window.test_start.isoformat(),
            'test_end': window.test_end.isoformat()
        }
        
        if extra_fields:
            result.update(extra_fields)
        
        return result
    
    def run_model(self,
                  model_factory: Callable[..., Any],
                  model_id: str,
                  context_lengths: List[int],
                  horizons: List[int],
                  train_on_val: bool = True,
                  model_kwargs: Optional[Dict[str, Any]] = None,
                  fit_kwargs: Optional[Dict[str, Any]] = None,
                  predict_kwargs: Optional[Dict[str, Any]] = None,
                  extra_result_fields: Optional[Dict[str, Any]] = None,
                  use_val: bool = False,
                  save_checkpoints: bool = False,
                  min_train_samples: int = 1,
                  save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Generic model runner for windowed experiments.
        
        model_factory should accept `name` plus any model_kwargs and return a model instance.
        """
        self._ensure_loaded()
        self._ensure_windows()
        
        model_kwargs = model_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        predict_kwargs = predict_kwargs or {}
        
        all_results = []
        
        for window in self.windows:
            self._print_window_header(window)
            
            for context_length in context_lengths:
                for horizon in horizons:
                    print(f"\n  Context: {context_length} days, Horizon: {horizon} days")
                    
                    try:
                        X_train, y_train, X_val, y_val, X_test, y_test, sample_dates = self._build_sequences_for_window(
                            window=window,
                            context_length=context_length,
                            horizon=horizon,
                            train_on_val=train_on_val,
                            use_val=use_val
                        )
                        
                        if len(X_test) == 0:
                            print("    Skipping: No test samples available")
                            continue
                        
                        if len(X_train) < min_train_samples:
                            if min_train_samples > 0:
                                print(f"    Skipping: Insufficient training data ({len(X_train)} samples)")
                                continue
                        
                        model_name = f"{model_id}_w{window.window_id}_c{context_length}_h{horizon}"
                        model = model_factory(name=model_name, **model_kwargs)
                        
                        fit_params = dict(
                            context_length=context_length,
                            horizon=horizon
                        )
                        if fit_kwargs:
                            fit_params.update(fit_kwargs)
                        if X_val is not None and y_val is not None:
                            fit_params["X_val"] = X_val
                            fit_params["y_val"] = y_val
                        # Pass tau_grid and logm_grid for GNN models
                        if hasattr(self, 'tau_grid') and self.tau_grid is not None:
                            fit_params["tau_grid"] = self.tau_grid
                        if hasattr(self, 'logm_grid') and self.logm_grid is not None:
                            fit_params["logm_grid"] = self.logm_grid
                        model.fit(
                            X_train if len(X_train) > 0 else None,
                            y_train if len(X_train) > 0 else None,
                            **fit_params
                        )

                        if save_checkpoints and hasattr(model, "save_checkpoint"):
                            checkpoint_dir = os.path.join(
                                os.path.dirname(__file__),
                                "..",
                                "models",
                                model_id,
                                "checkpoints"
                            )
                            checkpoint_dir = os.path.abspath(checkpoint_dir)
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            checkpoint_path = os.path.join(
                                checkpoint_dir,
                                f"{model_id}_w{window.window_id}_c{context_length}_h{horizon}.pt"
                            )
                            model.save_checkpoint(checkpoint_path)
                        
                        predictions = model.predict_horizon(X_test, horizon=horizon, **predict_kwargs)
                        
                        result = self._evaluate_and_package(
                            window=window,
                            context_length=context_length,
                            horizon=horizon,
                            predictions=predictions,
                            y_test=y_test,
                            extra_fields=extra_result_fields
                        )
                        all_results.append(result)
                        
                        if save_results:
                            forecast_file = os.path.join(
                                self.results_dir, 'forecasts',
                                f'{model_id}_w{window.window_id}_c{context_length}_h{horizon}.npz'
                            )
                            np.savez(forecast_file,
                                     predictions=predictions,
                                     true=y_test,
                                     dates=sample_dates)
                    
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                        continue
        
        if save_results:
            summary_file = os.path.join(self.results_dir, 'metrics', f'{model_id}_results.json')
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {summary_file}")
        
        return all_results
    
    def run_persistence(self, context_lengths: List[int] = [5, 21, 63],
                        horizons: List[int] = [1, 5, 21],
                        save_results: bool = True):
        """
        Run persistence model across all windows and configurations.
        """
        return self.run_model(
            model_factory=lambda name, **kwargs: PersistenceModel(name=name),
            model_id="persistence",
            context_lengths=context_lengths,
            horizons=horizons,
            train_on_val=True,
            min_train_samples=0,
            save_results=save_results
        )
    
    


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = ForecastingPipeline()
    
    # Load data
    pipeline.load_data()
    
    # Create rolling windows (10 years, 7:1:2 split, 6-month shifts)
    pipeline.create_windows(
        window_size_years=10.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        shift_months=6
    )
    
    # Run persistence baseline
    results = pipeline.run_persistence(
        context_lengths=[5, 21, 63],  # 1 week, 1 month, 3 months
        horizons=[1, 5, 21],  # 1 day, 1 week, 1 month
        save_results=True
    )
    
    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print(f"Total configurations tested: {len(results)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
