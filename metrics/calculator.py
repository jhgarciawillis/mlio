import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error
)
from datetime import datetime

from core import config
from core.exceptions import MetricError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class MetricsCalculator:
    """Handle metric calculations."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.custom_metrics: Dict[str, callable] = {}
        self.available_metrics = {
            'mse': mean_squared_error,
            'rmse': self._calculate_rmse,
            'mae': mean_absolute_error,
            'r2': r2_score,
            'explained_variance': explained_variance_score,
            'mape': mean_absolute_percentage_error,
            'median_ae': median_absolute_error,
            'max_error': max_error,
            'pearson_correlation': self._calculate_pearson,
            'spearman_correlation': self._calculate_spearman
        }
    
    @monitor_performance
    @handle_exceptions(MetricError)
    def calculate_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate specified metrics."""
        try:
            # Convert inputs to numpy arrays
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            
            # Use default metrics if none specified
            if metrics is None:
                metrics = list(self.available_metrics.keys())
            
            results = {}
            
            # Calculate standard metrics
            for metric in metrics:
                if metric in self.available_metrics:
                    if sample_weight is not None and metric in ['mse', 'rmse', 'mae', 'r2']:
                        value = self.available_metrics[metric](
                            y_true, y_pred, sample_weight=sample_weight
                        )
                    else:
                        value = self.available_metrics[metric](y_true, y_pred)
                    results[metric] = float(value)
                elif metric in self.custom_metrics:
                    value = self.custom_metrics[metric](y_true, y_pred)
                    results[metric] = float(value)
            
            # Add residual analysis
            results.update(self._analyze_residuals(y_true - y_pred))
            
            # Record calculation
            self._record_calculation(metrics, results)
            
            return results
            
        except Exception as e:
            raise MetricError(
                f"Error calculating metrics: {str(e)}"
            ) from e
    
    @monitor_performance
    def register_custom_metric(
        self,
        name: str,
        metric_function: callable,
        description: Optional[str] = None
    ) -> None:
        """Register custom metric function."""
        if name in self.available_metrics:
            raise MetricError(f"Cannot override default metric: {name}")
            
        self.custom_metrics[name] = metric_function
        logger.info(f"Registered custom metric: {name}")
    
    @monitor_performance
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        return list(self.available_metrics.keys()) + list(self.custom_metrics.keys())
    
    def _calculate_rmse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight
        )))
    
    def _calculate_pearson(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        return float(stats.pearsonr(y_true, y_pred)[0])
    
    def _calculate_spearman(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate Spearman correlation coefficient."""
        return float(stats.spearmanr(y_true, y_pred)[0])
    
    def _analyze_residuals(
        self,
        residuals: np.ndarray
    ) -> Dict[str, float]:
        """Analyze residual statistics."""
        return {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_skew': float(stats.skew(residuals)),
            'residual_kurtosis': float(stats.kurtosis(residuals))
        }
    
    def _record_calculation(
        self,
        metrics: List[str],
        results: Dict[str, float]
    ) -> None:
        """Record metric calculation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'metrics_calculated': metrics,
            'results': results
        }
        
        self.metrics_history.append(record)
        state_manager.set_state(
            f'metrics.history.{len(self.metrics_history)}',
            record
        )

# Create global metrics calculator instance
metrics_calculator = MetricsCalculator()