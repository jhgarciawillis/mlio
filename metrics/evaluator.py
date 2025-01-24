import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import scipy.stats as stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    median_absolute_error
)

from core import config
from core.exceptions import MetricError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter

class MetricsEvaluator:
    """Handle metric evaluation and analysis."""
    
    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.performance_thresholds: Dict[str, float] = {}
        
    @monitor_performance
    @handle_exceptions(MetricError)
    def evaluate_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        cluster_labels: Optional[np.ndarray] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate model performance metrics."""
        try:
            # Record operation start
            state_monitor.record_operation_start(
                'metric_evaluation',
                'evaluation',
                {'clustering_enabled': cluster_labels is not None}
            )
            
            evaluation_id = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if evaluation_config is None:
                evaluation_config = self._get_default_evaluation_config()
            
            results = {
                'overall_metrics': self._calculate_overall_metrics(y_true, y_pred),
                'error_analysis': self._analyze_errors(y_true, y_pred),
                'distribution_analysis': self._analyze_distributions(y_true, y_pred)
            }
            
            # Add cluster-specific analysis if clustering is enabled
            if cluster_labels is not None:
                results['cluster_metrics'] = self._calculate_cluster_metrics(
                    y_true, y_pred, cluster_labels
                )
            
            # Add visualizations
            results['visualizations'] = self._create_evaluation_visualizations(
                y_true, y_pred, cluster_labels
            )
            
            # Store results
            self.evaluation_results[evaluation_id] = results
            
            # Record evaluation
            self._record_evaluation(evaluation_id, results)
            
            # Record operation completion
            state_monitor.record_operation_end(
                'metric_evaluation',
                'completed',
                {'evaluation_id': evaluation_id}
            )
            
            return results
            
        except Exception as e:
            state_monitor.record_operation_end(
                'metric_evaluation',
                'failed',
                {'error': str(e)}
            )
            raise MetricError(f"Error evaluating metrics: {str(e)}") from e
    
    @monitor_performance
    def _calculate_overall_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'explained_variance': float(explained_variance_score(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
            'median_ae': float(median_absolute_error(y_true, y_pred)),
            'correlation': float(np.corrcoef(y_true, y_pred)[0, 1])
        }
    
    @monitor_performance
    def _calculate_cluster_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each cluster."""
        cluster_metrics = {}
        
        for label in np.unique(cluster_labels):
            mask = cluster_labels == label
            if np.sum(mask) > 0:  # Only calculate if cluster has samples
                cluster_metrics[int(label)] = self._calculate_overall_metrics(
                    y_true[mask], y_pred[mask]
                )
        
        return cluster_metrics
    
    @monitor_performance
    def _analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        return {
            'error_statistics': {
                'mean_error': float(np.mean(errors)),
                'median_error': float(np.median(errors)),
                'std_error': float(np.std(errors)),
                'max_error': float(np.max(abs_errors)),
                'min_error': float(np.min(abs_errors))
            },
            'error_distribution': {
                'skewness': float(stats.skew(errors)),
                'kurtosis': float(stats.kurtosis(errors)),
                'normality_test': self._test_error_normality(errors)
            },
            'error_percentiles': {
                f'p{p}': float(np.percentile(abs_errors, p))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
        }
    
    @monitor_performance
    def _analyze_distributions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze true vs predicted distributions."""
        return {
            'true_distribution': {
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true)),
                'median': float(np.median(y_true)),
                'skew': float(stats.skew(y_true)),
                'kurtosis': float(stats.kurtosis(y_true))
            },
            'pred_distribution': {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'median': float(np.median(y_pred)),
                'skew': float(stats.skew(y_pred)),
                'kurtosis': float(stats.kurtosis(y_pred))
            },
            'distribution_comparison': self._compare_distributions(y_true, y_pred)
        }
    
    def _test_error_normality(
        self,
        errors: np.ndarray
    ) -> Dict[str, float]:
        """Test normality of error distribution."""
        statistic, p_value = stats.normaltest(errors)
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_normal': float(p_value) > 0.05
        }
    
    def _compare_distributions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compare true and predicted distributions."""
        ks_stat, ks_pval = stats.ks_2samp(y_true, y_pred)
        return {
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_pval),
            'distributions_similar': float(ks_pval) > 0.05
        }
    
    def _create_evaluation_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Create evaluation visualizations."""
        visualizations = {}
        
        # Actual vs Predicted plot
        visualizations['actual_vs_predicted'] = plotter.create_plot(
            'scatter',
            pd.DataFrame({'Actual': y_true, 'Predicted': y_pred}),
            x='Actual',
            y='Predicted',
            title='Actual vs Predicted Values'
        )
        
        # Error distribution plot
        errors = y_true - y_pred
        visualizations['error_distribution'] = plotter.create_plot(
            'histogram',
            pd.DataFrame({'Error': errors}),
            x='Error',
            title='Error Distribution'
        )
        
        # Q-Q plot
        visualizations['qq_plot'] = plotter.create_plot(
            'qq',
            data=errors,
            title='Q-Q Plot of Errors'
        )
        
        # Add cluster-specific visualizations if clustering is enabled
        if cluster_labels is not None:
            cluster_vis = {}
            for label in np.unique(cluster_labels):
                mask = cluster_labels == label
                if np.sum(mask) > 0:
                    cluster_vis[f'cluster_{label}'] = plotter.create_plot(
                        'scatter',
                        pd.DataFrame({
                            'Actual': y_true[mask],
                            'Predicted': y_pred[mask]
                        }),
                        x='Actual',
                        y='Predicted',
                        title=f'Actual vs Predicted (Cluster {label})'
                    )
            visualizations['cluster_plots'] = cluster_vis
        
        return visualizations
    
    def _get_default_evaluation_config(self) -> Dict[str, Any]:
        """Get default evaluation configuration."""
        return {
            'calculate_overall_metrics': True,
            'analyze_errors': True,
            'analyze_distributions': True,
            'create_visualizations': True
        }
    
    def _record_evaluation(
        self,
        evaluation_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Record evaluation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_id': evaluation_id,
            'metrics': results['overall_metrics']
        }
        
        self.evaluation_history.append(record)
        state_manager.set_state(
            f'metrics.evaluation.history.{len(self.evaluation_history)}',
            record
        )

# Create global metrics evaluator instance
metrics_evaluator = MetricsEvaluator()