import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import scipy.stats as stats

from core import config
from core.exceptions import MetricError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from metrics import calculator
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
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate model performance metrics."""
        try:
            evaluation_id = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if evaluation_config is None:
                evaluation_config = self._get_default_evaluation_config()
            
            results = {
                'basic_metrics': metrics_calculator.calculate_metrics(
                    y_true, y_pred,
                    metrics=evaluation_config.get('metrics', None)
                ),
                'performance_analysis': self._analyze_performance(
                    y_true, y_pred
                ),
                'error_analysis': self._analyze_errors(
                    y_true, y_pred
                )
            }
            
            # Add threshold evaluation if thresholds are set
            if self.performance_thresholds:
                results['threshold_evaluation'] = self._evaluate_thresholds(
                    results['basic_metrics']
                )
            
            # Create visualizations
            results['visualizations'] = self._create_evaluation_visualizations(
                y_true, y_pred, results
            )
            
            # Store results
            self.evaluation_results[evaluation_id] = results
            
            # Record evaluation
            self._record_evaluation(evaluation_id, evaluation_config)
            
            return results
            
        except Exception as e:
            raise MetricError(
                f"Error evaluating metrics: {str(e)}"
            ) from e
    
    @monitor_performance
    def set_performance_thresholds(
        self,
        thresholds: Dict[str, float]
    ) -> None:
        """Set performance threshold values."""
        valid_metrics = metrics_calculator.get_available_metrics()
        invalid_metrics = set(thresholds.keys()) - set(valid_metrics)
        
        if invalid_metrics:
            raise MetricError(f"Invalid metrics in thresholds: {invalid_metrics}")
            
        self.performance_thresholds = thresholds
        logger.info(f"Set performance thresholds: {thresholds}")
    
    def _analyze_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze model performance patterns."""
        analysis = {
            'value_ranges': self._analyze_value_ranges(y_true, y_pred),
            'bias_analysis': self._analyze_prediction_bias(y_true, y_pred),
            'error_patterns': self._analyze_error_patterns(y_true, y_pred)
        }
        
        return analysis
    
    def _analyze_value_ranges(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze performance across value ranges."""
        # Create value bins
        bins = pd.qcut(y_true, q=10, duplicates='drop')
        
        range_analysis = {}
        for bin_label in bins.unique():
            mask = bins == bin_label
            if mask.any():
                range_metrics = metrics_calculator.calculate_metrics(
                    y_true[mask], y_pred[mask]
                )
                range_analysis[str(bin_label)] = range_metrics
        
        return range_analysis
    
    def _analyze_prediction_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction bias."""
        errors = y_pred - y_true
        
        return {
            'overall_bias': float(np.mean(errors)),
            'bias_std': float(np.std(errors)),
            'positive_bias_ratio': float(np.mean(errors > 0)),
            'negative_bias_ratio': float(np.mean(errors < 0))
        }
    
    def _analyze_error_patterns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze patterns in prediction errors."""
        errors = np.abs(y_pred - y_true)
        
        return {
            'error_distribution': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'skew': float(stats.skew(errors)),
                'kurtosis': float(stats.kurtosis(errors))
            },
            'percentile_errors': {
                str(p): float(np.percentile(errors, p))
                for p in [25, 50, 75, 90, 95, 99]
            }
        }
    
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
                'mean_absolute_error': float(np.mean(abs_errors)),
                'median_absolute_error': float(np.median(abs_errors)),
                'max_error': float(np.max(abs_errors)),
                'min_error': float(np.min(abs_errors))
            },
            'error_distribution': {
                'skewness': float(stats.skew(errors)),
                'kurtosis': float(stats.kurtosis(errors)),
                'normality_test': self._test_error_normality(errors)
            }
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
    
    def _evaluate_thresholds(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, bool]:
        """Evaluate metrics against set thresholds."""
        return {
            metric: metrics[metric] >= threshold
            for metric, threshold in self.performance_thresholds.items()
            if metric in metrics
        }
    
    def _create_evaluation_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create evaluation visualizations."""
        visualizations = {}
        
        # Actual vs Predicted
        visualizations['actual_vs_predicted'] = plotter.create_plot(
            'scatter',
            data=pd.DataFrame({
                'Actual': y_true,
                'Predicted': y_pred
            }),
            x='Actual',
            y='Predicted',
            title='Actual vs Predicted Values'
        )
        
        # Error Distribution
        errors = y_true - y_pred
        visualizations['error_distribution'] = plotter.create_plot(
            'histogram',
            data=pd.DataFrame({'Error': errors}),
            x='Error',
            title='Error Distribution'
        )
        
        # Error vs Predicted
        visualizations['error_vs_predicted'] = plotter.create_plot(
            'scatter',
            data=pd.DataFrame({
                'Predicted': y_pred,
                'Error': errors
            }),
            x='Predicted',
            y='Error',
            title='Error vs Predicted Values'
        )
        
        # Performance by Value Range
        range_metrics = pd.DataFrame(results['performance_analysis']['value_ranges']).T
        visualizations['range_performance'] = plotter.create_plot(
            'line',
            data=range_metrics.reset_index(),
            x='index',
            y='r2',
            title='R² Score by Value Range'
        )
        
        return visualizations
    
    def _get_default_evaluation_config(self) -> Dict[str, Any]:
        """Get default evaluation configuration."""
        return {
            'metrics': None,  # Use all available metrics
            'value_range_bins': 10,
            'error_percentiles': [25, 50, 75, 90, 95, 99],
            'visualization_config': {
                'width': config.ui.chart_width,
                'height': config.ui.chart_height
            }
        }
    
    def _record_evaluation(
        self,
        evaluation_id: str,
        config: Dict[str, Any]
    ) -> None:
        """Record evaluation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_id': evaluation_id,
            'configuration': config
        }
        
        self.evaluation_history.append(record)
        state_manager.set_state(
            f'metrics.evaluation.history.{len(self.evaluation_history)}',
            record
        )

# Create global metrics evaluator instance
metrics_evaluator = MetricsEvaluator()