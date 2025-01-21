import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import scipy.stats as stats

from core import config
from core.exceptions import PredictionEvaluationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter
from metrics import calculator

class PredictionEvaluator:
    """Handle prediction evaluation operations."""
    
    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        
    @monitor_performance
    @handle_exceptions(PredictionEvaluationError)
    def evaluate_predictions(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive prediction evaluation."""
        try:
            evaluation_id = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if evaluation_config is None:
                evaluation_config = self._get_default_evaluation_config()
            
            results = {}
            
            # Basic error analysis
            if evaluation_config.get('error_analysis', True):
                results['error_analysis'] = self._analyze_errors(y_true, y_pred)
            
            # Distribution analysis
            if evaluation_config.get('distribution_analysis', True):
                results['distribution_analysis'] = self._analyze_distributions(y_true, y_pred)
            
            # Segmented analysis
            if evaluation_config.get('segmented_analysis', True):
                results['segmented_analysis'] = self._analyze_segments(
                    y_true,
                    y_pred,
                    evaluation_config.get('n_segments', 10)
                )
            
            # Uncertainty analysis
            if evaluation_config.get('uncertainty_analysis', True):
                results['uncertainty_analysis'] = self._analyze_uncertainty(y_true, y_pred)
            
            # Create visualizations
            results['visualizations'] = self._create_evaluation_plots(y_true, y_pred)
            
            # Store results
            self.evaluation_results[evaluation_id] = results
            
            # Record evaluation
            self._record_evaluation(evaluation_id, evaluation_config)
            
            return results
            
        except Exception as e:
            raise PredictionEvaluationError(
                f"Error evaluating predictions: {str(e)}"
            ) from e
    
    @monitor_performance
    def _analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        error_analysis = {
            'basic_stats': {
                'mean_error': float(np.mean(errors)),
                'median_error': float(np.median(errors)),
                'std_error': float(np.std(errors)),
                'mean_absolute_error': float(np.mean(abs_errors)),
                'median_absolute_error': float(np.median(abs_errors)),
                'max_error': float(np.max(abs_errors)),
                'min_error': float(np.min(abs_errors))
            }
        }
        
        # Error distribution analysis
        normality_test = stats.normaltest(errors)
        error_analysis['normality_test'] = {
            'statistic': float(normality_test.statistic),
            'p_value': float(normality_test.pvalue),
            'is_normal': float(normality_test.pvalue) > 0.05
        }
        
        # Error percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        error_analysis['error_percentiles'] = {
            f'p{p}': float(np.percentile(abs_errors, p))
            for p in percentiles
        }
        
        return error_analysis
    
    @monitor_performance
    def _analyze_distributions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze true vs predicted distributions."""
        distribution_analysis = {
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
            }
        }
        
        # Distribution comparison tests
        ks_test = stats.ks_2samp(y_true, y_pred)
        distribution_analysis['distribution_comparison'] = {
            'ks_statistic': float(ks_test.statistic),
            'ks_p_value': float(ks_test.pvalue),
            'distributions_similar': float(ks_test.pvalue) > 0.05
        }
        
        return distribution_analysis
    
    @monitor_performance
    def _analyze_segments(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_segments: int = 10
    ) -> Dict[str, Any]:
        """Analyze prediction performance across different segments."""
        # Create segments based on true values
        segment_bounds = np.percentile(
            y_true,
            np.linspace(0, 100, n_segments + 1)
        )
        
        segment_analysis = {}
        for i in range(n_segments):
            mask = (y_true >= segment_bounds[i]) & (y_true < segment_bounds[i+1])
            if mask.any():
                segment_true = y_true[mask]
                segment_pred = y_pred[mask]
                
                segment_analysis[f'segment_{i+1}'] = {
                    'range': (float(segment_bounds[i]), float(segment_bounds[i+1])),
                    'count': int(np.sum(mask)),
                    'mean_error': float(np.mean(segment_true - segment_pred)),
                    'mean_absolute_error': float(np.mean(np.abs(segment_true - segment_pred))),
                    'std_error': float(np.std(segment_true - segment_pred))
                }
        
        return segment_analysis
    
    @monitor_performance
    def _analyze_uncertainty(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction uncertainty."""
        errors = y_true - y_pred
        
        # Calculate confidence intervals
        confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        uncertainty_analysis = {
            'confidence_intervals': {}
        }
        
        for level in confidence_levels:
            interval = stats.norm.interval(
                level,
                loc=np.mean(errors),
                scale=stats.sem(errors)
            )
            uncertainty_analysis['confidence_intervals'][f'level_{level}'] = {
                'lower': float(interval[0]),
                'upper': float(interval[1])
            }
        
        # Calculate prediction intervals
        prediction_std = np.std(errors)
        uncertainty_analysis['prediction_intervals'] = {
            f'level_{level}': {
                'lower': float(stats.norm.ppf((1-level)/2, scale=prediction_std)),
                'upper': float(stats.norm.ppf((1+level)/2, scale=prediction_std))
            }
            for level in confidence_levels
        }
        
        return uncertainty_analysis
    
    @monitor_performance
    def _create_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Create evaluation visualizations."""
        plots = {}
        
        # Actual vs Predicted scatter plot
        plots['actual_vs_predicted'] = plotter.create_plot(
            'scatter',
            data=pd.DataFrame({'Actual': y_true, 'Predicted': y_pred}),
            x='Actual',
            y='Predicted',
            title='Actual vs Predicted Values'
        )
        
        # Error distribution plot
        errors = y_true - y_pred
        plots['error_distribution'] = plotter.create_plot(
            'histogram',
            data=pd.DataFrame({'Error': errors}),
            x='Error',
            title='Error Distribution'
        )
        
        # Q-Q plot for error normality
        plots['qq_plot'] = plotter.create_plot(
            'qq',
            data=errors,
            title='Q-Q Plot of Prediction Errors'
        )
        
        # Error vs Predicted plot
        plots['error_vs_predicted'] = plotter.create_plot(
            'scatter',
            data=pd.DataFrame({'Predicted': y_pred, 'Error': errors}),
            x='Predicted',
            y='Error',
            title='Error vs Predicted Values'
        )
        
        return plots
    
    def _get_default_evaluation_config(self) -> Dict[str, Any]:
        """Get default evaluation configuration."""
        return {
            'error_analysis': True,
            'distribution_analysis': True,
            'segmented_analysis': True,
            'uncertainty_analysis': True,
            'n_segments': 10
        }
    
    def _record_evaluation(
        self,
        evaluation_id: str,
        evaluation_config: Dict[str, Any]
    ) -> None:
        """Record evaluation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_id': evaluation_id,
            'configuration': evaluation_config
        }
        
        self.evaluation_history.append(record)
        state_manager.set_state(
            f'prediction.evaluation.history.{len(self.evaluation_history)}',
            record
        )

# Create global prediction evaluator instance
prediction_evaluator = PredictionEvaluator()