import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import scipy.stats as stats
from sklearn.inspection import permutation_importance
from scipy.spatial.distance import cdist

from core import config
from core.exceptions import EvaluationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter
from metrics import calculator

class PredictionEvaluator:
    """Handle prediction evaluation operations with clustering support."""
    
    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.evaluation_plots: Dict[str, Dict[str, Any]] = {}
        self.evaluation_metadata: Dict[str, Dict[str, Any]] = {}
        
    @monitor_performance
    @handle_exceptions(EvaluationError)
    def evaluate_predictions(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        cluster_labels: Optional[np.ndarray] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate predictions with clustering support."""
        try:
            # Generate evaluation ID
            evaluation_id = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if evaluation_config is None:
                evaluation_config = self._get_default_evaluation_config()
            
            results = {}
            
            # Basic error analysis
            if evaluation_config.get('error_analysis', True):
                results['error_analysis'] = self._analyze_errors(
                    y_true, y_pred, cluster_labels
                )
            
            # Distribution analysis
            if evaluation_config.get('distribution_analysis', True):
                results['distribution_analysis'] = self._analyze_distributions(
                    y_true, y_pred, cluster_labels
                )
            
            # Segmented analysis
            if evaluation_config.get('segmented_analysis', True):
                results['segmented_analysis'] = self._analyze_segments(
                    y_true,
                    y_pred,
                    evaluation_config.get('n_segments', 10),
                    cluster_labels
                )
            
            # Cluster-specific analysis
            if cluster_labels is not None:
                results['cluster_analysis'] = self._analyze_cluster_performance(
                    y_true, y_pred, cluster_labels
                )
            
            # Uncertainty analysis
            if evaluation_config.get('uncertainty_analysis', True):
                results['uncertainty_analysis'] = self._analyze_uncertainty(
                    y_true, y_pred, cluster_labels
                )
            
            # Create visualizations
            results['visualizations'] = self._create_evaluation_plots(
                y_true, y_pred, cluster_labels
            )
            
            # Store results
            self.evaluation_results[evaluation_id] = results
            self.evaluation_metadata[evaluation_id] = {
                'timestamp': datetime.now().isoformat(),
                'config': evaluation_config,
                'clustering_used': cluster_labels is not None
            }
            
            # Record evaluation
            self._record_evaluation(evaluation_id, results)
            
            return results
            
        except Exception as e:
            raise EvaluationError(f"Error evaluating predictions: {str(e)}") from e
    
    @monitor_performance
    def _analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cluster_labels: Optional[np.ndarray]
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
            },
            'error_distribution': {
                'skewness': float(stats.skew(errors)),
                'kurtosis': float(stats.kurtosis(errors)),
                'normality_test': self._test_error_normality(errors)
            }
        }
        
        # Cluster-specific error analysis
        if cluster_labels is not None:
            cluster_errors = {}
            for cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                cluster_errors[f'cluster_{cluster_id}'] = {
                    'mean_error': float(np.mean(errors[mask])),
                    'std_error': float(np.std(errors[mask])),
                    'mean_absolute_error': float(np.mean(abs_errors[mask])),
                    'max_error': float(np.max(abs_errors[mask])),
                    'sample_size': int(np.sum(mask))
                }
            error_analysis['cluster_errors'] = cluster_errors
        
        return error_analysis
    
    @monitor_performance
    def _analyze_distributions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cluster_labels: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze prediction and true value distributions."""
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
        
        # Cluster-specific distribution analysis
        if cluster_labels is not None:
            cluster_distributions = {}
            for cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                cluster_distributions[f'cluster_{cluster_id}'] = {
                    'true_mean': float(np.mean(y_true[mask])),
                    'pred_mean': float(np.mean(y_pred[mask])),
                    'true_std': float(np.std(y_true[mask])),
                    'pred_std': float(np.std(y_pred[mask])),
                    'ks_test': self._perform_ks_test(
                        y_true[mask], y_pred[mask]
                    )
                }
            distribution_analysis['cluster_distributions'] = cluster_distributions
        
        return distribution_analysis
    
    @monitor_performance
    def _analyze_segments(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_segments: int,
        cluster_labels: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze prediction performance across segments."""
        segment_analysis = {}
        
        # Create segments based on true values
        segment_bounds = np.percentile(
            y_true,
            np.linspace(0, 100, n_segments + 1)
        )
        
        # Global segment analysis
        for i in range(n_segments):
            mask = (y_true >= segment_bounds[i]) & (y_true < segment_bounds[i+1])
            if mask.any():
                segment_analysis[f'segment_{i+1}'] = {
                    'range': (float(segment_bounds[i]), float(segment_bounds[i+1])),
                    'count': int(np.sum(mask)),
                    'mean_error': float(np.mean(y_true[mask] - y_pred[mask])),
                    'mean_absolute_error': float(np.mean(np.abs(y_true[mask] - y_pred[mask]))),
                    'std_error': float(np.std(y_true[mask] - y_pred[mask]))
                }
        
        # Cluster-specific segment analysis
        if cluster_labels is not None:
            cluster_segments = {}
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                cluster_segments[f'cluster_{cluster_id}'] = {}
                
                # Create cluster-specific segments
                cluster_bounds = np.percentile(
                    y_true[cluster_mask],
                    np.linspace(0, 100, n_segments + 1)
                )
                
                for i in range(n_segments):
                    segment_mask = (
                        (y_true[cluster_mask] >= cluster_bounds[i]) &
                        (y_true[cluster_mask] < cluster_bounds[i+1])
                    )
                    if segment_mask.any():
                        cluster_segments[f'cluster_{cluster_id}'][f'segment_{i+1}'] = {
                            'range': (float(cluster_bounds[i]), float(cluster_bounds[i+1])),
                            'count': int(np.sum(segment_mask)),
                            'mean_error': float(np.mean(
                                y_true[cluster_mask][segment_mask] -
                                y_pred[cluster_mask][segment_mask]
                            )),
                            'mean_absolute_error': float(np.mean(np.abs(
                                y_true[cluster_mask][segment_mask] -
                                y_pred[cluster_mask][segment_mask]
                            ))),
                            'std_error': float(np.std(
                                y_true[cluster_mask][segment_mask] -
                                y_pred[cluster_mask][segment_mask]
                            ))
                        }
            
            segment_analysis['cluster_segments'] = cluster_segments
        
        return segment_analysis
    
    @monitor_performance
    def _analyze_cluster_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction performance for each cluster."""
        cluster_analysis = {
            'metrics': {},
            'cross_cluster': {},
            'stability': {}
        }
        
        # Calculate metrics for each cluster
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_analysis['metrics'][f'cluster_{cluster_id}'] = {
                'size': int(np.sum(mask)),
                'metrics': calculator.calculate_metrics(
                    y_true[mask],
                    y_pred[mask]
                ),
                'error_distribution': self._analyze_error_distribution(
                    y_true[mask] - y_pred[mask]
                )
            }
        
        # Cross-cluster analysis
        for c1 in np.unique(cluster_labels):
            for c2 in np.unique(cluster_labels):
                if c1 < c2:
                    mask1 = cluster_labels == c1
                    mask2 = cluster_labels == c2
                    
                    # Compare error distributions
                    errors1 = y_true[mask1] - y_pred[mask1]
                    errors2 = y_true[mask2] - y_pred[mask2]
                    
                    cross_analysis = {
                        'ks_test': self._perform_ks_test(errors1, errors2),
                        'mean_difference': float(np.mean(errors1) - np.mean(errors2)),
                        'std_ratio': float(np.std(errors1) / np.std(errors2))
                    }
                    
                    cluster_analysis['cross_cluster'][f'cluster_{c1}_vs_{c2}'] = cross_analysis
        
        # Cluster stability analysis
        unique_clusters = np.unique(cluster_labels)
        stability_matrix = np.zeros((len(unique_clusters), len(unique_clusters)))
        
        for i, c1 in enumerate(unique_clusters):
            for j, c2 in enumerate(unique_clusters):
                mask1 = cluster_labels == c1
                mask2 = cluster_labels == c2
                errors1 = y_true[mask1] - y_pred[mask1]
                errors2 = y_true[mask2] - y_pred[mask2]
                
                if len(errors1) > 0 and len(errors2) > 0:
                    stability_matrix[i, j] = np.mean(cdist(
                        errors1.reshape(-1, 1),
                        errors2.reshape(-1, 1)
                    ))
        
        cluster_analysis['stability']['stability_matrix'] = stability_matrix.tolist()
        cluster_analysis['stability']['stability_score'] = float(
            np.mean(np.diag(stability_matrix))
        )
        
        return cluster_analysis
    
    @monitor_performance
    def _analyze_uncertainty(
       self,
       y_true: np.ndarray,
       y_pred: np.ndarray,
       cluster_labels: Optional[np.ndarray]
   ) -> Dict[str, Any]:
       """Analyze prediction uncertainty."""
       errors = y_true - y_pred
       
       uncertainty_analysis = {
           'confidence_intervals': {},
           'prediction_intervals': {},
           'error_bounds': {}
       }
       
       # Calculate confidence intervals
       confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
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
       for level in confidence_levels:
           uncertainty_analysis['prediction_intervals'][f'level_{level}'] = {
               'lower': float(stats.norm.ppf((1-level)/2, scale=prediction_std)),
               'upper': float(stats.norm.ppf((1+level)/2, scale=prediction_std))
           }
       
       # Calculate empirical error bounds
       percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
       uncertainty_analysis['error_bounds'] = {
           f'p{p}': float(np.percentile(np.abs(errors), p))
           for p in percentiles
       }
       
       # Cluster-specific uncertainty
       if cluster_labels is not None:
           cluster_uncertainty = {}
           for cluster_id in np.unique(cluster_labels):
               mask = cluster_labels == cluster_id
               cluster_errors = y_true[mask] - y_pred[mask]
               
               cluster_uncertainty[f'cluster_{cluster_id}'] = {
                   'confidence_intervals': {
                       f'level_{level}': {
                           'lower': float(stats.norm.interval(
                               level,
                               loc=np.mean(cluster_errors),
                               scale=stats.sem(cluster_errors)
                           )[0]),
                           'upper': float(stats.norm.interval(
                               level,
                               loc=np.mean(cluster_errors),
                               scale=stats.sem(cluster_errors)
                           )[1])
                       }
                       for level in confidence_levels
                   },
                   'error_bounds': {
                       f'p{p}': float(np.percentile(np.abs(cluster_errors), p))
                       for p in percentiles
                   }
               }
           
           uncertainty_analysis['cluster_uncertainty'] = cluster_uncertainty
       
       return uncertainty_analysis
   
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
   
    def _perform_ks_test(
       self,
       sample1: np.ndarray,
       sample2: np.ndarray
   ) -> Dict[str, float]:
       """Perform Kolmogorov-Smirnov test."""
       statistic, p_value = stats.ks_2samp(sample1, sample2)
       return {
           'statistic': float(statistic),
           'p_value': float(p_value),
           'samples_similar': float(p_value) > 0.05
       }
   
    def _analyze_error_distribution(
       self,
       errors: np.ndarray
   ) -> Dict[str, float]:
       """Analyze error distribution."""
       return {
           'mean': float(np.mean(errors)),
           'std': float(np.std(errors)),
           'skewness': float(stats.skew(errors)),
           'kurtosis': float(stats.kurtosis(errors)),
           'normality_test': self._test_error_normality(errors)
       }
   
    def _create_evaluation_plots(
       self,
       y_true: np.ndarray,
       y_pred: np.ndarray,
       cluster_labels: Optional[np.ndarray]
   ) -> Dict[str, Any]:
       """Create evaluation visualizations."""
       plots = {}
       
       # Actual vs Predicted scatter plot
       plots['actual_vs_predicted'] = plotter.create_plot(
           'scatter',
           data=pd.DataFrame({
               'Actual': y_true,
               'Predicted': y_pred,
               'Cluster': cluster_labels if cluster_labels is not None else None
           }),
           x='Actual',
           y='Predicted',
           color='Cluster' if cluster_labels is not None else None,
           title='Actual vs Predicted Values'
       )
       
       # Error distribution plot
       errors = y_true - y_pred
       plots['error_distribution'] = plotter.create_plot(
           'histogram',
           data=pd.DataFrame({
               'Error': errors,
               'Cluster': cluster_labels if cluster_labels is not None else None
           }),
           x='Error',
           color='Cluster' if cluster_labels is not None else None,
           title='Error Distribution'
       )
       
       # Q-Q plot
       plots['qq_plot'] = plotter.create_plot(
           'qq',
           data=errors,
           title='Q-Q Plot of Prediction Errors'
       )
       
       # Error vs Predicted plot
       plots['error_vs_predicted'] = plotter.create_plot(
           'scatter',
           data=pd.DataFrame({
               'Predicted': y_pred,
               'Error': errors,
               'Cluster': cluster_labels if cluster_labels is not None else None
           }),
           x='Predicted',
           y='Error',
           color='Cluster' if cluster_labels is not None else None,
           title='Error vs Predicted Values'
       )
       
       if cluster_labels is not None:
           # Cluster-specific plots
           for cluster_id in np.unique(cluster_labels):
               mask = cluster_labels == cluster_id
               plots[f'cluster_{cluster_id}_error_dist'] = plotter.create_plot(
                   'histogram',
                   data=pd.DataFrame({'Error': errors[mask]}),
                   x='Error',
                   title=f'Error Distribution - Cluster {cluster_id}'
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
       results: Dict[str, Any]
   ) -> None:
       """Record evaluation in history."""
       record = {
           'timestamp': datetime.now().isoformat(),
           'evaluation_id': evaluation_id,
           'results': results
       }
       
       self.evaluation_history.append(record)
       state_manager.set_state(
           f'prediction.evaluation.history.{len(self.evaluation_history)}',
           record
       )

# Create global prediction evaluator instance
prediction_evaluator = PredictionEvaluator()