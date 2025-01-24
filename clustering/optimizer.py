import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import KFold
import optuna

from core import config
from core.exceptions import ClusteringOptimizationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from clustering import clusterer

class ClusterOptimizer:
    """Handle cluster optimization operations."""
    
    def __init__(self):
        self.optimization_history: List[Dict[str, Any]] = []
        self.optimization_results: Dict[str, Dict[str, Any]] = {}
        self.studies: Dict[str, optuna.Study] = {}
        
        # Default parameter ranges
        self.param_ranges = {
            'kmeans': {
                'n_clusters': (2, 20),
                'n_init': (5, 15),
                'max_iter': (100, 500)
            },
            'dbscan': {
                'eps': (0.1, 2.0),
                'min_samples': (2, 10)
            },
            'gaussian_mixture': {
                'n_components': (2, 20),
                'max_iter': (100, 500)
            },
            'hierarchical': {
                'n_clusters': (2, 20)
            },
            'spectral': {
                'n_clusters': (2, 20),
                'n_neighbors': (2, 20)
            }
        }
        
        # Available metrics for optimization
        self.available_metrics = {
            'silhouette': silhouette_score,
            'calinski_harabasz': calinski_harabasz_score
        }
    
    @monitor_performance
    @handle_exceptions(ClusteringOptimizationError)
    def optimize_clustering(
        self,
        data: pd.DataFrame,
        method: str = 'kmeans',
        metric: str = 'silhouette',
        n_trials: int = 50,
        cv_folds: int = 5,
        param_ranges: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize clustering parameters."""
        try:
            # Record operation start
            state_monitor.record_operation_start(
                'clustering_optimization',
                'optimization',
                {'method': method, 'metric': metric}
            )
            
            optimization_id = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Validate method and metric
            if method not in self.param_ranges:
                raise ClusteringOptimizationError(f"Unsupported method: {method}")
            if metric not in self.available_metrics:
                raise ClusteringOptimizationError(f"Unsupported metric: {metric}")
            
            # Use provided param ranges or defaults
            ranges = param_ranges or self.param_ranges[method]
            
            # Create optimization study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=random_state)
            )
            
            # Define objective function
            objective = self._create_objective_function(
                data,
                method,
                metric,
                ranges,
                cv_folds
            )
            
            # Run optimization
            study.optimize(objective, n_trials=n_trials)
            
            # Store study
            self.studies[optimization_id] = study
            
            # Get best parameters
            best_params = study.best_params
            
            # Run clustering with best parameters
            best_results = clusterer.cluster_data(
                data,
                method=method,
                params=best_params
            )
            
            results = {
                'best_params': best_params,
                'best_score': float(study.best_value),
                'optimization_history': self._process_optimization_history(study),
                'best_clustering_results': best_results,
                'method': method,
                'metric': metric,
                'n_trials': n_trials,
                'cv_folds': cv_folds
            }
            
            # Store results
            self.optimization_results[optimization_id] = results
            
            # Record optimization
            self._record_optimization(optimization_id, results)
            
            # Record operation completion
            state_monitor.record_operation_end(
                'clustering_optimization',
                'completed',
                {
                    'best_score': float(study.best_value),
                    'n_trials': n_trials
                }
            )
            
            return results
            
        except Exception as e:
            state_monitor.record_operation_end(
                'clustering_optimization',
                'failed',
                {'error': str(e)}
            )
            raise ClusteringOptimizationError(
                f"Error optimizing clustering: {str(e)}"
            ) from e
    
    def _create_objective_function(
        self,
        data: pd.DataFrame,
        method: str,
        metric: str,
        param_ranges: Dict[str, Any],
        cv_folds: int
    ) -> callable:
        """Create objective function for optimization."""
        def objective(trial: optuna.Trial) -> float:
            # Define parameter space based on method
            params = self._suggest_parameters(trial, method, param_ranges)
            
            # Perform cross-validation
            scores = []
            kf = KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=config.random_state
            )
            
            for train_idx, val_idx in kf.split(data):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                # Cluster training data
                train_results = clusterer.cluster_data(
                    train_data,
                    method=method,
                    params=params
                )
                
                # Evaluate on validation data
                val_labels = clusterer.predict_clusters(
                    val_data,
                    train_results['clustering_id']
                )
                
                # Calculate score
                score = self.available_metrics[metric](val_data, val_labels)
                scores.append(score)
            
            return np.mean(scores)
        
        return objective
    
def _suggest_parameters(
        self,
        trial: optuna.Trial,
        method: str,
        param_ranges: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest parameters for optimization."""
        params = {}
        
        if method == 'kmeans':
            params.update({
                'n_clusters': trial.suggest_int(
                    'n_clusters',
                    param_ranges['n_clusters'][0],
                    param_ranges['n_clusters'][1]
                ),
                'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                'n_init': trial.suggest_int(
                    'n_init',
                    param_ranges['n_init'][0],
                    param_ranges['n_init'][1]
                ),
                'max_iter': trial.suggest_int(
                    'max_iter',
                    param_ranges['max_iter'][0],
                    param_ranges['max_iter'][1]
                )
            })
            
        elif method == 'dbscan':
            params.update({
                'eps': trial.suggest_float(
                    'eps',
                    param_ranges['eps'][0],
                    param_ranges['eps'][1]
                ),
                'min_samples': trial.suggest_int(
                    'min_samples',
                    param_ranges['min_samples'][0],
                    param_ranges['min_samples'][1]
                ),
                'metric': trial.suggest_categorical(
                    'metric',
                    ['euclidean', 'manhattan']
                )
            })
            
        elif method == 'gaussian_mixture':
            params.update({
                'n_components': trial.suggest_int(
                    'n_components',
                    param_ranges['n_components'][0],
                    param_ranges['n_components'][1]
                ),
                'covariance_type': trial.suggest_categorical(
                    'covariance_type',
                    ['full', 'tied', 'diag', 'spherical']
                ),
                'max_iter': trial.suggest_int(
                    'max_iter',
                    param_ranges['max_iter'][0],
                    param_ranges['max_iter'][1]
                )
            })
            
        elif method == 'hierarchical':
            params.update({
                'n_clusters': trial.suggest_int(
                    'n_clusters',
                    param_ranges['n_clusters'][0],
                    param_ranges['n_clusters'][1]
                ),
                'affinity': trial.suggest_categorical(
                    'affinity',
                    ['euclidean', 'manhattan', 'cosine']
                ),
                'linkage': trial.suggest_categorical(
                    'linkage',
                    ['ward', 'complete', 'average']
                )
            })
            
        elif method == 'spectral':
            params.update({
                'n_clusters': trial.suggest_int(
                    'n_clusters',
                    param_ranges['n_clusters'][0],
                    param_ranges['n_clusters'][1]
                ),
                'n_neighbors': trial.suggest_int(
                    'n_neighbors',
                    param_ranges['n_neighbors'][0],
                    param_ranges['n_neighbors'][1]
                ),
                'affinity': trial.suggest_categorical(
                    'affinity',
                    ['rbf', 'nearest_neighbors']
                )
            })
        
        return params
    
    def _process_optimization_history(
        self,
        study: optuna.Study
    ) -> List[Dict[str, Any]]:
        """Process optimization history for storage."""
        history = []
        for trial in study.trials:
            history.append({
                'number': trial.number,
                'params': trial.params,
                'value': float(trial.value) if trial.value is not None else None,
                'state': trial.state.name,
                'datetime': trial.datetime.isoformat()
            })
        return history
    
    def _record_optimization(
        self,
        optimization_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Record optimization in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'optimization_id': optimization_id,
            'method': results['method'],
            'metric': results['metric'],
            'best_score': results['best_score'],
            'best_params': results['best_params']
        }
        
        self.optimization_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'clustering.optimization.history.{len(self.optimization_history)}',
            record
        )
    
    @monitor_performance
    def get_optimization_summary(
        self,
        optimization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if optimization_id is not None:
            if optimization_id not in self.optimization_results:
                raise ClusteringOptimizationError(
                    f"Optimization ID not found: {optimization_id}"
                )
            return self._generate_optimization_summary(
                self.optimization_results[optimization_id]
            )
        
        # Get summary for all optimizations
        summaries = {}
        for opt_id, results in self.optimization_results.items():
            summaries[opt_id] = self._generate_optimization_summary(results)
        
        return summaries
    
    def _generate_optimization_summary(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary for optimization results."""
        return {
            'method': results['method'],
            'metric': results['metric'],
            'best_params': results['best_params'],
            'best_score': results['best_score'],
            'n_trials': results['n_trials'],
            'cv_folds': results['cv_folds'],
            'optimization_duration': self._calculate_optimization_duration(results)
        }
    
    def _calculate_optimization_duration(
        self,
        results: Dict[str, Any]
    ) -> float:
        """Calculate optimization duration in seconds."""
        history = results['optimization_history']
        if not history:
            return 0.0
        
        start_time = datetime.fromisoformat(history[0]['datetime'])
        end_time = datetime.fromisoformat(history[-1]['datetime'])
        
        return (end_time - start_time).total_seconds()
    
    @monitor_performance
    def plot_optimization_history(
        self,
        optimization_id: str
    ) -> Dict[str, Any]:
        """Create visualization of optimization history."""
        if optimization_id not in self.optimization_results:
            raise ClusteringOptimizationError(
                f"Optimization ID not found: {optimization_id}"
            )
        
        results = self.optimization_results[optimization_id]
        history = pd.DataFrame(results['optimization_history'])
        
        visualizations = {}
        
        # Optimization progress plot
        visualizations['progress'] = plotter.create_plot(
            'line',
            data=history,
            x='number',
            y='value',
            title='Optimization Progress'
        )
        
        # Parameter importance plot (if available)
        if optimization_id in self.studies:
            importance = optuna.importance.get_param_importances(
                self.studies[optimization_id]
            )
            importance_df = pd.DataFrame({
                'Parameter': list(importance.keys()),
                'Importance': list(importance.values())
            })
            
            visualizations['importance'] = plotter.create_plot(
                'bar',
                data=importance_df,
                x='Parameter',
                y='Importance',
                title='Parameter Importance'
            )
        
        return visualizations

# Create global cluster optimizer instance
cluster_optimizer = ClusterOptimizer()