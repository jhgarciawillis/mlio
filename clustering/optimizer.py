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
        
    @monitor_performance
    @handle_exceptions(ClusteringOptimizationError)
    def optimize_clustering(
        self,
        data: pd.DataFrame,
        method: str = 'kmeans',
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize clustering parameters."""
        try:
            optimization_id = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if optimization_config is None:
                optimization_config = self._get_default_optimization_config(method)
            
            # Create optimization study
            study = optuna.create_study(
                direction=optimization_config['direction'],
                sampler=optuna.samplers.TPESampler(
                    seed=config.random_state
                )
            )
            
            # Define objective function
            objective = self._create_objective_function(
                data,
                method,
                optimization_config
            )
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=optimization_config['n_trials'],
                show_progress_bar=True
            )
            
            # Store study
            self.studies[optimization_id] = study
            
            # Get best parameters
            best_params = study.best_params
            
            # Run clustering with best parameters
            best_results = clusterer.cluster_data(
                data,
                method=method,
                clustering_config=best_params
            )
            
            results = {
                'best_params': best_params,
                'best_score': float(study.best_value),
                'optimization_history': self._process_optimization_history(study),
                'best_clustering_results': best_results,
                'method': method,
                'config': optimization_config
            }
            
            # Store results
            self.optimization_results[optimization_id] = results
            
            # Record optimization
            self._record_optimization(optimization_id, results)
            
            return results
            
        except Exception as e:
            raise ClusteringOptimizationError(
                f"Error optimizing clustering: {str(e)}"
            ) from e
    
    def _create_objective_function(
        self,
        data: pd.DataFrame,
        method: str,
        config: Dict[str, Any]
    ) -> callable:
        """Create objective function for optimization."""
        def objective(trial: optuna.Trial) -> float:
            # Define parameter space based on method
            params = self._suggest_parameters(trial, method)
            
            # Perform cross-validation if enabled
            if config.get('cross_validation', False):
                scores = []
                kf = KFold(
                    n_splits=config['cv_splits'],
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
                        clustering_config=params
                    )
                    
                    # Evaluate on validation data
                    score = self._evaluate_clustering(
                        val_data,
                        train_results['model'],
                        config['metric']
                    )
                    scores.append(score)
                
                return np.mean(scores)
            else:
                # Cluster entire dataset
                results = clusterer.cluster_data(
                    data,
                    method=method,
                    clustering_config=params
                )
                
                return results['metrics'][config['metric']]
        
        return objective
    
    def _suggest_parameters(
        self,
        trial: optuna.Trial,
        method: str
    ) -> Dict[str, Any]:
        """Suggest parameters for optimization."""
        if method == 'kmeans':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, 20),
                'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                'n_init': trial.suggest_int('n_init', 5, 15),
                'max_iter': trial.suggest_int('max_iter', 100, 500)
            }
        elif method == 'dbscan':
            return {
                'eps': trial.suggest_float('eps', 0.1, 2.0),
                'min_samples': trial.suggest_int('min_samples', 2, 10),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
            }
        elif method == 'gaussian_mixture':
            return {
                'n_components': trial.suggest_int('n_components', 2, 20),
                'covariance_type': trial.suggest_categorical(
                    'covariance_type',
                    ['full', 'tied', 'diag', 'spherical']
                ),
                'max_iter': trial.suggest_int('max_iter', 100, 500)
            }
        else:
            raise ClusteringOptimizationError(f"Unsupported method for optimization: {method}")
    
    def _evaluate_clustering(
        self,
        data: pd.DataFrame,
        model: Any,
        metric: str
    ) -> float:
        """Evaluate clustering on validation data."""
        labels = model.predict(data)
        
        if metric == 'silhouette':
            return silhouette_score(data, labels)
        elif metric == 'calinski_harabasz':
            return calinski_harabasz_score(data, labels)
        else:
            raise ClusteringOptimizationError(f"Unsupported metric: {metric}")
    
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
                'state': trial.state.name
            })
        return history
    
    def _get_default_optimization_config(
        self,
        method: str
    ) -> Dict[str, Any]:
        """Get default optimization configuration."""
        return {
            'direction': 'maximize',
            'n_trials': 100,
            'metric': 'silhouette',
            'cross_validation': True,
            'cv_splits': 5
        }
    
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
            'best_params': results['best_params'],
            'best_score': results['best_score']
        }
        
        self.optimization_history.append(record)
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
            'best_params': results['best_params'],
            'best_score': results['best_score'],
            'n_trials': len(results['optimization_history']),
            'optimization_config': results['config']
        }

# Create global cluster optimizer instance
cluster_optimizer = ClusterOptimizer()