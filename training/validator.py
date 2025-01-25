import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    learning_curve,
    validation_curve,
    cross_val_score,
    BaseCrossValidator
)
from sklearn.metrics import make_scorer
import plotly.graph_objects as go

from core import config
from core.exceptions import ValidationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from metrics import calculator
from visualization import plotter
from clustering import validator as cluster_validator

class ModelValidator:
    """Handle model validation operations."""
    
    def __init__(self):
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self.learning_curves: Dict[str, Dict[str, Any]] = {}
        self.validation_curves: Dict[str, Dict[str, Any]] = {}
        
    @monitor_performance
    @handle_exceptions(ValidationError)
    def validate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: Optional[np.ndarray] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive model validation."""
        try:
            validation_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if validation_config is None:
                validation_config = self._get_default_validation_config()
            
            results = {}
            
            # Basic validation
            if validation_config.get('basic_validation', True):
                results['basic'] = self._perform_basic_validation(
                    model, X, y, cluster_labels
                )
            
            # Learning curves
            if validation_config.get('learning_curves', True):
                results['learning_curves'] = self._analyze_learning_curves(
                    model, X, y, cluster_labels
                )
            
            # Parameter validation
            if validation_config.get('parameter_validation', True):
                results['parameter_validation'] = self._validate_parameters(
                    model, X, y, validation_config.get('param_grid', {})
                )
            
            # Cluster-specific validation
            if cluster_labels is not None:
                results['cluster_validation'] = self._validate_clusters(
                    model, X, y, cluster_labels
                )
            
            # Feature importance analysis
            if validation_config.get('feature_importance', True):
                results['feature_importance'] = self._analyze_feature_importance(
                    model, X, y
                )
            
            # Model stability
            if validation_config.get('stability_analysis', True):
                results['stability'] = self._analyze_model_stability(
                    model, X, y, cluster_labels
                )
            
            # Store validation results
            self.validation_results[validation_id] = results
            
            # Record validation
            self._record_validation(validation_id, results)
            
            return results
        
        except Exception as e:
            raise ValidationError(f"Error validating model: {str(e)}") from e
    
    @monitor_performance
    def _perform_basic_validation(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform basic model validation."""
        validation_results = {}
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring=make_scorer(calculator.calculate_metrics)
        )
        
        validation_results['cv_scores'] = {
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'scores': cv_scores.tolist()
        }
        
        # Cluster-specific validation if applicable
        if cluster_labels is not None:
            cluster_scores = {}
            for cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                if np.sum(mask) >= 5:  # Minimum samples for CV
                    cluster_cv_scores = cross_val_score(
                        model,
                        X[mask],
                        y[mask],
                        cv=min(5, np.sum(mask)),
                        scoring=make_scorer(calculator.calculate_metrics)
                    )
                    cluster_scores[f'cluster_{cluster_id}'] = {
                        'mean': float(cluster_cv_scores.mean()),
                        'std': float(cluster_cv_scores.std()),
                        'scores': cluster_cv_scores.tolist()
                    }
            validation_results['cluster_scores'] = cluster_scores
        
        return validation_results
    
    @monitor_performance
    def _analyze_learning_curves(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze learning curves."""
        curves_results = {}
        
        # Global learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            n_jobs=-1,
            scoring=make_scorer(calculator.calculate_metrics)
        )
        
        curves_results['global'] = {
            'train_sizes': train_sizes.tolist(),
            'train_scores': {
                'mean': np.mean(train_scores, axis=1).tolist(),
                'std': np.std(train_scores, axis=1).tolist()
            },
            'val_scores': {
                'mean': np.mean(val_scores, axis=1).tolist(),
                'std': np.std(val_scores, axis=1).tolist()
            }
        }
        
        # Cluster-specific learning curves if applicable
        if cluster_labels is not None:
            cluster_curves = {}
            for cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                if np.sum(mask) >= 20:  # Minimum samples for learning curve
                    train_sizes_c, train_scores_c, val_scores_c = learning_curve(
                        model,
                        X[mask],
                        y[mask],
                        train_sizes=np.linspace(0.1, 1.0, 5),
                        cv=min(5, np.sum(mask) // 4),
                        n_jobs=-1,
                        scoring=make_scorer(calculator.calculate_metrics)
                    )
                    cluster_curves[f'cluster_{cluster_id}'] = {
                        'train_sizes': train_sizes_c.tolist(),
                        'train_scores': {
                            'mean': np.mean(train_scores_c, axis=1).tolist(),
                            'std': np.std(train_scores_c, axis=1).tolist()
                        },
                        'val_scores': {
                            'mean': np.mean(val_scores_c, axis=1).tolist(),
                            'std': np.std(val_scores_c, axis=1).tolist()
                        }
                    }
            curves_results['clusters'] = cluster_curves
        
        # Store learning curves
        self.learning_curves[str(datetime.now())] = curves_results
        
        return curves_results
    
    @monitor_performance
    def _validate_parameters(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Validate model parameters."""
        param_results = {}
        
        for param_name, param_range in param_grid.items():
            try:
                train_scores, test_scores = validation_curve(
                    model,
                    X,
                    y,
                    param_name=param_name,
                    param_range=param_range,
                    cv=5,
                    scoring=make_scorer(calculator.calculate_metrics),
                    n_jobs=-1
                )
                
                param_results[param_name] = {
                    'param_range': param_range,
                    'train_scores': {
                        'mean': np.mean(train_scores, axis=1).tolist(),
                        'std': np.std(train_scores, axis=1).tolist()
                    },
                    'test_scores': {
                        'mean': np.mean(test_scores, axis=1).tolist(),
                        'std': np.std(test_scores, axis=1).tolist()
                    }
                }
            except Exception as e:
                logger.warning(f"Could not validate parameter {param_name}: {str(e)}")
        
        # Store validation curves
        self.validation_curves[str(datetime.now())] = param_results
        
        return param_results
    
    @monitor_performance
    def _validate_clusters(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Validate cluster-specific model performance."""
        cluster_results = {
            'performance': {},
            'stability': {},
            'characteristics': {}
        }
        
        # Performance by cluster
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            if np.sum(mask) >= 5:
                cluster_scores = cross_val_score(
                    model,
                    X[mask],
                    y[mask],
                    cv=min(5, np.sum(mask)),
                    scoring=make_scorer(calculator.calculate_metrics)
                )
                cluster_results['performance'][f'cluster_{cluster_id}'] = {
                    'mean': float(cluster_scores.mean()),
                    'std': float(cluster_scores.std()),
                    'scores': cluster_scores.tolist(),
                    'size': int(np.sum(mask))
                }
        
        # Cluster stability analysis
        cluster_results['stability'] = cluster_validator.analyze_stability(X, cluster_labels)
        
        # Cluster characteristics
        cluster_results['characteristics'] = cluster_validator.analyze_characteristics(
            X, cluster_labels
        )
        
        return cluster_results
    
    @monitor_performance
    def _analyze_feature_importance(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Analyze feature importance."""
        importance_results = {}
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            importance_results['feature_importance'] = {
                feature: float(importance)
                for feature, importance in zip(X.columns, importance_scores)
            }
        
        # Permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=config.random_state
        )
        
        importance_results['permutation_importance'] = {
            feature: {
                'mean': float(mean),
                'std': float(std)
            }
            for feature, mean, std in zip(
                X.columns,
                perm_importance.importances_mean,
                perm_importance.importances_std
            )
        }
        
        return importance_results
    
    @monitor_performance
    def _analyze_model_stability(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze model stability."""
        stability_results = {
            'cross_validation': {},
            'bootstrap': {}
        }
        
        # Cross-validation stability
        cv_predictions = []
        kf = KFold(n_splits=5, shuffle=True, random_state=config.random_state)
        for train_idx, val_idx in kf.split(X):
            model_copy = clone(model)
            model_copy.fit(X.iloc[train_idx], y.iloc[train_idx])
            cv_predictions.append(model_copy.predict(X.iloc[val_idx]))
        
        stability_results['cross_validation'] = {
            'prediction_std': float(np.std([
                np.std(fold_preds) for fold_preds in cv_predictions
            ])),
            'score_std': float(np.std([
                calculator.calculate_metrics(
                    y.iloc[val_idx],
                    fold_preds
                )
                for val_idx, fold_preds in zip(
                    kf.split(X)[1],
                    cv_predictions
                )
            ]))
        }
        
        # Bootstrap stability
        n_bootstraps = 100
        bootstrap_predictions = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(len(X), len(X), replace=True)
            model_copy = clone(model)
            model_copy.fit(X.iloc[indices], y.iloc[indices])
            bootstrap_predictions.append(model_copy.predict(X))
        
        stability_results['bootstrap'] = {
            'prediction_std': float(np.std([
                np.std(boot_preds) for boot_preds in bootstrap_predictions
            ])),
            'score_std': float(np.std([
                calculator.calculate_metrics(y, boot_preds)
                for boot_preds in bootstrap_predictions
            ]))
        }
        
        # Cluster stability if applicable
        if cluster_labels is not None:
            stability_results['cluster_stability'] = {}
            for cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                if np.sum(mask) >= 20:
                    cluster_predictions = []
                    for _ in range(n_bootstraps):
                        indices = np.random.choice(
                            np.where(mask)[0],
                            np.sum(mask),
                            replace=True
                        )
                        model_copy = clone(model)
                        model_copy.fit(
                            X.iloc[indices],
                            y.iloc[indices]
                        )
                        cluster_predictions.append(
                            model_copy.predict(X[mask])
                        )
                    
                    stability_results['cluster_stability'][f'cluster_{cluster_id}'] = {
                        'prediction_std': float(np.std([
                            np.std(cluster_preds)
                            for cluster_preds in cluster_predictions
                        ])),
                        'score_std': float(np.std([
                            calculator.calculate_metrics(
                                y[mask],
                                cluster_preds
                            )
                            for cluster_preds in cluster_predictions
                        ]))
                    }
        
        return stability_results
    
    def _get_default_validation_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'basic_validation': True,
            'learning_curves': True,
            'parameter_validation': True,
            'feature_importance': True,
            'stability_analysis': True,
            'param_grid': {}
        }
    
    def _record_validation(
       self,
       validation_id: str,
       results: Dict[str, Any]
   ) -> None:
       """Record validation in history."""
       record = {
           'timestamp': datetime.now().isoformat(),
           'validation_id': validation_id,
           'results': results
       }
       
       self.validation_history.append(record)
       state_manager.set_state(
           f'training.validation.history.{len(self.validation_history)}',
           record
       )

# Create global model validator instance  
model_validator = ModelValidator()