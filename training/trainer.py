import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    cross_val_score,
    BaseCrossValidator,
    KFold
)
from sklearn.metrics import make_scorer
import joblib

from core import config
from core.exceptions import ModelTrainingError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from metrics import calculator
from clustering import clusterer

class ModelTrainer:
    """Handle model training operations with clustering integration."""
    
    def __init__(self):
        self.models: Dict[str, BaseEstimator] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.cv_results: Dict[str, Dict[str, Any]] = {}
        self.cluster_models: Dict[str, Dict[str, BaseEstimator]] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
    @monitor_performance
    @handle_exceptions(ModelTrainingError)
    def train_model(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cluster_labels: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Train model with optional clustering support."""
        try:
            if model_name is None:
                model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Split data for validation if not provided
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train,
                    test_size=0.2,
                    random_state=config.random_state
                )
            
            if cluster_labels is not None:
                # Train cluster-specific models
                cluster_specific_models = self._train_cluster_models(
                    model, X_train, y_train, cluster_labels, **kwargs
                )
                self.cluster_models[model_name] = cluster_specific_models
                
                # Train global model
                global_model = self._train_global_model(
                    model, X_train, y_train, cluster_labels, **kwargs
                )
                self.models[model_name] = global_model
                
                # Calculate metrics for both approaches
                cluster_metrics = self._evaluate_cluster_models(
                    cluster_specific_models,
                    X_val, y_val,
                    self._get_cluster_labels(X_val, cluster_labels)
                )
                
                global_metrics = calculator.calculate_metrics(
                    y_val,
                    global_model.predict(X_val)
                )
                
                metrics = {
                    'cluster_specific': cluster_metrics,
                    'global': global_metrics
                }
            else:
                # Train single model
                model.fit(X_train, y_train)
                self.models[model_name] = model
                
                # Calculate metrics
                metrics = {
                    'train': calculator.calculate_metrics(
                        y_train,
                        model.predict(X_train)
                    ),
                    'val': calculator.calculate_metrics(
                        y_val,
                        model.predict(X_val)
                    )
                }
            
            # Create metadata
            metadata = {
                'model_type': type(model).__name__,
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'features': list(X_train.columns),
                'metrics': metrics,
                'parameters': model.get_params(),
                'timestamp': datetime.now().isoformat(),
                'clustering_enabled': cluster_labels is not None
            }
            
            if cluster_labels is not None:
                metadata['n_clusters'] = len(np.unique(cluster_labels))
            
            self.model_metadata[model_name] = metadata
            
            # Record training
            self._record_training(model_name, metadata)
            
            return model, metadata
            
        except Exception as e:
            raise ModelTrainingError(f"Error training model: {str(e)}") from e
    
    def _train_cluster_models(
        self,
        base_model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: np.ndarray,
        **kwargs
    ) -> Dict[int, BaseEstimator]:
        """Train separate models for each cluster."""
        cluster_models = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            # Get cluster data
            mask = cluster_labels == cluster_id
            X_cluster = X[mask]
            y_cluster = y[mask]
            
            if len(X_cluster) > 0:
                # Create and train cluster-specific model
                model = self._clone_model(base_model)
                model.fit(X_cluster, y_cluster, **kwargs)
                cluster_models[cluster_id] = model
        
        return cluster_models
    
    def _train_global_model(
        self,
        base_model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: np.ndarray,
        **kwargs
    ) -> BaseEstimator:
        """Train global model with cluster information."""
        # Add cluster labels as a feature
        X_with_clusters = X.copy()
        X_with_clusters['cluster'] = cluster_labels
        
        # Train global model
        model = self._clone_model(base_model)
        model.fit(X_with_clusters, y, **kwargs)
        
        return model
    
    def _evaluate_cluster_models(
        self,
        cluster_models: Dict[int, BaseEstimator],
        X: pd.DataFrame,
        y: pd.Series,
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate cluster-specific models."""
        metrics = {}
        
        for cluster_id, model in cluster_models.items():
            # Get cluster data
            mask = cluster_labels == cluster_id
            X_cluster = X[mask]
            y_cluster = y[mask]
            
            if len(X_cluster) > 0:
                # Calculate metrics for this cluster
                predictions = model.predict(X_cluster)
                metrics[f'cluster_{cluster_id}'] = calculator.calculate_metrics(
                    y_cluster,
                    predictions
                )
        
        # Calculate overall metrics
        all_predictions = np.zeros_like(y)
        for cluster_id, model in cluster_models.items():
            mask = cluster_labels == cluster_id
            if np.any(mask):
                all_predictions[mask] = model.predict(X[mask])
        
        metrics['overall'] = calculator.calculate_metrics(y, all_predictions)
        
        return metrics
    
    @monitor_performance
    def cross_validate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Union[int, BaseCrossValidator] = 5,
        scoring: Union[str, Dict[str, str], List[str]] = 'r2',
        cluster_labels: Optional[np.ndarray] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation with optional clustering support."""
        if model_name is None:
            model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert string scoring to list
        if isinstance(scoring, str):
            scoring = [scoring]
        
        # Create scorers dictionary
        scorers = {}
        for score in scoring:
            if isinstance(score, str):
                scorers[score] = make_scorer(score)
        
        if cluster_labels is not None:
            # Perform cross-validation for each cluster
            cluster_cv_results = {}
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                if np.sum(mask) >= cv:  # Only if enough samples
                    cluster_model = self._clone_model(model)
                    cv_results = cross_validate(
                        cluster_model,
                        X[mask],
                        y[mask],
                        cv=cv,
                        scoring=scorers,
                        return_train_score=True,
                        n_jobs=-1
                    )
                    cluster_cv_results[f'cluster_{cluster_id}'] = cv_results
            
            # Add global model results
            global_cv_results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scorers,
                return_train_score=True,
                n_jobs=-1
            )
            
            cv_results = {
                'cluster_specific': cluster_cv_results,
                'global': global_cv_results
            }
        else:
            # Standard cross-validation
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scorers,
                return_train_score=True,
                n_jobs=-1
            )
        
        # Process results
        cv_summary = self._process_cv_results(cv_results)
        
        # Store results
        self.cv_results[model_name] = cv_summary
        
        return cv_summary
    
    def _process_cv_results(
        self,
        cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process cross-validation results."""
        if 'cluster_specific' in cv_results:
            # Process cluster-specific results
            processed = {
                'cluster_specific': {},
                'global': {}
            }
            
            # Process cluster results
            for cluster_id, results in cv_results['cluster_specific'].items():
                processed['cluster_specific'][cluster_id] = {
                    metric: {
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'scores': scores.tolist()
                    }
                    for metric, scores in results.items()
                }
            
            # Process global results
            global_results = cv_results['global']
            processed['global'] = {
                metric: {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
                for metric, scores in global_results.items()
            }
        else:
            # Process standard results
            processed = {
                metric: {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
                for metric, scores in cv_results.items()
            }
        
        return processed
    
    def _clone_model(
        self,
        model: BaseEstimator
    ) -> BaseEstimator:
        """Create a clone of the model."""
        from sklearn.base import clone
        return clone(model)
    
    def _get_cluster_labels(
        self,
        X: pd.DataFrame,
        train_labels: np.ndarray
    ) -> np.ndarray:
        """Get cluster labels for new data."""
        return clusterer.predict_clusters(X, train_labels)
    
    @monitor_performance
    def save_model(
        self,
        model_name: str,
        path: Optional[Path] = None
    ) -> None:
        """Save model and its metadata."""
        if model_name not in self.models:
            raise ModelTrainingError(f"Model not found: {model_name}")
        
        if path is None:
            path = config.directories.training_models
            path.mkdir(parents=True, exist_ok=True)
        
        # Save main model
        model_path = path / f"{model_name}.joblib"
        joblib.dump(self.models[model_name], model_path)
        
        # Save cluster models if they exist
        if model_name in self.cluster_models:
            cluster_path = path / f"{model_name}_clusters"
            cluster_path.mkdir(exist_ok=True)
            for cluster_id, model in self.cluster_models[model_name].items():
                cluster_model_path = cluster_path / f"cluster_{cluster_id}.joblib"
                joblib.dump(model, cluster_model_path)
        
        # Save metadata
        metadata_path = path / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata[model_name], f, indent=4)
        
        logger.info(f"Model {model_name} saved to {path}")
    
    @monitor_performance
    def load_model(
        self,
        model_name: str,
        path: Optional[Path] = None
    ) -> BaseEstimator:
        """Load model and its metadata."""
        if path is None:
            path = config.directories.training_models
        
        # Load main model
        model_path = path / f"{model_name}.joblib"
        if not model_path.exists():
            raise ModelTrainingError(f"Model file not found: {model_path}")
        
        self.models[model_name] = joblib.load(model_path)
        
        # Load cluster models if they exist
        cluster_path = path / f"{model_name}_clusters"
        if cluster_path.exists():
            self.cluster_models[model_name] = {}
            for cluster_model_path in cluster_path.glob("cluster_*.joblib"):
                cluster_id = int(cluster_model_path.stem.split('_')[1])
                self.cluster_models[model_name][cluster_id] = joblib.load(
                    cluster_model_path
                )
        
        # Load metadata
        metadata_path = path / f"{model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata[model_name] = json.load(f)
        
        logger.info(f"Model {model_name} loaded from {path}")
        return self.models[model_name]
    
    def _record_training(
        self,
        model_name: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Record training in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metadata': metadata
        }
        
        self.training_history.append(record)
        state_manager.set_state(
            f'training.history.{len(self.training_history)}',
            record
        )

# Create global model trainer instance
model_trainer = ModelTrainer()