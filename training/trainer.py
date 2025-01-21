import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import make_scorer
import joblib

from core import config
from core.exceptions import ModelTrainingError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from metrics import calculator

class ModelTrainer:
    """Handle model training operations."""
    
    def __init__(self):
        self.models: Dict[str, BaseEstimator] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.cv_results: Dict[str, Dict[str, Any]] = {}
        self.tuning_results: Dict[str, Dict[str, Any]] = {}
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
        model_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Train a model with validation."""
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
            
            # Train model
            model.fit(X_train, y_train)
            
            # Calculate metrics
            train_metrics = calculate_metrics(
                y_train,
                model.predict(X_train),
                prefix='train'
            )
            val_metrics = calculate_metrics(
                y_val,
                model.predict(X_val),
                prefix='val'
            )
            
            # Store model
            self.models[model_name] = model
            
            # Create metadata
            metadata = {
                'model_type': type(model).__name__,
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'features': list(X_train.columns),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'parameters': model.get_params(),
                'timestamp': datetime.now().isoformat()
            }
            self.model_metadata[model_name] = metadata
            
            # Record training
            self._record_training(model_name, metadata)
            
            return model, metadata
            
        except Exception as e:
            raise ModelTrainingError(
                f"Error training model: {str(e)}"
            ) from e
    
    @monitor_performance
    def cross_validate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: Union[str, Dict[str, str], List[str]] = 'r2',
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation."""
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
        
        # Perform cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scorers,
            return_train_score=True
        )
        
        # Process results
        cv_summary = {
            'cv_scores': {
                metric: {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
                for metric, scores in cv_results.items()
            },
            'cv_folds': cv,
            'scoring': scoring
        }
        
        # Store results
        self.cv_results[model_name] = cv_summary
        
        return cv_summary
    
    @monitor_performance
    def tune_hyperparameters(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, Any],
        method: str = 'grid',
        cv: int = 5,
        scoring: str = 'r2',
        n_iter: int = 10,
        model_name: Optional[str] = None
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Tune model hyperparameters."""
        if model_name is None:
            model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Select search method
        if method == 'grid':
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
        else:  # randomized
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
                random_state=config.random_state
            )
        
        # Perform search
        search.fit(X, y)
        
        # Process results
        tuning_results = {
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
            'cv_results': {
                'params': search.cv_results_['params'],
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': search.cv_results_['mean_train_score'].tolist(),
                'std_train_score': search.cv_results_['std_train_score'].tolist()
            },
            'method': method,
            'cv': cv,
            'scoring': scoring
        }
        
        # Store results
        self.tuning_results[model_name] = tuning_results
        
        return search.best_estimator_, tuning_results
    
    @monitor_performance
    def save_model(
        self,
        model_name: str,
        path: Optional[str] = None
    ) -> None:
        """Save model and its metadata."""
        if model_name not in self.models:
            raise ModelTrainingError(f"Model not found: {model_name}")
        
        if path is None:
            path = config.directories.training_models
        
        # Create directory if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = Path(path) / f"{model_name}.joblib"
        joblib.dump(self.models[model_name], model_path)
        
        # Save metadata
        metadata_path = Path(path) / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata[model_name], f, indent=4)
        
        logger.info(f"Model {model_name} saved to {path}")
    
    @monitor_performance
    def load_model(
        self,
        model_name: str,
        path: Optional[str] = None
    ) -> BaseEstimator:
        """Load model and its metadata."""
        if path is None:
            path = config.directories.training_models
        
        # Load model
        model_path = Path(path) / f"{model_name}.joblib"
        if not model_path.exists():
            raise ModelTrainingError(f"Model file not found: {model_path}")
        
        self.models[model_name] = joblib.load(model_path)
        
        # Load metadata
        metadata_path = Path(path) / f"{model_name}_metadata.json"
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