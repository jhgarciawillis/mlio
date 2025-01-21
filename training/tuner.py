import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    BaseCrossValidator,
    KFold
)
from scipy.stats import uniform, randint
import optuna

from core import config
from core.exceptions import TuningError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from metrics import calculator

class HyperparameterTuner:
    """Handle hyperparameter tuning operations."""
    
    def __init__(self):
        self.tuning_history: List[Dict[str, Any]] = []
        self.tuning_results: Dict[str, Dict[str, Any]] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.studies: Dict[str, optuna.Study] = {}
        
    @monitor_performance
    @handle_exceptions(TuningError)
    def tune_hyperparameters(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        method: str = 'grid',
        scoring: str = 'neg_mean_squared_error',
        cv: Union[int, BaseCrossValidator] = 5,
        n_iter: int = 50,
        random_state: Optional[int] = None,
        **kwargs
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Tune hyperparameters using specified method."""
        try:
            tuning_id = f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if method == 'grid':
                best_model, results = self._grid_search(
                    model, X, y, param_space, scoring, cv, **kwargs
                )
            elif method == 'random':
                best_model, results = self._random_search(
                    model, X, y, param_space, scoring, cv, n_iter, random_state, **kwargs
                )
            elif method == 'optuna':
                best_model, results = self._optuna_optimize(
                    model, X, y, param_space, scoring, cv, n_iter, random_state, **kwargs
                )
            else:
                raise TuningError(f"Unknown tuning method: {method}")
            
            # Store results
            self.tuning_results[tuning_id] = results
            self.best_params[tuning_id] = results['best_params']
            
            # Record tuning
            self._record_tuning(tuning_id, method, results)
            
            return best_model, results
            
        except Exception as e:
            raise TuningError(f"Error during hyperparameter tuning: {str(e)}") from e
    
    @monitor_performance
    def _grid_search(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, Any],
        scoring: str,
        cv: Union[int, BaseCrossValidator],
        **kwargs
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Perform grid search."""
        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            **kwargs
        )
        
        grid_search.fit(X, y)
        
        results = {
            'method': 'grid',
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'cv_results': self._process_cv_results(grid_search.cv_results_),
            'search_params': {
                'param_grid': param_grid,
                'scoring': scoring,
                'cv': cv,
                'kwargs': kwargs
            }
        }
        
        return grid_search.best_estimator_, results
    
    @monitor_performance
    def _random_search(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_distributions: Dict[str, Any],
        scoring: str,
        cv: Union[int, BaseCrossValidator],
        n_iter: int,
        random_state: Optional[int],
        **kwargs
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Perform random search."""
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            return_train_score=True,
            **kwargs
        )
        
        random_search.fit(X, y)
        
        results = {
            'method': 'random',
            'best_params': random_search.best_params_,
            'best_score': float(random_search.best_score_),
            'cv_results': self._process_cv_results(random_search.cv_results_),
            'search_params': {
                'param_distributions': param_distributions,
                'n_iter': n_iter,
                'scoring': scoring,
                'cv': cv,
                'random_state': random_state,
                'kwargs': kwargs
            }
        }
        
        return random_search.best_estimator_, results
    
    @monitor_performance
    def _optuna_optimize(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        scoring: str,
        cv: Union[int, BaseCrossValidator],
        n_trials: int,
        random_state: Optional[int],
        **kwargs
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Perform Optuna optimization."""
        def objective(trial):
            params = {}
            for param_name, param_config in param_space.items():
                if isinstance(param_config, tuple):
                    if isinstance(param_config[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config[0],
                            param_config[1]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config[0],
                            param_config[1]
                        )
                elif isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config
                    )
            
            model.set_params(**params)
            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
            return cv_scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # Get best model
        best_model = model.set_params(**study.best_params)
        best_model.fit(X, y)
        
        results = {
            'method': 'optuna',
            'best_params': study.best_params,
            'best_score': float(study.best_value),
            'study': study,
            'search_params': {
                'param_space': param_space,
                'n_trials': n_trials,
                'scoring': scoring,
                'cv': cv,
                'random_state': random_state,
                'kwargs': kwargs
            }
        }
        
        self.studies[str(datetime.now())] = study
        
        return best_model, results
    
    def _process_cv_results(self, cv_results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process cross-validation results."""
        processed = {
            'params': cv_results['params'],
            'mean_test_score': cv_results['mean_test_score'].tolist(),
            'std_test_score': cv_results['std_test_score'].tolist(),
            'mean_train_score': cv_results['mean_train_score'].tolist(),
            'std_train_score': cv_results['std_train_score'].tolist(),
            'rank_test_score': cv_results['rank_test_score'].tolist()
        }
        
        # Add split scores if available
        split_keys = [key for key in cv_results.keys() if key.startswith('split')]
        if split_keys:
            processed['split_scores'] = {
                key: cv_results[key].tolist() for key in split_keys
            }
        
        return processed
    
    def _record_tuning(
        self,
        tuning_id: str,
        method: str,
        results: Dict[str, Any]
    ) -> None:
        """Record tuning in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'tuning_id': tuning_id,
            'method': method,
            'best_params': results['best_params'],
            'best_score': results['best_score']
        }
        
        self.tuning_history.append(record)
        state_manager.set_state(
            f'training.tuning.history.{len(self.tuning_history)}',
            record
        )

# Create global hyperparameter tuner instance
hyperparameter_tuner = HyperparameterTuner()