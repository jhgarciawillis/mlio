import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    learning_curve,
    validation_curve,
    permutation_importance
)
from sklearn.inspection import partial_dependence

from core import config
from core.exceptions import ValidationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from metrics import calculator
from visualization import plotter

class ModelValidator:
    """Handle model validation operations."""
    
    def __init__(self):
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        
    @monitor_performance
    @handle_exceptions(ValidationError)
    def validate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
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
                results['basic'] = self._perform_basic_validation(model, X, y)
            
            # Learning curves
            if validation_config.get('learning_curves', True):
                results['learning_curves'] = self._analyze_learning_curves(model, X, y)
            
            # Feature importance
            if validation_config.get('feature_importance', True):
                results['feature_importance'] = self._analyze_feature_importance(model, X, y)
            
            # Parameter sensitivity
            if validation_config.get('parameter_sensitivity', True):
                results['parameter_sensitivity'] = self._analyze_parameter_sensitivity(
                    model, X, y, validation_config.get('param_ranges', {})
                )
            
            # Model complexity
            if validation_config.get('model_complexity', True):
                results['model_complexity'] = self._analyze_model_complexity(model, X, y)
            
            # Feature interactions
            if validation_config.get('feature_interactions', True):
                results['feature_interactions'] = self._analyze_feature_interactions(
                    model, X, y, validation_config.get('interaction_features', [])
                )
            
            # Store results
            self.validation_results[validation_id] = results
            
            # Record validation
            self._record_validation(validation_id, validation_config)
            
            return results
            
        except Exception as e:
            raise ValidationError(
                f"Error during model validation: {str(e)}"
            ) from e
    
    @monitor_performance
    def _perform_basic_validation(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Perform basic model validation."""
        # Cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=5,
            scoring=config.model.scoring_metrics,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Process results
        basic_results = {
            'cv_scores': {
                metric: {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
                for metric, scores in cv_results.items()
            }
        }
        
        # Create visualization
        cv_scores_df = pd.DataFrame(cv_results)
        fig = plotter.create_plot(
            'box',
            data=cv_scores_df,
            title='Cross-validation Scores Distribution'
        )
        basic_results['visualization'] = fig
        
        return basic_results
    
    @monitor_performance
    def _analyze_learning_curves(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Analyze learning curves."""
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        curves_results = {
            'train_sizes': train_sizes.tolist(),
            'train_scores': {
                'mean': np.mean(train_scores, axis=1).tolist(),
                'std': np.std(train_scores, axis=1).tolist()
            },
            'test_scores': {
                'mean': np.mean(test_scores, axis=1).tolist(),
                'std': np.std(test_scores, axis=1).tolist()
            }
        }
        
        # Create visualization
        fig = plotter.create_plot(
            'line',
            data=pd.DataFrame({
                'Train Size': np.repeat(train_sizes, 2),
                'Score': np.concatenate([
                    np.mean(train_scores, axis=1),
                    np.mean(test_scores, axis=1)
                ]),
                'Type': ['Train']*len(train_sizes) + ['Test']*len(train_sizes)
            }),
            x='Train Size',
            y='Score',
            color='Type',
            title='Learning Curves'
        )
        curves_results['visualization'] = fig
        
        return curves_results
    
    @monitor_performance
    def _analyze_feature_importance(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Analyze feature importance."""
        # Get feature importance using permutation importance
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=config.random_state,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        importance_results = {
            'importance_values': importance_df.to_dict('records'),
            'top_features': importance_df['feature'].head().tolist()
        }
        
        # Store feature importance
        self.feature_importance[str(datetime.now())] = importance_df
        
        # Create visualization
        fig = plotter.create_plot(
            'bar',
            data=importance_df,
            x='feature',
            y='importance_mean',
            error_y='importance_std',
            title='Feature Importance (Permutation)'
        )
        importance_results['visualization'] = fig
        
        return importance_results
    
    @monitor_performance
    def _analyze_parameter_sensitivity(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_ranges: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Analyze parameter sensitivity."""
        sensitivity_results = {}
        
        for param_name, param_range in param_ranges.items():
            train_scores, test_scores = validation_curve(
                model,
                X,
                y,
                param_name=param_name,
                param_range=param_range,
                cv=5,
                scoring=config.model.default_scoring,
                n_jobs=-1
            )
            
            sensitivity_results[param_name] = {
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
            
            # Create visualization
            fig = plotter.create_plot(
                'line',
                data=pd.DataFrame({
                    'Parameter Value': np.repeat(param_range, 2),
                    'Score': np.concatenate([
                        np.mean(train_scores, axis=1),
                        np.mean(test_scores, axis=1)
                    ]),
                    'Type': ['Train']*len(param_range) + ['Test']*len(param_range)
                }),
                x='Parameter Value',
                y='Score',
                color='Type',
                title=f'Parameter Sensitivity - {param_name}'
            )
            sensitivity_results[param_name]['visualization'] = fig
        
        return sensitivity_results
    
    @monitor_performance
    def _analyze_model_complexity(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Analyze model complexity."""
        # This is a placeholder for model-specific complexity analysis
        # Implementation would depend on the type of model being used
        complexity_results = {
            'model_type': type(model).__name__,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        # Add model-specific complexity metrics
        if hasattr(model, 'n_estimators'):
            complexity_results['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            complexity_results['max_depth'] = model.max_depth
        
        return complexity_results
    
    @monitor_performance
    def _analyze_feature_interactions(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        interaction_features: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Analyze feature interactions."""
        interaction_results = {}
        
        for feature1, feature2 in interaction_features:
            if feature1 in X.columns and feature2 in X.columns:
                # Calculate partial dependence
                pdp_result = partial_dependence(
                    model,
                    X,
                    features=[X.columns.get_loc(feature1), X.columns.get_loc(feature2)],
                    kind='average'
                )
                
                interaction_results[f"{feature1}_{feature2}"] = {
                    'partial_dependence': {
                        'values': pdp_result[1].tolist(),
                        'feature1_grid': pdp_result[2][0].tolist(),
                        'feature2_grid': pdp_result[2][1].tolist()
                    }
                }
                
                # Create visualization
                fig = plotter.create_plot(
                    'heatmap',
                    data=pd.DataFrame(
                        pdp_result[1],
                        index=pdp_result[2][0],
                        columns=pdp_result[2][1]
                    ),
                    title=f'Feature Interaction - {feature1} vs {feature2}'
                )
                interaction_results[f"{feature1}_{feature2}"]['visualization'] = fig
        
        return interaction_results
    
    def _get_default_validation_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'basic_validation': True,
            'learning_curves': True,
            'feature_importance': True,
            'parameter_sensitivity': True,
            'model_complexity': True,
            'feature_interactions': True,
            'param_ranges': {},
            'interaction_features': []
        }
    
    def _record_validation(
        self,
        validation_id: str,
        validation_config: Dict[str, Any]
    ) -> None:
        """Record validation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'validation_id': validation_id,
            'configuration': validation_config
        }
        
        self.validation_history.append(record)
        state_manager.set_state(
            f'training.validation.history.{len(self.validation_history)}',
            record
        )

# Create global model validator instance
model_validator = ModelValidator()