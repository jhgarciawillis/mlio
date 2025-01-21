import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    RFE,
    RFECV,
    mutual_info_regression,
    f_regression,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from datetime import datetime

from core import config
from core.exceptions import FeatureSelectionError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter

class FeatureSelector:
    """Handle feature selection operations."""
    
    def __init__(self):
        self.selection_history: List[Dict[str, Any]] = []
        self.selected_features: Dict[str, List[str]] = {}
        self.importance_scores: Dict[str, pd.DataFrame] = {}
        self.selection_metadata: Dict[str, Dict[str, Any]] = {}
    
    @monitor_performance
    @handle_exceptions(FeatureSelectionError)
    def select_features(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        method: str = 'mutual_info',
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select features using specified method."""
        try:
            selection_id = f"selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize selector based on method
            selector = self._get_selector(method, n_features, threshold, **kwargs)
            
            # Fit and transform data
            selected_data = selector.fit_transform(data, target)
            
            # Get selected feature names
            if hasattr(selector, 'get_support'):
                selected_cols = data.columns[selector.get_support()].tolist()
            else:
                selected_cols = data.columns.tolist()
            
            # Get feature importance scores if available
            importance_scores = self._get_importance_scores(selector, data.columns)
            
            # Store results
            self.selected_features[selection_id] = selected_cols
            if importance_scores is not None:
                self.importance_scores[selection_id] = importance_scores
            
            # Create metadata
            metadata = {
                'method': method,
                'n_features_in': data.shape[1],
                'n_features_selected': len(selected_cols),
                'selected_features': selected_cols,
                'parameters': kwargs
            }
            self.selection_metadata[selection_id] = metadata
            
            # Create visualizations
            self._create_selection_visualizations(
                selection_id,
                importance_scores,
                selected_cols
            )
            
            # Record selection
            self._record_feature_selection(selection_id, metadata)
            
            return pd.DataFrame(selected_data, columns=selected_cols), metadata
            
        except Exception as e:
            raise FeatureSelectionError(
                f"Error selecting features: {str(e)}"
            ) from e
    
    def _get_selector(
        self,
        method: str,
        n_features: Optional[int],
        threshold: Optional[float],
        **kwargs
    ) -> Any:
        """Get feature selector based on method."""
        if method == 'mutual_info':
            return SelectKBest(
                score_func=mutual_info_regression,
                k=n_features or 'all'
            )
        elif method == 'f_regression':
            return SelectKBest(
                score_func=f_regression,
                k=n_features or 'all'
            )
        elif method == 'variance':
            return VarianceThreshold(threshold=threshold or 0.0)
        elif method == 'lasso':
            return SelectFromModel(
                Lasso(alpha=kwargs.get('alpha', 1.0)),
                threshold=threshold
            )
        elif method == 'random_forest':
            return SelectFromModel(
                RandomForestRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    random_state=config.random_state
                ),
                threshold=threshold
            )
        elif method == 'rfe':
            return RFE(
                estimator=RandomForestRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    random_state=config.random_state
                ),
                n_features_to_select=n_features
            )
        elif method == 'rfecv':
            return RFECV(
                estimator=RandomForestRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    random_state=config.random_state
                ),
                min_features_to_select=kwargs.get('min_features', 1),
                cv=kwargs.get('cv', 5)
            )
        else:
            raise FeatureSelectionError(f"Unknown selection method: {method}")
    
    def _get_importance_scores(
        self,
        selector: Any,
        feature_names: pd.Index
    ) -> Optional[pd.DataFrame]:
        """Get feature importance scores from selector."""
        if hasattr(selector, 'scores_'):
            scores = selector.scores_
        elif hasattr(selector, 'feature_importances_'):
            scores = selector.feature_importances_
        elif hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
            scores = selector.estimator_.feature_importances_
        else:
            return None
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': scores
        }).sort_values('importance', ascending=False)
    
    @monitor_performance
    def _create_selection_visualizations(
        self,
        selection_id: str,
        importance_scores: Optional[pd.DataFrame],
        selected_features: List[str]
    ) -> None:
        """Create visualizations for feature selection results."""
        if importance_scores is not None:
            # Feature importance plot
            fig_importance = plotter.create_plot(
                'bar',
                data=importance_scores,
                x='feature',
                y='importance',
                title='Feature Importance Scores'
            )
            self.selection_metadata[selection_id]['visualizations'] = {
                'importance_plot': fig_importance
            }
            
            # Selected vs. Not Selected features
            importance_scores['selected'] = importance_scores['feature'].isin(selected_features)
            fig_comparison = plotter.create_plot(
                'scatter',
                data=importance_scores,
                x='feature',
                y='importance',
                color='selected',
                title='Selected vs. Not Selected Features'
            )
            self.selection_metadata[selection_id]['visualizations']['comparison_plot'] = fig_comparison
    
    @monitor_performance
    def get_optimal_features(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        max_features: Optional[int] = None,
        cv_folds: int = 5
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Get optimal feature subset using cross-validation."""
        try:
            selector = RFECV(
                estimator=RandomForestRegressor(random_state=config.random_state),
                min_features_to_select=1,
                cv=cv_folds
            )
            
            selector.fit(data, target)
            optimal_features = data.columns[selector.support_].tolist()
            
            results = {
                'n_features': len(optimal_features),
                'cv_scores': selector.grid_scores_.tolist(),
                'optimal_score': float(np.max(selector.grid_scores_))
            }
            
            # Create CV scores plot
            fig = plotter.create_plot(
                'line',
                data=pd.DataFrame({
                    'n_features': range(1, len(selector.grid_scores_) + 1),
                    'cv_score': selector.grid_scores_
                }),
                x='n_features',
                y='cv_score',
                title='Cross-validation Scores vs. Number of Features'
            )
            results['visualization'] = fig
            
            return optimal_features, results
            
        except Exception as e:
            raise FeatureSelectionError(
                f"Error finding optimal features: {str(e)}"
            ) from e
    
    def _record_feature_selection(
        self,
        selection_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Record feature selection in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'selection_id': selection_id,
            'metadata': metadata
        }
        
        self.selection_history.append(record)
        state_manager.set_state(
            f'features.selection.history.{len(self.selection_history)}',
            record
        )

# Create global feature selector instance
feature_selector = FeatureSelector()