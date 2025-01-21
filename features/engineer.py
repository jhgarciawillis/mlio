import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.decomposition import PCA

from core import config
from core.exceptions import FeatureEngineeringError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter

class FeatureEngineer:
    """Handle feature engineering operations."""
    
    def __init__(self):
        self.feature_history: List[Dict[str, Any]] = []
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.engineered_features: Dict[str, pd.DataFrame] = {}
        self.feature_metadata: Dict[str, Dict[str, Any]] = {}
    
    @monitor_performance
    @handle_exceptions(FeatureEngineeringError)
    def generate_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate features based on configuration."""
        try:
            # Get default config if none provided
            if config is None:
                config = self._get_default_config()
            
            feature_set_id = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            new_features = pd.DataFrame(index=data.index)
            metadata = {'feature_types': {}}
            
            # Generate different types of features
            if config.get('polynomial', {}).get('enabled', True):
                poly_features = self._generate_polynomial_features(
                    data,
                    config['polynomial']
                )
                new_features = pd.concat([new_features, poly_features], axis=1)
                metadata['feature_types']['polynomial'] = poly_features.columns.tolist()
            
            if config.get('interaction', {}).get('enabled', True):
                interact_features = self._generate_interaction_features(
                    data,
                    config['interaction']
                )
                new_features = pd.concat([new_features, interact_features], axis=1)
                metadata['feature_types']['interaction'] = interact_features.columns.tolist()
            
            if config.get('time', {}).get('enabled', True):
                time_features = self._generate_time_features(
                    data,
                    config['time']
                )
                new_features = pd.concat([new_features, time_features], axis=1)
                metadata['feature_types']['time'] = time_features.columns.tolist()
            
            if config.get('aggregate', {}).get('enabled', True):
                agg_features = self._generate_aggregate_features(
                    data,
                    config['aggregate']
                )
                new_features = pd.concat([new_features, agg_features], axis=1)
                metadata['feature_types']['aggregate'] = agg_features.columns.tolist()
            
            if config.get('text', {}).get('enabled', True):
                text_features = self._generate_text_features(
                    data,
                    config['text']
                )
                new_features = pd.concat([new_features, text_features], axis=1)
                metadata['feature_types']['text'] = text_features.columns.tolist()
            
            # Store results
            self.engineered_features[feature_set_id] = new_features
            self.feature_metadata[feature_set_id] = metadata
            
            # Record operation
            self._record_feature_generation(feature_set_id, config)
            
            return new_features, metadata
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Error generating features: {str(e)}"
            ) from e
    
    @monitor_performance
    def _generate_polynomial_features(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate polynomial features."""
        degree = config.get('degree', 2)
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns)
        
        if not columns.empty:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(data[columns])
            feature_names = poly.get_feature_names_out(columns)
            
            return pd.DataFrame(
                poly_features[:,len(columns):],  # Exclude original features
                columns=feature_names[len(columns):],
                index=data.index
            )
        
        return pd.DataFrame(index=data.index)
    
    @monitor_performance
    def _generate_interaction_features(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate interaction features."""
        columns = config.get('columns', data.select_dtypes(include=[np.number]).columns)
        max_interactions = config.get('max_interactions', 2)
        
        interactions = pd.DataFrame(index=data.index)
        if not columns.empty:
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    if j - i <= max_interactions:
                        col1, col2 = columns[i], columns[j]
                        name = f"{col1}_x_{col2}"
                        interactions[name] = data[col1] * data[col2]
        
        return interactions
    
    @monitor_performance
    def _generate_time_features(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate time-based features."""
        datetime_columns = [
            col for col in data.columns
            if pd.api.types.is_datetime64_any_dtype(data[col])
        ]
        
        time_features = pd.DataFrame(index=data.index)
        for col in datetime_columns:
            time_features[f"{col}_year"] = data[col].dt.year
            time_features[f"{col}_month"] = data[col].dt.month
            time_features[f"{col}_day"] = data[col].dt.day
            time_features[f"{col}_dayofweek"] = data[col].dt.dayofweek
            time_features[f"{col}_hour"] = data[col].dt.hour
            time_features[f"{col}_minute"] = data[col].dt.minute
            
            if config.get('include_cyclical', True):
                # Add cyclical encoding for periodic features
                time_features[f"{col}_month_sin"] = np.sin(2 * np.pi * data[col].dt.month/12)
                time_features[f"{col}_month_cos"] = np.cos(2 * np.pi * data[col].dt.month/12)
                time_features[f"{col}_hour_sin"] = np.sin(2 * np.pi * data[col].dt.hour/24)
                time_features[f"{col}_hour_cos"] = np.cos(2 * np.pi * data[col].dt.hour/24)
        
        return time_features
    
    @monitor_performance
    def _generate_aggregate_features(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate aggregate features."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        group_by_columns = config.get('group_by', [])
        
        agg_features = pd.DataFrame(index=data.index)
        if group_by_columns and not numeric_columns.empty:
            for col in numeric_columns:
                grouped = data.groupby(group_by_columns)[col]
                agg_features[f"{col}_mean"] = grouped.transform('mean')
                agg_features[f"{col}_std"] = grouped.transform('std')
                agg_features[f"{col}_min"] = grouped.transform('min')
                agg_features[f"{col}_max"] = grouped.transform('max')
        
        return agg_features
    
    @monitor_performance
    def _generate_text_features(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate text-based features."""
        text_columns = config.get('columns', [])
        text_features = pd.DataFrame(index=data.index)
        
        for col in text_columns:
            if col in data.columns:
                text_features[f"{col}_length"] = data[col].str.len()
                text_features[f"{col}_word_count"] = data[col].str.split().str.len()
                text_features[f"{col}_unique_chars"] = data[col].str.nunique()
                
                if config.get('advanced', False):
                    # Add more advanced text features
                    text_features[f"{col}_uppercase_ratio"] = (
                        data[col].str.count(r'[A-Z]') / data[col].str.len()
                    )
                    text_features[f"{col}_digit_ratio"] = (
                        data[col].str.count(r'[0-9]') / data[col].str.len()
                    )
                    text_features[f"{col}_space_ratio"] = (
                        data[col].str.count(r'\s') / data[col].str.len()
                    )
        
        return text_features
    
    @monitor_performance
    def select_features(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        n_features: int,
        method: str = 'mutual_info'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Select most important features."""
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        else:  # f_regression
            selector = SelectKBest(score_func=f_regression, k=n_features)
        
        selected_features = selector.fit_transform(data, target)
        feature_scores = pd.Series(
            selector.scores_,
            index=data.columns
        ).sort_values(ascending=False)
        
        # Store feature importance
        self.feature_importance[method] = pd.DataFrame({
            'feature': feature_scores.index,
            'importance': feature_scores.values
        })
        
        # Create importance plot
        fig = plotter.create_plot(
            'bar',
            data=self.feature_importance[method],
            x='feature',
            y='importance',
            title=f'Feature Importance ({method})'
        )
        
        return (
            pd.DataFrame(selected_features, columns=data.columns[selector.get_support()]),
            feature_scores
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration."""
        return {
            'polynomial': {
                'enabled': True,
                'degree': 2,
                'columns': []
            },
            'interaction': {
                'enabled': True,
                'max_interactions': 2,
                'columns': []
            },
            'time': {
                'enabled': True,
                'include_cyclical': True
            },
            'aggregate': {
                'enabled': True,
                'group_by': []
            },
            'text': {
                'enabled': True,
                'columns': [],
                'advanced': False
            }
        }
    
    def _record_feature_generation(
        self,
        feature_set_id: str,
        config: Dict[str, Any]
    ) -> None:
        """Record feature generation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'feature_set_id': feature_set_id,
            'configuration': config,
            'n_features': len(self.engineered_features[feature_set_id].columns)
        }
        
        self.feature_history.append(record)
        state_manager.set_state(
            f'features.history.{len(self.feature_history)}',
            record
        )

# Create global feature engineer instance
feature_engineer = FeatureEngineer()