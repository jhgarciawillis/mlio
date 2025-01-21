import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin

from core import config
from core.exceptions import FeatureGenerationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class BaseFeatureGenerator(BaseEstimator, TransformerMixin):
    """Base class for feature generators."""
    
    def __init__(self):
        self.feature_names: List[str] = []
        self.n_features_generated: int = 0
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseFeatureGenerator':
        """Fit generator to data."""
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by generating features."""
        raise NotImplementedError
        
    def get_feature_names(self) -> List[str]:
        """Get names of generated features."""
        return self.feature_names

class MathematicalFeatureGenerator(BaseFeatureGenerator):
    """Generate mathematical features."""
    
    def __init__(
        self,
        operations: List[str] = ['square', 'cube', 'sqrt', 'log', 'abs']
    ):
        super().__init__()
        self.operations = operations
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate mathematical features."""
        result = pd.DataFrame(index=X.index)
        
        for col in X.select_dtypes(include=[np.number]).columns:
            series = X[col]
            if 'square' in self.operations:
                result[f"{col}_squared"] = series ** 2
            if 'cube' in self.operations:
                result[f"{col}_cubed"] = series ** 3
            if 'sqrt' in self.operations:
                result[f"{col}_sqrt"] = np.sqrt(np.abs(series))
            if 'log' in self.operations:
                result[f"{col}_log"] = np.log1p(np.abs(series))
            if 'abs' in self.operations:
                result[f"{col}_abs"] = np.abs(series)
                
        self.feature_names = result.columns.tolist()
        self.n_features_generated = len(self.feature_names)
        return result

class DateTimeFeatureGenerator(BaseFeatureGenerator):
    """Generate datetime features."""
    
    def __init__(
        self,
        cyclical: bool = True,
        time_components: bool = True,
        intervals: bool = True
    ):
        super().__init__()
        self.cyclical = cyclical
        self.time_components = time_components
        self.intervals = intervals
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate datetime features."""
        result = pd.DataFrame(index=X.index)
        
        for col in X.select_dtypes(include=['datetime64']).columns:
            dt = X[col]
            
            if self.time_components:
                result[f"{col}_year"] = dt.dt.year
                result[f"{col}_month"] = dt.dt.month
                result[f"{col}_day"] = dt.dt.day
                result[f"{col}_hour"] = dt.dt.hour
                result[f"{col}_minute"] = dt.dt.minute
                result[f"{col}_dayofweek"] = dt.dt.dayofweek
                result[f"{col}_quarter"] = dt.dt.quarter
                result[f"{col}_dayofyear"] = dt.dt.dayofyear
                result[f"{col}_weekofyear"] = dt.dt.isocalendar().week
            
            if self.cyclical:
                result[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
                result[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
                result[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
                result[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
                result[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
                result[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
            
            if self.intervals and col not in X.columns[0]:
                first_date = X[X.columns[0]]
                if pd.api.types.is_datetime64_any_dtype(first_date):
                    result[f"{col}_days_since_first"] = (dt - first_date).dt.days
                    result[f"{col}_weeks_since_first"] = (dt - first_date).dt.days / 7
                    result[f"{col}_months_since_first"] = (dt - first_date).dt.days / 30.44
        
        self.feature_names = result.columns.tolist()
        self.n_features_generated = len(self.feature_names)
        return result

class CategoryFeatureGenerator(BaseFeatureGenerator):
    """Generate categorical features."""
    
    def __init__(
        self,
        encoding_methods: List[str] = ['count', 'frequency', 'label'],
        handle_unknown: str = 'ignore'
    ):
        super().__init__()
        self.encoding_methods = encoding_methods
        self.handle_unknown = handle_unknown
        self.encoding_maps = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoryFeatureGenerator':
        """Fit category encodings."""
        for col in X.select_dtypes(include=['object', 'category']).columns:
            self.encoding_maps[col] = {}
            if 'count' in self.encoding_methods:
                self.encoding_maps[col]['count'] = X[col].value_counts()
            if 'frequency' in self.encoding_methods:
                self.encoding_maps[col]['frequency'] = X[col].value_counts(normalize=True)
            if 'label' in self.encoding_methods:
                self.encoding_maps[col]['label'] = {
                    val: i for i, val in enumerate(X[col].unique())
                }
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate categorical features."""
        result = pd.DataFrame(index=X.index)
        
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if 'count' in self.encoding_methods:
                result[f"{col}_count"] = X[col].map(self.encoding_maps[col]['count'])
            if 'frequency' in self.encoding_methods:
                result[f"{col}_freq"] = X[col].map(self.encoding_maps[col]['frequency'])
            if 'label' in self.encoding_methods:
                result[f"{col}_label"] = X[col].map(self.encoding_maps[col]['label'])
            
            # Handle unknown values
            if self.handle_unknown == 'ignore':
                result = result.fillna(-1)
        
        self.feature_names = result.columns.tolist()
        self.n_features_generated = len(self.feature_names)
        return result

class TextFeatureGenerator(BaseFeatureGenerator):
    """Generate text features."""
    
    def __init__(
        self,
        basic_features: bool = True,
        advanced_features: bool = False,
        custom_patterns: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        self.basic_features = basic_features
        self.advanced_features = advanced_features
        self.custom_patterns = custom_patterns or {}
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate text features."""
        result = pd.DataFrame(index=X.index)
        
        for col in X.select_dtypes(include=['object']):
            if self.basic_features:
                result[f"{col}_length"] = X[col].str.len()
                result[f"{col}_word_count"] = X[col].str.split().str.len()
                result[f"{col}_unique_chars"] = X[col].str.unique().str.len()
                result[f"{col}_capital_ratio"] = (
                    X[col].str.count(r'[A-Z]') / X[col].str.len()
                )
            
            if self.advanced_features:
                result[f"{col}_digit_ratio"] = (
                    X[col].str.count(r'[0-9]') / X[col].str.len()
                )
                result[f"{col}_space_ratio"] = (
                    X[col].str.count(r'\s') / X[col].str.len()
                )
                result[f"{col}_special_ratio"] = (
                    X[col].str.count(r'[^a-zA-Z0-9\s]') / X[col].str.len()
                )
            
            for pattern_name, pattern in self.custom_patterns.items():
                result[f"{col}_{pattern_name}"] = X[col].str.count(pattern)
        
        self.feature_names = result.columns.tolist()
        self.n_features_generated = len(self.feature_names)
        return result

class FeatureGenerator:
    """Main feature generator class."""
    
    def __init__(self):
        self.generators: Dict[str, BaseFeatureGenerator] = {
            'mathematical': MathematicalFeatureGenerator(),
            'datetime': DateTimeFeatureGenerator(),
            'category': CategoryFeatureGenerator(),
            'text': TextFeatureGenerator()
        }
        self.generation_history: List[Dict[str, Any]] = []
    
    @monitor_performance
    @handle_exceptions(FeatureGenerationError)
    def generate_features(
        self,
        data: pd.DataFrame,
        generators: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate features using specified generators."""
        try:
            if generators is None:
                generators = list(self.generators.keys())
            
            generation_id = f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            new_features = pd.DataFrame(index=data.index)
            metadata = {'generators': {}}
            
            for generator_name in generators:
                if generator_name in self.generators:
                    generator = self.generators[generator_name]
                    generator.fit(data)
                    features = generator.transform(data)
                    
                    new_features = pd.concat([new_features, features], axis=1)
                    
                    metadata['generators'][generator_name] = {
                        'n_features': generator.n_features_generated,
                        'feature_names': generator.feature_names
                    }
            
            # Record generation
            self._record_generation(generation_id, metadata)
            
            return new_features, metadata
            
        except Exception as e:
            raise FeatureGenerationError(
                f"Error generating features: {str(e)}"
            ) from e
    
    def _record_generation(
        self,
        generation_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Record feature generation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'generation_id': generation_id,
            'metadata': metadata
        }
        
        self.generation_history.append(record)
        state_manager.set_state(
            f'features.generation.history.{len(self.generation_history)}',
            record
        )

# Create global feature generator instance
feature_generator = FeatureGenerator()