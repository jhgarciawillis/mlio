import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from core import config
from core.exceptions import PreprocessingError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class DataPreprocessor:
    """Handle all data preprocessing operations."""
    
    def __init__(self):
        self.preprocessing_history: List[Dict[str, Any]] = []
        self.preprocessing_pipelines: Dict[str, Pipeline] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_names: Dict[str, List[str]] = {}
        
        # Initialize available preprocessors
        self.available_scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'power': PowerTransformer,
            'quantile': QuantileTransformer
        }
        
        self.available_encoders = {
            'onehot': OneHotEncoder,
            'label': LabelEncoder,
            'ordinal': OrdinalEncoder
        }
        
        self.available_imputers = {
            'simple': SimpleImputer,
            'knn': KNNImputer
        }
    
    @monitor_performance
    @handle_exceptions(PreprocessingError)
    def create_preprocessing_pipeline(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        scaling_method: str = 'standard',
        encoding_method: str = 'onehot',
        imputation_method: str = 'simple',
        pipeline_name: str = 'default',
        **kwargs
    ) -> Pipeline:
        """Create preprocessing pipeline."""
        try:
            # Record operation start
            state_monitor.record_operation_start(
                'preprocessing_pipeline_creation',
                'preprocessing'
            )
            
            # Validate methods
            self._validate_methods(
                scaling_method,
                encoding_method,
                imputation_method
            )
            
            # Create transformers for numeric and categorical features
            numeric_transformer = self._create_numeric_pipeline(
                scaling_method,
                imputation_method,
                **kwargs
            )
            
            categorical_transformer = self._create_categorical_pipeline(
                encoding_method,
                imputation_method,
                **kwargs
            )
            
            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor)
            ])
            
            # Store pipeline and feature names
            self.preprocessing_pipelines[pipeline_name] = pipeline
            self.feature_names[pipeline_name] = numeric_features + categorical_features
            
            # Record creation in history
            self._record_pipeline_creation(
                pipeline_name,
                {
                    'numeric_features': numeric_features,
                    'categorical_features': categorical_features,
                    'scaling_method': scaling_method,
                    'encoding_method': encoding_method,
                    'imputation_method': imputation_method,
                    'additional_params': kwargs
                }
            )
            
            # Record operation completion
            state_monitor.record_operation_end(
                'preprocessing_pipeline_creation',
                'completed'
            )
            
            return pipeline
            
        except Exception as e:
            state_monitor.record_operation_end(
                'preprocessing_pipeline_creation',
                'failed',
                {'error': str(e)}
            )
            raise PreprocessingError(
                f"Error creating preprocessing pipeline: {str(e)}"
            ) from e
    
    @monitor_performance
    def _create_numeric_pipeline(
        self,
        scaling_method: str,
        imputation_method: str,
        **kwargs
    ) -> Pipeline:
        """Create preprocessing pipeline for numeric features."""
        steps = []
        
        # Add imputer
        imputer = self._get_imputer(
            imputation_method,
            numeric=True,
            **kwargs.get('imputer_params', {})
        )
        steps.append(('imputer', imputer))
        
        # Add scaler
        scaler = self._get_scaler(
            scaling_method,
            **kwargs.get('scaler_params', {})
        )
        steps.append(('scaler', scaler))
        
        return Pipeline(steps)
    
    @monitor_performance
    def _create_categorical_pipeline(
        self,
        encoding_method: str,
        imputation_method: str,
        **kwargs
    ) -> Pipeline:
        """Create preprocessing pipeline for categorical features."""
        steps = []
        
        # Add imputer
        imputer = self._get_imputer(
            imputation_method,
            numeric=False,
            **kwargs.get('imputer_params', {})
        )
        steps.append(('imputer', imputer))
        
        # Add encoder
        encoder = self._get_encoder(
            encoding_method,
            **kwargs.get('encoder_params', {})
        )
        steps.append(('encoder', encoder))
        
        return Pipeline(steps)
    
    def _get_scaler(
        self,
        method: str,
        **kwargs
    ) -> Any:
        """Get scaler instance."""
        if method not in self.available_scalers:
            raise PreprocessingError(
                f"Invalid scaling method: {method}",
                details={'available_methods': list(self.available_scalers.keys())}
            )
        
        return self.available_scalers[method](**kwargs)
    
    def _get_encoder(
        self,
        method: str,
        **kwargs
    ) -> Any:
        """Get encoder instance."""
        if method not in self.available_encoders:
            raise PreprocessingError(
                f"Invalid encoding method: {method}",
                details={'available_methods': list(self.available_encoders.keys())}
            )
        
        return self.available_encoders[method](**kwargs)
    
    def _get_imputer(
        self,
        method: str,
        numeric: bool = True,
        **kwargs
    ) -> Any:
        """Get imputer instance."""
        if method not in self.available_imputers:
            raise PreprocessingError(
                f"Invalid imputation method: {method}",
                details={'available_methods': list(self.available_imputers.keys())}
            )
        
        if method == 'simple':
            if numeric:
                kwargs.setdefault('strategy', 'mean')
            else:
                kwargs.setdefault('strategy', 'most_frequent')
        
        return self.available_imputers[method](**kwargs)
    
    def _validate_methods(
        self,
        scaling_method: str,
        encoding_method: str,
        imputation_method: str
    ) -> None:
        """Validate preprocessing methods."""
        if scaling_method not in self.available_scalers:
            raise PreprocessingError(f"Invalid scaling method: {scaling_method}")
        
        if encoding_method not in self.available_encoders:
            raise PreprocessingError(f"Invalid encoding method: {encoding_method}")
        
        if imputation_method not in self.available_imputers:
            raise PreprocessingError(f"Invalid imputation method: {imputation_method}")
    
    def _record_pipeline_creation(
        self,
        pipeline_name: str,
        config: Dict[str, Any]
    ) -> None:
        """Record pipeline creation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_name': pipeline_name,
            'configuration': config
        }
        
        self.preprocessing_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'preprocessing.pipelines.{pipeline_name}',
            record
        )
    
    @monitor_performance
    @handle_exceptions(PreprocessingError)
    def fit_transform(
        self,
        data: pd.DataFrame,
        pipeline_name: str = 'default'
    ) -> pd.DataFrame:
        """Fit and transform data using specified pipeline."""
        try:
            # Record operation start
            state_monitor.record_operation_start('preprocessing_fit_transform', 'preprocessing')
            
            pipeline = self.preprocessing_pipelines.get(pipeline_name)
            if pipeline is None:
                raise PreprocessingError(f"Pipeline not found: {pipeline_name}")
            
            # Fit and transform data
            transformed_data = pipeline.fit_transform(data)
            
            # Get feature names
            feature_names = self.feature_names[pipeline_name]
            
            # Create DataFrame with proper column names
            transformed_df = pd.DataFrame(
                transformed_data,
                index=data.index,
                columns=feature_names
            )
            
            # Record operation completion
            state_monitor.record_operation_end(
                'preprocessing_fit_transform',
                'completed',
                {'shape': transformed_df.shape}
            )
            
            return transformed_df
            
        except Exception as e:
            state_monitor.record_operation_end(
                'preprocessing_fit_transform',
                'failed',
                {'error': str(e)}
            )
            raise PreprocessingError(f"Error in fit_transform: {str(e)}") from e
    
    @monitor_performance
    @handle_exceptions(PreprocessingError)
    def transform(
        self,
        data: pd.DataFrame,
        pipeline_name: str = 'default'
    ) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        try:
            # Record operation start
            state_monitor.record_operation_start('preprocessing_transform', 'preprocessing')
            
            pipeline = self.preprocessing_pipelines.get(pipeline_name)
            if pipeline is None:
                raise PreprocessingError(f"Pipeline not found: {pipeline_name}")
            
            # Transform data
            transformed_data = pipeline.transform(data)
            
            # Get feature names
            feature_names = self.feature_names[pipeline_name]
            
            # Create DataFrame with proper column names
            transformed_df = pd.DataFrame(
                transformed_data,
                index=data.index,
                columns=feature_names
            )
            
            # Record operation completion
            state_monitor.record_operation_end(
                'preprocessing_transform',
                'completed',
                {'shape': transformed_df.shape}
            )
            
            return transformed_df
            
        except Exception as e:
            state_monitor.record_operation_end(
                'preprocessing_transform',
                'failed',
                {'error': str(e)}
            )
            raise PreprocessingError(f"Error in transform: {str(e)}") from e
    
    @monitor_performance
    def save_pipeline(
        self,
        pipeline_name: str,
        path: Optional[Path] = None
    ) -> None:
        """Save preprocessing pipeline."""
        if path is None:
            path = config.directories.preprocessors
            path.mkdir(parents=True, exist_ok=True)
        
        pipeline = self.preprocessing_pipelines.get(pipeline_name)
        if pipeline is None:
            raise PreprocessingError(f"Pipeline not found: {pipeline_name}")
        
        # Save pipeline
        pipeline_path = path / f"{pipeline_name}_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        
        # Save feature names
        feature_names_path = path / f"{pipeline_name}_features.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names[pipeline_name], f)
        
        # Save preprocessing history
        history_path = path / f"{pipeline_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.preprocessing_history, f, indent=4)
        
        logger.info(f"Preprocessing pipeline saved to {path}")
    
    @monitor_performance
    def load_pipeline(
        self,
        pipeline_name: str,
        path: Optional[Path] = None
    ) -> None:
        """Load preprocessing pipeline."""
        if path is None:
            path = config.directories.preprocessors
        
        # Load pipeline
        pipeline_path = path / f"{pipeline_name}_pipeline.joblib"
        if not pipeline_path.exists():
            raise PreprocessingError(f"Pipeline file not found: {pipeline_path}")
        
        self.preprocessing_pipelines[pipeline_name] = joblib.load(pipeline_path)
        
        # Load feature names
        feature_names_path = path / f"{pipeline_name}_features.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names[pipeline_name] = json.load(f)
        
        # Load preprocessing history
        history_path = path / f"{pipeline_name}_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.preprocessing_history = json.load(f)
        
        logger.info(f"Preprocessing pipeline loaded from {path}")
    
    @monitor_performance
    def get_preprocessing_summary(
        self,
        pipeline_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of preprocessing operations."""
        if pipeline_name:
            pipeline = self.preprocessing_pipelines.get(pipeline_name)
            if pipeline is None:
                raise PreprocessingError(f"Pipeline not found: {pipeline_name}")
            
            return {
                'pipeline_name': pipeline_name,
                'feature_names': self.feature_names[pipeline_name],
                'history': [
                    record for record in self.preprocessing_history
                    if record['pipeline_name'] == pipeline_name
                ]
            }
        
        return {
            'total_pipelines': len(self.preprocessing_pipelines),
            'pipelines': list(self.preprocessing_pipelines.keys()),
            'history_length': len(self.preprocessing_history),
            'last_operation': self.preprocessing_history[-1] if self.preprocessing_history else None
        }
    
    @monitor_performance
    def reset_pipeline(
        self,
        pipeline_name: str
    ) -> None:
        """Reset specified pipeline to initial state."""
        if pipeline_name in self.preprocessing_pipelines:
            del self.preprocessing_pipelines[pipeline_name]
        if pipeline_name in self.feature_names:
            del self.feature_names[pipeline_name]
            
        # Update state
        state_manager.set_state(
            f'preprocessing.pipelines.{pipeline_name}',
            None
        )
        
        logger.info(f"Pipeline {pipeline_name} reset")

# Create global preprocessor instance
preprocessor = DataPreprocessor()