import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Type
from datetime import datetime

# Importing model classes
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Base directory configuration
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DirectoryConfig:
    """Directory configuration with automatic path creation."""
    base_dir: Path = BASE_DIR
    
    # Mode-specific directories
    analysis_dir: Path = field(default_factory=lambda: BASE_DIR / "Analysis")
    training_dir: Path = field(default_factory=lambda: BASE_DIR / "Training")
    prediction_dir: Path = field(default_factory=lambda: BASE_DIR / "Prediction")
    
    # Analysis subdirectories
    analysis_outputs: Path = field(default_factory=lambda: BASE_DIR / "Analysis/Outputs")
    analysis_reports: Path = field(default_factory=lambda: BASE_DIR / "Analysis/Reports")
    analysis_visualizations: Path = field(default_factory=lambda: BASE_DIR / "Analysis/Visualizations")
    analysis_state: Path = field(default_factory=lambda: BASE_DIR / "Analysis/State")
    analysis_config: Path = field(default_factory=lambda: BASE_DIR / "Analysis/Config")
    
    # Training subdirectories
    training_outputs: Path = field(default_factory=lambda: BASE_DIR / "Training/Outputs")
    training_models: Path = field(default_factory=lambda: BASE_DIR / "Training/Models")
    training_reports: Path = field(default_factory=lambda: BASE_DIR / "Training/Reports")
    training_visualizations: Path = field(default_factory=lambda: BASE_DIR / "Training/Visualizations")
    training_state: Path = field(default_factory=lambda: BASE_DIR / "Training/State")
    training_config: Path = field(default_factory=lambda: BASE_DIR / "Training/Config")
    
    # Prediction subdirectories
    prediction_outputs: Path = field(default_factory=lambda: BASE_DIR / "Prediction/Outputs")
    prediction_reports: Path = field(default_factory=lambda: BASE_DIR / "Prediction/Reports")
    prediction_visualizations: Path = field(default_factory=lambda: BASE_DIR / "Prediction/Visualizations")
    prediction_results: Path = field(default_factory=lambda: BASE_DIR / "Prediction/Results")
    prediction_state: Path = field(default_factory=lambda: BASE_DIR / "Prediction/State")

    def __post_init__(self):
        """Create all directories after initialization."""
        self._create_directories()
    
    def _create_directories(self):
        """Create all configured directories."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)

@dataclass
class ValidationConfig:
    """Configuration for data and input validation."""
    default_min_rows: int = 1
    default_max_rows: int = 100000
    allow_nulls: bool = False
    numeric_column_types: List[Type] = field(default_factory=lambda: [int, float, np.number])
    categorical_column_types: List[Type] = field(default_factory=lambda: [str, object, 'category'])
    min_string_length: int = 0
    max_string_length: int = 1000
    max_file_size: int = 200 * 1024 * 1024  # 200 MB

@dataclass
class FileConfig:
    """File-related configurations."""
    # Mandatory model files
    scaler_file: str = "scaler.joblib"
    imputer_file: str = "imputer.joblib"
    trained_model_file: str = "trained_model.joblib"
    cluster_file: str = "cluster.joblib"
    
    # Report files
    data_quality_report: str = "data_quality_report.xlsx"
    performance_metrics_report: str = "performance_metrics.xlsx"
    error_analysis_report: str = "error_analysis.xlsx"
    predictions_file: str = "predictions.xlsx"
    
    # Visualization files
    analysis_visualizations_pdf: str = "analysis_visualizations.pdf"
    training_visualizations_pdf: str = "training_visualizations.pdf"
    prediction_visualizations_pdf: str = "prediction_visualizations.pdf"
    
    # Sheet configuration
    sheet_name_max_length: int = 31
    sheet_name_truncate_suffix: str = "_cluster_db"
    
    # File upload configuration
    allowed_extensions: Set[str] = field(default_factory=lambda: {'csv', 'xlsx'})

@dataclass
class ProcessingConfig:
    """Data processing and analysis parameters."""
    # Analysis parameters
    analysis_threshold: float = 0.05
    analysis_metrics: List[str] = field(default_factory=lambda: [
        'mean', 'median', 'std', 'min', 'max', 'skew', 'kurtosis'
    ])
    correlation_methods: List[str] = field(default_factory=lambda: [
        'pearson', 'spearman', 'kendall'
    ])
    
    # Outlier parameters
    outlier_threshold: float = 3.0
    
    # Feature engineering parameters
    statistical_agg_functions: List[str] = field(default_factory=lambda: [
        'mean', 'median', 'std'
    ])
    top_k_features: int = 20
    max_interaction_degree: int = 2
    polynomial_degree: int = 2
    feature_selection_score_func: str = 'f_regression'
    
    # Random state
    random_state: int = 42

@dataclass
class ClusteringConfig:
    """Clustering-related configurations."""
    available_methods: List[str] = field(default_factory=lambda: [
        'None', 'DBSCAN', 'KMeans', 'GaussianMixture'
    ])
    default_method: str = 'None'
    
    validation_config: Dict[str, Any] = field(default_factory=lambda: {
        'min_samples_per_cluster': 5,
        'max_clusters': 20,
        'required_numeric_features': True
    })
    
    dbscan_params: Dict[str, Any] = field(default_factory=lambda: {
        'eps': 0.5,
        'min_samples': 5
    })
    
    kmeans_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_clusters': 5
    })

@dataclass
class ModelConfig:
    """Model-related configurations."""
    model_classes: Dict[str, Any] = field(default_factory=lambda: {
        'rf': RandomForestRegressor,
        'xgb': XGBRegressor,
        'lgbm': LGBMRegressor,
        'ada': AdaBoostRegressor,
        'catboost': CatBoostRegressor,
        'knn': KNeighborsRegressor
    })
    
    model_validation_config: Dict[str, Any] = field(default_factory=lambda: {
        'required_params': {'model_type', 'hyperparameters'},
        'param_type_constraints': {
            'model_type': str,
            'hyperparameters': dict
        },
        'param_range_constraints': {}
    })
    
    hyperparameter_grids: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'rf': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'xgb': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9]
        },
        'lgbm': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 62, 124]
        },
        'ada': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        },
        'catboost': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8]
        },
        'knn': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    })
    
    # Training parameters
    cv_splits: int = 5
    randomized_search_iterations: int = 10
    ensemble_cv_splits: int = 10
    ensemble_cv_shuffle: bool = True
    scoring_metrics: List[str] = field(default_factory=lambda: [
        'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    ])

@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration."""
    feature_validation_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_generated_features': 100,
        'max_interaction_degree': 3,
        'required_feature_types': ['numeric', 'categorical']
    })

class ConfigManager:
    """Central configuration management class."""
    def __init__(self):
        # Initialize all configurations
        self.directories = DirectoryConfig()
        self.files = FileConfig()
        self.processing = ProcessingConfig()
        self.clustering = ClusteringConfig()
        self.model = ModelConfig()
        self.validation = ValidationConfig()
        self.feature_engineering = FeatureEngineeringConfig()
        
        # Runtime configurations
        self.current_mode: str = 'Analysis'
        self.file_path: Optional[str] = None
        self.sheet_name: Optional[str] = None
        self.target_column: Optional[str] = None
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.unused_columns: List[str] = []
        self.all_columns: List[str] = []
        
        # Initialization timestamps
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters with type-safe and hierarchical updates."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                
                # Handle nested configuration updates
                if isinstance(attr, (DirectoryConfig, FileConfig, ProcessingConfig, 
                                     ClusteringConfig, ModelConfig, ValidationConfig)):
                    for subkey, subvalue in value.items():
                        if hasattr(attr, subkey):
                            setattr(attr, subkey, subvalue)
                else:
                    setattr(self, key, value)
        
        # Update timestamps and columns
        self.last_updated = datetime.now()
        self._update_columns()
    
    def _update_columns(self):
        """Update column-related configurations."""
        self.all_columns = list(set(
            self.numerical_columns +
            self.categorical_columns +
            ([self.target_column] if self.target_column else []) +
            self.unused_columns
        ))
    
    def validate(self) -> bool:
        """Validate entire configuration."""
        try:
            # Validate key configurations
            assert self.current_mode in ['Analysis', 'Training', 'Prediction']
            assert isinstance(self.train_size, float) and 0 < self.train_size < 1
            
            return True
        except AssertionError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            'directories': self.directories.__dict__,
            'files': self.files.__dict__,
            'processing': self.processing.__dict__,
            'clustering': self.clustering.__dict__,
            'model': self.model.__dict__,
            'validation': self.validation.__dict__,
            'feature_engineering': self.feature_engineering.__dict__,
            'runtime': {
                'current_mode': self.current_mode,
                'file_path': self.file_path,
                'sheet_name': self.sheet_name,
                'target_column': self.target_column,
                'numerical_columns': self.numerical_columns,
                'categorical_columns': self.categorical_columns,
                'unused_columns': self.unused_columns,
                'all_columns': self.all_columns
            },
            'metadata': {
                'created_at': self.created_at.isoformat(),
                'last_updated': self.last_updated.isoformat()
            }
        }

# Create global configuration instance
config = ConfigManager()