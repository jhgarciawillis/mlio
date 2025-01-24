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
    clustering_dir: Path = field(default_factory=lambda: BASE_DIR / "Clustering")
    
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
    
    # Clustering subdirectories
    clustering_outputs: Path = field(default_factory=lambda: BASE_DIR / "Clustering/Outputs")
    clustering_models: Path = field(default_factory=lambda: BASE_DIR / "Clustering/Models")
    clustering_reports: Path = field(default_factory=lambda: BASE_DIR / "Clustering/Reports")
    clustering_visualizations: Path = field(default_factory=lambda: BASE_DIR / "Clustering/Visualizations")
    clustering_state: Path = field(default_factory=lambda: BASE_DIR / "Clustering/State")

    def __post_init__(self):
        """Create all directories after initialization."""
        self._create_directories()
    
    def _create_directories(self):
        """Create all configured directories."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)

@dataclass
class ClusteringConfig:
    """Configuration for clustering operations."""
    available_methods: List[str] = field(default_factory=lambda: [
        'kmeans', 'dbscan', 'gaussian_mixture', 'hierarchical', 'spectral'
    ])
    default_method: str = 'kmeans'
    
    # Default parameters for each method
    kmeans_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_clusters': 5,
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300
    })
    
    dbscan_params: Dict[str, Any] = field(default_factory=lambda: {
        'eps': 0.5,
        'min_samples': 5,
        'metric': 'euclidean'
    })
    
    gaussian_mixture_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_components': 5,
        'covariance_type': 'full',
        'max_iter': 100
    })
    
    hierarchical_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_clusters': 5,
        'affinity': 'euclidean',
        'linkage': 'ward'
    })
    
    spectral_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_clusters': 5,
        'affinity': 'rbf',
        'assign_labels': 'kmeans'
    })
    
    # Optimization parameters
    optimization_metric: str = 'silhouette'
    max_clusters: int = 20
    min_clusters: int = 2
    optimization_trials: int = 50
    
    # Validation parameters
    validation_metrics: List[str] = field(default_factory=lambda: [
        'silhouette', 'calinski_harabasz', 'davies_bouldin'
    ])
    stability_trials: int = 10
    cross_validation_splits: int = 5

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    # Theme settings
    primary_color: str = '#1f77b4'
    secondary_color: str = '#ff7f0e'
    background_color: str = '#ffffff'
    text_color: str = '#2f2f2f'
    grid_color: str = '#e6e6e6'
    
    # Font settings
    font_family: str = 'Arial'
    font_size: int = 12
    title_font_size: int = 16
    axis_font_size: int = 10
    
    # Plot settings
    plot_width: int = 800
    plot_height: int = 500
    dpi: int = 100
    
    # Dashboard settings
    dashboard_width: str = 'wide'
    dashboard_padding: int = 20
    max_plots_per_row: int = 3
    
    # Animation settings
    animation_duration: int = 500
    transition_easing: str = 'cubic-in-out'
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ['png', 'html', 'pdf'])
    image_quality: int = 95

@dataclass
class UIConfig:
    """Configuration for UI settings."""
    # Theme
    dark_mode: bool = False
    primary_color: str = '#1f77b4'
    accent_color: str = '#ff7f0e'
    
    # Layout
    sidebar_width: int = 300
    content_width: str = 'wide'
    show_footer: bool = True
    
    # Components
    button_style: str = 'primary'
    input_style: str = 'outlined'
    table_height: int = 400
    
    # Interaction
    animation_speed: float = 0.3
    tooltip_delay: int = 500
    confirmation_dialogs: bool = True
    
    # Notifications
    show_notifications: bool = True
    notification_duration: int = 3
    notification_position: str = 'top-right'

@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    # Performance monitoring
    monitor_performance: bool = True
    monitor_memory: bool = True
    monitor_cpu: bool = True
    sampling_interval: float = 1.0
    
    # Logging
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    max_log_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    
    # Alerts
    memory_threshold: float = 90.0  # Percentage
    cpu_threshold: float = 80.0  # Percentage
    response_time_threshold: float = 5.0  # Seconds

class ConfigManager:
    """Central configuration management class."""
    def __init__(self):
        # Initialize all configurations
        self.directories = DirectoryConfig()
        self.clustering = ClusteringConfig()
        self.visualization = VisualizationConfig()
        self.ui = UIConfig()
        self.monitoring = MonitoringConfig()
        
        # Runtime configurations
        self.current_mode: str = 'Analysis'
        self.file_path: Optional[str] = None
        self.sheet_name: Optional[str] = None
        self.target_column: Optional[str] = None
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.unused_columns: List[str] = []
        self.all_columns: List[str] = []
        
        # Model configurations
        self.model_classes: Dict[str, Any] = {
            'rf': RandomForestRegressor,
            'xgb': XGBRegressor,
            'lgbm': LGBMRegressor,
            'ada': AdaBoostRegressor,
            'catboost': CatBoostRegressor,
            'knn': KNeighborsRegressor
        }
        
        # Initialization timestamps
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.last_updated = datetime.now()
        self._update_columns()
    
    def _update_columns(self) -> None:
        """Update column-related configurations."""
        self.all_columns = list(set(
            self.numerical_columns +
            self.categorical_columns +
            ([self.target_column] if self.target_column else []) +
            self.unused_columns
        ))
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = self.directories.base_dir / 'config.json'
        
        config_dict = {
            'clustering': self.clustering.__dict__,
            'visualization': self.visualization.__dict__,
            'ui': self.ui.__dict__,
            'monitoring': self.monitoring.__dict__,
            'runtime': {
                'current_mode': self.current_mode,
                'numerical_columns': self.numerical_columns,
                'categorical_columns': self.categorical_columns,
                'target_column': self.target_column
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def load_config(self, path: Optional[Path] = None) -> None:
        """Load configuration from file."""
        if path is None:
            path = self.directories.base_dir / 'config.json'
        
        if not path.exists():
            return
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Update configurations
        for key, value in config_dict.get('clustering', {}).items():
            setattr(self.clustering, key, value)
        
        for key, value in config_dict.get('visualization', {}).items():
            setattr(self.visualization, key, value)
            
        for key, value in config_dict.get('ui', {}).items():
            setattr(self.ui, key, value)
            
        for key, value in config_dict.get('monitoring', {}).items():
            setattr(self.monitoring, key, value)
        
        # Update runtime configurations
        runtime = config_dict.get('runtime', {})
        self.current_mode = runtime.get('current_mode', self.current_mode)
        self.numerical_columns = runtime.get('numerical_columns', self.numerical_columns)
        self.categorical_columns = runtime.get('categorical_columns', self.categorical_columns)
        self.target_column = runtime.get('target_column', self.target_column)
        
        self._update_columns()

# Create global configuration instance
config = ConfigManager()