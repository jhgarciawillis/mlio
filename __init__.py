"""
MLTrainer2: A comprehensive machine learning training and analysis framework.
"""

from mltrainer2.core.config import config, ConfigManager
from mltrainer2.core.exceptions import (
    MLTrainerException,
    ConfigurationError,
    DataError,
    ValidationError,
    FeatureEngineeringError,
    ClusteringError,
    ModelError,
    PredictionError,
    VisualizationError,
    StateError,
    UIError
)
from mltrainer2.core.state_manager import state_manager
from mltrainer2.core.state_monitoring import state_monitor

# Data handling
from mltrainer2.data.loader import data_loader
from mltrainer2.data.preprocessor import preprocessor
from mltrainer2.data.validator import data_validator
from mltrainer2.data.cleaner import data_cleaner
from mltrainer2.data.exporter import exporter

# Analysis
from mltrainer2.analysis.analyzer import analyzer
from mltrainer2.analysis.statistics import statistical_analyzer
from mltrainer2.analysis.quality import quality_analyzer

# Feature engineering
from mltrainer2.features.engineer import feature_engineer
from mltrainer2.features.generator import feature_generator
from mltrainer2.features.selector import feature_selector

# Clustering
from mltrainer2.clustering.clusterer import clusterer
from mltrainer2.clustering.optimizer import cluster_optimizer
from mltrainer2.clustering.validator import cluster_validator
from mltrainer2.clustering.manager import cluster_manager

# Training
from mltrainer2.training.trainer import model_trainer
from mltrainer2.training.tuner import hyperparameter_tuner
from mltrainer2.training.validator import model_validator

# Metrics
from mltrainer2.metrics.calculator import metrics_calculator
from mltrainer2.metrics.evaluator import metrics_evaluator
from mltrainer2.metrics.reporter import metrics_reporter

# Prediction
from mltrainer2.prediction.predictor import model_predictor
from mltrainer2.prediction.evaluator import prediction_evaluator
from mltrainer2.prediction.explainer import prediction_explainer
from mltrainer2.prediction.reporter import prediction_reporter

# Visualization
from mltrainer2.visualization.plotter import plotter
from mltrainer2.visualization.styler import style_manager
from mltrainer2.visualization.dashboard import dashboard_manager

# UI Components
from mltrainer2.ui.components import ui_components
from mltrainer2.ui.forms import form_manager
from mltrainer2.ui.handlers import event_handler
from mltrainer2.ui.inputs import input_manager
from mltrainer2.ui.layouts import layout_manager
from mltrainer2.ui.callbacks import callback_manager
from mltrainer2.ui.notifications import notification_manager
from mltrainer2.ui.settings import ui_settings
from mltrainer2.ui.validators import ui_validator

__version__ = "2.0.0"
__author__ = "MLTrainer Team"

__all__ = [
    # Core
    'config', 'ConfigManager', 'MLTrainerException', 'state_manager', 'state_monitor',
    
    # Data
    'data_loader', 'preprocessor', 'data_validator', 'data_cleaner', 'exporter',
    
    # Analysis
    'analyzer', 'statistical_analyzer', 'quality_analyzer',
    
    # Features
    'feature_engineer', 'feature_generator', 'feature_selector',
    
    # Clustering
    'clusterer', 'cluster_optimizer', 'cluster_validator', 'cluster_manager',
    
    # Training
    'model_trainer', 'hyperparameter_tuner', 'model_validator',
    
    # Metrics
    'metrics_calculator', 'metrics_evaluator', 'metrics_reporter',
    
    # Prediction
    'model_predictor', 'prediction_evaluator', 'prediction_explainer', 'prediction_reporter',
    
    # Visualization
    'plotter', 'style_manager', 'dashboard_manager',
    
    # UI
    'ui_components', 'form_manager', 'event_handler', 'input_manager', 'layout_manager',
    'callback_manager', 'notification_manager', 'ui_settings', 'ui_validator'
]