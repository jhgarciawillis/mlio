import streamlit as st
import pandas as pd
import numpy as np
import traceback
import logging
from typing import Optional, Any, Dict, Callable

# Core imports
from core.config import config
from core.exceptions import MLTrainerException
from core.state_manager import state_manager
from core.state_monitoring import state_monitor

# Data handling imports
from data.loader import data_loader
from data.cleaner import data_cleaner
from data.preprocessor import preprocessor
from data.validator import data_validator
from data.exporter import exporter

# Analysis imports
from analysis.analyzer import analyzer
from analysis.quality import quality_analyzer
from analysis.statistics import statistical_analyzer

# Feature imports
from features.engineer import feature_engineer
from features.generator import feature_generator
from features.selector import feature_selector

# Clustering imports
from clustering.clusterer import clusterer
from clustering.optimizer import cluster_optimizer
from clustering.validator import cluster_validator

# Training imports
from training.trainer import model_trainer
from training.tuner import hyperparameter_tuner
from training.validator import model_validator

# Metrics imports
from metrics.calculator import metrics_calculator
from metrics.evaluator import metrics_evaluator
from metrics.reporter import performance_reporter

# Prediction imports
from prediction.predictor import model_predictor
from prediction.evaluator import prediction_evaluator
from prediction.explainer import prediction_explainer
from prediction.reporter import prediction_reporter

# Visualization imports
from visualization.plotter import plotter
from visualization.dashboard import dashboard_manager
from visualization.styler import style_manager

# UI imports
from ui.components import ui_components
from ui.forms import form_manager
from ui.handlers import event_handler
from ui.inputs import input_manager
from ui.layouts import layout_manager
from ui.notifications import notification_manager
from ui.settings import ui_settings
from ui.validators import ui_validator

# Utility imports
from utils import (
    monitor_performance, 
    handle_exceptions, 
    log_execution,
    setup_directory, 
    setup_logging, 
    create_timestamp
)
from utils.validators import (
    validate_dataframe, 
    validate_file_path, 
    validate_input
)

def initialize_application():
    """Initialize the entire machine learning training application."""
    try:
        # Set up logging
        setup_logging(
            log_file=config.directories.base_dir / 'app.log',
            level=logging.INFO
        )

        # Initialize state management
        state_monitor.start_monitoring()

        # Configure application layout
        st.set_page_config(
            page_title=config.ui.app_name,
            page_icon=config.ui.app_icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Set up global event handlers
        setup_global_event_handlers()

    except Exception as e:
        st.error(f"Application Initialization Error: {str(e)}")
        logging.critical("Application initialization failed", exc_info=True)

def setup_global_event_handlers():
    """Set up global event handlers and callbacks."""
    # Example of registering a global interaction handler
    event_handler.register_handler(
        'user_interaction', 
        lambda event: notification_manager.show_info(
            f"Interaction detected: {event.get('type', 'Unknown')}"
        )
    )

def create_data_upload_section() -> Optional[pd.DataFrame]:
    """Comprehensive data upload section with validation."""
    st.header("Data Upload")
    
    # File upload with UI components
    uploaded_file = ui_components.create_data_upload_section(
        allowed_extensions=['csv', 'xlsx']
    )
    
    if uploaded_file:
        try:
            # Validate file
            is_valid, error_msg = ui_validator.validate_file(
                uploaded_file, 
                validators=['type', 'size'],
                allowed_types=['csv', 'xlsx'],
                max_size=10 * 1024 * 1024  # 10 MB
            )
            
            if not is_valid:
                notification_manager.show_error(error_msg)
                return None
            
            # Load data
            data = data_loader.load_file(uploaded_file)
            
            # Validate data
            data_validator.validate_dataset(data)
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            return data
        
        except Exception as e:
            notification_manager.show_error(f"Data Processing Error: {str(e)}")
            logging.error("Data upload and processing failed", exc_info=True)
            return None
    
    return None

def run_analysis_workflow(data: pd.DataFrame):
    """Comprehensive data analysis workflow."""
    st.header("Data Analysis")
    
    # Create analysis layout
    layout_manager.create_analysis_layout(data)
    
    # Analysis configuration form
    analysis_config = form_manager.create_form(
        "Analysis Configuration",
        [
            {
                'type': 'select',
                'label': 'Analysis Type',
                'key': 'analysis_type',
                'options': ['Basic', 'Detailed', 'Advanced']
            },
            {
                'type': 'multiselect',
                'label': 'Selected Features',
                'key': 'features',
                'options': list(data.columns)
            }
        ],
        key="analysis_config_form"
    )
    
    if analysis_config:
        try:
            # Perform comprehensive analysis
            analysis_results = analyzer.analyze_dataset(data)
            quality_results = quality_analyzer.analyze_data_quality(data)
            statistical_results = statistical_analyzer.compare_groups(
                data, 
                group_column=analysis_config.get('features', [])[0] if analysis_config.get('features') else None,
                value_column=None
            )
            
            # Display results
            st.subheader("Analysis Results")
            st.json(analysis_results)
            
            # Visualizations
            dashboard_manager.create_dashboard("Analysis Dashboard")
            dashboard_manager.add_plot(
                'data_distribution', 
                'histogram', 
                data, 
                x=analysis_config.get('features', [])[0] if analysis_config.get('features') else data.columns[0]
            )
            dashboard_manager.render_dashboard()
            
            # Export option
            if st.button("Export Analysis Results"):
                exporter.export_data(
                    analysis_results, 
                    config.directories.analysis_outputs / "analysis_results.xlsx"
                )
        
        except Exception as e:
            notification_manager.show_error(f"Analysis Error: {str(e)}")
            logging.error("Analysis workflow failed", exc_info=True)

def run_feature_engineering_workflow(data: pd.DataFrame):
    """Comprehensive feature engineering workflow."""
    st.header("Feature Engineering")
    
    # Feature selection form
    feature_selection = form_manager.create_feature_selection_form(
        list(data.columns)
    )
    
    if feature_selection:
        try:
            # Generate features
            generated_features, gen_metadata = feature_generator.generate_features(
                data, 
                generators=['mathematical', 'datetime']
            )
            
            # Feature engineering
            engineered_features, eng_metadata = feature_engineer.generate_features(
                pd.concat([data, generated_features], axis=1)
            )
            
            # Feature selection
            selected_features, selection_metadata = feature_selector.select_features(
                engineered_features, 
                data[feature_selection['target']],
                n_features=20
            )
            
            # Display results
            st.subheader("Feature Engineering Results")
            st.write(f"Total Generated Features: {generated_features.shape[1]}")
            st.write(f"Selected Features: {selected_features.shape[1]}")
            
            # Visualization
            if 'feature_importance' in selection_metadata:
                st.plotly_chart(selection_metadata['feature_importance']['visualization'])
        
        except Exception as e:
            notification_manager.show_error(f"Feature Engineering Error: {str(e)}")
            logging.error("Feature engineering workflow failed", exc_info=True)

def run_training_workflow(data: pd.DataFrame):
    """Comprehensive model training workflow."""
    st.header("Model Training")
    
    # Model configuration form
    model_config = form_manager.create_model_config_form({
        'model_type': {
            'type': 'select',
            'options': list(config.model.model_classes.keys())
        },
        'train_size': {
            'type': 'slider',
            'min': 0.5,
            'max': 0.9,
            'default': 0.8
        }
    })
    
    if model_config:
        try:
            # Perform model training
            from sklearn.model_selection import train_test_split
            X = data.drop(columns=[model_config['target_column']])
            y = data[model_config['target_column']]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                train_size=model_config.get('train_size', 0.8),
                random_state=config.random_state
            )
            
            # Hyperparameter tuning
            tuned_model, tuning_results = hyperparameter_tuner.tune_hyperparameters(
                config.model.model_classes[model_config['model_type']](),
                X_train,
                y_train,
                method='grid'
            )
            
            # Train model
            trained_model, training_metadata = model_trainer.train_model(
                tuned_model,
                X_train,
                y_train,
                X_test,
                y_test
            )
            
            # Validate model
            validation_results = model_validator.validate_model(
                trained_model,
                X_test,
                y_test
            )
            
            # Display results
            st.subheader("Model Training Results")
            st.json(training_metadata)
            
            # Performance metrics
            metrics = metrics_calculator.calculate_metrics(y_test, trained_model.predict(X_test))
            st.subheader("Performance Metrics")
            st.json(metrics)
        
        except Exception as e:
            notification_manager.show_error(f"Training Error: {str(e)}")
            logging.error("Training workflow failed", exc_info=True)

def run_prediction_workflow(data: pd.DataFrame):
    """Comprehensive prediction workflow."""
    st.header("Prediction")
    
    # Model selection
    model_path = st.file_uploader("Upload Trained Model", type=['.joblib', '.pkl'])
    
    if model_path:
        try:
            # Load model
            model = model_predictor.load_model(str(model_path))
            
            # Prediction configuration
            pred_config = form_manager.create_form(
                "Prediction Configuration",
                [
                    {
                        'type': 'select',
                        'label': 'Prediction Column',
                        'key': 'prediction_column',
                        'options': list(data.columns)
                    }
                ],
                key="prediction_config_form"
            )
            
            if pred_config:
                # Make predictions
                predictions, metadata = model_predictor.predict(
                    model, 
                    data.drop(columns=[pred_config['prediction_column']])
                )
                
                # Evaluate predictions
                evaluation_results = prediction_evaluator.evaluate_predictions(
                    data[pred_config['prediction_column']], 
                    predictions
                )
                
                # Generate explanations
                explanations = prediction_explainer.explain_predictions(
                    model,
                    data.drop(columns=[pred_config['prediction_column']])
                )
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(pd.DataFrame({
                    'Actual': data[pred_config['prediction_column']],
                    'Predicted': predictions
                }))
                
                # Performance metrics
                st.subheader("Prediction Metrics")
                metrics = metrics_calculator.calculate_metrics(
                    data[pred_config['prediction_column']], 
                    predictions
                )
                st.json(metrics)
                
                # Generate report
                report = prediction_reporter.generate_report(
                    predictions, 
                    data[pred_config['prediction_column']]
                )
                
                if st.button("Export Prediction Report"):
                    prediction_reporter.export_report(
                        report['metadata']['report_id'], 
                        format='docx'
                    )
        
        except Exception as e:
            notification_manager.show_error(f"Prediction Error: {str(e)}")
            logging.error("Prediction workflow failed", exc_info=True)

ddef main():
    """Main application entry point."""
    try:
        # Initialize application
        initialize_application()
        
        # Mode selection
        st.sidebar.title("ML Trainer Workflow")
        mode = st.sidebar.radio(
            "Select Workflow Mode", 
            ["Data Upload", "Analysis", "Feature Engineering", "Training", "Prediction"]
        )
        
        # Data upload as a prerequisite for other modes
        data = None
        if mode != "Data Upload":
            data = create_data_upload_section()
        
        # Run appropriate workflow
        workflow_map = {
            "Data Upload": create_data_upload_section,
            "Analysis": lambda: run_analysis_workflow(data) if data is not None else None,
            "Feature Engineering": lambda: run_feature_engineering_workflow(data) if data is not None else None,
            "Training": lambda: run_training_workflow(data) if data is not None else None,
            "Prediction": lambda: run_prediction_workflow(data) if data is not None else None
        }
        
        # Execute selected workflow
        if mode in workflow_map:
            workflow_map[mode]()
    
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}")
        logging.critical("Application failed", exc_info=True)
    
    finally:
        # Ensure monitoring is stopped
        state_monitor.stop_monitoring()

if __name__ == "__main__":
    main()