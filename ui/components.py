import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from datetime import datetime
import json
import plotly.graph_objects as go

from core import config
from core.exceptions import UIError
from core.state_manager import state_manager
from core.state_monitoring import state_monitor
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter, dashboard_manager, style_manager
from metrics import calculator
from clustering import clusterer, cluster_optimizer, cluster_validator

class UIComponents:
    """Handle UI component creation and management with enhanced clustering support."""
    
    def __init__(self):
        self.component_history: List[Dict[str, Any]] = []
        self.active_components: Dict[str, Any] = {}
        self.workflow_states: Dict[str, Any] = {}
        
    @monitor_performance
    @handle_exceptions(UIError)
    def create_sidebar(
        self,
        title: str,
        components_config: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create sidebar with specified components."""
        # Record operation start
        state_monitor.record_operation_start(
            'sidebar_creation',
            'ui_component', 
            {'title': title}
        )
        
        with st.sidebar:
            st.title(title)
            
            component_values = {}
            for comp_config in components_config:
                value = self._create_sidebar_component(comp_config)
                component_values[comp_config['key']] = value
            
            # Record operation completion
            state_monitor.record_operation_end(
                'sidebar_creation',
                'completed'
            )
            
            return component_values
    
    @monitor_performance
    def create_data_upload_section(
        self,
        allowed_extensions: List[str],
        key: str = "data_upload"
    ) -> Optional[Any]:
        """Create enhanced data upload section."""
        st.subheader("Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=allowed_extensions,
            key=key
        )
        
        if uploaded_file is not None:
            preview_container = st.expander("Preview Uploaded Data", expanded=True)
            with preview_container:
                try:
                    # Create file info section
                    st.write("File Information:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Filename: {uploaded_file.name}")
                        st.write(f"Size: {self._format_size(uploaded_file.size)}")
                    with col2:
                        st.write(f"Type: {uploaded_file.type}")
                        
                    # Add to component tracking
                    self._track_component(key, "file_upload", uploaded_file.name)
                    
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")
                    logger.error(f"Upload error: {str(e)}", exc_info=True)
                    return None
                
        return uploaded_file
    
    @monitor_performance
    def create_clustering_section(
        self,
        data: pd.DataFrame,
        available_methods: Optional[Dict[str, Any]] = None,
        key: str = "clustering"
    ) -> Dict[str, Any]:
        """Create enhanced clustering configuration section."""
        st.subheader("Clustering Configuration")
        
        # Get available methods or use defaults
        if available_methods is None:
            available_methods = {
                'kmeans': {'name': 'K-Means', 'params': {'n_clusters': (2, 20)}},
                'dbscan': {'name': 'DBSCAN', 'params': {'eps': (0.1, 2.0), 'min_samples': (2, 20)}},
                'gaussian_mixture': {'name': 'Gaussian Mixture', 'params': {'n_components': (2, 20)}},
                'hierarchical': {'name': 'Hierarchical', 'params': {'n_clusters': (2, 20)}},
                'spectral': {'name': 'Spectral', 'params': {'n_clusters': (2, 20)}}
            }
        
        # Method selection
        method = st.selectbox(
            "Select Clustering Method",
            options=list(available_methods.keys()),
            format_func=lambda x: available_methods[x]['name'],
            key=f"{key}_method"
        )
        
        # Dynamic parameters based on method
        params = available_methods[method].get('params', {})
        param_values = self.create_parameter_inputs(
            params,
            key=f"{key}_params"
        )
        
        # Feature selection
        feature_cols = st.multiselect(
            "Select Features for Clustering",
            options=list(data.columns),
            key=f"{key}_features"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                optimization = st.checkbox(
                    "Enable Parameter Optimization",
                    key=f"{key}_optimize"
                )
                
                scaler = st.selectbox(
                    "Scaling Method",
                    options=["standard", "minmax", "robust", "none"],
                    index=0,
                    key=f"{key}_scaler"
                )
            
            with col2:
                validation = st.checkbox(
                    "Enable Cluster Validation",
                    key=f"{key}_validate"
                )
                
                if optimization:
                    opt_trials = st.slider(
                        "Number of Optimization Trials",
                        min_value=10,
                        max_value=100,
                        value=50,
                        key=f"{key}_trials"
                    )
                else:
                    opt_trials = None
                    
                max_clusters = st.slider(
                    "Maximum Number of Clusters",
                    min_value=2,
                    max_value=50,
                    value=20,
                    key=f"{key}_max_clusters"
                )
        
        # Visualization options
        with st.expander("Visualization Options"):
            visualization_config = {
                'show_cluster_map': st.checkbox("Show Cluster Map", value=True, key=f"{key}_viz_map"),
                'show_silhouette': st.checkbox("Show Silhouette Plot", value=True, key=f"{key}_viz_silhouette"),
                'show_elbow': st.checkbox("Show Elbow Plot", value=True, key=f"{key}_viz_elbow"),
                'show_feature_importance': st.checkbox("Show Feature Importance", value=True, key=f"{key}_viz_importance"),
                'show_statistics': st.checkbox("Show Cluster Statistics", value=True, key=f"{key}_viz_stats"),
                'dimension_reduction': st.selectbox(
                    "Dimension Reduction Method",
                    options=["PCA", "TSNE", "UMAP"],
                    index=0,
                    key=f"{key}_dim_reduction"
                )
            }
        
        config = {
            'method': method,
            'parameters': param_values,
            'features': feature_cols,
            'scaling': scaler,
            'optimization': {
                'enabled': optimization,
                'n_trials': opt_trials,
                'max_clusters': max_clusters
            },
            'validation': validation,
            'visualization': visualization_config
        }
        
        self._track_component(key, "clustering", config)
        
        return config
    
    @monitor_performance
    def create_analysis_section(
        self,
        data: pd.DataFrame,
        analysis_types: Optional[List[str]] = None,
        key: str = "analysis"
    ) -> Dict[str, Any]:
        """Create enhanced analysis configuration section."""
        st.subheader("Analysis Configuration")
        
        if analysis_types is None:
            analysis_types = ["Statistical", "Correlation", "Distribution", "Missing Data", "Outliers", "Feature Importance"]
        
        selected_analyses = st.multiselect(
            "Select Analysis Types",
            options=analysis_types,
            default=["Statistical", "Correlation"],
            key=f"{key}_types"
        )
        
        config = {}
        for analysis_type in selected_analyses:
            with st.expander(f"{analysis_type} Configuration"):
                config[analysis_type] = self._create_analysis_config(
                    analysis_type,
                    data,
                    key=f"{key}_{analysis_type.lower().replace(' ', '_')}"
                )
        
        # Add visualization options
        with st.expander("Visualization Options"):
            viz_config = {
                'plot_width': st.slider("Plot Width", 400, 1200, 800, key=f"{key}_plot_width"),
                'plot_height': st.slider("Plot Height", 300, 800, 500, key=f"{key}_plot_height"),
                'theme': st.selectbox(
                    "Visualization Theme",
                    options=["default", "dark", "light"],
                    index=0,
                    key=f"{key}_theme"
                ),
                'interactive': st.checkbox("Interactive Plots", value=True, key=f"{key}_interactive"),
                'export_format': st.selectbox(
                    "Export Format",
                    options=["html", "png", "pdf", "json"],
                    index=0,
                    key=f"{key}_export_format"
                )
            }
            config['visualization'] = viz_config
        
        self._track_component(key, "analysis", config)
        
        return config
    
    @monitor_performance
    def create_model_selection_section(
        self,
        available_models: Optional[Dict[str, Any]] = None,
        key: str = "model_selection"
    ) -> Dict[str, Any]:
        """Create enhanced model selection section."""
        st.subheader("Model Selection")
        
        if available_models is None:
            available_models = {
                "Regression": {
                    "Linear Regression": {"name": "Linear Regression", "params": {"fit_intercept": [True, False]}},
                    "Random Forest": {"name": "Random Forest", "params": {"n_estimators": (10, 500), "max_depth": (2, 30)}},
                    "XGBoost": {"name": "XGBoost", "params": {"n_estimators": (10, 500), "learning_rate": (0.01, 0.3)}},
                    "LightGBM": {"name": "LightGBM", "params": {"n_estimators": (10, 500), "learning_rate": (0.01, 0.3)}}
                },
                "Classification": {
                    "Logistic Regression": {"name": "Logistic Regression", "params": {"C": (0.01, 10.0)}},
                    "Random Forest": {"name": "Random Forest", "params": {"n_estimators": (10, 500), "max_depth": (2, 30)}},
                    "XGBoost": {"name": "XGBoost", "params": {"n_estimators": (10, 500), "learning_rate": (0.01, 0.3)}},
                    "LightGBM": {"name": "LightGBM", "params": {"n_estimators": (10, 500), "learning_rate": (0.01, 0.3)}}
                }
            }
        
        selected_models = {}
        
        # Create tabs for different model categories
        categories = list(available_models.keys())
        tabs = st.tabs(categories)
        
        for tab, category in zip(tabs, categories):
            with tab:
                models = available_models[category]
                selected = st.multiselect(
                    f"Select {category} models",
                    options=list(models.keys()),
                    key=f"{key}_{category.lower()}"
                )
                selected_models[category] = selected
                
                # Model specific configurations
                if selected:
                    selected_models[f"{category}_config"] = {}
                    for model in selected:
                        with st.expander(f"{model} Configuration"):
                            selected_models[f"{category}_config"][model] = (
                                self._create_model_config(models[model], key=f"{key}_{model.lower().replace(' ', '_')}")
                            )
        
        # Cluster-aware training section
        with st.expander("Cluster-Aware Training"):
            cluster_aware = st.checkbox("Enable Cluster-Aware Training", key=f"{key}_cluster_aware")
            
            if cluster_aware:
                cluster_method = st.selectbox(
                    "Clustering Method",
                    options=["kmeans", "dbscan", "gaussian_mixture", "hierarchical", "spectral"],
                    key=f"{key}_cluster_method"
                )
                
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=20,
                    value=5,
                    key=f"{key}_n_clusters"
                )
                
                cluster_strategy = st.selectbox(
                    "Cluster Training Strategy",
                    options=["separate_models", "feature_injection", "hierarchical_models"],
                    key=f"{key}_cluster_strategy"
                )
                
                selected_models["cluster_aware"] = {
                    "enabled": cluster_aware,
                    "method": cluster_method,
                    "n_clusters": n_clusters,
                    "strategy": cluster_strategy
                }
        
        # Hyperparameter tuning section
        with st.expander("Hyperparameter Tuning"):
            tuning_enabled = st.checkbox("Enable Hyperparameter Tuning", key=f"{key}_tuning")
            
            if tuning_enabled:
                tuning_method = st.selectbox(
                    "Tuning Method",
                    options=["grid", "random", "optuna"],
                    key=f"{key}_tuning_method"
                )
                
                n_trials = st.slider(
                    "Number of Trials",
                    min_value=10,
                    max_value=200,
                    value=50,
                    key=f"{key}_n_trials"
                )
                
                cv_folds = st.slider(
                    "Cross-Validation Folds",
                    min_value=2,
                    max_value=10,
                    value=5,
                    key=f"{key}_cv_folds"
                )
                
                selected_models["tuning"] = {
                    "enabled": tuning_enabled,
                    "method": tuning_method,
                    "n_trials": n_trials,
                    "cv_folds": cv_folds
                }
        
        self._track_component(key, "model_selection", selected_models)
        
        return selected_models
    
    @monitor_performance
    def create_evaluation_section(
        self,
        metrics: Dict[str, Any],
        visualizations: Dict[str, Any],
        cluster_info: Optional[Dict[str, Any]] = None,
        key: str = "evaluation"
    ) -> None:
        """Create enhanced evaluation results section."""
        st.subheader("Evaluation Results")
        
        # Tabs for different views
        if cluster_info is not None:
            tabs = st.tabs(["Overall Metrics", "Cluster Metrics", "Visualizations", "Explanations", "Export"])
        else:
            tabs = st.tabs(["Overall Metrics", "Visualizations", "Explanations", "Export"])
        
        # Overall metrics tab
        with tabs[0]:
            self._display_metrics(metrics)
            
            # Learning curve
            if 'learning_curve' in visualizations:
                st.subheader("Learning Curve")
                st.plotly_chart(visualizations['learning_curve'], use_container_width=True)
        
        # Cluster metrics tab (if cluster info available)
        tab_index = 1
        if cluster_info is not None:
            with tabs[tab_index]:
                self._display_cluster_metrics(metrics, cluster_info)
            tab_index += 1
        
        # Visualizations tab
        with tabs[tab_index]:
            self.create_visualization_section(
                visualizations,
                key=f"{key}_viz"
            )
            tab_index += 1
        
        # Explanations tab
        with tabs[tab_index]:
            self._display_explanations(visualizations.get('explanations', {}))
            tab_index += 1
        
        # Export tab
        with tabs[tab_index]:
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    options=["HTML", "PDF", "JSON", "CSV", "Excel"],
                    key=f"{key}_export_format"
                )
                
            with col2:
                if st.button("Export Results", key=f"{key}_export"):
                    self._export_evaluation(metrics, visualizations, export_format, key)
        
        self._track_component(key, "evaluation", {'metrics': metrics.keys()})
    
    @monitor_performance
    def create_workflow_section(
        self,
        workflow_type: str,
        available_steps: Optional[List[str]] = None,
        key: str = "workflow"
    ) -> Dict[str, Any]:
        """Create enhanced workflow configuration section."""
        st.subheader(f"{workflow_type} Workflow")
        
        if available_steps is None:
            if workflow_type == "Training":
                available_steps = ["Data Preparation", "Feature Engineering", "Feature Selection", "Model Training", "Evaluation"]
            elif workflow_type == "Prediction":
                available_steps = ["Data Preparation", "Prediction", "Evaluation", "Explanation"]
            elif workflow_type == "Clustering":
                available_steps = ["Data Preparation", "Clustering", "Validation", "Visualization"]
            else:
                available_steps = ["Data Preparation", "Analysis", "Visualization"]
        
        selected_steps = st.multiselect(
            "Select Workflow Steps",
            options=available_steps,
            default=available_steps,
            key=f"{key}_steps"
        )
        
        step_config = {}
        for step in selected_steps:
            with st.expander(f"{step} Configuration"):
                step_config[step] = self._create_step_config(
                    step,
                    workflow_type,
                    key=f"{key}_{step.lower().replace(' ', '_')}"
                )
        
        # Advanced configuration
        with st.expander("Advanced Workflow Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                parallel = st.checkbox("Enable Parallel Processing", key=f"{key}_parallel")
                checkpoint = st.checkbox("Enable Checkpointing", key=f"{key}_checkpoint")
            
            with col2:
                timeout = st.number_input(
                    "Timeout (seconds)",
                    min_value=0,
                    value=3600,
                    key=f"{key}_timeout"
                )
                
                monitoring = st.checkbox("Enable Performance Monitoring", value=True, key=f"{key}_monitoring")
        
        workflow_config = {
            'workflow_type': workflow_type,
            'selected_steps': selected_steps,
            'step_config': step_config,
            'advanced': {
                'parallel': parallel,
                'checkpoint': checkpoint,
                'timeout': timeout,
                'monitoring': monitoring
            }
        }
        
        self._track_component(key, "workflow", workflow_config)
        self.workflow_states[key] = workflow_config
        
        return workflow_config
    
    @monitor_performance
    def create_monitoring_section(
        self,
        monitoring_data: Dict[str, Any],
        key: str = "monitoring"
    ) -> None:
        """Create enhanced monitoring display section."""
        st.subheader("Performance Monitoring")
        
        tabs = st.tabs(["Metrics", "Resources", "Operations", "Errors"])
        
        with tabs[0]:
            self._display_performance_metrics(
                monitoring_data.get('metrics', {})
            )
        
        with tabs[1]:
            self._display_resource_usage(
                monitoring_data.get('resources', {})
            )
            
        with tabs[2]:
            self._display_operations(
                monitoring_data.get('operations', {})
            )
            
        with tabs[3]:
            self._display_errors(
                monitoring_data.get('errors', {})
            )
    
    @monitor_performance
    def create_parameter_inputs(
        self,
        parameters: Dict[str, Dict[str, Any]],
        key: str = "parameters"
    ) -> Dict[str, Any]:
        """Create enhanced parameter input section."""
        parameter_values = {}
        
        for param_name, param_config in parameters.items():
            param_type = param_config.get('type', 'number')
            
            if isinstance(param_config, tuple):
                # Handle range tuples for numeric parameters
                if isinstance(param_config[0], int):
                    param_type = 'int_range'
                    min_val, max_val = param_config
                else:
                    param_type = 'float_range'
                    min_val, max_val = param_config
            
            if param_type == 'number':
                value = st.number_input(
                    param_name,
                    min_value=param_config.get('min'),
                    max_value=param_config.get('max'),
                    value=param_config.get('default', 0),
                    step=param_config.get('step', 1),
                    key=f"{key}_{param_name}"
                )
            elif param_type == 'int_range':
                value = st.slider(
                    param_name,
                    min_value=min_val,
                    max_value=max_val,
                    value=min_val,
                    step=1,
                    key=f"{key}_{param_name}"
                )
            elif param_type == 'float_range':
                value = st.slider(
                    param_name,
                    min_value=min_val,
                    max_value=max_val,
                    value=min_val,
                    step=(max_val - min_val) / 100,
                    key=f"{key}_{param_name}"
                )
            elif param_type == 'select':
                value = st.selectbox(
                    param_name,
                    options=param_config['options'],
                    index=param_config.get('default_index', 0),
                    key=f"{key}_{param_name}"
                )
            elif param_type == 'multiselect':
                value = st.multiselect(
                    param_name,
                    options=param_config['options'],
                    default=param_config.get('default', []),
                    key=f"{key}_{param_name}"
                )
            elif param_type == 'checkbox':
                value = st.checkbox(
                    param_name,
                    value=param_config.get('default', False),
                    key=f"{key}_{param_name}"
                )
            else:
                value = None
                st.warning(f"Unsupported parameter type: {param_type}")
            
            parameter_values[param_name] = value
        
        self._track_component(key, "parameters", parameter_values)
        
        return parameter_values
    
    @monitor_performance
    def create_results_section(
        self,
        results: Dict[str, Any],
        key: str = "results"
    ) -> None:
        """Create enhanced results display section."""
        st.subheader("Results")
        
        # Create tabs for different result types
        tabs = st.tabs(list(results.keys()))
        
        for tab, (result_type, result_data) in zip(tabs, results.items()):
            with tab:
                if isinstance(result_data, pd.DataFrame):
                    st.dataframe(result_data, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Rows: {len(result_data)}")
                    with col2:
                        st.write(f"Columns: {len(result_data.columns)}")
                        
                elif isinstance(result_data, (dict, list)):
                    st.json(result_data)
                elif isinstance(result_data, (go.Figure, dict)) and 'data' in result_data:
                    st.plotly_chart(result_data, use_container_width=True)
                else:
                    st.write(result_data)
        
        # Add download options
        st.subheader("Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download as CSV", key=f"{key}_csv"):
                self._create_download_button(results, key, 'csv')
        
        with col2:
            if st.button("Download as JSON", key=f"{key}_json"):
                self._create_download_button(results, key, 'json')
                
        with col3:
            if st.button("Download as Excel", key=f"{key}_excel"):
                self._create_download_button(results, key, 'excel')
    
    @monitor_performance
    def create_visualization_section(
        self,
        figures: Dict[str, Any],
        key: str = "visualizations"
    ) -> None:
        """Create enhanced visualization section."""
        st.subheader("Visualizations")
        
        # Get figure names and create tabs
        fig_names = list(figures.keys())
        if not fig_names:
            st.info("No visualizations available")
            return
            
        # Group visualizations into categories if many
        if len(fig_names) > 5:
            # Try to find categories in names
            categories = set()
            for name in fig_names:
                parts = name.split('_')
                if len(parts) > 1:
                    categories.add(parts[0])
            
            if len(categories) > 1:
                # Use categories for tabs
                category_tabs = st.tabs(list(categories))
                for cat_tab, category in zip(category_tabs, categories):
                    with cat_tab:
                        cat_figs = {k: v for k, v in figures.items() if k.startswith(category)}
                        for fig_name, fig in cat_figs.items():
                            with st.expander(f"{fig_name}"):
                                st.plotly_chart(fig, use_container_width=True)
                                self._add_visualization_download(fig, fig_name, key)
                return
        
        # Create tabs for each visualization
        tabs = st.tabs(fig_names)
        
        for tab, fig_name in zip(tabs, fig_names):
            with tab:
                fig = figures[fig_name]
                st.plotly_chart(fig, use_container_width=True)
                self._add_visualization_download(fig, fig_name, key)
    
    def _add_visualization_download(
        self,
        fig: Any,
        fig_name: str,
        key: str
    ) -> None:
        """Add download options for visualization."""
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Download {fig_name} as PNG", key=f"{key}_{fig_name}_png"):
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    "Download PNG",
                    img_bytes,
                    f"{fig_name}.png",
                    "image/png"
                )
        
        with col2:
            if st.button(f"Download {fig_name} as HTML", key=f"{key}_{fig_name}_html"):
                html = fig.to_html(include_plotlyjs="cdn")
                st.download_button(
                    "Download HTML",
                    html,
                    f"{fig_name}.html",
                    "text/html"
                )
    
    def _create_sidebar_component(
        self,
        config: Dict[str, Any]
    ) -> Any:
        """Create individual sidebar component."""
        component_type = config['type']
        
        if component_type == 'selectbox':
            return st.sidebar.selectbox(
                config['label'],
                options=config['options'],
                index=config.get('default_index', 0),
                key=config['key']
            )
        elif component_type == 'multiselect':
            return st.sidebar.multiselect(
                config['label'],
                options=config['options'],
                default=config.get('default', []),
                key=config['key']
            )
        elif component_type == 'slider':
            return st.sidebar.slider(
                config['label'],
                min_value=config['min'],
                max_value=config['max'],
                value=config.get('default'),
                step=config.get('step', 1),
                key=config['key']
            )
        elif component_type == 'checkbox':
            return st.sidebar.checkbox(
                config['label'],
                value=config.get('default', False),
                key=config['key']
            )
        elif component_type == 'radio':
            return st.sidebar.radio(
                config['label'],
                options=config['options'],
                index=config.get('default_index', 0),
                key=config['key']
            )
        elif component_type == 'button':
            return st.sidebar.button(
                config['label'],
                key=config['key']
            )
        elif component_type == 'divider':
            st.sidebar.divider()
            return None
        elif component_type == 'text':
            st.sidebar.text(config['content'])
            return None
        elif component_type == 'markdown':
            st.sidebar.markdown(config['content'])
            return None
        else:
            st.sidebar.warning(f"Unknown component type: {component_type}")
            return None
    
    def _create_analysis_config(
        self,
        analysis_type: str,
        data: pd.DataFrame,
        key: str
    ) -> Dict[str, Any]:
        """Create configuration for specific analysis type."""
        config = {}
        
        if analysis_type == "Statistical":
            config['descriptive'] = st.checkbox("Descriptive Statistics", value=True, key=f"{key}_descriptive")
            config['inferential'] = st.checkbox("Inferential Statistics", value=False, key=f"{key}_inferential")
            
            if config['inferential']:
                config['tests'] = st.multiselect(
                    "Select Statistical Tests",
                    options=['t-test', 'anova', 'chi-square', 'correlation', 'regression'],
                    key=f"{key}_tests"
                )
        
        elif analysis_type == "Correlation":
            config['method'] = st.selectbox(
                "Correlation Method",
                options=['pearson', 'spearman', 'kendall'],
                key=f"{key}_method"
            )
            config['threshold'] = st.slider(
                "Correlation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"{key}_threshold"
            )
            config['cluster_correlation'] = st.checkbox(
                "Analyze Cluster-specific Correlations",
                value=False,
                key=f"{key}_cluster"
            )
        
        elif analysis_type == "Distribution":
            config['bins'] = st.slider(
                "Number of Bins",
                min_value=5,
                max_value=100,
                value=20,
                key=f"{key}_bins"
            )
            config['kde'] = st.checkbox(
                "Show KDE",
                value=True,
                key=f"{key}_kde"
            )
            config['normality_test'] = st.checkbox(
                "Perform Normality Tests",
                value=True,
                key=f"{key}_normality"
            )
        
        elif analysis_type == "Missing Data":
            config['threshold'] = st.slider(
                "Missing Data Threshold (%)",
                min_value=0,
                max_value=100,
                value=50,
                key=f"{key}_threshold"
            )
            config['imputation'] = st.checkbox(
                "Show Imputation Recommendations",
                value=True,
                key=f"{key}_imputation"
            )
            config['pattern_analysis'] = st.checkbox(
                "Analyze Missing Patterns",
                value=True,
                key=f"{key}_patterns"
            )
        
        elif analysis_type == "Outliers":
            config['method'] = st.selectbox(
                "Outlier Detection Method",
                options=['zscore', 'iqr', 'isolation_forest', 'local_outlier_factor'],
                key=f"{key}_method"
            )
            
            if config['method'] == 'zscore':
                config['threshold'] = st.slider(
                    "Z-Score Threshold",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                    key=f"{key}_threshold"
                )
            elif config['method'] == 'iqr':
                config['multiplier'] = st.slider(
                    "IQR Multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    key=f"{key}_multiplier"
                )
        
        elif analysis_type == "Feature Importance":
            config['method'] = st.selectbox(
                "Importance Method",
                options=['mutual_info', 'f_score', 'random_forest', 'xgboost', 'permutation'],
                key=f"{key}_method"
            )
            config['n_features'] = st.slider(
                "Number of Features",
                min_value=1,
                max_value=min(20, len(data.columns)),
                value=10,
                key=f"{key}_n_features"
            )
        
        return config
    
    def _create_model_config(
        self,
        model_info: Dict[str, Any],
        key: str
    ) -> Dict[str, Any]:
        """Create configuration for specific model."""
        config = {}
        
        if 'parameters' in model_info:
            config['parameters'] = self.create_parameter_inputs(
                model_info['parameters'],
                key=f"{key}_params"
            )
        
        if 'features' in model_info:
            config['features'] = st.multiselect(
                "Select Features",
                options=model_info['features'],
                key=f"{key}_features"
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                config['early_stopping'] = st.checkbox(
                    "Early Stopping",
                    value=True,
                    key=f"{key}_early_stopping"
                )
                
                config['feature_importance'] = st.checkbox(
                    "Calculate Feature Importance",
                    value=True,
                    key=f"{key}_importance"
                )
            
            with col2:
                config['validation_size'] = st.slider(
                    "Validation Size",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    key=f"{key}_val_size"
                )
                
                config['random_state'] = st.number_input(
                    "Random State",
                    min_value=0,
                    value=42,
                    key=f"{key}_random_state"
                )
        
        return config
    
    def _create_step_config(
        self,
        step: str,
        workflow_type: str,
        key: str
    ) -> Dict[str, Any]:
        """Create configuration for specific workflow step."""
        config = {}
        
        if step == "Data Preparation":
            config['scaling'] = st.selectbox(
                "Scaling Method",
                options=['standard', 'minmax', 'robust', 'none'],
                key=f"{key}_scaling"
            )
            
            config['handle_missing'] = st.selectbox(
                "Handle Missing Values",
                options=['drop', 'mean', 'median', 'most_frequent', 'knn'],
                key=f"{key}_missing"
            )
            
            config['handle_outliers'] = st.checkbox(
                "Handle Outliers",
                key=f"{key}_outliers"
            )
            
            if config['handle_outliers']:
                config['outlier_method'] = st.selectbox(
                    "Outlier Method",
                    options=['zscore', 'iqr', 'isolation_forest'],
                    key=f"{key}_outlier_method"
                )
                
            config['categorical_encoding'] = st.selectbox(
                "Categorical Encoding",
                options=['onehot', 'label', 'target', 'frequency', 'none'],
                key=f"{key}_encoding"
            )
            
        elif step == "Feature Engineering":
            config['polynomial'] = st.checkbox(
                "Polynomial Features",
                key=f"{key}_polynomial"
            )
            
            if config['polynomial']:
                config['poly_degree'] = st.slider(
                    "Polynomial Degree",
                    min_value=2,
                    max_value=5,
                    value=2,
                    key=f"{key}_poly_degree"
                )
            
            config['interactions'] = st.checkbox(
                "Create Interaction Features",
                key=f"{key}_interactions"
            )
            
            config['transformations'] = st.multiselect(
                "Apply Transformations",
                options=['log', 'sqrt', 'exp', 'power', 'box-cox', 'yeo-johnson'],
                key=f"{key}_transformations"
            )
            
            config['date_features'] = st.checkbox(
                "Extract DateTime Features",
                key=f"{key}_date_features"
            )
            
            config['text_features'] = st.checkbox(
                "Extract Text Features",
                key=f"{key}_text_features"
            )
            
        elif step == "Feature Selection":
            config['method'] = st.selectbox(
                "Feature Selection Method",
                options=['mutual_info', 'f_score', 'rfe', 'lasso', 'tree', 'permutation'],
                key=f"{key}_method"
            )
            
            config['n_features'] = st.slider(
                "Number of Features to Select",
                min_value=1,
                max_value=50,
                value=10,
                key=f"{key}_n_features"
            )
            
            config['cross_validation'] = st.checkbox(
                "Use Cross-Validation",
                value=True,
                key=f"{key}_cv"
            )
            
            if config['cross_validation']:
                config['cv_folds'] = st.slider(
                    "CV Folds",
                    min_value=2,
                    max_value=10,
                    value=5,
                    key=f"{key}_cv_folds"
                )
            
        elif step == "Model Training":
            config['cv_folds'] = st.slider(
                "Cross-validation Folds",
                min_value=2,
                max_value=10,
                value=5,
                key=f"{key}_cv"
            )
            
            config['optimize_hyperparameters'] = st.checkbox(
                "Optimize Hyperparameters",
                key=f"{key}_optimize"
            )
            
            if config['optimize_hyperparameters']:
                config['optimization_method'] = st.selectbox(
                    "Optimization Method",
                    options=['grid', 'random', 'optuna'],
                    key=f"{key}_opt_method"
                )
                
                config['n_trials'] = st.slider(
                    "Number of Trials",
                    min_value=10,
                    max_value=200,
                    value=50,
                    key=f"{key}_n_trials"
                )
            
            config['cluster_aware'] = st.checkbox(
                "Cluster-Aware Training",
                key=f"{key}_cluster_aware"
            )
            
            if config['cluster_aware']:
                config['cluster_method'] = st.selectbox(
                    "Clustering Method",
                    options=['kmeans', 'dbscan', 'gaussian_mixture'],
                    key=f"{key}_cluster_method"
                )
                
                config['n_clusters'] = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=20,
                    value=5,
                    key=f"{key}_n_clusters"
                )
            
        elif step == "Evaluation":
            config['metrics'] = st.multiselect(
                "Select Evaluation Metrics",
                options=['mse', 'rmse', 'mae', 'r2', 'mape', 'accuracy', 'precision', 'recall', 'f1'],
                default=['mse', 'rmse', 'r2'] if workflow_type == "Training" else ['rmse', 'r2'],
                key=f"{key}_metrics"
            )
            
            config['visualization'] = st.checkbox(
                "Create Visualizations",
                value=True,
                key=f"{key}_viz"
            )
            
            config['cluster_evaluation'] = st.checkbox(
                "Evaluate per Cluster",
                value=True if workflow_type == "Clustering" else False,
                key=f"{key}_cluster_eval"
            )
            
        elif step == "Clustering":
            config['method'] = st.selectbox(
                "Clustering Method",
                options=['kmeans', 'dbscan', 'gaussian_mixture', 'hierarchical', 'spectral'],
                key=f"{key}_method"
            )
            
            if config['method'] == 'kmeans' or config['method'] == 'gaussian_mixture':
                config['n_clusters'] = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=20,
                    value=5,
                    key=f"{key}_n_clusters"
                )
            elif config['method'] == 'dbscan':
                config['eps'] = st.slider(
                    "Epsilon",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    key=f"{key}_eps"
                )
                config['min_samples'] = st.slider(
                    "Min Samples",
                    min_value=2,
                    max_value=20,
                    value=5,
                    key=f"{key}_min_samples"
                )
            
            config['preprocessing'] = st.selectbox(
                "Preprocessing",
                options=['standard', 'minmax', 'robust', 'none'],
                key=f"{key}_preprocessing"
            )
            
            config['optimize'] = st.checkbox(
                "Optimize Parameters",
                value=True,
                key=f"{key}_optimize"
            )
            
        elif step == "Prediction":
            config['batch_size'] = st.slider(
                "Batch Size",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key=f"{key}_batch_size"
            )
            
            config['use_clusters'] = st.checkbox(
                "Use Cluster-Specific Models",
                key=f"{key}_use_clusters"
            )
            
            config['confidence_intervals'] = st.checkbox(
                "Calculate Confidence Intervals",
                value=True,
                key=f"{key}_confidence"
            )
            
        elif step == "Explanation":
            config['methods'] = st.multiselect(
                "Explanation Methods",
                options=['shap', 'lime', 'pdp', 'feature_importance', 'ice'],
                default=['shap', 'pdp'],
                key=f"{key}_methods"
            )
            
            if 'shap' in config['methods']:
                config['shap_sample_size'] = st.slider(
                    "SHAP Sample Size",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    key=f"{key}_shap_samples"
                )
            
            config['global_explanations'] = st.checkbox(
                "Include Global Explanations",
                value=True,
                key=f"{key}_global"
            )
            
            config['local_explanations'] = st.checkbox(
                "Include Local Explanations",
                value=True,
                key=f"{key}_local"
            )
            
            if config['local_explanations']:
                config['n_samples'] = st.slider(
                    "Number of Samples to Explain",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key=f"{key}_n_samples"
                )
            
        elif step == "Visualization":
            config['plot_types'] = st.multiselect(
                "Plot Types",
                options=['scatter', 'line', 'bar', 'histogram', 'box', 'violin', 'heatmap', '3d'],
                default=['scatter', 'histogram'],
                key=f"{key}_plot_types"
            )
            
            config['interactive'] = st.checkbox(
                "Interactive Plots",
                value=True,
                key=f"{key}_interactive"
            )
            
            config['theme'] = st.selectbox(
                "Plot Theme",
                options=['default', 'dark', 'light'],
                key=f"{key}_theme"
            )
            
            config['export_formats'] = st.multiselect(
                "Export Formats",
                options=['html', 'png', 'pdf', 'svg'],
                default=['html', 'png'],
                key=f"{key}_export_formats"
            )
        
        return config
    
    def _display_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Display metrics in organized format."""
        # First display main metrics in a clean layout
        st.subheader("Key Metrics")
        
        main_metrics = ["r2", "rmse", "mae", "mse", "accuracy", "f1"]
        cols = st.columns(3)
        displayed_count = 0
        
        for metric_name in main_metrics:
            if metric_name in metrics:
                col_idx = displayed_count % 3
                with cols[col_idx]:
                    st.metric(
                        metric_name.upper(),
                        f"{metrics[metric_name]:.4f}"
                    )
                displayed_count += 1
        
        # Then show all metrics in an organized way
        st.subheader("All Metrics")
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                st.write(f"**{metric_name.replace('_', ' ').title()}**")
                for sub_name, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        st.write(f"- {sub_name.replace('_', ' ').title()}: {sub_value:.4f}")
                    else:
                        st.write(f"- {sub_name.replace('_', ' ').title()}: {sub_value}")
            elif isinstance(value, (int, float)):
                st.write(f"**{metric_name.replace('_', ' ').title()}**: {value:.4f}")
            else:
                st.write(f"**{metric_name.replace('_', ' ').title()}**: {value}")
    
    def _display_cluster_metrics(
        self,
        metrics: Dict[str, Any],
        cluster_info: Dict[str, Any]
    ) -> None:
        """Display cluster-specific metrics."""
        st.subheader("Cluster Performance")
        
        # Get cluster metrics
        cluster_metrics = {}
        for key, value in metrics.items():
            if key.startswith('cluster_'):
                cluster_metrics[key] = value
        
        if not cluster_metrics:
            st.info("No cluster-specific metrics available")
            return
        
        # Create comparison table
        cluster_data = []
        for cluster_name, cluster_metric in cluster_metrics.items():
            cluster_id = cluster_name.split('_')[1]
            row = {'Cluster': cluster_id}
            
            # Add metrics
            for metric_name, metric_value in cluster_metric.items():
                if isinstance(metric_value, (int, float)):
                    row[metric_name] = metric_value
            
            # Add cluster size if available
            if 'sizes' in cluster_info and cluster_id in cluster_info['sizes']:
                row['Size'] = cluster_info['sizes'][cluster_id]
            
            cluster_data.append(row)
        
        if cluster_data:
            cluster_df = pd.DataFrame(cluster_data)
            st.dataframe(cluster_df, use_container_width=True)
            
            # Add visualizations
            metrics_to_plot = [col for col in cluster_df.columns if col not in ['Cluster', 'Size']]
            if metrics_to_plot:
                selected_metric = st.selectbox(
                    "Select Metric to Visualize",
                    options=metrics_to_plot
                )
                
                fig = plotter.create_plot(
                    'bar',
                    data=cluster_df,
                    x='Cluster',
                    y=selected_metric,
                    title=f'{selected_metric} by Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_explanations(
        self,
        explanations: Dict[str, Any]
    ) -> None:
        """Display model explanations."""
        if not explanations:
            st.info("No explanation data available")
            return
        
        # Create tabs for different explanation types
        explanation_types = list(explanations.keys())
        if not explanation_types:
            return
            
        tabs = st.tabs(explanation_types)
        
        for tab, exp_type in zip(tabs, explanation_types):
            with tab:
                exp_data = explanations[exp_type]
                
                if exp_type == 'shap':
                    self._display_shap_explanations(exp_data)
                elif exp_type == 'lime':
                    self._display_lime_explanations(exp_data)
                elif exp_type == 'pdp':
                    self._display_pdp_explanations(exp_data)
                elif exp_type == 'feature_importance':
                    self._display_feature_importance(exp_data)
                else:
                    st.json(exp_data)
    
    def _display_shap_explanations(
        self,
        shap_data: Dict[str, Any]
    ) -> None:
        """Display SHAP explanations."""
        st.subheader("SHAP Feature Importance")
        
        if 'visualizations' in shap_data and 'summary_plot' in shap_data['visualizations']:
            st.plotly_chart(shap_data['visualizations']['summary_plot'], use_container_width=True)
        
        # Show feature importance ranking
        if 'summary' in shap_data and 'feature_importance_ranking' in shap_data['summary']:
            importance_df = pd.DataFrame(shap_data['summary']['feature_importance_ranking'])
            st.dataframe(importance_df, use_container_width=True)
        
        # Individual feature dependence plots
        st.subheader("Feature Dependence")
        if 'visualizations' in shap_data:
            dependence_plots = {
                k: v for k, v in shap_data['visualizations'].items() 
                if k.startswith('dependence_')
            }
            
            if dependence_plots:
                feature_names = [k.replace('dependence_', '') for k in dependence_plots.keys()]
                selected_feature = st.selectbox(
                    "Select Feature",
                    options=feature_names
                )
                
                plot_key = f"dependence_{selected_feature}"
                if plot_key in shap_data['visualizations']:
                    st.plotly_chart(shap_data['visualizations'][plot_key], use_container_width=True)
    
    def _display_lime_explanations(
        self,
        lime_data: Dict[str, Any]
    ) -> None:
        """Display LIME explanations."""
        st.subheader("LIME Explanations")
        
        if 'explanations' in lime_data:
            # Let user select sample
            sample_indices = [exp['sample_idx'] for exp in lime_data['explanations']]
            selected_sample = st.selectbox(
                "Select Sample",
                options=sample_indices
            )
            
            # Find the explanation for selected sample
            selected_exp = None
            for exp in lime_data['explanations']:
                if exp['sample_idx'] == selected_sample:
                    selected_exp = exp
                    break
            
            if selected_exp:
                st.write(f"Local Prediction: {selected_exp['local_prediction']:.4f}")
                st.write(f"Explanation Score: {selected_exp['score']:.4f}")
                
                # Create feature contribution chart
                features_df = pd.DataFrame(
                    selected_exp['features'],
                    columns=['Feature', 'Contribution']
                )
                
                fig = plotter.create_plot(
                    'bar',
                    data=features_df,
                    x='Contribution',
                    y='Feature',
                    title='Feature Contributions',
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Show summary if available
        if 'summary' in lime_data:
            st.subheader("LIME Summary")
            
            if 'average_feature_importance' in lime_data['summary']:
                importance_df = pd.DataFrame({
                    'Feature': list(lime_data['summary']['average_feature_importance'].keys()),
                    'Importance': list(lime_data['summary']['average_feature_importance'].values())
                }).sort_values('Importance', ascending=False)
                
                fig = plotter.create_plot(
                    'bar',
                    data=importance_df,
                    x='Feature',
                    y='Importance',
                    title='Average Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_pdp_explanations(
        self,
        pdp_data: Dict[str, Any]
    ) -> None:
        """Display PDP explanations."""
        st.subheader("Partial Dependence Plots")
        
        # Single feature PDPs
        if 'single_feature_pdp' in pdp_data:
            features = list(pdp_data['single_feature_pdp'].keys())
            selected_feature = st.selectbox(
                "Select Feature",
                options=features
            )
            
            pdp_result = pdp_data['single_feature_pdp'][selected_feature]
            pdp_df = pd.DataFrame({
                'Feature Value': pdp_result['feature_values'],
                'PDP Value': pdp_result['values']
            })
            
            fig = plotter.create_plot(
                'line',
                data=pdp_df,
                x='Feature Value',
                y='PDP Value',
                title=f'Partial Dependence Plot for {selected_feature}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Interaction PDPs
        if 'interaction_pdp' in pdp_data and pdp_data['interaction_pdp']:
            st.subheader("Feature Interactions")
            
            interactions = list(pdp_data['interaction_pdp'].keys())
            selected_interaction = st.selectbox(
                "Select Interaction",
                options=interactions
            )
            
            if selected_interaction in pdp_data['interaction_pdp']:
                st.write(f"Interaction: {selected_interaction.replace('_', ' x ')}")
                st.json(pdp_data['interaction_pdp'][selected_interaction])
    
    def _display_feature_importance(
        self,
        importance_data: Dict[str, Any]
    ) -> None:
        """Display feature importance."""
        st.subheader("Feature Importance")
        
        # Permutation importance
        if 'permutation_importance' in importance_data:
            st.write("Permutation Importance")
            
            if 'feature_ranking' in importance_data['permutation_importance']:
                importance_df = pd.DataFrame(
                    importance_data['permutation_importance']['feature_ranking']
                ).sort_values('importance', ascending=False)
                
                fig = plotter.create_plot(
                    'bar',
                    data=importance_df,
                    x='feature',
                    y='importance',
                    title='Permutation Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Model-specific importance
        if 'model_importance' in importance_data:
            st.write("Model Feature Importance")
            
            if 'feature_ranking' in importance_data['model_importance']:
                importance_df = pd.DataFrame(
                    importance_data['model_importance']['feature_ranking']
                ).sort_values('importance', ascending=False)
                
                fig = plotter.create_plot(
                    'bar',
                    data=importance_df,
                    x='feature',
                    y='importance',
                    title='Model Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_performance_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Display performance monitoring metrics."""
        if 'execution_time' in metrics:
            st.metric("Execution Time (s)", f"{metrics['execution_time']:.2f}")
        if 'memory_usage' in metrics:
            st.metric("Memory Usage (MB)", f"{metrics['memory_usage']:.2f}")
        if 'cpu_usage' in metrics:
            st.metric("CPU Usage (%)", f"{metrics['cpu_usage']:.2f}")
            
        # Performance history plot if available
        if 'history' in metrics:
            metrics_df = pd.DataFrame(metrics['history'])
            if 'timestamp' in metrics_df.columns:
                metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
                
                # Plot time series
                fig = plotter.create_plot(
                    'line',
                    data=metrics_df,
                    x='timestamp',
                    y=['execution_time', 'memory_usage', 'cpu_usage'],
                    title='Performance History'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_resource_usage(
        self,
        resources: Dict[str, Any]
    ) -> None:
        """Display system resource usage."""
        if 'memory' in resources:
            memory_percent = resources['memory'].get('percent', 0)
            st.metric("Memory Usage (%)", f"{memory_percent:.1f}")
            st.progress(memory_percent / 100)
            
        if 'cpu' in resources:
            cpu_percent = resources['cpu'].get('percent', 0)
            st.metric("CPU Usage (%)", f"{cpu_percent:.1f}")
            st.progress(cpu_percent / 100)
            
        if 'disk' in resources:
            disk_percent = resources['disk'].get('percent', 0)
            st.metric("Disk Usage (%)", f"{disk_percent:.1f}")
            st.progress(disk_percent / 100)
            
        # Resource history plot if available
        if 'history' in resources:
            history_df = pd.DataFrame(resources['history'])
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                
                fig = plotter.create_plot(
                    'line',
                    data=history_df,
                    x='timestamp',
                    y=['memory_percent', 'cpu_percent', 'disk_percent'],
                    title='Resource Usage History'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_operations(
       self,
       operations: Dict[str, Any]
   ) -> None:
       """Display operations monitoring."""
       if 'active' in operations:
           st.subheader("Active Operations")
           for op_name, op_info in operations['active'].items():
               duration = op_info.get('duration', 0)
               status = op_info.get('status', 'Unknown')
               
               st.write(f"**{op_name}**")
               st.write(f"Status: {status}")
               st.write(f"Duration: {duration:.2f}s")
               
               if 'progress' in op_info:
                   st.progress(op_info['progress'])
       
       if 'history' in operations:
           st.subheader("Operation History")
           history_df = pd.DataFrame(operations['history'])
           if not history_df.empty:
               if 'timestamp' in history_df.columns:
                   history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
               
               st.dataframe(
                   history_df.sort_values('timestamp', ascending=False),
                   use_container_width=True
               )
               
               # Operation duration chart
               if 'duration' in history_df.columns and 'operation' in history_df.columns:
                   fig = plotter.create_plot(
                       'bar',
                       data=history_df,
                       x='operation',
                       y='duration',
                       title='Operation Durations'
                   )
                   st.plotly_chart(fig, use_container_width=True)
   
   def _display_errors(
       self,
       errors: Dict[str, Any]
   ) -> None:
       """Display error monitoring."""
       if not errors:
           st.info("No errors recorded")
           return
           
       if 'recent' in errors:
           st.subheader("Recent Errors")
           for error in errors['recent']:
               with st.expander(f"{error.get('type', 'Error')}: {error.get('message', 'Unknown error')}"):
                   st.write(f"Timestamp: {error.get('timestamp', 'Unknown')}")
                   st.write(f"Location: {error.get('location', 'Unknown')}")
                   
                   if 'traceback' in error:
                       st.code(error['traceback'])
                   
                   if 'details' in error:
                       st.json(error['details'])
       
       if 'summary' in errors:
           st.subheader("Error Summary")
           summary_df = pd.DataFrame({
               'Error Type': list(errors['summary'].keys()),
               'Count': list(errors['summary'].values())
           })
           
           fig = plotter.create_plot(
               'bar',
               data=summary_df,
               x='Error Type',
               y='Count',
               title='Error Frequency'
           )
           st.plotly_chart(fig, use_container_width=True)
   
   def _track_component(
       self,
       key: str,
       component_type: str,
       value: Any
   ) -> None:
       """Track component creation and state."""
       self.active_components[key] = {
           'type': component_type,
           'value': value,
           'timestamp': datetime.now().isoformat()
       }
       
       # Record in history
       self.component_history.append({
           'key': key,
           'type': component_type,
           'timestamp': datetime.now().isoformat()
       })
       
       # Update state
       state_manager.set_state(
           f'ui.components.{key}',
           self.active_components[key]
       )
   
   def _format_size(
       self,
       size: int
   ) -> str:
       """Format size in bytes to human-readable format."""
       for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
           if size < 1024:
               return f"{size:.1f} {unit}"
           size /= 1024
       return f"{size:.1f} PB"
   
   def _create_download_button(
       self,
       data: Any,
       key: str,
       format: str = 'csv'
   ) -> None:
       """Create download button for results."""
       if format == 'csv':
           if isinstance(data, pd.DataFrame):
               csv = data.to_csv(index=False)
               st.download_button(
                   "Download CSV",
                   csv,
                   f"{key}_results.csv",
                   "text/csv",
                   key=f"download_{key}_csv"
               )
           elif isinstance(data, dict):
               # Convert dict of dataframes to CSV
               if all(isinstance(v, pd.DataFrame) for v in data.values()):
                   result_df = pd.concat(data.values(), keys=data.keys())
                   csv = result_df.to_csv()
                   st.download_button(
                       "Download CSV",
                       csv,
                       f"{key}_results.csv",
                       "text/csv",
                       key=f"download_{key}_csv"
                   )
               else:
                   st.error("Cannot convert this data to CSV format")
           else:
               st.error("Cannot convert this data to CSV format")
       elif format == 'json':
           if isinstance(data, pd.DataFrame):
               json_str = data.to_json(orient='records', indent=2)
               st.download_button(
                   "Download JSON",
                   json_str,
                   f"{key}_results.json",
                   "application/json",
                   key=f"download_{key}_json"
               )
           elif isinstance(data, (dict, list)):
               json_str = json.dumps(data, indent=2, default=str)
               st.download_button(
                   "Download JSON",
                   json_str,
                   f"{key}_results.json",
                   "application/json",
                   key=f"download_{key}_json"
               )
           else:
               st.error("Cannot convert this data to JSON format")
       elif format == 'excel':
           if isinstance(data, pd.DataFrame):
               output = io.BytesIO()
               with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                   data.to_excel(writer, sheet_name='Results', index=False)
                   
               st.download_button(
                   "Download Excel",
                   output.getvalue(),
                   f"{key}_results.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   key=f"download_{key}_excel"
               )
           elif isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
               output = io.BytesIO()
               with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                   for sheet_name, df in data.items():
                       sheet_name = str(sheet_name)[:31]  # Excel sheet name limit
                       df.to_excel(writer, sheet_name=sheet_name, index=False)
               
               st.download_button(
                   "Download Excel",
                   output.getvalue(),
                   f"{key}_results.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   key=f"download_{key}_excel"
               )
           else:
               st.error("Cannot convert this data to Excel format")
   
   def _export_evaluation(
       self,
       metrics: Dict[str, Any],
       visualizations: Dict[str, Any],
       format: str,
       key: str
   ) -> None:
       """Export evaluation results."""
       # Prepare export data
       export_data = {
           'metrics': metrics,
           'timestamp': datetime.now().isoformat()
       }
       
       if format == 'JSON':
           json_str = json.dumps(export_data, indent=2, default=str)
           st.download_button(
               "Download JSON",
               json_str,
               f"{key}_evaluation.json",
               "application/json"
           )
       elif format == 'HTML':
           # Create HTML report
           html_content = f"""
           <html>
           <head>
               <title>Evaluation Report</title>
               <style>
                   body {{ font-family: Arial, sans-serif; margin: 20px; }}
                   h1, h2 {{ color: #4a4a4a; }}
                   table {{ border-collapse: collapse; width: 100%; }}
                   th, td {{ border: 1px solid #ddd; padding: 8px; }}
                   th {{ background-color: #f2f2f2; }}
               </style>
           </head>
           <body>
               <h1>Evaluation Report</h1>
               <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
               
               <h2>Metrics</h2>
               <table>
                   <tr><th>Metric</th><th>Value</th></tr>
           """
           
           for name, value in metrics.items():
               if isinstance(value, (int, float)):
                   html_content += f"<tr><td>{name}</td><td>{value:.4f}</td></tr>"
               else:
                   html_content += f"<tr><td>{name}</td><td>{value}</td></tr>"
           
           html_content += """
               </table>
           </body>
           </html>
           """
           
           st.download_button(
               "Download HTML",
               html_content,
               f"{key}_evaluation.html",
               "text/html"
           )
       elif format == 'Excel':
           # Create Excel report
           output = io.BytesIO()
           
           with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
               # Metrics sheet
               metric_data = []
               for name, value in metrics.items():
                   if isinstance(value, (int, float)):
                       metric_data.append({'Metric': name, 'Value': value})
                   elif isinstance(value, dict):
                       for subname, subvalue in value.items():
                           if isinstance(subvalue, (int, float)):
                               metric_data.append({'Metric': f"{name}_{subname}", 'Value': subvalue})
               
               pd.DataFrame(metric_data).to_excel(writer, sheet_name='Metrics', index=False)
               
           st.download_button(
               "Download Excel",
               output.getvalue(),
               f"{key}_evaluation.xlsx",
               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
           )
       elif format == 'PDF':
           st.warning("PDF export not yet implemented")
       else:
           st.error(f"Unsupported export format: {format}")

# Create global UI components instance
ui_components = UIComponents()