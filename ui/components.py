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
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter
from metrics import calculator

class UIComponents:
    """Handle UI component creation and management."""
    
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
        with st.sidebar:
            st.title(title)
            
            component_values = {}
            for comp_config in components_config:
                value = self._create_sidebar_component(comp_config)
                component_values[comp_config['key']] = value
            
            return component_values
    
    @monitor_performance
    def create_data_upload_section(
        self,
        allowed_extensions: List[str],
        key: str = "data_upload"
    ) -> Optional[Any]:
        """Create data upload section."""
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
                    return None
                
        return uploaded_file
    
    @monitor_performance
    def create_clustering_section(
        self,
        data: pd.DataFrame,
        available_methods: Dict[str, Any],
        key: str = "clustering"
    ) -> Dict[str, Any]:
        """Create clustering configuration section."""
        st.subheader("Clustering Configuration")
        
        method = st.selectbox(
            "Select Clustering Method",
            options=list(available_methods.keys()),
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
            optimization = st.checkbox(
                "Enable Parameter Optimization",
                key=f"{key}_optimize"
            )
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
        
        config = {
            'method': method,
            'parameters': param_values,
            'features': feature_cols,
            'optimization': {
                'enabled': optimization,
                'n_trials': opt_trials
            },
            'validation': validation
        }
        
        self._track_component(key, "clustering", config)
        
        return config
    
    @monitor_performance
    def create_analysis_section(
        self,
        data: pd.DataFrame,
        analysis_types: List[str],
        key: str = "analysis"
    ) -> Dict[str, Any]:
        """Create analysis configuration section."""
        st.subheader("Analysis Configuration")
        
        selected_analyses = st.multiselect(
            "Select Analysis Types",
            options=analysis_types,
            key=f"{key}_types"
        )
        
        config = {}
        for analysis_type in selected_analyses:
            with st.expander(f"{analysis_type} Configuration"):
                config[analysis_type] = self._create_analysis_config(
                    analysis_type,
                    data,
                    key=f"{key}_{analysis_type}"
                )
        
        self._track_component(key, "analysis", config)
        
        return config
    
    @monitor_performance
    def create_model_selection_section(
        self,
        available_models: Dict[str, Any],
        key: str = "model_selection"
    ) -> Dict[str, Any]:
        """Create model selection section."""
        st.subheader("Model Selection")
        
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
                    key=f"{key}_{category}"
                )
                selected_models[category] = selected
                
                # Model specific configurations
                if selected:
                    selected_models[f"{category}_config"] = {}
                    for model in selected:
                        with st.expander(f"{model} Configuration"):
                            selected_models[f"{category}_config"][model] = (
                                self._create_model_config(models[model], key=f"{key}_{model}")
                            )
        
        self._track_component(key, "model_selection", selected_models)
        
        return selected_models
    
    @monitor_performance
    def create_evaluation_section(
        self,
        metrics: Dict[str, Any],
        visualizations: Dict[str, Any],
        key: str = "evaluation"
    ) -> None:
        """Create evaluation results section."""
        st.subheader("Evaluation Results")
        
        # Metrics display
        with st.expander("Metrics", expanded=True):
            self._display_metrics(metrics)
        
        # Visualizations
        with st.expander("Visualizations", expanded=True):
            self.create_visualization_section(
                visualizations,
                key=f"{key}_viz"
            )
        
        # Export options
        if st.button("Export Results"):
            self._export_evaluation(metrics, visualizations, key)
    
    @monitor_performance
    def create_workflow_section(
        self,
        workflow_type: str,
        available_steps: List[str],
        key: str = "workflow"
    ) -> Dict[str, Any]:
        """Create workflow configuration section."""
        st.subheader(f"{workflow_type} Workflow")
        
        selected_steps = st.multiselect(
            "Select Workflow Steps",
            options=available_steps,
            key=f"{key}_steps"
        )
        
        step_config = {}
        for step in selected_steps:
            with st.expander(f"{step} Configuration"):
                step_config[step] = self._create_step_config(
                    step,
                    key=f"{key}_{step}"
                )
        
        workflow_config = {
            'selected_steps': selected_steps,
            'step_config': step_config
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
        """Create monitoring display section."""
        st.subheader("Performance Monitoring")
        
        metrics_tab, resources_tab = st.tabs(["Metrics", "Resources"])
        
        with metrics_tab:
            self._display_performance_metrics(
                monitoring_data.get('metrics', {})
            )
        
        with resources_tab:
            self._display_resource_usage(
                monitoring_data.get('resources', {})
            )
    
    @monitor_performance
    def create_parameter_inputs(
        self,
        parameters: Dict[str, Dict[str, Any]],
        key: str = "parameters"
    ) -> Dict[str, Any]:
        """Create parameter input section."""
        parameter_values = {}
        
        for param_name, param_config in parameters.items():
            param_type = param_config.get('type', 'number')
            
            if param_type == 'number':
                value = st.number_input(
                    param_name,
                    min_value=param_config.get('min'),
                    max_value=param_config.get('max'),
                    value=param_config.get('default'),
                    step=param_config.get('step', 1),
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
            elif param_type == 'slider':
                value = st.slider(
                    param_name,
                    min_value=param_config['min'],
                    max_value=param_config['max'],
                    value=param_config.get('default'),
                    step=param_config.get('step', 1),
                    key=f"{key}_{param_name}"
                )
            
            parameter_values[param_name] = value
        
        self._track_component(key, "parameters", parameter_values)
        
        return parameter_values
    
    @monitor_performance
    def create_results_section(
        self,
        results: Dict[str, Any],
        key: str = "results"
    ) -> None:
        """Create results display section."""
        st.subheader("Results")
        
        # Create tabs for different result types
        tabs = st.tabs(list(results.keys()))
        
        for tab, (result_type, result_data) in zip(tabs, results.items()):
            with tab:
                if isinstance(result_data, pd.DataFrame):
                    st.dataframe(result_data)
                elif isinstance(result_data, (dict, list)):
                    st.json(result_data)
                else:
                    st.write(result_data)
        
        # Add download button
        if st.button("Download Results"):
            self._create_download_button(results, key)
    
    @monitor_performance
    def create_visualization_section(
        self,
        figures: Dict[str, Any],
        key: str = "visualizations"
    ) -> None:
        """Create visualization section."""
        st.subheader("Visualizations")
        
        # Create tabs for different visualizations
        tabs = st.tabs(list(figures.keys()))
        
        for tab, (fig_name, fig) in zip(tabs, figures.items()):
            with tab:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add download button for individual visualization
                if st.button(f"Download {fig_name}"):
                    self._download_visualization(fig, fig_name)
    
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
    
    def _create_analysis_config(
        self,
        analysis_type: str,
        data: pd.DataFrame,
        key: str
    ) -> Dict[str, Any]:
        """Create configuration for specific analysis type."""
        config = {}
        
        if analysis_type == 'statistical':
            config['tests'] = st.multiselect(
                "Select Statistical Tests",
                options=['t-test', 'anova', 'chi-square', 'correlation'],
                key=f"{key}_tests"
            )
        elif analysis_type == 'clustering':
            config['n_clusters'] = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=20,
                value=5,
                key=f"{key}_clusters"
            )
        elif analysis_type == 'feature_importance':
            config['method'] = st.selectbox(
                "Feature Importance Method",
                options=['mutual_info', 'f_regression', 'permutation'],
                key=f"{key}_method"
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
        
        return config
    
    def _create_step_config(
       self,
       step: str,
       key: str
   ) -> Dict[str, Any]:
       """Create configuration for specific workflow step."""
       config = {}
       
       if step == 'preprocessing':
           config['scaling'] = st.selectbox(
               "Scaling Method",
               options=['standard', 'minmax', 'robust'],
               key=f"{key}_scaling"
           )
           config['handle_missing'] = st.selectbox(
               "Handle Missing Values",
               options=['drop', 'mean', 'median', 'most_frequent'],
               key=f"{key}_missing"
           )
           config['handle_outliers'] = st.checkbox(
               "Handle Outliers",
               key=f"{key}_outliers"
           )
           
       elif step == 'feature_engineering':
           config['create_interactions'] = st.checkbox(
               "Create Interaction Features",
               key=f"{key}_interactions"
           )
           config['polynomial_features'] = st.checkbox(
               "Create Polynomial Features",
               key=f"{key}_polynomial"
           )
           if config['polynomial_features']:
               config['poly_degree'] = st.slider(
                   "Polynomial Degree",
                   min_value=2,
                   max_value=5,
                   value=2,
                   key=f"{key}_poly_degree"
               )
               
       elif step == 'feature_selection':
           config['method'] = st.selectbox(
               "Feature Selection Method",
               options=['mutual_info', 'f_regression', 'rfe', 'lasso'],
               key=f"{key}_method"
           )
           config['n_features'] = st.slider(
               "Number of Features to Select",
               min_value=1,
               max_value=50,
               value=10,
               key=f"{key}_n_features"
           )
           
       elif step == 'model_training':
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
           
       elif step == 'evaluation':
           config['metrics'] = st.multiselect(
               "Select Evaluation Metrics",
               options=['mse', 'rmse', 'mae', 'r2', 'mape'],
               key=f"{key}_metrics"
           )
           config['create_visualizations'] = st.checkbox(
               "Create Visualizations",
               key=f"{key}_viz"
           )
       
       return config
   
    def _display_metrics(
       self,
       metrics: Dict[str, Any]
   ) -> None:
       """Display metrics in organized format."""
       for metric_name, value in metrics.items():
           if isinstance(value, dict):
               st.subheader(metric_name.replace('_', ' ').title())
               for sub_name, sub_value in value.items():
                   st.write(f"{sub_name.replace('_', ' ').title()}: {sub_value}")
           else:
               st.write(f"{metric_name.replace('_', ' ').title()}: {value}")
   
    def _display_performance_metrics(
       self,
       metrics: Dict[str, Any]
   ) -> None:
       """Display performance monitoring metrics."""
       if 'execution_time' in metrics:
           st.metric("Execution Time (s)", metrics['execution_time'])
       if 'memory_usage' in metrics:
           st.metric("Memory Usage (MB)", metrics['memory_usage'])
       if 'cpu_usage' in metrics:
           st.metric("CPU Usage (%)", metrics['cpu_usage'])
           
       # Performance history plot if available
       if 'history' in metrics:
           fig = plotter.create_plot(
               'line',
               data=pd.DataFrame(metrics['history']),
               x='timestamp',
               y=['execution_time', 'memory_usage', 'cpu_usage'],
               title='Performance History'
           )
           st.plotly_chart(fig)
   
    def _display_resource_usage(
       self,
       resources: Dict[str, Any]
   ) -> None:
       """Display system resource usage."""
       if 'memory' in resources:
           st.metric("Memory Usage (%)", resources['memory'].get('percent'))
           st.progress(resources['memory'].get('percent', 0) / 100)
           
       if 'cpu' in resources:
           st.metric("CPU Usage (%)", resources['cpu'].get('percent'))
           st.progress(resources['cpu'].get('percent', 0) / 100)
           
       if 'disk' in resources:
           st.metric("Disk Usage (%)", resources['disk'].get('percent'))
           st.progress(resources['disk'].get('percent', 0) / 100)
   
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
       for unit in ['B', 'KB', 'MB', 'GB']:
           if size < 1024:
               return f"{size:.1f} {unit}"
           size /= 1024
       return f"{size:.1f} TB"
   
    def _create_download_button(
       self,
       data: Any,
       key: str
   ) -> None:
       """Create download button for results."""
       if isinstance(data, pd.DataFrame):
           csv = data.to_csv(index=False)
           st.download_button(
               "Download CSV",
               csv,
               f"{key}_results.csv",
               "text/csv",
               key=f"download_{key}"
           )
       else:
           json_str = json.dumps(data, indent=2)
           st.download_button(
               "Download JSON",
               json_str,
               f"{key}_results.json",
               "application/json",
               key=f"download_{key}"
           )
   
    def _download_visualization(
       self,
       fig: Any,
       name: str
   ) -> None:
       """Create download button for visualization."""
       img_bytes = fig.to_image(format="png")
       st.download_button(
           "Download PNG",
           img_bytes,
           f"{name}.png",
           "image/png"
       )
   
    def _export_evaluation(
       self,
       metrics: Dict[str, Any],
       visualizations: Dict[str, Any],
       key: str
   ) -> None:
       """Export evaluation results."""
       # Create combined results dictionary
       results = {
           'metrics': metrics,
           'timestamp': datetime.now().isoformat()
       }
       
       # Save visualizations as base64 encoded images
       results['visualizations'] = {}
       for name, fig in visualizations.items():
           results['visualizations'][name] = fig.to_image(format="png").decode()
       
       # Create download button
       self._create_download_button(results, f"{key}_evaluation")

# Create global UI components instance
ui_components = UIComponents()