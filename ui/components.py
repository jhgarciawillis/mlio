import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path

from core import config
from core.exceptions import UIError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class UIComponents:
    """Handle UI component creation and management."""
    
    def __init__(self):
        self.component_history: List[Dict[str, Any]] = []
        self.active_components: Dict[str, Any] = {}
        
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
            preview_container = st.expander("Preview Uploaded Data")
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
        
        # Track component
        self._track_component(key, "model_selection", selected_models)
        
        return selected_models
    
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
        
        # Track component
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
    
    def _format_size(self, size: int) -> str:
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

# Create global UI components instance
ui_components = UIComponents()