import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime

from core import config
from core.exceptions import UIError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from ui.components import ui_components

class LayoutManager:
    """Handle UI layout management."""
    
    def __init__(self):
        self.layout_history: List[Dict[str, Any]] = []
        self.active_layouts: Dict[str, Dict[str, Any]] = {}
        self.section_callbacks: Dict[str, Callable] = {}
        
    @monitor_performance
    @handle_exceptions(UIError)
    def create_main_layout(
        self,
        title: str,
        sidebar_config: Optional[Dict[str, Any]] = None,
        layout_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create main application layout."""
        # Set page config
        st.set_page_config(
            page_title=title,
            page_icon=config.ui.app_icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Create main title
        st.title(title)
        
        # Create sidebar if config provided
        if sidebar_config:
            self._create_sidebar_layout(sidebar_config)
        
        # Create main layout
        if layout_config:
            self._create_content_layout(layout_config)
            
        # Track layout creation
        self._track_layout("main", {
            'title': title,
            'sidebar_config': sidebar_config,
            'layout_config': layout_config
        })
    
    @monitor_performance
    def create_analysis_layout(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> None:
        """Create analysis mode layout."""
        tabs = st.tabs([
            "Data Overview",
            "Statistical Analysis",
            "Visualizations",
            "Data Quality"
        ])
        
        with tabs[0]:
            self._create_data_overview_section(data)
            
        with tabs[1]:
            self._create_statistical_analysis_section(data)
            
        with tabs[2]:
            self._create_visualization_section(data)
            
        with tabs[3]:
            self._create_data_quality_section(data)
            
        # Track layout
        self._track_layout("analysis", {
            'has_data': data is not None
        })
    
    @monitor_performance
    def create_training_layout(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> None:
        """Create training mode layout."""
        tabs = st.tabs([
            "Data Preparation",
            "Model Configuration",
            "Training Progress",
            "Results"
        ])
        
        with tabs[0]:
            self._create_data_preparation_section(data)
            
        with tabs[1]:
            self._create_model_configuration_section()
            
        with tabs[2]:
            self._create_training_progress_section()
            
        with tabs[3]:
            self._create_training_results_section()
            
        # Track layout
        self._track_layout("training", {
            'has_data': data is not None
        })
    
    @monitor_performance
    def create_prediction_layout(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> None:
        """Create prediction mode layout."""
        tabs = st.tabs([
            "Model Selection",
            "Prediction Input",
            "Results",
            "Analysis"
        ])
        
        with tabs[0]:
            self._create_model_selection_section()
            
        with tabs[1]:
            self._create_prediction_input_section(data)
            
        with tabs[2]:
            self._create_prediction_results_section()
            
        with tabs[3]:
            self._create_prediction_analysis_section()
            
        # Track layout
        self._track_layout("prediction", {
            'has_data': data is not None
        })
    
    def _create_sidebar_layout(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Create sidebar layout."""
        with st.sidebar:
            # Mode selection if provided
            if 'modes' in config:
                selected_mode = st.selectbox(
                    "Select Mode",
                    options=config['modes'],
                    key="mode_selector"
                )
                
                # Update state
                state_manager.set_state('ui.current_mode', selected_mode)
            
            # Additional sidebar components
            if 'components' in config:
                ui_components.create_sidebar(
                    config.get('title', 'Settings'),
                    config['components']
                )
    
    def _create_content_layout(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Create main content layout."""
        # Create sections based on configuration
        for section_config in config.get('sections', []):
            self._create_section(section_config)
    
    def _create_section(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Create individual section."""
        section_type = config['type']
        
        if section_type == 'columns':
            self._create_column_section(config)
        elif section_type == 'tabs':
            self._create_tab_section(config)
        elif section_type == 'expander':
            self._create_expander_section(config)
        else:
            # Default to regular section
            self._create_regular_section(config)
    
    def _create_column_section(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Create column-based section."""
        cols = st.columns(config.get('n_columns', 2))
        
        for i, col_config in enumerate(config.get('content', [])):
            with cols[i % len(cols)]:
                self._create_section(col_config)
    
    def _create_tab_section(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Create tab-based section."""
        tabs = st.tabs(config.get('tab_names', []))
        
        for i, tab_config in enumerate(config.get('content', [])):
            with tabs[i]:
                self._create_section(tab_config)
    
    def _create_expander_section(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Create expander section."""
        with st.expander(config.get('title', 'Expander')):
            for section_config in config.get('content', []):
                self._create_section(section_config)
    
    def _create_regular_section(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Create regular section."""
        if 'title' in config:
            st.subheader(config['title'])
            
        if 'content' in config:
            if isinstance(config['content'], str):
                st.write(config['content'])
            elif isinstance(config['content'], dict):
                self._create_section(config['content'])
            elif isinstance(config['content'], list):
                for item in config['content']:
                    self._create_section(item)
    
    def register_section_callback(
        self,
        section_name: str,
        callback: Callable
    ) -> None:
        """Register callback for section creation."""
        self.section_callbacks[section_name] = callback
    
    def _track_layout(
        self,
        layout_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Track layout creation."""
        layout_id = f"{layout_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_layouts[layout_id] = {
            'type': layout_type,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        # Record in history
        self.layout_history.append({
            'layout_id': layout_id,
            'type': layout_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update state
        state_manager.set_state(
            f'ui.layouts.{layout_id}',
            self.active_layouts[layout_id]
        )

# Create global layout manager instance
layout_manager = LayoutManager()