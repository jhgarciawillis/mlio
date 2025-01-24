import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from core import config
from core.exceptions import UIError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from ui.components import ui_components
from ui.settings import ui_settings

class LayoutManager:
    """Handle UI layout management."""
    
    def __init__(self):
        self.layout_history: List[Dict[str, Any]] = []
        self.active_layouts: Dict[str, Dict[str, Any]] = {}
        self.section_callbacks: Dict[str, Callable] = {}
        
        # Layout templates
        self.templates = {
            'analysis': self._create_analysis_template(),
            'training': self._create_training_template(),
            'prediction': self._create_prediction_template(),
            'clustering': self._create_clustering_template(),
            'metrics': self._create_metrics_template(),
            'evaluation': self._create_evaluation_template()
        }
    
    @monitor_performance
    @handle_exceptions(UIError)
    def create_main_layout(
        self,
        title: str,
        sidebar_config: Optional[Dict[str, Any]] = None,
        layout_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create main application layout."""
        # Record operation start
        state_monitor.record_operation_start(
            'layout_creation',
            'layout',
            {'title': title}
        )
        
        try:
            # Set page config
            st.set_page_config(
                page_title=title,
                page_icon=config.ui.app_icon,
                layout=ui_settings.settings['layout']['content_width'],
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
            
            # Record operation completion
            state_monitor.record_operation_end('layout_creation', 'completed')
            
        except Exception as e:
            state_monitor.record_operation_end('layout_creation', 'failed', {'error': str(e)})
            raise
    
    @monitor_performance
    def create_analysis_layout(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> None:
        """Create analysis mode layout."""
        with st.expander("Data Overview", expanded=True):
            if data is not None:
                ui_components.create_data_overview(data)
            else:
                st.info("Please upload data to begin analysis")
        
        tabs = st.tabs([
            "Statistical Analysis",
            "Feature Analysis",
            "Clustering Analysis",
            "Quality Analysis"
        ])
        
        with tabs[0]:
            self._create_statistical_analysis_section(data)
        
        with tabs[1]:
            self._create_feature_analysis_section(data)
        
        with tabs[2]:
            self._create_clustering_analysis_section(data)
        
        with tabs[3]:
            self._create_quality_analysis_section(data)
    
    @monitor_performance
    def create_clustering_layout(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> None:
        """Create clustering mode layout."""
        st.header("Clustering Analysis")
        
        if data is not None:
            # Clustering configuration
            with st.expander("Clustering Configuration", expanded=True):
                clustering_config = ui_components.create_clustering_config_form()
            
            # Feature selection
            with st.expander("Feature Selection"):
                selected_features = ui_components.create_feature_selection_form(
                    data.columns.tolist()
                )
            
            # Clustering execution
            if clustering_config and selected_features:
                if st.button("Run Clustering"):
                    with st.spinner("Performing clustering analysis..."):
                        self._execute_clustering(
                            data,
                            clustering_config,
                            selected_features
                        )
            
            # Results visualization
            if 'clustering_results' in st.session_state:
                self._display_clustering_results(st.session_state.clustering_results)
        else:
            st.info("Please upload data to begin clustering analysis")
    
    @monitor_performance
    def create_metrics_layout(
        self,
        metrics_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create metrics visualization layout."""
        st.header("Performance Metrics")
        
        if metrics_data is not None:
            # Overview metrics
            with st.expander("Overview Metrics", expanded=True):
                ui_components.display_metrics_overview(metrics_data)
            
            # Detailed analysis
            tabs = st.tabs([
                "Error Analysis",
                "Distribution Analysis",
                "Cluster Analysis",
                "Recommendations"
            ])
            
            with tabs[0]:
                ui_components.display_error_analysis(metrics_data)
            
            with tabs[1]:
                ui_components.display_distribution_analysis(metrics_data)
            
            with tabs[2]:
                ui_components.display_cluster_analysis(metrics_data)
            
            with tabs[3]:
                ui_components.display_recommendations(metrics_data)
        else:
            st.info("No metrics data available")
    
    def _create_statistical_analysis_section(
        self,
        data: Optional[pd.DataFrame]
    ) -> None:
        """Create statistical analysis section."""
        if data is not None:
            st.subheader("Statistical Analysis")
            
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Descriptive Statistics", "Correlation Analysis", "Distribution Analysis"]
            )
            
            if analysis_type == "Descriptive Statistics":
                ui_components.display_descriptive_statistics(data)
            elif analysis_type == "Correlation Analysis":
                ui_components.display_correlation_analysis(data)
            else:
                ui_components.display_distribution_analysis(data)
    
    def _create_feature_analysis_section(
        self,
        data: Optional[pd.DataFrame]
    ) -> None:
        """Create feature analysis section."""
        if data is not None:
            st.subheader("Feature Analysis")
            
            feature_options = st.multiselect(
                "Select Features for Analysis",
                data.columns.tolist()
            )
            
            if feature_options:
                ui_components.display_feature_analysis(data[feature_options])
    
    def _create_clustering_analysis_section(
        self,
        data: Optional[pd.DataFrame]
    ) -> None:
        """Create clustering analysis section."""
        if data is not None:
            st.subheader("Clustering Analysis")
            
            # Create clustering configuration form
            clustering_params = ui_components.create_clustering_config_form()
            
            if clustering_params:
                if st.button("Run Clustering Analysis"):
                    with st.spinner("Performing clustering..."):
                        results = self._execute_clustering(data, clustering_params)
                        ui_components.display_clustering_results(results)
    
    def _create_quality_analysis_section(
        self,
        data: Optional[pd.DataFrame]
    ) -> None:
        """Create quality analysis section."""
        if data is not None:
            st.subheader("Data Quality Analysis")
            
            quality_metrics = ui_components.analyze_data_quality(data)
            ui_components.display_quality_metrics(quality_metrics)
    
    def _execute_clustering(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute clustering analysis."""
        # Implement clustering execution logic here
        pass
    
    def _display_clustering_results(
        self,
        results: Dict[str, Any]
    ) -> None:
        """Display clustering results."""
        # Implement clustering results display logic here
        pass
    
    def _record_layout(
        self,
        layout_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Record layout creation."""
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