import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json
from pathlib import Path

from core import config
from core.exceptions import UIError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import style_manager

class UISettings:
    """Handle UI settings and configuration."""
    
    def __init__(self):
        # Initialize settings with defaults
        self.settings = {
            'theme': self._get_default_theme(),
            'layout': self._get_default_layout(),
            'display': self._get_default_display(),
            'interaction': self._get_default_interaction(),
            'notifications': self._get_default_notifications(),
            'clustering': self._get_default_clustering()
        }
        self.settings_history: List[Dict[str, Any]] = []
        self.custom_settings: Dict[str, Any] = {}
        
    @monitor_performance
    @handle_exceptions(UIError)
    def create_settings_page(self) -> None:
        """Create settings configuration page."""
        st.title("Settings")
        
        tabs = st.tabs([
            "Theme",
            "Layout",
            "Display",
            "Interaction",
            "Notifications",
            "Clustering"
        ])
        
        with tabs[0]:
            self._create_theme_settings()
        
        with tabs[1]:
            self._create_layout_settings()
            
        with tabs[2]:
            self._create_display_settings()
            
        with tabs[3]:
            self._create_interaction_settings()
            
        with tabs[4]:
            self._create_notification_settings()
            
        with tabs[5]:
            self._create_clustering_settings()
        
        if st.button("Save Settings"):
            self.save_settings()
            st.success("Settings saved successfully!")
    
    def _create_theme_settings(self) -> None:
        """Create theme settings section."""
        st.subheader("Theme Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.settings['theme']['primary_color'] = st.color_picker(
                "Primary Color",
                value=self.settings['theme']['primary_color']
            )
            
            self.settings['theme']['background_color'] = st.color_picker(
                "Background Color",
                value=self.settings['theme']['background_color']
            )
            
            self.settings['theme']['accent_color'] = st.color_picker(
                "Accent Color",
                value=self.settings['theme']['accent_color']
            )
        
        with col2:
            self.settings['theme']['font'] = st.selectbox(
                "Font",
                options=["Arial", "Helvetica", "Times New Roman"],
                index=["Arial", "Helvetica", "Times New Roman"].index(
                    self.settings['theme']['font']
                )
            )
            
            self.settings['theme']['font_size'] = st.slider(
                "Font Size",
                min_value=8,
                max_value=24,
                value=self.settings['theme']['font_size']
            )
            
            self.settings['theme']['dark_mode'] = st.checkbox(
                "Dark Mode",
                value=self.settings['theme'].get('dark_mode', False)
            )
    
    def _create_layout_settings(self) -> None:
        """Create layout settings section."""
        st.subheader("Layout Settings")
        
        self.settings['layout']['sidebar_position'] = st.radio(
            "Sidebar Position",
            options=["Left", "Right"],
            index=["Left", "Right"].index(
                self.settings['layout']['sidebar_position']
            )
        )
        
        self.settings['layout']['content_width'] = st.slider(
            "Content Width",
            min_value=400,
            max_value=2000,
            value=self.settings['layout']['content_width']
        )
        
        self.settings['layout']['show_footer'] = st.checkbox(
            "Show Footer",
            value=self.settings['layout']['show_footer']
        )
        
        self.settings['layout']['max_plots_per_row'] = st.number_input(
            "Maximum Plots per Row",
            min_value=1,
            max_value=4,
            value=self.settings['layout']['max_plots_per_row']
        )
    
    def _create_display_settings(self) -> None:
        """Create display settings section."""
        st.subheader("Display Settings")
        
        self.settings['display']['animation_speed'] = st.slider(
            "Animation Speed",
            min_value=0.1,
            max_value=2.0,
            value=self.settings['display']['animation_speed']
        )
        
        self.settings['display']['table_height'] = st.number_input(
            "Table Height",
            min_value=100,
            max_value=1000,
            value=self.settings['display']['table_height']
        )
        
        self.settings['display']['plot_width'] = st.number_input(
            "Plot Width",
            min_value=400,
            max_value=2000,
            value=self.settings['display']['plot_width']
        )
        
        self.settings['display']['plot_height'] = st.number_input(
            "Plot Height",
            min_value=300,
            max_value=1500,
            value=self.settings['display']['plot_height']
        )
    
    def _create_interaction_settings(self) -> None:
        """Create interaction settings section."""
        st.subheader("Interaction Settings")
        
        self.settings['interaction']['auto_scroll'] = st.checkbox(
            "Auto Scroll",
            value=self.settings['interaction']['auto_scroll']
        )
        
        self.settings['interaction']['confirmation_dialogs'] = st.checkbox(
            "Show Confirmation Dialogs",
            value=self.settings['interaction']['confirmation_dialogs']
        )
        
        self.settings['interaction']['tooltip_delay'] = st.slider(
            "Tooltip Delay (ms)",
            min_value=0,
            max_value=2000,
            value=self.settings['interaction']['tooltip_delay']
        )
    
    def _create_notification_settings(self) -> None:
        """Create notification settings section."""
        st.subheader("Notification Settings")
        
        self.settings['notifications']['show_notifications'] = st.checkbox(
            "Show Notifications",
            value=self.settings['notifications']['show_notifications']
        )
        
        self.settings['notifications']['notification_duration'] = st.slider(
            "Notification Duration (s)",
            min_value=1,
            max_value=10,
            value=self.settings['notifications']['notification_duration']
        )
        
        self.settings['notifications']['notification_position'] = st.selectbox(
            "Notification Position",
            options=['top-right', 'top-left', 'bottom-right', 'bottom-left'],
            index=['top-right', 'top-left', 'bottom-right', 'bottom-left'].index(
                self.settings['notifications']['notification_position']
            )
        )
        
        self.settings['notifications']['sound_enabled'] = st.checkbox(
            "Enable Sound",
            value=self.settings['notifications'].get('sound_enabled', False)
        )
    
    def _create_clustering_settings(self) -> None:
        """Create clustering settings section."""
        st.subheader("Clustering Settings")
        
        self.settings['clustering']['default_method'] = st.selectbox(
            "Default Clustering Method",
            options=['kmeans', 'dbscan', 'gaussian_mixture', 'hierarchical', 'spectral'],
            index=['kmeans', 'dbscan', 'gaussian_mixture', 'hierarchical', 'spectral'].index(
                self.settings['clustering']['default_method']
            )
        )
        
        self.settings['clustering']['max_clusters'] = st.number_input(
            "Maximum Number of Clusters",
            min_value=2,
            max_value=50,
            value=self.settings['clustering']['max_clusters']
        )
        
        self.settings['clustering']['visualization_enabled'] = st.checkbox(
            "Enable Clustering Visualizations",
            value=self.settings['clustering'].get('visualization_enabled', True)
        )
        
        self.settings['clustering']['auto_optimize'] = st.checkbox(
            "Auto-optimize Clustering Parameters",
            value=self.settings['clustering'].get('auto_optimize', True)
        )
    
    def _get_default_theme(self) -> Dict[str, Any]:
        """Get default theme settings."""
        return {
            'primary_color': '#1f77b4',
            'background_color': '#ffffff',
            'accent_color': '#ff7f0e',
            'font': 'Arial',
            'font_size': 14,
            'dark_mode': False
        }
    
    def _get_default_layout(self) -> Dict[str, Any]:
        """Get default layout settings."""
        return {
            'sidebar_position': 'Left',
            'content_width': 1200,
            'show_footer': True,
            'max_plots_per_row': 2
        }
    
    def _get_default_display(self) -> Dict[str, Any]:
        """Get default display settings."""
        return {
            'animation_speed': 1.0,
            'table_height': 400,
            'plot_width': 800,
            'plot_height': 500
        }
    
    def _get_default_interaction(self) -> Dict[str, Any]:
        """Get default interaction settings."""
        return {
            'auto_scroll': True,
            'confirmation_dialogs': True,
            'tooltip_delay': 500
        }
    
    def _get_default_notifications(self) -> Dict[str, Any]:
        """Get default notification settings."""
        return {
            'show_notifications': True,
            'notification_duration': 3,
            'notification_position': 'top-right',
            'sound_enabled': False
        }
    
    def _get_default_clustering(self) -> Dict[str, Any]:
        """Get default clustering settings."""
        return {
            'default_method': 'kmeans',
            'max_clusters': 20,
            'visualization_enabled': True,
            'auto_optimize': True
        }
    
    @monitor_performance
    def save_settings(self) -> None:
        """Save current settings."""
        # Record settings change
        self._track_settings_change()
        
        # Save to state manager
        state_manager.set_state('ui.settings', self.settings)
        
        # Apply theme
        style_manager.set_theme(self.settings['theme'])
        
        # Save to disk
        self._save_settings_to_disk()
    
    @monitor_performance
    def load_settings(self) -> None:
        """Load saved settings."""
        try:
            # Try loading from state manager first
            saved_settings = state_manager.get_state('ui.settings')
            if saved_settings:
                self.settings.update(saved_settings)
                return
            
            # Try loading from disk
            self._load_settings_from_disk()
            
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            self._reset_to_defaults()
    
    def _track_settings_change(self) -> None:
        """Track settings changes."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'settings': self.settings.copy()
        }
        
        self.settings_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.settings.history.{len(self.settings_history)}',
            record
        )
    
    def _save_settings_to_disk(self) -> None:
        """Save settings to disk."""
        settings_path = config.directories.base_dir / 'ui_settings.json'
        with open(settings_path, 'w') as f:
            json.dump(self.settings, f, indent=4)
    
    def _load_settings_from_disk(self) -> None:
        """Load settings from disk."""
        settings_path = config.directories.base_dir / 'ui_settings.json'
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                self.settings.update(json.load(f))
    
    def _reset_to_defaults(self) -> None:
        """Reset settings to defaults."""
        self.settings = {
            'theme': self._get_default_theme(),
            'layout': self._get_default_layout(),
            'display': self._get_default_display(),
            'interaction': self._get_default_interaction(),
            'notifications': self._get_default_notifications(),
            'clustering': self._get_default_clustering()
        }

# Create global UI settings instance
ui_settings = UISettings()