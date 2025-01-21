import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import json
from pathlib import Path

from core import config
from core.exceptions import SettingsError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class UISettings:
    """Handle UI settings and configuration."""
    
    def __init__(self):
        self.settings: Dict[str, Any] = {
            'theme': self._get_default_theme(),
            'layout': self._get_default_layout(),
            'display': self._get_default_display(),
            'interaction': self._get_default_interaction(),
            'notifications': self._get_default_notifications()
        }
        self.settings_history: List[Dict[str, Any]] = []
        self.custom_settings: Dict[str, Any] = {}
        
    @monitor_performance
    @handle_exceptions(SettingsError)
    def create_settings_page(self) -> None:
        """Create settings configuration page."""
        st.title("Settings")
        
        tabs = st.tabs([
            "Theme",
            "Layout",
            "Display",
            "Interaction",
            "Notifications"
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
    
    def _create_display_settings(self) -> None:
        """Create display settings section."""
        st.subheader("Display Settings")
        
        self.settings['display']['dark_mode'] = st.checkbox(
            "Dark Mode",
            value=self.settings['display']['dark_mode']
        )
        
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
        
        self.settings['notifications']['sound_enabled'] = st.checkbox(
            "Enable Sound",
            value=self.settings['notifications']['sound_enabled']
        )
    
    @monitor_performance
    def save_settings(self) -> None:
        """Save current settings."""
        # Track settings change
        self._track_settings_change()
        
        # Save to state manager
        state_manager.set_state('ui.settings', self.settings)
        
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
    
    def _get_default_theme(self) -> Dict[str, Any]:
        """Get default theme settings."""
        return {
            'primary_color': '#1f77b4',
            'background_color': '#ffffff',
            'font': 'Arial',
            'font_size': 14
        }
    
    def _get_default_layout(self) -> Dict[str, Any]:
        """Get default layout settings."""
        return {
            'sidebar_position': 'Left',
            'content_width': 1200,
            'show_footer': True
        }
    
    def _get_default_display(self) -> Dict[str, Any]:
        """Get default display settings."""
        return {
            'dark_mode': False,
            'animation_speed': 1.0,
            'table_height': 400
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
            'sound_enabled': False
        }
    
    def _track_settings_change(self) -> None:
        """Track settings changes."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'settings': self.settings.copy()
        }
        
        self.settings_history.append(record)
    
    def _save_settings_to_disk(self) -> None:
        """Save settings to disk."""
        settings_path = Path(config.directories.base_dir) / 'settings.json'
        with open(settings_path, 'w') as f:
            json.dump(self.settings, f, indent=4)
    
    def _load_settings_from_disk(self) -> None:
        """Load settings from disk."""
        settings_path = Path(config.directories.base_dir) / 'settings.json'
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                loaded_settings = json.load(f)
                self.settings.update(loaded_settings)
    
    def _reset_to_defaults(self) -> None:
        """Reset settings to defaults."""
        self.settings = {
            'theme': self._get_default_theme(),
            'layout': self._get_default_layout(),
            'display': self._get_default_display(),
            'interaction': self._get_default_interaction(),
            'notifications': self._get_default_notifications()
        }

# Create global UI settings instance
ui_settings = UISettings()