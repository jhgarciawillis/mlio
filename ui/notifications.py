import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import time
import asyncio
import queue
from dataclasses import dataclass

from core import config
from core.exceptions import NotificationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

@dataclass
class Notification:
    """Notification data structure."""
    message: str
    type: str  # 'info', 'success', 'warning', 'error'
    duration: int
    id: str
    timestamp: datetime
    dismissible: bool = True
    action: Optional[Callable] = None
    action_label: Optional[str] = None

class NotificationManager:
    """Handle UI notifications and alerts."""
    
    def __init__(self):
        self.notifications: List[Notification] = []
        self.notification_history: List[Dict[str, Any]] = []
        self.notification_queue = queue.Queue()
        self.active_notifications: Dict[str, Notification] = {}
        
    @monitor_performance
    @handle_exceptions(NotificationError)
    def show_notification(
        self,
        message: str,
        notification_type: str = 'info',
        duration: int = 3,
        dismissible: bool = True,
        action: Optional[Callable] = None,
        action_label: Optional[str] = None
    ) -> None:
        """Show notification."""
        notification_id = f"notification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        notification = Notification(
            message=message,
            type=notification_type,
            duration=duration,
            id=notification_id,
            timestamp=datetime.now(),
            dismissible=dismissible,
            action=action,
            action_label=action_label
        )
        
        self.notifications.append(notification)
        self.notification_queue.put(notification)
        
        # Track notification
        self._track_notification(notification)
        
        # Display notification
        self._display_notification(notification)
    
    @monitor_performance
    def show_success(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Show success notification."""
        self.show_notification(message, 'success', **kwargs)
    
    @monitor_performance
    def show_error(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Show error notification."""
        self.show_notification(message, 'error', **kwargs)
    
    @monitor_performance
    def show_warning(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Show warning notification."""
        self.show_notification(message, 'warning', **kwargs)
    
    @monitor_performance
    def show_info(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Show info notification."""
        self.show_notification(message, 'info', **kwargs)
    
    @monitor_performance
    @handle_exceptions(NotificationError)
    def create_toast(
        self,
        message: str,
        icon: Optional[str] = None,
        duration: int = 3
    ) -> None:
        """Create toast notification."""
        st.toast(
            message,
            icon=icon
        )
        
        # Track toast
        self._track_notification(Notification(
            message=message,
            type='toast',
            duration=duration,
            id=f"toast_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now()
        ))
    
    def _display_notification(
        self,
        notification: Notification
    ) -> None:
        """Display notification in UI."""
        self.active_notifications[notification.id] = notification
        
        if notification.type == 'success':
            st.success(notification.message, icon="✅")
        elif notification.type == 'error':
            st.error(notification.message, icon="🚨")
        elif notification.type == 'warning':
            st.warning(notification.message, icon="⚠️")
        else:  # info
            st.info(notification.message, icon="ℹ️")
        
        # Add action button if provided
        if notification.action and notification.action_label:
            if st.button(notification.action_label, key=f"action_{notification.id}"):
                notification.action()
        
        # Add dismiss button if dismissible
        if notification.dismissible:
            if st.button("Dismiss", key=f"dismiss_{notification.id}"):
                self.dismiss_notification(notification.id)
    
    @monitor_performance
    def dismiss_notification(
        self,
        notification_id: str
    ) -> None:
        """Dismiss notification."""
        if notification_id in self.active_notifications:
            notification = self.active_notifications.pop(notification_id)
            self._track_notification_dismissal(notification)
    
    @monitor_performance
    def clear_notifications(self) -> None:
        """Clear all notifications."""
        self.active_notifications.clear()
        self.notification_queue = queue.Queue()
    
    @monitor_performance
    async def process_notification_queue(self) -> None:
        """Process notification queue asynchronously."""
        while True:
            try:
                if not self.notification_queue.empty():
                    notification = self.notification_queue.get()
                    self._display_notification(notification)
                    
                    # Auto-dismiss after duration
                    if notification.duration > 0:
                        await asyncio.sleep(notification.duration)
                        self.dismiss_notification(notification.id)
                        
                await asyncio.sleep(0.1)  # Small delay to prevent CPU hogging
                
            except Exception as e:
                logger.error(f"Error processing notification queue: {str(e)}")
    
    def _track_notification(
        self,
        notification: Notification
    ) -> None:
        """Track notification creation."""
        record = {
            'id': notification.id,
            'type': notification.type,
            'message': notification.message,
            'timestamp': notification.timestamp.isoformat(),
            'status': 'created'
        }
        
        self.notification_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.notifications.{notification.id}',
            record
        )
    
    def _track_notification_dismissal(
        self,
        notification: Notification
    ) -> None:
        """Track notification dismissal."""
        record = {
            'id': notification.id,
            'type': notification.type,
            'message': notification.message,
            'timestamp': datetime.now().isoformat(),
            'status': 'dismissed'
        }
        
        self.notification_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.notifications.{notification.id}',
            record
        )
    
    @monitor_performance
    def get_notification_history(
        self,
        notification_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get notification history."""
        if notification_type:
            return [
                record for record in self.notification_history
                if record['type'] == notification_type
            ]
        return self.notification_history
    
    @monitor_performance
    def get_active_notifications(
        self,
        notification_type: Optional[str] = None
    ) -> Dict[str, Notification]:
        """Get active notifications."""
        if notification_type:
            return {
                notification_id: notification
                for notification_id, notification in self.active_notifications.items()
                if notification.type == notification_type
            }
        return self.active_notifications

# Create global notification manager instance
notification_manager = NotificationManager()

# Start notification queue processing
asyncio.create_task(notification_manager.process_notification_queue())