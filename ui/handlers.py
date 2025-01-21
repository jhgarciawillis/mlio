import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core import config
from core.exceptions import HandlerError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from ui.callbacks import callback_manager

class EventHandler:
    """Handle UI events and state changes."""
    
    def __init__(self):
        self.event_history: List[Dict[str, Any]] = []
        self.active_handlers: Dict[str, Callable] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    @monitor_performance
    @handle_exceptions(HandlerError)
    def register_handler(
        self,
        event_type: str,
        handler: Callable,
        description: Optional[str] = None
    ) -> None:
        """Register event handler."""
        if event_type in self.active_handlers:
            raise HandlerError(f"Handler already registered for event: {event_type}")
        
        self.active_handlers[event_type] = {
            'handler': handler,
            'description': description or handler.__doc__,
            'registered_at': datetime.now().isoformat()
        }
        
        # Track registration
        self._track_handler_registration(event_type)
    
    @monitor_performance
    @handle_exceptions(HandlerError)
    async def handle_event(
        self,
        event_type: str,
        event_data: Any
    ) -> None:
        """Handle event asynchronously."""
        if event_type not in self.active_handlers:
            raise HandlerError(f"No handler registered for event: {event_type}")
        
        handler_info = self.active_handlers[event_type]
        
        try:
            # Execute handler asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                handler_info['handler'],
                event_data
            )
            
            # Track successful execution
            self._track_handler_execution(event_type, True)
            
        except Exception as e:
            # Track failed execution
            self._track_handler_execution(event_type, False, str(e))
            raise HandlerError(f"Error handling event: {str(e)}")
    
    @monitor_performance
    def handle_state_change(
        self,
        state_path: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        """Handle state changes."""
        event_data = {
            'path': state_path,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Trigger state change event
        asyncio.create_task(
            self.handle_event('state_change', event_data)
        )
    
    @monitor_performance
    def handle_user_interaction(
        self,
        interaction_type: str,
        interaction_data: Dict[str, Any]
    ) -> None:
        """Handle user interactions."""
        event_data = {
            'type': interaction_type,
            'data': interaction_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Trigger user interaction event
        asyncio.create_task(
            self.handle_event('user_interaction', event_data)
        )
    
    @monitor_performance
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle errors."""
        event_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Trigger error event
        asyncio.create_task(
            self.handle_event('error', event_data)
        )
    
    @monitor_performance
    def handle_session_state_change(
        self,
        key: str,
        value: Any
    ) -> None:
        """Handle Streamlit session state changes."""
        if key not in st.session_state:
            st.session_state[key] = value
            state_change = {'old_value': None, 'new_value': value}
        else:
            old_value = st.session_state[key]
            st.session_state[key] = value
            state_change = {'old_value': old_value, 'new_value': value}
        
        # Track state change
        self._track_state_change(key, state_change)
    
    @monitor_performance
    def create_state_handler(
        self,
        state_key: str,
        handler: Callable
    ) -> None:
        """Create handler for specific state key."""
        def state_change_handler(state_data: Dict[str, Any]) -> None:
            if state_data['path'] == state_key:
                handler(
                    state_data['old_value'],
                    state_data['new_value']
                )
        
        self.register_handler(
            f"state_change_{state_key}",
            state_change_handler,
            f"Handler for state changes in {state_key}"
        )
    
    @monitor_performance
    def create_error_handler(
        self,
        error_type: Type[Exception],
        handler: Callable
    ) -> None:
        """Create handler for specific error type."""
        def error_handler(error_data: Dict[str, Any]) -> None:
            if error_data['error_type'] == error_type.__name__:
                handler(error_data)
        
        self.register_handler(
            f"error_{error_type.__name__}",
            error_handler,
            f"Handler for {error_type.__name__} errors"
        )
    
    def _track_handler_registration(
        self,
        event_type: str
    ) -> None:
        """Track handler registration."""
        record = {
            'type': 'registration',
            'event_type': event_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.event_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.handlers.registrations.{event_type}',
            record
        )
    
    def _track_handler_execution(
        self,
        event_type: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Track handler execution."""
        record = {
            'type': 'execution',
            'event_type': event_type,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.event_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.handlers.executions.{event_type}',
            record
        )
    
    def _track_state_change(
        self,
        key: str,
        change: Dict[str, Any]
    ) -> None:
        """Track state changes."""
        record = {
            'type': 'state_change',
            'key': key,
            'change': change,
            'timestamp': datetime.now().isoformat()
        }
        
        self.event_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.handlers.state_changes.{key}',
            record
        )
    
    @monitor_performance
    def get_handler_history(
        self,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get handler history."""
        if event_type:
            return [
                record for record in self.event_history
                if record['event_type'] == event_type
            ]
        return self.event_history
    
    @monitor_performance
    def clear_handler_history(
        self,
        event_type: Optional[str] = None
    ) -> None:
        """Clear handler history."""
        if event_type:
            self.event_history = [
                record for record in self.event_history
                if record['event_type'] != event_type
            ]
        else:
            self.event_history = []
            
        # Update state
        state_manager.set_state('ui.handlers.history', self.event_history)

# Create global event handler instance
event_handler = EventHandler()