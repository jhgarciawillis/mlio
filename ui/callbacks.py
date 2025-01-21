import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import inspect

from core import config
from core.exceptions import CallbackError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class CallbackManager:
    """Handle UI callbacks and event handling."""
    
    def __init__(self):
        self.callbacks: Dict[str, Callable] = {}
        self.callback_history: List[Dict[str, Any]] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        
    @monitor_performance
    @handle_exceptions(CallbackError)
    def register_callback(
        self,
        callback_id: str,
        callback_func: Callable,
        description: Optional[str] = None
    ) -> None:
        """Register a callback function."""
        if callback_id in self.callbacks:
            raise CallbackError(f"Callback ID already exists: {callback_id}")
        
        # Validate callback function
        if not callable(callback_func):
            raise CallbackError("Callback must be a callable")
        
        # Store callback
        self.callbacks[callback_id] = {
            'function': callback_func,
            'description': description or callback_func.__doc__,
            'signature': inspect.signature(callback_func),
            'registered_at': datetime.now().isoformat()
        }
        
        # Track registration
        self._track_callback_registration(callback_id)
    
    @monitor_performance
    @handle_exceptions(CallbackError)
    def execute_callback(
        self,
        callback_id: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute registered callback function."""
        if callback_id not in self.callbacks:
            raise CallbackError(f"Callback not found: {callback_id}")
        
        callback_info = self.callbacks[callback_id]
        
        try:
            # Validate arguments
            sig = callback_info['signature']
            sig.bind(*args, **kwargs)
            
            # Execute callback
            result = callback_info['function'](*args, **kwargs)
            
            # Track execution
            self._track_callback_execution(callback_id, True)
            
            return result
            
        except Exception as e:
            # Track failed execution
            self._track_callback_execution(callback_id, False, str(e))
            raise CallbackError(f"Error executing callback: {str(e)}")
    
    @monitor_performance
    def register_event_handler(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """Register event handler for specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    @monitor_performance
    def trigger_event(
        self,
        event_type: str,
        event_data: Any
    ) -> None:
        """Trigger event and execute registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler: {str(e)}")
    
    @monitor_performance
    def create_button_callback(
        self,
        label: str,
        callback_func: Callable,
        key: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Create button with callback."""
        if st.button(label, key=key):
            try:
                callback_func(**kwargs)
                return True
            except Exception as e:
                st.error(f"Error executing callback: {str(e)}")
                return False
        return False
    
    @monitor_performance
    def create_form_callback(
        self,
        form_key: str,
        submit_callback: Callable,
        validate_callback: Optional[Callable] = None
    ) -> None:
        """Register callbacks for form submission and validation."""
        callback_id = f"form_{form_key}"
        
        # Register validation callback if provided
        if validate_callback:
            self.register_callback(
                f"{callback_id}_validate",
                validate_callback,
                "Form validation callback"
            )
        
        # Register submission callback
        self.register_callback(
            f"{callback_id}_submit",
            submit_callback,
            "Form submission callback"
        )
    
    @monitor_performance
    def create_selectbox_callback(
        self,
        label: str,
        options: List[Any],
        callback_func: Callable,
        key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create selectbox with callback."""
        selected = st.selectbox(label, options, key=key)
        
        try:
            callback_func(selected, **kwargs)
        except Exception as e:
            st.error(f"Error executing callback: {str(e)}")
        
        return selected
    
    def _track_callback_registration(
        self,
        callback_id: str
    ) -> None:
        """Track callback registration."""
        record = {
            'type': 'registration',
            'callback_id': callback_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.callback_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.callbacks.registrations.{callback_id}',
            record
        )
    
    def _track_callback_execution(
        self,
        callback_id: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Track callback execution."""
        record = {
            'type': 'execution',
            'callback_id': callback_id,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.callback_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.callbacks.executions.{callback_id}',
            record
        )
    
    @monitor_performance
    def get_callback_history(
        self,
        callback_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get callback history."""
        if callback_id:
            return [
                record for record in self.callback_history
                if record['callback_id'] == callback_id
            ]
        return self.callback_history
    
    @monitor_performance
    def clear_callback_history(
        self,
        callback_id: Optional[str] = None
    ) -> None:
        """Clear callback history."""
        if callback_id:
            self.callback_history = [
                record for record in self.callback_history
                if record['callback_id'] != callback_id
            ]
        else:
            self.callback_history = []
            
        # Update state
        state_manager.set_state('ui.callbacks.history', self.callback_history)

# Create global callback manager instance
callback_manager = CallbackManager()