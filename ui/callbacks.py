import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import inspect
import asyncio
import functools

from core import config
from core.exceptions import CallbackError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class CallbackManager:
    """Handle UI callbacks and event handling."""
    
    def __init__(self):
        self.callbacks: Dict[str, Dict[str, Any]] = {}
        self.callback_history: List[Dict[str, Any]] = []
        self.event_handlers: Dict[str, List[Dict[str, Any]]] = {}
        self.callback_queues: Dict[str, asyncio.Queue] = {}
        
    @monitor_performance
    @handle_exceptions(CallbackError)
    def register_callback(
        self,
        callback_id: str,
        callback_func: Callable,
        description: Optional[str] = None,
        priority: int = 0,
        debounce_ms: Optional[int] = None
    ) -> None:
        """Register a callback function with advanced options."""
        if callback_id in self.callbacks:
            raise CallbackError(f"Callback ID already exists: {callback_id}")
        
        # Validate callback function
        if not callable(callback_func):
            raise CallbackError("Callback must be a callable")
        
        # Add debouncing if requested
        if debounce_ms is not None:
            callback_func = self._debounce(callback_func, debounce_ms)
        
        # Store callback
        self.callbacks[callback_id] = {
            'function': callback_func,
            'description': description or callback_func.__doc__,
            'signature': inspect.signature(callback_func),
            'priority': priority,
            'registered_at': datetime.now().isoformat(),
            'execution_count': 0,
            'last_execution': None,
            'average_execution_time': 0.0
        }
        
        # Create callback queue if needed
        self.callback_queues[callback_id] = asyncio.Queue()
        
        # Track registration
        self._track_callback_registration(callback_id)
    
    @monitor_performance
    @handle_exceptions(CallbackError)
    async def execute_callback(
        self,
        callback_id: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute registered callback function asynchronously."""
        if callback_id not in self.callbacks:
            raise CallbackError(f"Callback not found: {callback_id}")
        
        callback_info = self.callbacks[callback_id]
        
        try:
            # Validate arguments
            sig = callback_info['signature']
            sig.bind(*args, **kwargs)
            
            # Add to callback queue
            await self.callback_queues[callback_id].put({
                'args': args,
                'kwargs': kwargs,
                'timestamp': datetime.now().isoformat()
            })
            
            # Execute callback
            start_time = datetime.now()
            result = await self._execute_callback_async(
                callback_info['function'],
                *args,
                **kwargs
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update callback statistics
            self._update_callback_stats(callback_id, execution_time)
            
            # Track successful execution
            self._track_callback_execution(callback_id, True, execution_time)
            
            return result
            
        except Exception as e:
            # Track failed execution
            self._track_callback_execution(callback_id, False, error=str(e))
            raise CallbackError(f"Error executing callback: {str(e)}") from e
    
    async def _execute_callback_async(
        self,
        callback: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute callback asynchronously."""
        if asyncio.iscoroutinefunction(callback):
            return await callback(*args, **kwargs)
        else:
            return callback(*args, **kwargs)
    
    def _debounce(
        self,
        func: Callable,
        wait_ms: int
    ) -> Callable:
        """Create a debounced version of a callback."""
        timer = None
        
        @functools.wraps(func)
        def debounced(*args, **kwargs):
            nonlocal timer
            
            def call_later():
                func(*args, **kwargs)
            
            if timer is not None:
                timer.cancel()
            
            timer = asyncio.create_task(
                asyncio.sleep(wait_ms / 1000)
            )
            timer.add_done_callback(lambda _: call_later())
            
        return debounced
    
    @monitor_performance
    def register_event_handler(
        self,
        event_type: str,
        handler: Callable,
        priority: int = 0,
        description: Optional[str] = None
    ) -> None:
        """Register event handler with priority."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append({
            'handler': handler,
            'priority': priority,
            'description': description or handler.__doc__,
            'registered_at': datetime.now().isoformat()
        })
        
        # Sort handlers by priority
        self.event_handlers[event_type].sort(
            key=lambda x: x['priority'],
            reverse=True
        )
    
    @monitor_performance
    def trigger_event(
        self,
        event_type: str,
        event_data: Any
    ) -> None:
        """Trigger event and execute registered handlers."""
        if event_type in self.event_handlers:
            for handler_info in self.event_handlers[event_type]:
                try:
                    handler_info['handler'](event_data)
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
            'callback_info': self.callbacks[callback_id],
            'timestamp': datetime.now().isoformat()
        }
        
        self.callback_history.append(record)
        state_manager.set_state(
            f'ui.callbacks.registrations.{callback_id}',
            record
        )
    
    def _track_callback_execution(
        self,
        callback_id: str,
        success: bool,
        execution_time: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Track callback execution."""
        record = {
            'type': 'execution',
            'callback_id': callback_id,
            'success': success,
            'execution_time': execution_time,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.callback_history.append(record)
        state_manager.set_state(
            f'ui.callbacks.executions.{callback_id}',
            record
        )
    
    def _update_callback_stats(
        self,
        callback_id: str,
        execution_time: float
    ) -> None:
        """Update callback execution statistics."""
        callback_info = self.callbacks[callback_id]
        callback_info['execution_count'] += 1
        callback_info['last_execution'] = datetime.now().isoformat()
        
        # Update moving average of execution time
        n = callback_info['execution_count']
        current_avg = callback_info['average_execution_time']
        callback_info['average_execution_time'] = (
            (current_avg * (n - 1) + execution_time) / n
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