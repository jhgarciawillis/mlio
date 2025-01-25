import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect

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
        self.active_handlers: Dict[str, Dict[str, Any]] = {}
        self.handler_queues: Dict[str, asyncio.Queue] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize handler queues for different event types
        self.initialize_handler_queues()
        
    def initialize_handler_queues(self) -> None:
        """Initialize event queues for different event types."""
        event_types = [
            'state_change',
            'user_interaction',
            'error',
            'data_update',
            'model_update',
            'clustering_update',
            'analysis_update',
            'visualization_update'
        ]
        
        for event_type in event_types:
            self.handler_queues[event_type] = asyncio.Queue()
    
    @monitor_performance
    @handle_exceptions(HandlerError)
    def register_handler(
        self,
        event_type: str,
        handler: Callable,
        handler_name: Optional[str] = None,
        description: Optional[str] = None,
        priority: int = 0
    ) -> None:
        """Register event handler with priority."""
        if handler_name is None:
            handler_name = f"{event_type}_{len(self.active_handlers)}"
        
        if handler_name in self.active_handlers:
            raise HandlerError(f"Handler already registered: {handler_name}")
        
        # Validate handler signature
        sig = inspect.signature(handler)
        if len(sig.parameters) < 1:
            raise HandlerError("Handler must accept at least one parameter (event_data)")
        
        self.active_handlers[handler_name] = {
            'handler': handler,
            'event_type': event_type,
            'description': description or handler.__doc__,
            'priority': priority,
            'registered_at': datetime.now().isoformat()
        }
        
        # Track registration
        self._record_handler_registration(handler_name)
        
        logger.info(f"Registered handler: {handler_name} for event type: {event_type}")
    
    @monitor_performance
    @handle_exceptions(HandlerError)
    async def handle_event(
        self,
        event_type: str,
        event_data: Any,
        priority_threshold: Optional[int] = None
    ) -> None:
        """Handle event asynchronously with priority support."""
        # Get relevant handlers for event type
        relevant_handlers = {
            name: info for name, info in self.active_handlers.items()
            if info['event_type'] == event_type and
            (priority_threshold is None or info['priority'] >= priority_threshold)
        }
        
        if not relevant_handlers:
            logger.warning(f"No handlers registered for event type: {event_type}")
            return
        
        # Sort handlers by priority
        sorted_handlers = sorted(
            relevant_handlers.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )
        
        try:
            # Execute handlers in priority order
            for handler_name, handler_info in sorted_handlers:
                handler = handler_info['handler']
                
                # Execute handler asynchronously
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    lambda: self._execute_handler(handler, event_data, handler_name)
                )
            
            # Add event to queue for processing
            await self.handler_queues[event_type].put({
                'timestamp': datetime.now().isoformat(),
                'event_data': event_data,
                'handlers': [h[0] for h in sorted_handlers]
            })
            
        except Exception as e:
            error_data = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'event_type': event_type,
                'event_data': event_data
            }
            await self.handle_error(error_data)
            raise HandlerError(f"Error handling event: {str(e)}") from e
    
    def _execute_handler(
        self,
        handler: Callable,
        event_data: Any,
        handler_name: str
    ) -> None:
        """Execute individual handler with tracking."""
        try:
            start_time = datetime.now()
            handler(event_data)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self._track_handler_execution(
                handler_name,
                True,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._track_handler_execution(
                handler_name,
                False,
                error=str(e)
            )
            raise
    
    @monitor_performance
    def handle_state_change(
        self,
        state_path: str,
        old_value: Any,
        new_value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle state changes."""
        event_data = {
            'path': state_path,
            'old_value': old_value,
            'new_value': new_value,
            'metadata': metadata or {},
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
        interaction_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle user interactions."""
        event_data = {
            'type': interaction_type,
            'data': interaction_data,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Trigger user interaction event
        asyncio.create_task(
            self.handle_event('user_interaction', event_data)
        )
    
    @monitor_performance
    async def handle_error(
        self,
        error_data: Dict[str, Any]
    ) -> None:
        """Handle errors with appropriate logging and notification."""
        # Add timestamp if not present
        if 'timestamp' not in error_data:
            error_data['timestamp'] = datetime.now().isoformat()
        
        # Log error
        logger.error(
            f"Error in handler: {error_data.get('error_message')}",
            extra={'error_data': error_data}
        )
        
        # Update state with error information
        state_manager.set_state(
            f'errors.handlers.{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            error_data
        )
        
        # Trigger error event
        await self.handle_event('error', error_data)
        
        # Show notification if UI is available
        if st._is_running_with_streamlit:
            st.error(f"Error: {error_data.get('error_message')}")
    
    @monitor_performance
    def create_handler_chain(
        self,
        handlers: List[Tuple[str, Callable]],
        chain_name: str
    ) -> None:
        """Create a chain of handlers to be executed in sequence."""
        for i, (event_type, handler) in enumerate(handlers):
            handler_name = f"{chain_name}_step_{i}"
            self.register_handler(
                event_type,
                handler,
                handler_name=handler_name,
                priority=len(handlers) - i  # Higher priority for earlier handlers
            )
    
    @monitor_performance
    def create_conditional_handler(
        self,
        condition: Callable[[Any], bool],
        true_handler: Callable,
        false_handler: Optional[Callable] = None,
        handler_name: str = "conditional_handler"
    ) -> None:
        """Create a conditional handler based on a condition."""
        def conditional_wrapper(event_data: Any) -> None:
            if condition(event_data):
                true_handler(event_data)
            elif false_handler is not None:
                false_handler(event_data)
        
        self.register_handler(
            'conditional',
            conditional_wrapper,
            handler_name=handler_name
        )
    
    def _track_handler_registration(
        self,
        handler_name: str
    ) -> None:
        """Track handler registration."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'handler_name': handler_name,
            'handler_info': self.active_handlers[handler_name]
        }
        
        self.event_history.append(record)
        state_manager.set_state(
            f'handlers.registrations.{handler_name}',
            record
        )
    
    def _track_handler_execution(
        self,
        handler_name: str,
        success: bool,
        execution_time: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Track handler execution results."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'handler_name': handler_name,
            'success': success,
            'execution_time': execution_time,
            'error': error
        }
        
        self.event_history.append(record)
        state_manager.set_state(
            f'handlers.executions.{handler_name}',
            record
        )
    
    @monitor_performance
    async def process_event_queues(self) -> None:
        """Process event queues asynchronously."""
        while True:
            # Process each event type queue
            for event_type, queue in self.handler_queues.items():
                try:
                    while not queue.empty():
                        event = await queue.get()
                        logger.debug(f"Processing queued {event_type} event: {event}")
                        
                        # Process the event
                        await self.handle_event(
                            event_type,
                            event['event_data']
                        )
                        
                        queue.task_done()
                        
                except Exception as e:
                    logger.error(f"Error processing {event_type} queue: {str(e)}")
                    
            # Small delay to prevent CPU overuse
            await asyncio.sleep(0.1)
    
    @monitor_performance
    def get_handler_status(
        self,
        handler_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get status of handlers."""
        if handler_name:
            return {
                'handler_info': self.active_handlers.get(handler_name),
                'history': [
                    record for record in self.event_history
                    if record.get('handler_name') == handler_name
                ]
            }
        
        return {
            'active_handlers': len(self.active_handlers),
            'event_types': list(self.handler_queues.keys()),
            'queue_sizes': {
                event_type: queue.qsize()
                for event_type, queue in self.handler_queues.items()
            },
            'recent_events': self.event_history[-10:]
        }

# Create global event handler instance
event_handler = EventHandler()

# Start event queue processing
asyncio.create_task(event_handler.process_event_queues())