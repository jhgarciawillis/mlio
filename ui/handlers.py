import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect

from core import config
from core.exceptions import HandlerError
from core.state_manager import state_manager
from core.state_monitoring import state_monitor
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from ui.callbacks import callback_manager
from clustering import clusterer, cluster_optimizer, cluster_validator

class EventHandler:
    """Handle UI events and state changes with enhanced clustering support."""
    
    def __init__(self):
        self.event_history: List[Dict[str, Any]] = []
        self.active_handlers: Dict[str, Dict[str, Any]] = {}
        self.handler_queues: Dict[str, asyncio.Queue] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.performance.max_workers)
        
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
            'clustering_update',  # Added for clustering support
            'analysis_update',
            'visualization_update',
            'performance_alert',  # Added for performance monitoring
            'feature_engineering_update',  # Added for feature engineering
            'metrics_update'  # Added for metrics updates
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
            # Start performance monitoring for this event
            state_monitor.record_operation_start(
                f"event_handling_{event_type}",
                "event_processing",
                {"handler_count": len(sorted_handlers)}
            )
            
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
            
            # Complete performance monitoring for this event
            state_monitor.record_operation_end(
                f"event_handling_{event_type}",
                "completed",
                {"handler_count": len(sorted_handlers)}
            )
            
        except Exception as e:
            error_data = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'event_type': event_type,
                'event_data': event_data
            }
            state_monitor.record_operation_end(
                f"event_handling_{event_type}",
                "failed",
                error_data
            )
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
            # Track operation start
            state_monitor.record_operation_start(
                f"handler_execution_{handler_name}",
                "event_handler",
                {"handler": handler_name}
            )
            
            start_time = datetime.now()
            handler(event_data)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self._track_handler_execution(
                handler_name,
                True,
                execution_time=execution_time
            )
            
            # Track operation completion
            state_monitor.record_operation_end(
                f"handler_execution_{handler_name}",
                "completed",
                {"execution_time": execution_time}
            )
            
        except Exception as e:
            self._track_handler_execution(
                handler_name,
                False,
                error=str(e)
            )
            
            # Track operation failure
            state_monitor.record_operation_end(
                f"handler_execution_{handler_name}",
                "failed",
                {"error": str(e)}
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
    def handle_clustering_update(
        self,
        update_type: str,
        clustering_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle clustering updates."""
        event_data = {
            'type': update_type,
            'data': clustering_data,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Trigger clustering update event
        asyncio.create_task(
            self.handle_event('clustering_update', event_data)
        )
    
    @monitor_performance
    def handle_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle performance monitoring alerts."""
        event_data = {
            'metric': metric_name,
            'value': current_value,
            'threshold': threshold,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Trigger performance alert event
        asyncio.create_task(
            self.handle_event('performance_alert', event_data, priority=10)  # High priority
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
        
        # Track in state monitoring
        state_monitor.record_error(
            error_data.get('error_type', 'Unknown'),
            error_data.get('error_message', 'Unknown error'),
            error_data
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
    
    @monitor_performance
    def create_clustering_handlers(self) -> None:
        """Create clustering-specific event handlers."""
        # Register handler for cluster creation
        self.register_handler(
            'clustering_update',
            self._handle_cluster_creation,
            handler_name='cluster_creation_handler',
            description='Handles cluster creation events',
            priority=10
        )
        
        # Register handler for cluster optimization
        self.register_handler(
            'clustering_update',
            self._handle_cluster_optimization,
            handler_name='cluster_optimization_handler',
            description='Handles cluster optimization events',
            priority=8
        )
        
        # Register handler for cluster validation
        self.register_handler(
            'clustering_update',
            self._handle_cluster_validation,
            handler_name='cluster_validation_handler',
            description='Handles cluster validation events',
            priority=6
        )
        
        # Register handler for cluster visualization
        self.register_handler(
            'clustering_update',
            self._handle_cluster_visualization,
            handler_name='cluster_visualization_handler',
            description='Handles cluster visualization events',
            priority=4
        )
    
    def _handle_cluster_creation(self, event_data: Dict[str, Any]) -> None:
        """Handle cluster creation events."""
        if event_data.get('type') != 'creation':
            return
        
        try:
            cluster_data = event_data.get('data', {})
            method = cluster_data.get('method', 'kmeans')
            params = cluster_data.get('params', {})
            
            # Update UI with cluster creation status
            if st._is_running_with_streamlit:
                with st.spinner(f"Creating clusters with {method}..."):
                    clusterer.cluster_data(
                        cluster_data.get('data'),
                        method=method,
                        params=params
                    )
                st.success(f"Clusters created successfully with {method}")
        
        except Exception as e:
            logger.error(f"Error in cluster creation: {str(e)}")
            if st._is_running_with_streamlit:
                st.error(f"Error creating clusters: {str(e)}")
    
    def _handle_cluster_optimization(self, event_data: Dict[str, Any]) -> None:
        """Handle cluster optimization events."""
        if event_data.get('type') != 'optimization':
            return
        
        try:
            optimization_data = event_data.get('data', {})
            method = optimization_data.get('method', 'kmeans')
            metric = optimization_data.get('metric', 'silhouette')
            
            # Update UI with optimization status
            if st._is_running_with_streamlit:
                with st.spinner(f"Optimizing clusters with {method}..."):
                    results = cluster_optimizer.optimize_clustering(
                        optimization_data.get('data'),
                        method=method,
                        metric=metric,
                        n_trials=optimization_data.get('n_trials', 50)
                    )
                st.success(f"Cluster optimization complete. Best score: {results['best_score']:.4f}")
        
        except Exception as e:
            logger.error(f"Error in cluster optimization: {str(e)}")
            if st._is_running_with_streamlit:
                st.error(f"Error optimizing clusters: {str(e)}")
    
    def _handle_cluster_validation(self, event_data: Dict[str, Any]) -> None:
        """Handle cluster validation events."""
        if event_data.get('type') != 'validation':
            return
        
        try:
            validation_data = event_data.get('data', {})
            
            # Update UI with validation status
            if st._is_running_with_streamlit:
                with st.spinner("Validating clusters..."):
                    results = cluster_validator.validate_clustering(
                        validation_data.get('data'),
                        validation_data.get('labels')
                    )
                
                st.success("Cluster validation complete")
                
                # Display validation metrics
                if 'internal' in results:
                    st.subheader("Validation Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': list(results['internal'].keys()),
                        'Value': list(results['internal'].values())
                    })
                    st.dataframe(metrics_df)
        
        except Exception as e:
            logger.error(f"Error in cluster validation: {str(e)}")
            if st._is_running_with_streamlit:
                st.error(f"Error validating clusters: {str(e)}")
    
    def _handle_cluster_visualization(self, event_data: Dict[str, Any]) -> None:
        """Handle cluster visualization events."""
        if event_data.get('type') != 'visualization':
            return
        
        try:
            visualization_data = event_data.get('data', {})
            
            # Update UI with visualization
            if st._is_running_with_streamlit:
                if 'visualizations' in visualization_data:
                    for name, fig in visualization_data['visualizations'].items():
                        st.subheader(f"Cluster Visualization: {name}")
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error in cluster visualization: {str(e)}")
            if st._is_running_with_streamlit:
                st.error(f"Error visualizing clusters: {str(e)}")
    
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
    
    @monitor_performance
    def register_model_handlers(self) -> None:
        """Register handlers for model-related events."""
        # Training completion handler
        self.register_handler(
            'model_update',
            self._handle_model_training_completion,
            handler_name='model_training_completion_handler',
            description='Handles model training completion events',
            priority=10
        )
        
        # Model validation handler
        self.register_handler(
            'model_update',
            self._handle_model_validation,
            handler_name='model_validation_handler',
            description='Handles model validation events',
            priority=8
        )
        
        # Prediction handler
        self.register_handler(
            'model_update',
            self._handle_model_prediction,
            handler_name='model_prediction_handler',
            description='Handles model prediction events',
            priority=6
        )
    
    def _handle_model_training_completion(self, event_data: Dict[str, Any]) -> None:
        """Handle model training completion events."""
        if event_data.get('type') != 'training_complete':
            return
        
        try:
            training_data = event_data.get('data', {})
            model_name = training_data.get('model_name', 'unknown')
            metrics = training_data.get('metrics', {})
            
            # Update UI with training completion
            if st._is_running_with_streamlit:
                st.success(f"Model {model_name} training completed")
                
                # Display metrics
                st.subheader("Training Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': list(metrics.values())
                })
                st.dataframe(metrics_df)
        
        except Exception as e:
            logger.error(f"Error handling model training completion: {str(e)}")
            if st._is_running_with_streamlit:
                st.error(f"Error handling training completion: {str(e)}")
    
    def _handle_model_validation(self, event_data: Dict[str, Any]) -> None:
        """Handle model validation events."""
        if event_data.get('type') != 'validation':
            return
        
        try:
            validation_data = event_data.get('data', {})
            model_name = validation_data.get('model_name', 'unknown')
            validation_results = validation_data.get('results', {})
            
            # Update UI with validation results
            if st._is_running_with_streamlit:
                st.success(f"Model {model_name} validation completed")
                
                # Display validation results
                if 'basic' in validation_results:
                    st.subheader("Validation Results")
                    cv_scores = validation_results['basic'].get('cv_scores', {})
                    st.write(f"Mean CV Score: {cv_scores.get('mean', 0):.4f}")
                    st.write(f"Standard Deviation: {cv_scores.get('std', 0):.4f}")
        
        except Exception as e:
            logger.error(f"Error handling model validation: {str(e)}")
            if st._is_running_with_streamlit:
                st.error(f"Error handling model validation: {str(e)}")
    
    def _handle_model_prediction(self, event_data: Dict[str, Any]) -> None:
        """Handle model prediction events."""
        if event_data.get('type') != 'prediction':
            return
        
        try:
            prediction_data = event_data.get('data', {})
            model_name = prediction_data.get('model_name', 'unknown')
            predictions = prediction_data.get('predictions', [])
            
            # Update UI with predictions
            if st._is_running_with_streamlit:
                st.success(f"Model {model_name} predictions completed")
                
                # Display predictions
                if len(predictions) > 0:
                    st.subheader("Predictions")
                    predictions_df = pd.DataFrame({
                        'Index': range(len(predictions)),
                        'Prediction': predictions
                    })
                    st.dataframe(predictions_df)
        
        except Exception as e:
            logger.error(f"Error handling model prediction: {str(e)}")
            if st._is_running_with_streamlit:
                st.error(f"Error handling model prediction: {str(e)}")

# Create global event handler instance
event_handler = EventHandler()

# Register clustering handlers
event_handler.create_clustering_handlers()

# Register model handlers
event_handler.register_model_handlers()

# Start event queue processing
asyncio.create_task(event_handler.process_event_queues())