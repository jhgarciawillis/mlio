import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import inspect
import asyncio
import functools

from core import config
from core.exceptions import CallbackError
from core.state_manager import state_manager
from core.state_monitoring import state_monitor
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from clustering import clusterer, cluster_optimizer, cluster_validator
from metrics import calculator, evaluator, reporter

class CallbackManager:
    """Handle UI callbacks and event handling with enhanced clustering support."""
    
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
        
        # Monitor registration
        state_monitor.record_operation_start(
            f"callback_registration_{callback_id}",
            "callback_registration",
            {"callback_id": callback_id, "priority": priority}
        )
        state_monitor.record_operation_end(
            f"callback_registration_{callback_id}",
            "completed",
            {"callback_id": callback_id}
        )
    
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
            # Monitor execution
            state_monitor.record_operation_start(
                f"callback_execution_{callback_id}",
                "callback_execution",
                {"callback_id": callback_id}
            )
            
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
            
            # Monitor execution completion
            state_monitor.record_operation_end(
                f"callback_execution_{callback_id}",
                "completed",
                {"execution_time": execution_time}
            )
            
            return result
            
        except Exception as e:
            # Track failed execution
            self._track_callback_execution(callback_id, False, error=str(e))
            
            # Monitor execution failure
            state_monitor.record_operation_end(
                f"callback_execution_{callback_id}",
                "failed",
                {"error": str(e)}
            )
            
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
        
        # Monitor registration
        state_monitor.record_operation_start(
            f"event_handler_registration_{event_type}",
            "event_handler_registration",
            {"event_type": event_type, "priority": priority}
        )
        state_monitor.record_operation_end(
            f"event_handler_registration_{event_type}",
            "completed",
            {"event_type": event_type}
        )
    
    @monitor_performance
    def trigger_event(
        self,
        event_type: str,
        event_data: Any
    ) -> None:
        """Trigger event and execute registered handlers."""
        if event_type in self.event_handlers:
            # Monitor event triggering
            state_monitor.record_operation_start(
                f"event_trigger_{event_type}",
                "event_triggering",
                {"event_type": event_type}
            )
            
            executed_handlers = 0
            for handler_info in self.event_handlers[event_type]:
                try:
                    handler_info['handler'](event_data)
                    executed_handlers += 1
                except Exception as e:
                    logger.error(f"Error in event handler: {str(e)}")
                    state_monitor.record_error(
                        "EventHandlerError",
                        str(e),
                        {"event_type": event_type, "event_data": event_data}
                    )
            
            # Monitor event completion
            state_monitor.record_operation_end(
                f"event_trigger_{event_type}",
                "completed",
                {"handlers_executed": executed_handlers}
            )
    
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
                # Monitor button callback
                state_monitor.record_operation_start(
                    f"button_callback_{key or label}",
                    "button_callback",
                    {"label": label, "key": key}
                )
                
                result = callback_func(**kwargs)
                
                # Monitor completion
                state_monitor.record_operation_end(
                    f"button_callback_{key or label}",
                    "completed",
                    {"label": label, "key": key}
                )
                
                return result
            except Exception as e:
                # Monitor failure
                state_monitor.record_operation_end(
                    f"button_callback_{key or label}",
                    "failed",
                    {"error": str(e)}
                )
                
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
            # Monitor selectbox callback
            state_monitor.record_operation_start(
                f"selectbox_callback_{key or label}",
                "selectbox_callback",
                {"label": label, "key": key}
            )
            
            result = callback_func(selected, **kwargs)
            
            # Monitor completion
            state_monitor.record_operation_end(
                f"selectbox_callback_{key or label}",
                "completed",
                {"label": label, "key": key}
            )
            
            return selected
        except Exception as e:
            # Monitor failure
            state_monitor.record_operation_end(
                f"selectbox_callback_{key or label}",
                "failed",
                {"error": str(e)}
            )
            
            st.error(f"Error executing callback: {str(e)}")
            return selected
    
    @monitor_performance
    def create_clustering_callbacks(self) -> None:
        """Register callbacks for clustering operations."""
        # Cluster creation callback
        self.register_callback(
            "clustering_create",
            self._clustering_create_callback,
            "Create clusters based on configuration",
            priority=10
        )
        
        # Cluster optimization callback
        self.register_callback(
            "clustering_optimize",
            self._clustering_optimize_callback,
            "Optimize clustering parameters",
            priority=8
        )
        
        # Cluster validation callback
        self.register_callback(
            "clustering_validate",
            self._clustering_validate_callback,
            "Validate clustering results",
            priority=6
        )
        
        # Cluster visualization callback
        self.register_callback(
            "clustering_visualize",
            self._clustering_visualize_callback,
            "Visualize clustering results",
            priority=4
        )
    
    def _clustering_create_callback(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Callback for cluster creation."""
        try:
            # Monitor clustering operation
            state_monitor.record_operation_start(
                "clustering_creation",
                "clustering",
                {"method": config.get("method", "kmeans")}
            )
            
            # Create clusters
            results = clusterer.cluster_data(
                data,
                method=config.get("method", "kmeans"),
                params=config.get("params", {}),
                scale_data=config.get("scale_data", True)
            )
            
            # Monitor completion
            state_monitor.record_operation_end(
                "clustering_creation",
                "completed",
                {"n_clusters": results.get("n_clusters", 0)}
            )
            
            return results
        
        except Exception as e:
            # Monitor failure
            state_monitor.record_operation_end(
                "clustering_creation",
                "failed",
                {"error": str(e)}
            )
            
            raise CallbackError(f"Error creating clusters: {str(e)}") from e
    
    def _clustering_optimize_callback(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Callback for cluster optimization."""
        try:
            # Monitor optimization operation
            state_monitor.record_operation_start(
                "clustering_optimization",
                "clustering",
                {"method": config.get("method", "kmeans")}
            )
            
            # Optimize clustering
            results = cluster_optimizer.optimize_clustering(
                data,
                method=config.get("method", "kmeans"),
                metric=config.get("metric", "silhouette"),
                n_trials=config.get("n_trials", 50),
                cv_folds=config.get("cv_folds", 5),
                param_ranges=config.get("param_ranges", None),
                random_state=config.get("random_state", None)
            )
            
            # Monitor completion
            state_monitor.record_operation_end(
                "clustering_optimization",
                "completed",
                {"best_score": results.get("best_score", 0)}
            )
            
            return results
        
        except Exception as e:
            # Monitor failure
            state_monitor.record_operation_end(
                "clustering_optimization",
                "failed",
                {"error": str(e)}
            )
            
            raise CallbackError(f"Error optimizing clusters: {str(e)}") from e
    
    def _clustering_validate_callback(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Callback for cluster validation."""
        try:
            # Monitor validation operation
            state_monitor.record_operation_start(
                "clustering_validation",
                "clustering",
                {"n_clusters": len(np.unique(labels))}
            )
            
            # Validate clustering
            results = cluster_validator.validate_clustering(
                data,
                labels,
                config
            )
            
            # Monitor completion
            state_monitor.record_operation_end(
                "clustering_validation",
                "completed",
                {"validation_metrics": list(results.get("internal", {}).keys())}
            )
            
            return results
        
        except Exception as e:
            # Monitor failure
            state_monitor.record_operation_end(
                "clustering_validation",
                "failed",
                {"error": str(e)}
            )
            
            raise CallbackError(f"Error validating clusters: {str(e)}") from e
    
    def _clustering_visualize_callback(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Callback for cluster visualization."""
        try:
            # Monitor visualization operation
            state_monitor.record_operation_start(
                "clustering_visualization",
                "clustering",
                {"n_clusters": len(np.unique(labels))}
            )
            
            # Create visualizations
            visualizations = {}
            
            # Add 2D scatter plot
            from sklearn.decomposition import PCA
            if data.shape[1] > 2:
                pca = PCA(n_components=2)
                coords = pca.fit_transform(data)
            else:
                coords = data.values
            
            scatter_data = pd.DataFrame(
                coords,
                columns=['Component 1', 'Component 2']
            )
            scatter_data['Cluster'] = labels
            
            from visualization import plotter
            visualizations['scatter_2d'] = plotter.create_plot(
                'scatter',
                data=scatter_data,
                x='Component 1',
                y='Component 2',
                color='Cluster',
                title='Cluster Assignments (2D Projection)'
            )
            
            # Add cluster sizes plot
            unique_labels, counts = np.unique(labels, return_counts=True)
            size_data = pd.DataFrame({
                'Cluster': unique_labels,
                'Size': counts
            })
            
            visualizations['cluster_sizes'] = plotter.create_plot(
                'bar',
                data=size_data,
                x='Cluster',
                y='Size',
                title='Cluster Sizes'
            )
            
            # Monitor completion
            state_monitor.record_operation_end(
                "clustering_visualization",
                "completed",
                {"visualizations": list(visualizations.keys())}
            )
            
            return {"visualizations": visualizations}
        
        except Exception as e:
            # Monitor failure
            state_monitor.record_operation_end(
                "clustering_visualization",
                "failed",
                {"error": str(e)}
            )
            
            raise CallbackError(f"Error visualizing clusters: {str(e)}") from e
    
    @monitor_performance
    def create_metrics_callbacks(self) -> None:
        """Register callbacks for metrics operations."""
        # Metrics calculation callback
        self.register_callback(
            "metrics_calculate",
            self._metrics_calculate_callback,
            "Calculate metrics for predictions",
            priority=8
        )
        
        # Metrics evaluation callback
        self.register_callback(
            "metrics_evaluate",
            self._metrics_evaluate_callback,
            "Evaluate metrics in detail",
            priority=6
        )
        
        # Metrics reporting callback
        self.register_callback(
            "metrics_report",
            self._metrics_report_callback,
            "Generate metrics report",
            priority=4
        )
    
    def _metrics_calculate_callback(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Callback for metrics calculation."""
        try:
            # Monitor metrics calculation
            state_monitor.record_operation_start(
                "metrics_calculation",
                "metrics",
                {"metrics": metrics}
            )
            
            # Calculate metrics
            results = calculator.calculate_metrics(
                y_true,
                y_pred,
                metrics=metrics
            )
            
            # Monitor completion
            state_monitor.record_operation_end(
                "metrics_calculation",
                "completed",
                {"metrics_calculated": list(results.keys())}
            )
            
            return results
        
        except Exception as e:
            # Monitor failure
            state_monitor.record_operation_end(
                "metrics_calculation",
                "failed",
                {"error": str(e)}
            )
            
            raise CallbackError(f"Error calculating metrics: {str(e)}") from e
    
    def _metrics_evaluate_callback(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        cluster_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Callback for metrics evaluation."""
        try:
            # Monitor metrics evaluation
            state_monitor.record_operation_start(
                "metrics_evaluation",
                "metrics",
                {"clustered": cluster_labels is not None}
            )
            
            # Evaluate metrics
            results = evaluator.evaluate_metrics(
                y_true,
                y_pred,
                cluster_labels=cluster_labels
            )
            
            # Monitor completion
            state_monitor.record_operation_end(
                "metrics_evaluation",
                "completed",
                {"evaluation_aspects": list(results.keys())}
            )
            
            return results
        
        except Exception as e:
            # Monitor failure
            state_monitor.record_operation_end(
                "metrics_evaluation",
                "failed",
                {"error": str(e)}
            )
            
            raise CallbackError(f"Error evaluating metrics: {str(e)}") from e
    
    def _metrics_report_callback(
        self,
        metrics_data: Dict[str, Any],
        report_config: Optional[Dict[str, Any]] = None,
        cluster_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Callback for metrics reporting."""
        try:
            # Monitor report generation
            state_monitor.record_operation_start(
                "metrics_report_generation",
                "metrics",
                {"report_sections": report_config.get("sections", []) if report_config else None}
            )
            
            # Generate report
            report = reporter.generate_report(
                metrics_data,
                report_config,
                cluster_info
            )
            
            # Monitor completion
            state_monitor.record_operation_end(
                "metrics_report_generation",
                "completed",
                {"report_sections": list(report.get("sections", {}).keys())}
            )
            
            return report
        
        except Exception as e:
            # Monitor failure
            state_monitor.record_operation_end(
                "metrics_report_generation",
                "failed",
                {"error": str(e)}
            )
            
            raise CallbackError(f"Error generating metrics report: {str(e)}") from e
    
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

# Register clustering callbacks
callback_manager.create_clustering_callbacks()

# Register metrics callbacks
callback_manager.create_metrics_callbacks()