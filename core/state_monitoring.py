import psutil
import time
import asyncio
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from core import config
from core.exceptions import StateError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class StateMonitor:
    """Monitor and track application state changes and performance."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.state_changes: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'response_time': [],
            'operation_count': []
        }
        self.error_counts: Dict[str, int] = {}
        self.active_operations: Dict[str, datetime] = {}
        self.monitor_thread: Optional[asyncio.Task] = None
        self.should_monitor = False
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': config.monitoring.cpu_threshold,
            'memory_usage': config.monitoring.memory_threshold,
            'response_time': config.monitoring.response_time_threshold
        }
        
        # Initialize monitoring
        self.start_monitoring()
    
    @monitor_performance
    def start_monitoring(self) -> None:
        """Start monitoring thread."""
        if not self.monitor_thread or self.monitor_thread.done():
            self.should_monitor = True
            self.monitor_thread = asyncio.create_task(self._monitoring_loop())
            logger.info("State monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring thread."""
        self.should_monitor = False
        if self.monitor_thread and not self.monitor_thread.done():
            self.monitor_thread.cancel()
            logger.info("State monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.should_monitor:
            try:
                # Collect current metrics
                await self._collect_metrics()
                
                # Check thresholds
                await self._check_thresholds()
                
                # Check for long-running operations
                await self._check_long_running_operations()
                
                # Sleep for monitoring interval
                await asyncio.sleep(config.monitoring.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(config.monitoring.sampling_interval)
    
    @monitor_performance
    async def _collect_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.performance_metrics['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.performance_metrics['memory_usage'].append(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.performance_metrics['disk_usage'].append(disk.percent)
            
            # Operation metrics
            self.performance_metrics['operation_count'].append(len(self.active_operations))
            
            # Calculate response time
            if self.active_operations:
                response_times = [
                    (datetime.now() - start_time).total_seconds()
                    for start_time in self.active_operations.values()
                ]
                avg_response_time = np.mean(response_times)
                self.performance_metrics['response_time'].append(avg_response_time)
            
            # Trim metrics lists if too long
            max_metrics = 1000  # Store last 1000 measurements
            for metric_list in self.performance_metrics.values():
                if len(metric_list) > max_metrics:
                    metric_list.pop(0)
            
            # Update state
            state_manager.set_state('monitoring.metrics', self.get_metrics_summary())
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
    
    async def _check_thresholds(self) -> None:
        """Check if metrics exceed thresholds."""
        for metric, threshold in self.thresholds.items():
            if self.performance_metrics[metric]:
                current_value = self.performance_metrics[metric][-1]
                if current_value > threshold:
                    logger.warning(
                        f"{metric} exceeds threshold: {current_value:.2f}% > {threshold}%"
                    )
                    
                    # Update state with warning
                    state_manager.set_state(
                        f'monitoring.warnings.{metric}',
                        {
                            'timestamp': datetime.now().isoformat(),
                            'value': current_value,
                            'threshold': threshold
                        }
                    )
    
    async def _check_long_running_operations(self) -> None:
        """Check for operations that might be running too long."""
        current_time = datetime.now()
        timeout = config.monitoring.response_time_threshold
        
        for op_name, start_time in list(self.active_operations.items()):
            duration = (current_time - start_time).total_seconds()
            if duration > timeout:
                logger.warning(f"Long-running operation detected: {op_name}")
                
                # Update state with warning
                state_manager.set_state(
                    f'monitoring.long_running.{op_name}',
                    {
                        'timestamp': current_time.isoformat(),
                        'duration': duration,
                        'timeout': timeout
                    }
                )
    
    @monitor_performance
    def record_operation_start(
        self,
        operation_name: str,
        operation_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record start of an operation."""
        self.active_operations[operation_name] = datetime.now()
        
        # Record in state changes
        state_change = {
            'operation': operation_name,
            'type': operation_type,
            'status': 'started',
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.state_changes.append(state_change)
        
        # Update state
        state_manager.set_state(
            f'monitoring.operations.{operation_name}',
            state_change
        )
    
    @monitor_performance
    def record_operation_end(
        self,
        operation_name: str,
        status: str = 'completed',
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record end of an operation."""
        if operation_name in self.active_operations:
            start_time = self.active_operations.pop(operation_name)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record in state changes
            state_change = {
                'operation': operation_name,
                'status': status,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'result': result or {}
            }
            self.state_changes.append(state_change)
            
            # Update state
            state_manager.set_state(
                f'monitoring.operations.{operation_name}',
                state_change
            )
    
    @monitor_performance
    def record_error(
        self,
        error_type: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record error occurrence."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_record = {
            'type': error_type,
            'message': error_message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        # Update state with error
        state_manager.set_state(
            f'monitoring.errors.{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            error_record
        )
    
    @monitor_performance
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[metric_name] = {
                    'current': float(values[-1]),
                    'mean': float(np.mean(values)),
                    'max': float(np.max(values)),
                    'min': float(np.min(values)),
                    'std': float(np.std(values)) if len(values) > 1 else 0.0
                }
        return summary
    
    @monitor_performance
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of state changes."""
        return {
            'total_changes': len(self.state_changes),
            'active_operations': len(self.active_operations),
            'error_counts': self.error_counts,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'last_state_change': self.state_changes[-1] if self.state_changes else None
        }
    
    @monitor_performance
    def save_monitoring_data(
        self,
        path: Optional[Path] = None
    ) -> None:
        """Save monitoring data to disk."""
        if path is None:
            path = config.directories.base_dir / 'monitoring'
        
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics
        metrics_df = pd.DataFrame(self.performance_metrics)
        metrics_df.to_csv(path / f'metrics_{timestamp}.csv', index=False)
        
        # Save state changes
        pd.DataFrame(self.state_changes).to_csv(
            path / f'state_changes_{timestamp}.csv',
            index=False
        )
        
        # Save error counts
        pd.DataFrame(
            [{'error_type': k, 'count': v} for k, v in self.error_counts.items()]
        ).to_csv(path / f'error_counts_{timestamp}.csv', index=False)
        
        logger.info(f"Monitoring data saved to {path}")

# Create global state monitor instance
state_monitor = StateMonitor()