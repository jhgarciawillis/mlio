import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
import joblib

from core import config
from core.exceptions import ClusteringError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from clustering import clusterer, optimizer, validator
from clustering.algorithms import *

class ClusterManager:
    """Centralize clustering operations management."""
    
    def __init__(self):
        self.clustering_operations: Dict[str, Dict[str, Any]] = {}
        self.active_models: Dict[str, Any] = {}
        self.operation_history: List[Dict[str, Any]] = []
        self.clustering_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize components
        self.clusterer = clusterer
        self.optimizer = cluster_optimizer
        self.validator = cluster_validator
        
        # Available algorithms
        self.algorithms = {
            'kmeans': CustomKMeans,
            'dbscan': CustomDBSCAN,
            'gaussian_mixture': CustomGaussianMixture
        }
    
    @monitor_performance
    @handle_exceptions(ClusteringError)
    def create_clustering_operation(
        self,
        name: str,
        data: pd.DataFrame,
        algorithm: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new clustering operation."""
        if name in self.clustering_operations:
            raise ClusteringError(f"Operation name already exists: {name}")
            
        if algorithm not in self.algorithms:
            raise ClusteringError(f"Unknown algorithm: {algorithm}")
            
        operation_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        operation = {
            'data': data,
            'algorithm': algorithm,
            'config': config or {},
            'status': 'created',
            'results': None,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        self.clustering_operations[operation_id] = operation
        self._record_operation('creation', operation_id)
        
        return operation_id
    
    @monitor_performance
    def execute_clustering(
        self,
        operation_id: str,
        optimize: bool = False,
        validate: bool = True
    ) -> Dict[str, Any]:
        """Execute clustering operation."""
        if operation_id not in self.clustering_operations:
            raise ClusteringError(f"Operation not found: {operation_id}")
            
        operation = self.clustering_operations[operation_id]
        operation['status'] = 'running'
        
        try:
            # Get algorithm and configuration
            algorithm_class = self.algorithms[operation['algorithm']]
            config = operation['config']
            
            # Optimize if requested
            if optimize:
                best_config = self.optimizer.optimize_clustering(
                    operation['data'],
                    operation['algorithm'],
                    config
                )
                config.update(best_config)
            
            # Perform clustering
            model = algorithm_class(**config)
            results = self.clusterer.cluster_data(
                operation['data'],
                model=model,
                algorithm=operation['algorithm']
            )
            
            # Validate if requested
            if validate:
                validation_results = self.validator.validate_clustering(
                    operation['data'],
                    results['labels'],
                    operation['algorithm']
                )
                results['validation'] = validation_results
            
            # Store results
            operation['results'] = results
            operation['status'] = 'completed'
            operation['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Store model
            self.active_models[operation_id] = model
            
            self._record_operation('execution', operation_id)
            
            return results
            
        except Exception as e:
            operation['status'] = 'failed'
            operation['metadata']['error'] = str(e)
            self._record_operation('failure', operation_id)
            raise
    
    @monitor_performance
    def get_operation_status(
        self,
        operation_id: str
    ) -> Dict[str, Any]:
        """Get status of clustering operation."""
        if operation_id not in self.clustering_operations:
            raise ClusteringError(f"Operation not found: {operation_id}")
            
        operation = self.clustering_operations[operation_id]
        return {
            'status': operation['status'],
            'metadata': operation['metadata']
        }
    
    @monitor_performance
    def get_clustering_results(
        self,
        operation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get results of clustering operation."""
        if operation_id not in self.clustering_operations:
            raise ClusteringError(f"Operation not found: {operation_id}")
            
        return self.clustering_operations[operation_id].get('results')
    
    @monitor_performance
    def save_model(
        self,
        operation_id: str,
        path: Optional[Path] = None
    ) -> None:
        """Save clustering model."""
        if operation_id not in self.active_models:
            raise ClusteringError(f"No active model for operation: {operation_id}")
            
        if path is None:
            path = config.directories.models / 'clustering'
            path.mkdir(parents=True, exist_ok=True)
        
        model_path = path / f"{operation_id}_model.joblib"
        config_path = path / f"{operation_id}_config.joblib"
        
        # Save model and configuration
        joblib.dump(self.active_models[operation_id], model_path)
        joblib.dump(
            self.clustering_operations[operation_id]['config'],
            config_path
        )
        
        logger.info(f"Model saved to {model_path}")
    
    @monitor_performance
    def load_model(
        self,
        path: Path,
        operation_id: Optional[str] = None
    ) -> str:
        """Load clustering model."""
        if not path.exists():
            raise ClusteringError(f"Model path not found: {path}")
            
        # Generate operation ID if not provided
        if operation_id is None:
            operation_id = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Load model and configuration
        model = joblib.load(path)
        config_path = path.parent / f"{path.stem.replace('_model', '_config')}.joblib"
        
        if config_path.exists():
            config = joblib.load(config_path)
        else:
            config = {}
        
        # Create operation entry
        self.clustering_operations[operation_id] = {
            'data': None,
            'algorithm': type(model).__name__.lower(),
            'config': config,
            'status': 'loaded',
            'results': None,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'loaded_from': str(path)
            }
        }
        
        # Store model
        self.active_models[operation_id] = model
        
        self._record_operation('loading', operation_id)
        
        return operation_id
    
    def _record_operation(
        self,
        operation_type: str,
        operation_id: str
    ) -> None:
        """Record clustering operation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'operation_id': operation_id
        }
        
        self.operation_history.append(record)
        state_manager.set_state(
            f'clustering.manager.history.{len(self.operation_history)}',
            record
        )

# Create global cluster manager instance
cluster_manager = ClusterManager()