import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import joblib

from core import config
from core.exceptions import PredictionError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from metrics import calculator
from clustering import clusterer

class ModelPredictor:
    """Handle prediction operations with clustering support."""
    
    def __init__(self):
        self.prediction_history: List[Dict[str, Any]] = []
        self.prediction_results: Dict[str, pd.DataFrame] = {}
        self.prediction_metadata: Dict[str, Dict[str, Any]] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.cluster_models: Dict[str, Dict[str, Any]] = {}
        self.preprocessors: Dict[str, Any] = {}
        
    @monitor_performance
    @handle_exceptions(PredictionError)
    def predict(
        self,
        model: Any,
        X: pd.DataFrame,
        cluster_labels: Optional[np.ndarray] = None,
        preprocessor: Optional[Any] = None,
        prediction_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions using specified model with clustering support."""
        try:
            if prediction_id is None:
                prediction_id = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Preprocess data if preprocessor provided
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X
            
            # Make predictions based on clustering
            if cluster_labels is not None and prediction_id in self.cluster_models:
                predictions = self._predict_with_clusters(
                    X_processed,
                    cluster_labels,
                    prediction_id
                )
            else:
                predictions = model.predict(X_processed)
            
            # Create metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'input_shape': X.shape,
                'feature_names': list(X.columns),
                'preprocessor_used': preprocessor is not None,
                'clustering_used': cluster_labels is not None,
                'prediction_statistics': self._calculate_prediction_stats(predictions)
            }
            
            if cluster_labels is not None:
                metadata['cluster_statistics'] = self._calculate_cluster_stats(
                    predictions,
                    cluster_labels
                )
            
            # Store results
            self.prediction_results[prediction_id] = pd.DataFrame({
                'predictions': predictions,
                'cluster_labels': cluster_labels if cluster_labels is not None else [-1] * len(predictions)
            })
            self.prediction_metadata[prediction_id] = metadata
            
            # Record prediction
            self._record_prediction(prediction_id, metadata)
            
            return predictions, metadata
            
        except Exception as e:
            raise PredictionError(f"Error making predictions: {str(e)}") from e
    
    @monitor_performance
    def _predict_with_clusters(
        self,
        X: pd.DataFrame,
        cluster_labels: np.ndarray,
        prediction_id: str
    ) -> np.ndarray:
        """Make predictions using cluster-specific models."""
        predictions = np.zeros(len(X))
        cluster_models = self.cluster_models[prediction_id]
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id in cluster_models:
                mask = cluster_labels == cluster_id
                predictions[mask] = cluster_models[cluster_id].predict(X[mask])
        
        return predictions
    
    @monitor_performance
    @handle_exceptions(PredictionError)
    def predict_batch(
        self,
        model: Any,
        X: pd.DataFrame,
        batch_size: int = 1000,
        cluster_labels: Optional[np.ndarray] = None,
        preprocessor: Optional[Any] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions in batches with clustering support."""
        try:
            predictions = []
            n_batches = (len(X) + batch_size - 1) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))
                batch_X = X.iloc[start_idx:end_idx]
                
                batch_clusters = None
                if cluster_labels is not None:
                    batch_clusters = cluster_labels[start_idx:end_idx]
                
                batch_predictions, _ = self.predict(
                    model,
                    batch_X,
                    cluster_labels=batch_clusters,
                    preprocessor=preprocessor,
                    prediction_id=f"batch_{i}",
                    **kwargs
                )
                predictions.append(batch_predictions)
            
            combined_predictions = np.concatenate(predictions)
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'input_shape': X.shape,
                'batch_size': batch_size,
                'n_batches': n_batches,
                'clustering_used': cluster_labels is not None,
                'prediction_statistics': self._calculate_prediction_stats(combined_predictions)
            }
            
            if cluster_labels is not None:
                metadata['cluster_statistics'] = self._calculate_cluster_stats(
                    combined_predictions,
                    cluster_labels
                )
            
            return combined_predictions, metadata
            
        except Exception as e:
            raise PredictionError(f"Error in batch prediction: {str(e)}") from e
    
    @monitor_performance
    def load_model(
        self,
        model_path: Union[str, Path],
        model_id: Optional[str] = None,
        load_clusters: bool = True
    ) -> Any:
        """Load model and optional cluster-specific models."""
        try:
            if model_id is None:
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Load main model
            model = joblib.load(model_path)
            self.loaded_models[model_id] = model
            
            # Load cluster models if requested
            if load_clusters:
                cluster_path = Path(model_path).parent / 'clusters'
                if cluster_path.exists():
                    self.cluster_models[model_id] = {}
                    for cluster_file in cluster_path.glob('cluster_*.joblib'):
                        cluster_id = int(cluster_file.stem.split('_')[1])
                        self.cluster_models[model_id][cluster_id] = joblib.load(cluster_file)
            
            logger.info(f"Model loaded: {model_id}")
            return model
            
        except Exception as e:
            raise PredictionError(f"Error loading model: {str(e)}") from e
    
    @monitor_performance
    def load_preprocessor(
        self,
        preprocessor_path: Union[str, Path],
        preprocessor_id: Optional[str] = None
    ) -> Any:
        """Load preprocessor from disk."""
        try:
            preprocessor = joblib.load(preprocessor_path)
            
            if preprocessor_id is None:
                preprocessor_id = f"preprocessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.preprocessors[preprocessor_id] = preprocessor
            logger.info(f"Preprocessor loaded: {preprocessor_id}")
            
            return preprocessor
            
        except Exception as e:
            raise PredictionError(f"Error loading preprocessor: {str(e)}") from e
    
    def _calculate_prediction_stats(
        self,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate statistics for predictions."""
        return {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75))
        }
    
    def _calculate_cluster_stats(
        self,
        predictions: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each cluster."""
        cluster_stats = {}
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_predictions = predictions[mask]
            
            cluster_stats[f'cluster_{cluster_id}'] = {
                'count': int(np.sum(mask)),
                'mean': float(np.mean(cluster_predictions)),
                'std': float(np.std(cluster_predictions)),
                'min': float(np.min(cluster_predictions)),
                'max': float(np.max(cluster_predictions)),
                'median': float(np.median(cluster_predictions))
            }
        
        return cluster_stats
    
    def _record_prediction(
        self,
        prediction_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Record prediction in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction_id': prediction_id,
            'metadata': metadata
        }
        
        self.prediction_history.append(record)
        state_manager.set_state(
            f'prediction.history.{len(self.prediction_history)}',
            record
        )

# Create global model predictor instance
model_predictor = ModelPredictor()