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

class ModelPredictor:
    """Handle prediction operations."""
    
    def __init__(self):
        self.prediction_history: List[Dict[str, Any]] = []
        self.prediction_results: Dict[str, pd.DataFrame] = {}
        self.prediction_metadata: Dict[str, Dict[str, Any]] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.preprocessors: Dict[str, Any] = {}
    
    @monitor_performance
    @handle_exceptions(PredictionError)
    def predict(
        self,
        model: Any,
        X: pd.DataFrame,
        preprocessor: Optional[Any] = None,
        prediction_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions using specified model."""
        try:
            if prediction_id is None:
                prediction_id = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Preprocess data if preprocessor provided
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X
            
            # Make predictions
            predictions = model.predict(X_processed)
            
            # Create metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'input_shape': X.shape,
                'feature_names': list(X.columns),
                'preprocessor_used': preprocessor is not None
            }
            
            # Store results
            self.prediction_results[prediction_id] = pd.DataFrame({
                'predictions': predictions
            })
            self.prediction_metadata[prediction_id] = metadata
            
            # Record prediction
            self._record_prediction(prediction_id, metadata)
            
            return predictions, metadata
            
        except Exception as e:
            raise PredictionError(
                f"Error making predictions: {str(e)}"
            ) from e
    
    @monitor_performance
    @handle_exceptions(PredictionError)
    def predict_batch(
        self,
        model: Any,
        X: pd.DataFrame,
        batch_size: int = 1000,
        preprocessor: Optional[Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions in batches."""
        try:
            predictions = []
            n_batches = (len(X) + batch_size - 1) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))
                batch_X = X.iloc[start_idx:end_idx]
                
                batch_predictions, _ = self.predict(
                    model,
                    batch_X,
                    preprocessor=preprocessor,
                    prediction_id=f"batch_{i}"
                )
                predictions.append(batch_predictions)
            
            combined_predictions = np.concatenate(predictions)
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'input_shape': X.shape,
                'batch_size': batch_size,
                'n_batches': n_batches
            }
            
            return combined_predictions, metadata
            
        except Exception as e:
            raise PredictionError(
                f"Error in batch prediction: {str(e)}"
            ) from e
    
    @monitor_performance
    def load_model(
        self,
        model_path: str,
        model_id: Optional[str] = None
    ) -> Any:
        """Load model from disk."""
        try:
            model = joblib.load(model_path)
            
            if model_id is None:
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.loaded_models[model_id] = model
            logger.info(f"Model loaded: {model_id}")
            
            return model
            
        except Exception as e:
            raise PredictionError(
                f"Error loading model: {str(e)}"
            ) from e
    
    @monitor_performance
    def load_preprocessor(
        self,
        preprocessor_path: str,
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
            raise PredictionError(
                f"Error loading preprocessor: {str(e)}"
            ) from e
    
    @monitor_performance
    def get_prediction_statistics(
        self,
        prediction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics about predictions."""
        if prediction_id is not None:
            predictions = self.prediction_results.get(prediction_id)
            if predictions is None:
                raise PredictionError(f"Prediction ID not found: {prediction_id}")
                
            return self._calculate_prediction_stats(predictions['predictions'])
        
        # Get statistics for all predictions
        all_stats = {}
        for pred_id, predictions in self.prediction_results.items():
            all_stats[pred_id] = self._calculate_prediction_stats(
                predictions['predictions']
            )
        
        return all_stats
    
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