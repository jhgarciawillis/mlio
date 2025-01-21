import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.cluster import (
    KMeans, 
    DBSCAN, 
    AgglomerativeClustering,
    SpectralClustering,
    Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler

from core import config
from core.exceptions import ClusteringError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter

class Clusterer:
    """Handle clustering operations."""
    
    def __init__(self):
        self.clustering_history: List[Dict[str, Any]] = []
        self.clustering_results: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Available algorithms
        self.algorithm_mapping = {
            'kmeans': CustomKMeans,
            'dbscan': CustomDBSCAN,
            'gaussian_mixture': CustomGaussianMixture
        }
    
    @monitor_performance
    @handle_exceptions(ClusteringError)
    def cluster_data(
        self,
        data: pd.DataFrame,
        method: str,
        clustering_config: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Perform clustering on data."""
        try:
            clustering_id = f"clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if clustering_config is None:
                clustering_config = self._get_default_clustering_config(method)
            
            # Use provided model or create new one
            if model is None:
                if method not in self.algorithm_mapping:
                    raise ClusteringError(f"Unknown clustering method: {method}")
                model = self.algorithm_mapping[method](**clustering_config)
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            self.scalers[clustering_id] = scaler
            
            # Fit model and get labels
            labels = model.fit_predict(scaled_data)
            self.models[clustering_id] = model
            
            # Calculate metrics
            metrics = self._calculate_clustering_metrics(scaled_data, labels)
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(data, labels)
            
            # Create visualizations
            visualizations = self._create_clustering_visualizations(
                data, labels, scaled_data
            )
            
            results = {
                'labels': labels,
                'n_clusters': len(np.unique(labels)),
                'metrics': metrics,
                'analysis': cluster_analysis,
                'visualizations': visualizations,
                'method': method,
                'config': clustering_config
            }
            
            # Store results
            self.clustering_results[clustering_id] = results
            
            # Record clustering
            self._record_clustering(clustering_id, results)
            
            return results
            
        except Exception as e:
            raise ClusteringError(
                f"Error performing clustering: {str(e)}"
            ) from e
    
    @monitor_performance
    def predict_clusters(
        self,
        data: pd.DataFrame,
        clustering_id: str
    ) -> np.ndarray:
        """Predict clusters for new data."""
        if clustering_id not in self.models:
            raise ClusteringError(f"Clustering model not found: {clustering_id}")
        
        # Scale data using stored scaler
        scaled_data = self.scalers[clustering_id].transform(data)
        
        # Predict clusters
        return self.models[clustering_id].predict(scaled_data)
    
    def _calculate_clustering_metrics(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        # Only calculate if more than one cluster
        if len(np.unique(labels)) > 1:
            try:
                metrics['silhouette'] = float(silhouette_score(data, labels))
            except:
                metrics['silhouette'] = None
                
            try:
                metrics['calinski_harabasz'] = float(calinski_harabasz_score(data, labels))
            except:
                metrics['calinski_harabasz'] = None
                
            try:
                metrics['davies_bouldin'] = float(davies_bouldin_score(data, labels))
            except:
                metrics['davies_bouldin'] = None
        
        return metrics
    
    def _analyze_clusters(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        analysis = {
            'cluster_sizes': {},
            'cluster_stats': {},
            'feature_importance': {}
        }
        
        unique_labels = np.unique(labels)
        
        # Analyze cluster sizes
        for label in unique_labels:
            mask = labels == label
            analysis['cluster_sizes'][int(label)] = int(np.sum(mask))
            
            # Calculate statistics for each feature
            cluster_data = data[mask]
            analysis['cluster_stats'][int(label)] = {
                col: {
                    'mean': float(cluster_data[col].mean()),
                    'std': float(cluster_data[col].std()),
                    'min': float(cluster_data[col].min()),
                    'max': float(cluster_data[col].max()),
                    'median': float(cluster_data[col].median())
                }
                for col in data.columns
            }
        
        # Calculate feature importance for clustering
        for col in data.columns:
            f_scores = []
            for label in unique_labels:
                mask = labels == label
                if mask.any():
                    cluster_values = data[col][mask]
                    other_values = data[col][~mask]
                    try:
                        f_stat = float(
                            np.var(cluster_values) / np.var(other_values)
                            if np.var(other_values) > 0 else 0
                        )
                        f_scores.append(f_stat)
                    except:
                        f_scores.append(0)
            
            analysis['feature_importance'][col] = float(np.mean(f_scores))
        
        return analysis
    
    def _create_clustering_visualizations(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        scaled_data: np.ndarray
    ) -> Dict[str, Any]:
        """Create clustering visualizations."""
        visualizations = {}
        
        # 2D scatter plot of first two features
        visualizations['feature_scatter'] = plotter.create_plot(
            'scatter',
            data=pd.DataFrame({
                'x': data.iloc[:, 0],
                'y': data.iloc[:, 1],
                'Cluster': labels
            }),
            x='x',
            y='y',
            color='Cluster',
            title='Cluster Assignments (First Two Features)'
        )
        
        # Cluster sizes
        cluster_sizes = pd.Series(labels).value_counts()
        visualizations['cluster_sizes'] = plotter.create_plot(
            'bar',
            data=pd.DataFrame({
                'Cluster': cluster_sizes.index,
                'Size': cluster_sizes.values
            }),
            x='Cluster',
            y='Size',
            title='Cluster Sizes'
        )
        
        # Feature importance heatmap
        feature_importance = self._calculate_feature_importance_by_cluster(
            data, labels
        )
        visualizations['feature_importance'] = plotter.create_plot(
            'heatmap',
            data=feature_importance,
            title='Feature Importance by Cluster'
        )
        
        return visualizations
    
    def _calculate_feature_importance_by_cluster(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> pd.DataFrame:
        """Calculate feature importance for each cluster."""
        unique_labels = np.unique(labels)
        importance_matrix = []
        
        for label in unique_labels:
            mask = labels == label
            cluster_importance = {}
            
            for col in data.columns:
                cluster_values = data[col][mask]
                other_values = data[col][~mask]
                
                try:
                    importance = np.abs(
                        np.mean(cluster_values) - np.mean(other_values)
                    ) / np.std(data[col])
                except:
                    importance = 0
                    
                cluster_importance[col] = importance
                
            importance_matrix.append(cluster_importance)
            
        return pd.DataFrame(importance_matrix)
    
    def _get_default_clustering_config(
        self,
        method: str
    ) -> Dict[str, Any]:
        """Get default configuration for clustering method."""
        configs = {
            'kmeans': {
                'n_clusters': 5,
                'random_state': config.random_state
            },
            'dbscan': {
                'eps': 0.5,
                'min_samples': 5
            },
            'gaussian_mixture': {
                'n_components': 5,
                'random_state': config.random_state
            }
        }
        
        return configs.get(method, {})
    
    def _record_clustering(
        self,
        clustering_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Record clustering in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'clustering_id': clustering_id,
            'method': results['method'],
            'n_clusters': results['n_clusters'],
            'metrics': results['metrics']
        }
        
        self.clustering_history.append(record)
        state_manager.set_state(
            f'clustering.history.{len(self.clustering_history)}',
            record
        )

# Create global clusterer instance
clusterer = Clusterer()