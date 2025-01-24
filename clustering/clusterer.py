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
        
        # Available algorithms with default parameters
        self.algorithms = {
            'kmeans': {
                'class': KMeans,
                'params': {
                    'n_clusters': 5,
                    'random_state': config.random_state
                }
            },
            'dbscan': {
                'class': DBSCAN,
                'params': {
                    'eps': 0.5,
                    'min_samples': 5
                }
            },
            'gaussian_mixture': {
                'class': GaussianMixture,
                'params': {
                    'n_components': 5,
                    'random_state': config.random_state
                }
            },
            'hierarchical': {
                'class': AgglomerativeClustering,
                'params': {
                    'n_clusters': 5
                }
            },
            'spectral': {
                'class': SpectralClustering,
                'params': {
                    'n_clusters': 5,
                    'random_state': config.random_state
                }
            },
            'birch': {
                'class': Birch,
                'params': {
                    'n_clusters': 5
                }
            }
        }
    
    @monitor_performance
    @handle_exceptions(ClusteringError)
    def cluster_data(
        self,
        data: pd.DataFrame,
        method: str = 'kmeans',
        params: Optional[Dict[str, Any]] = None,
        scale_data: bool = True
    ) -> Dict[str, Any]:
        """Perform clustering on data."""
        try:
            # Record operation start
            state_monitor.record_operation_start(
                'clustering',
                'clustering_process',
                {'method': method}
            )
            
            clustering_id = f"clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Validate method
            if method not in self.algorithms:
                raise ClusteringError(
                    f"Unknown clustering method: {method}",
                    details={'available_methods': list(self.algorithms.keys())}
                )
            
            # Get algorithm and parameters
            algorithm = self.algorithms[method]
            algorithm_params = algorithm['params'].copy()
            if params:
                algorithm_params.update(params)
            
            # Scale data if requested
            if scale_data:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                self.scalers[clustering_id] = scaler
            else:
                scaled_data = data.values
            
            # Create and fit model
            model = algorithm['class'](**algorithm_params)
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
            
            # Store results
            results = {
                'labels': labels,
                'n_clusters': len(np.unique(labels)),
                'metrics': metrics,
                'analysis': cluster_analysis,
                'visualizations': visualizations,
                'method': method,
                'params': algorithm_params
            }
            
            self.clustering_results[clustering_id] = results
            
            # Record operation completion
            state_monitor.record_operation_end(
                'clustering',
                'completed',
                {
                    'n_clusters': len(np.unique(labels)),
                    'metrics': metrics
                }
            )
            
            # Record clustering
            self._record_clustering(clustering_id, results)
            
            return results
            
        except Exception as e:
            state_monitor.record_operation_end(
                'clustering',
                'failed',
                {'error': str(e)}
            )
            raise ClusteringError(f"Error performing clustering: {str(e)}") from e
    
    @monitor_performance
    def predict_clusters(
        self,
        data: pd.DataFrame,
        clustering_id: str
    ) -> np.ndarray:
        """Predict clusters for new data."""
        if clustering_id not in self.models:
            raise ClusteringError(f"Clustering model not found: {clustering_id}")
        
        # Scale data if scaler exists
        if clustering_id in self.scalers:
            data = self.scalers[clustering_id].transform(data)
        
        return self.models[clustering_id].predict(data)
    
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
        
        # Add cluster distribution metrics
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = {
            int(label): int(count) for label, count in zip(unique_labels, counts)
        }
        metrics['cluster_proportions'] = {
            int(label): float(count/len(labels))
            for label, count in zip(unique_labels, counts)
        }
        
        return metrics
    
    def _analyze_clusters(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        analysis = {
            'cluster_profiles': {},
            'feature_importance': {},
            'cluster_separation': {},
            'cluster_stability': {}
        }
        
        unique_labels = np.unique(labels)
        
        # Analyze each cluster
        for label in unique_labels:
            mask = labels == label
            cluster_data = data[mask]
            
            # Calculate cluster profile
            analysis['cluster_profiles'][int(label)] = {
                'size': int(np.sum(mask)),
                'proportion': float(np.mean(mask)),
                'centroid': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict(),
                'min': cluster_data.min().to_dict(),
                'max': cluster_data.max().to_dict()
            }
            
            # Calculate feature importance for cluster
            analysis['feature_importance'][int(label)] = self._calculate_feature_importance(
                data, mask
            )
            
            # Calculate cluster separation
            analysis['cluster_separation'][int(label)] = self._calculate_cluster_separation(
                data, labels, label
            )
        
        # Calculate cluster stability
        analysis['cluster_stability'] = self._calculate_cluster_stability(
            data, labels
        )
        
        return analysis
    
    def _calculate_feature_importance(
        self,
        data: pd.DataFrame,
        cluster_mask: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance for cluster."""
        importance_scores = {}
        
        for column in data.columns:
            # Calculate effect size (Cohen's d)
            cluster_values = data[cluster_mask][column]
            other_values = data[~cluster_mask][column]
            
            if len(cluster_values) > 0 and len(other_values) > 0:
                pooled_std = np.sqrt(
                    (np.var(cluster_values) + np.var(other_values)) / 2
                )
                if pooled_std > 0:
                    effect_size = abs(
                        np.mean(cluster_values) - np.mean(other_values)
                    ) / pooled_std
                    importance_scores[column] = float(effect_size)
                else:
                    importance_scores[column] = 0.0
            else:
                importance_scores[column] = 0.0
        
        return importance_scores
    
    def _calculate_cluster_separation(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        current_label: int
    ) -> Dict[str, float]:
        """Calculate separation between clusters."""
        separation = {}
        current_mask = labels == current_label
        
        for other_label in np.unique(labels):
            if other_label != current_label:
                other_mask = labels == other_label
                
                # Calculate distance between cluster centroids
                current_centroid = data[current_mask].mean()
                other_centroid = data[other_mask].mean()
                
                separation[int(other_label)] = float(
                    np.linalg.norm(current_centroid - other_centroid)
                )
        
        return separation
    
    def _calculate_cluster_stability(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate cluster stability through bootstrapping."""
        n_boots = 10
        stability_scores = {}
        
        for label in np.unique(labels):
            boot_labels = []
            mask = labels == label
            
            for _ in range(n_boots):
                # Bootstrap sample
                boot_idx = np.random.choice(
                    len(data),
                    size=len(data),
                    replace=True
                )
                boot_data = data.iloc[boot_idx]
                boot_mask = mask[boot_idx]
                
                if np.sum(boot_mask) > 0:
                    stability = float(
                        np.mean(boot_mask == mask[boot_idx])
                    )
                    boot_labels.append(stability)
            
            stability_scores[int(label)] = float(np.mean(boot_labels))
        
        return stability_scores
    
    def _create_clustering_visualizations(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        scaled_data: np.ndarray
    ) -> Dict[str, Any]:
        """Create clustering visualizations."""
        visualizations = {}
        
        # 2D scatter plot using PCA if needed
        if data.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(scaled_data)
        else:
            coords_2d = scaled_data
        
        # Create scatter plot
        scatter_data = pd.DataFrame(
            coords_2d,
            columns=['Component 1', 'Component 2']
        )
        scatter_data['Cluster'] = labels
        
        visualizations['scatter_2d'] = plotter.create_plot(
            'scatter',
            data=scatter_data,
            x='Component 1',
            y='Component 2',
            color='Cluster',
            title='Cluster Assignments (2D Projection)'
        )
        
        # Cluster sizes bar plot
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
        
        # Feature importance heatmap
        feature_importance = pd.DataFrame(
            [
                self._calculate_feature_importance(data, labels == label)
                for label in unique_labels
            ],
            index=[f'Cluster {label}' for label in unique_labels]
        )
        
        visualizations['feature_importance'] = plotter.create_plot(
            'heatmap',
            data=feature_importance,
            title='Feature Importance by Cluster'
        )
        
        return visualizations
    
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
        
        # Update state
        state_manager.set_state(
            f'clustering.history.{len(self.clustering_history)}',
            record
        )
    
    @monitor_performance
    def get_clustering_summary(
        self,
        clustering_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of clustering operations."""
        if clustering_id:
            if clustering_id not in self.clustering_results:
                raise ClusteringError(f"Clustering ID not found: {clustering_id}")
            return {
                'clustering_id': clustering_id,
                'results': self.clustering_results[clustering_id],
                'history': [
                    record for record in self.clustering_history
                    if record['clustering_id'] == clustering_id
                ]
            }
        
        return {
            'total_clusterings': len(self.clustering_results),
            'available_methods': list(self.algorithms.keys()),
            'history_length': len(self.clustering_history),
            'last_clustering': self.clustering_history[-1] if self.clustering_history else None
        }

# Create global clusterer instance
clusterer = Clusterer()