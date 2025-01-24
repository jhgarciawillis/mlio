import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.model_selection import KFold
from scipy.stats import f_oneway
import plotly.graph_objects as go

from core import config
from core.exceptions import ClusterValidationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter

class ClusterValidator:
    """Handle cluster validation operations."""
    
    def __init__(self):
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        
    @monitor_performance
    @handle_exceptions(ClusterValidationError)
    def validate_clustering(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive cluster validation."""
        try:
            # Record operation start
            state_monitor.record_operation_start(
                'cluster_validation',
                'validation',
                {'n_clusters': len(np.unique(labels))}
            )
            
            validation_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if validation_config is None:
                validation_config = self._get_default_validation_config()
            
            results = {}
            
            # Internal validation
            if validation_config.get('internal_validation', True):
                results['internal'] = self._perform_internal_validation(
                    data,
                    labels
                )
            
            # Stability analysis
            if validation_config.get('stability_analysis', True):
                results['stability'] = self._analyze_stability(
                    data,
                    labels,
                    validation_config.get('n_splits', 5)
                )
            
            # Feature analysis
            if validation_config.get('feature_analysis', True):
                results['feature_analysis'] = self._analyze_features(
                    data,
                    labels
                )
            
            # Cluster characteristics
            if validation_config.get('cluster_characteristics', True):
                results['characteristics'] = self._analyze_cluster_characteristics(
                    data,
                    labels
                )
            
            # Create visualizations
            results['visualizations'] = self._create_validation_visualizations(
                data,
                labels,
                results
            )
            
            # Store results
            self.validation_results[validation_id] = results
            
            # Record validation
            self._record_validation(validation_id, validation_config)
            
            # Record operation completion
            state_monitor.record_operation_end(
                'cluster_validation',
                'completed',
                {'validation_id': validation_id}
            )
            
            return results
            
        except Exception as e:
            state_monitor.record_operation_end(
                'cluster_validation',
                'failed',
                {'error': str(e)}
            )
            raise ClusterValidationError(
                f"Error validating clustering: {str(e)}"
            ) from e
    
    @monitor_performance
    def _perform_internal_validation(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Perform internal cluster validation."""
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
        
        # Intra-cluster distances
        metrics['intra_cluster_distances'] = self._calculate_intra_cluster_distances(
            data,
            labels
        )
        
        # Inter-cluster distances
        metrics['inter_cluster_distances'] = self._calculate_inter_cluster_distances(
            data,
            labels
        )
        
        return metrics
    
    @monitor_performance
    def _analyze_stability(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        n_splits: int
    ) -> Dict[str, Any]:
        """Analyze clustering stability."""
        stability_results = {
            'fold_metrics': [],
            'label_consistency': {},
            'cluster_persistence': {}
        }
        
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_state
        )
        
        # Analyze across folds
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            
            # Calculate metrics for fold
            fold_metrics = self._perform_internal_validation(
                val_data,
                val_labels
            )
            
            stability_results['fold_metrics'].append({
                'fold': fold,
                'metrics': fold_metrics
            })
            
            # Analyze label consistency
            for label in np.unique(labels):
                if label not in stability_results['label_consistency']:
                    stability_results['label_consistency'][label] = []
                
                consistency = self._calculate_label_consistency(
                    train_data[train_labels == label],
                    val_data,
                    val_labels
                )
                stability_results['label_consistency'][label].append(consistency)
        
        # Calculate cluster persistence
        for label in np.unique(labels):
            mask = labels == label
            persistence = self._calculate_cluster_persistence(
                data[mask],
                n_splits
            )
            stability_results['cluster_persistence'][int(label)] = persistence
        
        # Calculate stability statistics
        stability_results['statistics'] = self._calculate_stability_statistics(
            stability_results
        )
        
        return stability_results
    
    @monitor_performance
    def _analyze_features(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze feature importance and distributions."""
        feature_analysis = {
            'importance': {},
            'distributions': {},
            'discriminative_power': {}
        }
        
        # Calculate feature importance using ANOVA F-value
        for column in data.columns:
            groups = [group[column].values for _, group in data.groupby(labels)]
            f_stat, p_value = f_oneway(*groups)
            feature_analysis['importance'][column] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value)
            }
        
        # Analyze feature distributions per cluster
        for column in data.columns:
            feature_analysis['distributions'][column] = {}
            for label in np.unique(labels):
                cluster_data = data[labels == label][column]
                feature_analysis['distributions'][column][int(label)] = {
                    'mean': float(cluster_data.mean()),
                    'std': float(cluster_data.std()),
                    'median': float(cluster_data.median()),
                    'q25': float(cluster_data.quantile(0.25)),
                    'q75': float(cluster_data.quantile(0.75))
                }
        
        # Calculate discriminative power
        feature_analysis['discriminative_power'] = self._calculate_discriminative_power(
            data,
            labels
        )
        
        return feature_analysis
    
    @monitor_performance
    def _analyze_cluster_characteristics(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""
        characteristics = {}
        
        for label in np.unique(labels):
            cluster_data = data[labels == label]
            
            # Basic statistics
            characteristics[int(label)] = {
                'size': int(len(cluster_data)),
                'proportion': float(len(cluster_data) / len(data)),
                'density': self._calculate_cluster_density(cluster_data),
                'compactness': self._calculate_cluster_compactness(cluster_data),
                'features': {}
            }
            
            # Feature-level statistics
            for column in data.columns:
                characteristics[int(label)]['features'][column] = {
                    'mean': float(cluster_data[column].mean()),
                    'std': float(cluster_data[column].std()),
                    'min': float(cluster_data[column].min()),
                    'max': float(cluster_data[column].max()),
                    'unique_values': int(cluster_data[column].nunique()),
                    'completeness': float(cluster_data[column].notna().mean())
                }
        
        return characteristics
    
    def _calculate_intra_cluster_distances(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate average intra-cluster distances."""
        distances = {}
        
        for label in np.unique(labels):
            cluster_data = data[labels == label].values
            if len(cluster_data) > 1:
                # Calculate pairwise distances within cluster
                dists = []
                for i in range(len(cluster_data)):
                    for j in range(i + 1, len(cluster_data)):
                        dist = np.linalg.norm(cluster_data[i] - cluster_data[j])
                        dists.append(dist)
                distances[int(label)] = float(np.mean(dists))
            else:
                distances[int(label)] = 0.0
        
        return distances
    
def _calculate_inter_cluster_distances(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate average inter-cluster distances."""
        distances = {}
        
        unique_labels = np.unique(labels)
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                cluster1_data = data[labels == label1].values
                cluster2_data = data[labels == label2].values
                
                # Calculate distances between clusters
                dists = []
                for point1 in cluster1_data:
                    for point2 in cluster2_data:
                        dist = np.linalg.norm(point1 - point2)
                        dists.append(dist)
                        
                distances[f"{label1}_{label2}"] = float(np.mean(dists))
        
        return distances
    
    def _calculate_label_consistency(
        self,
        reference_data: pd.DataFrame,
        comparison_data: pd.DataFrame,
        comparison_labels: np.ndarray
    ) -> float:
        """Calculate label consistency between folds."""
        # Calculate centroid of reference data
        centroid = reference_data.mean().values
        
        # Find nearest points in comparison data
        distances = np.linalg.norm(comparison_data - centroid, axis=1)
        nearest_indices = np.argsort(distances)[:len(reference_data)]
        
        # Calculate label consistency
        nearest_labels = comparison_labels[nearest_indices]
        most_common_label = pd.Series(nearest_labels).mode().iloc[0]
        
        return float(np.mean(nearest_labels == most_common_label))
    
    def _calculate_cluster_persistence(
        self,
        cluster_data: pd.DataFrame,
        n_splits: int
    ) -> float:
        """Calculate cluster persistence across splits."""
        persistence_scores = []
        
        for _ in range(n_splits):
            # Random split
            mask = np.random.choice(
                [True, False],
                size=len(cluster_data),
                p=[0.5, 0.5]
            )
            
            if np.sum(mask) > 0 and np.sum(~mask) > 0:
                split1 = cluster_data[mask]
                split2 = cluster_data[~mask]
                
                # Calculate centroids
                centroid1 = split1.mean()
                centroid2 = split2.mean()
                
                # Calculate similarity between splits
                similarity = 1 / (1 + np.linalg.norm(centroid1 - centroid2))
                persistence_scores.append(similarity)
        
        return float(np.mean(persistence_scores)) if persistence_scores else 0.0
    
    def _calculate_discriminative_power(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate discriminative power of features."""
        discriminative_power = {}
        
        for column in data.columns:
            # Calculate ratio of between-cluster to within-cluster variance
            cluster_means = [
                data[labels == label][column].mean()
                for label in np.unique(labels)
            ]
            
            total_mean = data[column].mean()
            
            between_cluster_var = np.sum([
                len(data[labels == label]) * (mean - total_mean) ** 2
                for label, mean in zip(np.unique(labels), cluster_means)
            ]) / len(data)
            
            within_cluster_var = np.mean([
                data[labels == label][column].var()
                for label in np.unique(labels)
                if len(data[labels == label]) > 1
            ])
            
            if within_cluster_var > 0:
                discriminative_power[column] = float(
                    between_cluster_var / within_cluster_var
                )
            else:
                discriminative_power[column] = float('inf')
        
        return discriminative_power
    
    def _calculate_stability_statistics(
        self,
        stability_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate stability statistics."""
        stats = {
            'mean_consistency': {},
            'consistency_std': {},
            'mean_persistence': float(np.mean(list(
                stability_results['cluster_persistence'].values()
            ))),
            'fold_stability': {}
        }
        
        # Calculate consistency statistics
        for label, consistencies in stability_results['label_consistency'].items():
            stats['mean_consistency'][int(label)] = float(np.mean(consistencies))
            stats['consistency_std'][int(label)] = float(np.std(consistencies))
        
        # Calculate fold stability
        for fold_result in stability_results['fold_metrics']:
            metrics = fold_result['metrics']
            fold = fold_result['fold']
            stats['fold_stability'][fold] = float(metrics.get('silhouette', 0))
        
        return stats
    
    def _create_validation_visualizations(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create validation visualizations."""
        visualizations = {}
        
        # Silhouette plot
        if 'internal' in results and results['internal'].get('silhouette') is not None:
            visualizations['silhouette'] = self._create_silhouette_plot(data, labels)
        
        # Feature importance plot
        if 'feature_analysis' in results:
            importance_df = pd.DataFrame([
                {
                    'feature': feature,
                    'importance': -np.log10(info['p_value'])
                }
                for feature, info in results['feature_analysis']['importance'].items()
            ])
            
            visualizations['feature_importance'] = plotter.create_plot(
                'bar',
                data=importance_df.sort_values('importance', ascending=False),
                x='feature',
                y='importance',
                title='Feature Importance (-log10 p-value)'
            )
        
        # Stability plot
        if 'stability' in results:
            stability_df = pd.DataFrame({
                'Cluster': list(results['stability']['cluster_persistence'].keys()),
                'Persistence': list(results['stability']['cluster_persistence'].values())
            })
            
            visualizations['stability'] = plotter.create_plot(
                'bar',
                data=stability_df,
                x='Cluster',
                y='Persistence',
                title='Cluster Stability'
            )
        
        # Characteristics plot
        if 'characteristics' in results:
            visualizations['characteristics'] = self._create_characteristics_plot(
                results['characteristics']
            )
        
        return visualizations
    
    def _get_default_validation_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'internal_validation': True,
            'stability_analysis': True,
            'feature_analysis': True,
            'cluster_characteristics': True,
            'n_splits': 5
        }
    
    def _record_validation(
        self,
        validation_id: str,
        validation_config: Dict[str, Any]
    ) -> None:
        """Record validation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'validation_id': validation_id,
            'configuration': validation_config
        }
        
        self.validation_history.append(record)
        state_manager.set_state(
            f'clustering.validation.history.{len(self.validation_history)}',
            record
        )

# Create global cluster validator instance
cluster_validator = ClusterValidator()