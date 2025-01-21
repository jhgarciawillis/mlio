import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import (
    KMeans, 
    DBSCAN, 
    AgglomerativeClustering,
    SpectralClustering,
    Birch
)
from sklearn.mixture import GaussianMixture

from core import config
from core.exceptions import ClusteringError
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class BaseClusteringAlgorithm(BaseEstimator, ClusterMixin):
    """Base class for clustering algorithms."""
    
    def __init__(self):
        self.fitted = False
        self.scaler = StandardScaler()
        self.labels_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseClusteringAlgorithm':
        """Fit clustering algorithm."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if not self.fitted:
            raise ClusteringError("Model must be fitted before predicting")
        return self._predict(X)
        
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Internal prediction method."""
        raise NotImplementedError
        
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and predict in one step."""
        return self.fit(X).predict(X)
        
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """Validate and preprocess input data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not isinstance(X, np.ndarray):
            raise ClusteringError("Input must be numpy array or pandas DataFrame")
        return X

class CustomKMeans(BaseClusteringAlgorithm):
    """Custom K-Means implementation."""
    
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.inertia_ = None
        
    @monitor_performance
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CustomKMeans':
        """Fit K-Means clustering."""
        X = self._validate_data(X)
        X = self.scaler.fit_transform(X)
        
        # Initialize centroids
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(X.shape[0])[:self.n_clusters]
        self.centroids_ = X[idx].copy()
        
        for _ in range(self.max_iter):
            old_centroids = self.centroids_.copy()
            
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids_[:, np.newaxis])**2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:
                    self.centroids_[k] = X[self.labels_ == k].mean(axis=0)
            
            # Check convergence
            if np.sum((old_centroids - self.centroids_)**2) < self.tol:
                break
        
        # Calculate inertia
        self.inertia_ = self._calculate_inertia(X)
        self.fitted = True
        
        return self
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict closest cluster for new data."""
        X = self._validate_data(X)
        X = self.scaler.transform(X)
        distances = np.sqrt(((X - self.centroids_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _calculate_inertia(self, X: np.ndarray) -> float:
        """Calculate within-cluster sum of squares."""
        distances = np.sqrt(((X - self.centroids_[self.labels_])**2).sum(axis=1))
        return float(distances.sum())

class CustomDBSCAN(BaseClusteringAlgorithm):
    """Custom DBSCAN implementation."""
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.core_samples_mask_ = None
        
    @monitor_performance
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CustomDBSCAN':
        """Fit DBSCAN clustering."""
        X = self._validate_data(X)
        X = self.scaler.fit_transform(X)
        
        n_samples = X.shape[0]
        
        # Calculate distance matrix
        if self.metric == 'euclidean':
            distances = np.sqrt(((X - X[:, np.newaxis])**2).sum(axis=2))
        else:
            raise ClusteringError(f"Unsupported metric: {self.metric}")
        
        # Find core samples
        neighbors = distances <= self.eps
        core_samples = np.sum(neighbors, axis=1) >= self.min_samples
        self.core_samples_mask_ = core_samples
        
        # Initialize labels
        self.labels_ = np.full(n_samples, -1)
        current_cluster = 0
        
        # Find clusters
        for i in range(n_samples):
            if not core_samples[i] or self.labels_[i] != -1:
                continue
                
            # Start new cluster
            cluster_samples = self._expand_cluster(neighbors, i, current_cluster)
            self.labels_[cluster_samples] = current_cluster
            current_cluster += 1
        
        self.fitted = True
        return self
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict clusters for new data."""
        X = self._validate_data(X)
        X = self.scaler.transform(X)
        
        labels = np.full(X.shape[0], -1)
        core_samples = np.where(self.core_samples_mask_)[0]
        
        for i, point in enumerate(X):
            # Find distances to core samples
            distances = np.sqrt(((point - self.scaler.transform(core_samples))**2).sum(axis=1))
            
            # Assign to nearest cluster if within eps
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] <= self.eps:
                labels[i] = self.labels_[core_samples[min_dist_idx]]
        
        return labels
    
    def _expand_cluster(
        self,
        neighbors: np.ndarray,
        point_idx: int,
        cluster_label: int
    ) -> np.ndarray:
        """Expand cluster from core point."""
        cluster_points = {point_idx}
        stack = [point_idx]
        
        while stack:
            current = stack.pop()
            current_neighbors = np.where(neighbors[current])[0]
            
            for neighbor in current_neighbors:
                if neighbor not in cluster_points:
                    cluster_points.add(neighbor)
                    if self.core_samples_mask_[neighbor]:
                        stack.append(neighbor)
        
        return np.array(list(cluster_points))

class CustomGaussianMixture(BaseClusteringAlgorithm):
    """Custom Gaussian Mixture Model implementation."""
    
    def __init__(
        self,
        n_components: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        
    @monitor_performance
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CustomGaussianMixture':
        """Fit Gaussian Mixture Model."""
        X = self._validate_data(X)
        X = self.scaler.fit_transform(X)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        rng = np.random.RandomState(self.random_state)
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[rng.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        # EM algorithm
        for _ in range(self.max_iter):
            old_means = self.means_.copy()
            
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Check convergence
            if np.all(np.abs(old_means - self.means_) < self.tol):
                break
        
        self.fitted = True
        self.labels_ = self.predict(X)
        
        return self
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        X = self._validate_data(X)
        X = self.scaler.transform(X)
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """E-step: compute responsibilities."""
        weighted_log_prob = np.zeros((X.shape[0], self.n_components))
        
        for k in range(self.n_components):
            # Compute log probability under each Gaussian
            diff = X - self.means_[k]
            log_prob = -0.5 * (
                np.sum(np.dot(diff, np.linalg.inv(self.covariances_[k])) * diff, axis=1) +
                np.log(np.linalg.det(self.covariances_[k])) +
                X.shape[1] * np.log(2 * np.pi)
            )
            weighted_log_prob[:, k] = np.log(self.weights_[k]) + log_prob
        
        # Normalize responsibilities
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return np.exp(log_resp)
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """M-step: update parameters."""
        weights = responsibilities.sum(axis=0)
        self.weights_ = weights / weights.sum()
        
        self.means_ = np.dot(responsibilities.T, X) / weights[:, np.newaxis]
        
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(responsibilities[:, k] * diff.T, diff) / weights[k]

def logsumexp(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute log of sum of exponentials."""
    arr_max = np.max(arr, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(arr - arr_max), axis=axis)) + arr_max.squeeze(axis=axis)