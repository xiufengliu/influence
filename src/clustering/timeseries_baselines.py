
"""
Time series clustering baselines for comparison with the proposed method.

This module implements established time series clustering algorithms including
k-Shape and DTW-based clustering to serve as competitive baselines.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from src.clustering.base_clustering import BaseClustering

try:
    from tslearn.clustering import TimeSeriesKMeans, KShape
    from tslearn.metrics import dtw
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    print("Warning: tslearn not available. Time series baselines will use approximations.")

class KShapeClustering(BaseClustering):
    """
    k-Shape clustering algorithm for time series.
    Uses tslearn if available, otherwise falls back to standard k-means.
    """
    def __init__(self, n_clusters=3, random_state=42, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kwargs)
        
        if TSLEARN_AVAILABLE:
            self.model = KShape(n_clusters=self.n_clusters, random_state=self.random_state, **kwargs)
        else:
            # Fallback to standard k-means if tslearn not available
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, **kwargs)
        
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X, y=None):
        """Fit the k-Shape clustering model."""
        # Reshape data if needed for time series
        if len(X.shape) == 2 and not TSLEARN_AVAILABLE:
            # For standard k-means fallback, flatten time series
            X_reshaped = X.reshape(X.shape[0], -1)
            self.labels_ = self.model.fit_predict(X_reshaped)
            self.cluster_centers_ = self.model.cluster_centers_
        else:
            self.labels_ = self.model.fit_predict(X)
            self.cluster_centers_ = self.model.cluster_centers_
        return self

    def predict(self, X):
        """Predict cluster labels."""
        if len(X.shape) == 2 and not TSLEARN_AVAILABLE:
            X_reshaped = X.reshape(X.shape[0], -1)
            return self.model.predict(X_reshaped)
        return self.model.predict(X)

    def get_cluster_centers(self):
        """Get cluster centers."""
        return self.cluster_centers_


class DTWKMeansClustering(BaseClustering):
    """
    K-means clustering with Dynamic Time Warping (DTW) distance.
    Uses tslearn if available, otherwise approximates with Euclidean distance.
    """
    def __init__(self, n_clusters=3, random_state=42, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kwargs)
        
        if TSLEARN_AVAILABLE:
            self.model = TimeSeriesKMeans(n_clusters=self.n_clusters,
                                          metric="dtw",
                                          random_state=self.random_state,
                                          **kwargs)
        else:
            # Fallback to standard k-means if tslearn not available
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, **kwargs)
        
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X, y=None):
        """Fit the DTW k-means clustering model."""
        if len(X.shape) == 2 and not TSLEARN_AVAILABLE:
            # For standard k-means fallback, flatten time series
            X_reshaped = X.reshape(X.shape[0], -1)
            self.labels_ = self.model.fit_predict(X_reshaped)
            self.cluster_centers_ = self.model.cluster_centers_
        else:
            self.labels_ = self.model.fit_predict(X)
            self.cluster_centers_ = self.model.cluster_centers_
        return self

    def predict(self, X):
        """Predict cluster labels."""
        if len(X.shape) == 2 and not TSLEARN_AVAILABLE:
            X_reshaped = X.reshape(X.shape[0], -1)
            return self.model.predict(X_reshaped)
        return self.model.predict(X)

    def get_cluster_centers(self):
        """Get cluster centers."""
        return self.cluster_centers_


class DTWKMedoidsClustering(BaseClustering):
    """
    K-medoids clustering with DTW distance for time series.
    """
    def __init__(self, n_clusters=3, random_state=42, max_iter=300, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kwargs)
        self.max_iter = max_iter
        self.labels_ = None
        self.medoid_indices_ = None
        self.cluster_centers_ = None

    def _dtw_distance(self, x, y):
        """Compute DTW distance between two time series."""
        if TSLEARN_AVAILABLE:
            return dtw(x, y)
        else:
            # Fallback to Euclidean distance if DTW not available
            return np.linalg.norm(x - y)

    def fit(self, X, y=None):
        """Fit k-medoids clustering with DTW distance."""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize medoids randomly
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        for iteration in range(self.max_iter):
            # Assign points to nearest medoid
            distances = np.zeros((n_samples, self.n_clusters))
            for i, medoid_idx in enumerate(medoid_indices):
                for j in range(n_samples):
                    distances[j, i] = self._dtw_distance(X[j], X[medoid_idx])
            
            labels = np.argmin(distances, axis=1)
            
            # Update medoids
            new_medoid_indices = []
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]
                if len(cluster_points) == 0:
                    new_medoid_indices.append(medoid_indices[k])
                    continue
                
                # Find point that minimizes total distance within cluster
                min_total_distance = float('inf')
                best_medoid = medoid_indices[k]
                
                for candidate in cluster_points:
                    total_distance = 0
                    for point in cluster_points:
                        total_distance += self._dtw_distance(X[candidate], X[point])
                    
                    if total_distance < min_total_distance:
                        min_total_distance = total_distance
                        best_medoid = candidate
                
                new_medoid_indices.append(best_medoid)
            
            new_medoid_indices = np.array(new_medoid_indices)
            
            # Check for convergence
            if np.array_equal(medoid_indices, new_medoid_indices):
                break
                
            medoid_indices = new_medoid_indices
        
        self.medoid_indices_ = medoid_indices
        self.labels_ = labels
        self.cluster_centers_ = X[medoid_indices]
        return self

    def predict(self, X):
        """Predict cluster labels for new data."""
        if self.medoid_indices_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i, medoid_idx in enumerate(self.medoid_indices_):
            for j in range(n_samples):
                distances[j, i] = self._dtw_distance(X[j], self.cluster_centers_[i])
        
        return np.argmin(distances, axis=1)

    def get_cluster_centers(self):
        """Get cluster centers (medoids)."""
        return self.cluster_centers_
        return self.cluster_centers_
