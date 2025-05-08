"""
Spectral clustering implementation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from sklearn.cluster import SpectralClustering as SklearnSpectralClustering

from src.clustering.base_clustering import BaseClustering


class SpectralClustering(BaseClustering):
    """
    Spectral clustering implementation.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random seed for reproducibility.
    affinity : str, default='rbf'
        Affinity type. Options: 'rbf', 'nearest_neighbors', 'precomputed'.
    n_neighbors : int, default=10
        Number of neighbors for nearest_neighbors affinity.
    gamma : float, default=1.0
        Kernel coefficient for rbf affinity.
    """
    
    def __init__(self, n_clusters=3, random_state=42, affinity='rbf', 
                 n_neighbors=10, gamma=1.0, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kwargs)
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.logger = logging.getLogger(__name__)
        self.model = None
    
    def fit(self, Z):
        """
        Fit the spectral clustering algorithm to the influence space.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
            
        Returns
        -------
        self
            Fitted clustering instance.
        """
        self.logger.info(f"Fitting spectral clustering with {self.n_clusters} clusters...")
        
        # Create and fit spectral clustering model
        self.model = SklearnSpectralClustering(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors,
            gamma=self.gamma
        )
        
        self.model.fit(Z)
        
        # Store labels
        self.labels_ = self.model.labels_
        
        # Compute cluster centers (not provided by SpectralClustering)
        self._compute_cluster_centers(Z)
        
        self.is_fitted = True
        
        self.logger.info("Spectral clustering fitted successfully")
        return self
    
    def predict(self, Z):
        """
        Predict cluster labels for new data.
        
        Note: SpectralClustering does not implement predict(). This method
        assigns each instance to the nearest cluster center.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
            
        Returns
        -------
        numpy.ndarray
            Cluster labels.
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before predict().")
            raise ValueError("Model is not fitted. Call fit() before predict().")
        
        # Assign each instance to the nearest cluster center
        distances = np.zeros((Z.shape[0], self.n_clusters))
        
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(Z - self.cluster_centers_[i], axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _compute_cluster_centers(self, Z):
        """
        Compute cluster centers as the mean of instances in each cluster.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
        """
        self.cluster_centers_ = np.zeros((self.n_clusters, Z.shape[1]))
        
        for i in range(self.n_clusters):
            mask = self.labels_ == i
            if np.any(mask):
                self.cluster_centers_[i] = np.mean(Z[mask], axis=0)
    
    def get_cluster_centers(self):
        """
        Get cluster centers.
        
        Returns
        -------
        numpy.ndarray
            Cluster centers.
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before get_cluster_centers().")
            raise ValueError("Model is not fitted. Call fit() before get_cluster_centers().")
        
        return self.cluster_centers_
    
    def get_affinity_matrix(self):
        """
        Get the affinity matrix.
        
        Returns
        -------
        numpy.ndarray
            Affinity matrix.
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before get_affinity_matrix().")
            raise ValueError("Model is not fitted. Call fit() before get_affinity_matrix().")
        
        return self.model.affinity_matrix_
