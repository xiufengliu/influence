"""
K-means clustering implementation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from sklearn.cluster import KMeans

from src.clustering.base_clustering import BaseClustering


class KMeansClustering(BaseClustering):
    """
    K-means clustering implementation.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random seed for reproducibility.
    n_init : int, default=10
        Number of initializations.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    """
    
    def __init__(self, n_clusters=3, random_state=42, n_init=10, max_iter=300, tol=1e-4, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kwargs)
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.logger = logging.getLogger(__name__)
        self.model = None
    
    def fit(self, Z):
        """
        Fit the K-means clustering algorithm to the influence space.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
            
        Returns
        -------
        self
            Fitted clustering instance.
        """
        self.logger.info(f"Fitting K-means with {self.n_clusters} clusters...")
        
        # Create and fit K-means model
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol
        )
        
        self.model.fit(Z)
        
        # Store cluster centers and labels
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.is_fitted = True
        
        self.logger.info("K-means fitted successfully")
        return self
    
    def predict(self, Z):
        """
        Predict cluster labels for new data.
        
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
        
        return self.model.predict(Z)
    
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
    
    def get_inertia(self):
        """
        Get the sum of squared distances to the nearest centroid.
        
        Returns
        -------
        float
            Inertia.
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before get_inertia().")
            raise ValueError("Model is not fitted. Call fit() before get_inertia().")
        
        return self.model.inertia_
