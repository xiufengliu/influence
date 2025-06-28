"""
Base clustering class for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod

import config


class BaseClustering(ABC):
    """
    Abstract base class for clustering algorithms.
    """
    
    def __init__(self, n_clusters=3, random_state=42, **kwargs):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        self.cluster_centers_ = None
        self.labels_ = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, Z):
        """Fit the clustering algorithm to the influence space."""
        pass
    
    @abstractmethod
    def predict(self, Z):
        """Predict cluster labels for new data."""
        pass
    
    def fit_predict(self, Z, t=None, c=None):
        """Fit the clustering algorithm and predict cluster labels with constraints."""
        self.logger.info(f"Fitting {self.__class__.__name__} with {self.n_clusters} clusters...")
        
        # Initial clustering
        self.fit(Z)
        labels = self.labels_.copy()
        
        if t is None and c is None:
            return labels

        # Iterative refinement with constraints
        max_iter = 10
        for iteration in range(max_iter):
            changes = 0
            
            # Update labels based on combined cost
            new_labels = self._update_labels(Z, labels, t, c)
            
            # Update centroids
            self._update_centroids(Z, new_labels)
            
            # Check for convergence
            changes = np.sum(new_labels != labels)
            self.logger.info(f"Iteration {iteration+1}: {changes} changes")
            if changes == 0:
                break
            labels = new_labels
        
        self.labels_ = labels
        self.logger.info(f"Clustering completed with {self.n_clusters} clusters")
        return labels

    def _update_labels(self, Z, labels, t, c):
        """Update labels based on a combined cost function."""
        new_labels = labels.copy()
        for i in range(len(Z)):
            costs = []
            for k in range(self.n_clusters):
                cost = self._calculate_cost(Z[i], k, i, labels, t, c)
                costs.append(cost)
            new_labels[i] = np.argmin(costs)
        return new_labels

    def _calculate_cost(self, z_i, k, i, labels, t, c):
        """Calculate the cost of assigning a point to a cluster."""
        alpha = config.TEMPORAL_PARAMS['alpha']
        beta = config.TEMPORAL_PARAMS['beta']
        gamma = config.TEMPORAL_PARAMS['gamma']

        # Cohesion cost
        cohesion_cost = alpha * np.linalg.norm(z_i - self.cluster_centers_[k])

        # Temporal cost
        temporal_cost = 0
        if t is not None and i > 0 and i < len(t) - 1:
            prev_cluster = labels[i-1]
            next_cluster = labels[i+1]
            temporal_cost = beta * (int(k != prev_cluster) + int(k != next_cluster))

        # Contextual cost
        contextual_cost = 0
        if c is not None:
            context = c[i]
            # This is a simplified contextual cost. A more sophisticated version
            # would consider the distribution of clusters in the context.
            # For now, we penalize if the cluster is not the dominant one in the context.
            context_mask = np.all(c == context, axis=1)
            if np.any(context_mask):
                context_labels = labels[context_mask]
                if len(context_labels) > 0:
                    dominant_cluster = np.bincount(context_labels).argmax()
                    if k != dominant_cluster:
                        contextual_cost = gamma

        return cohesion_cost + temporal_cost + contextual_cost

    def _update_centroids(self, Z, labels):
        """Update cluster centroids."""
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                self.cluster_centers_[k] = Z[mask].mean(axis=0)
