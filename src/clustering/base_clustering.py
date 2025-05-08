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
    
    This class defines the interface for all clustering algorithms used in the framework.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random seed for reproducibility.
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
        """
        Fit the clustering algorithm to the influence space.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
            
        Returns
        -------
        self
            Fitted clustering instance.
        """
        pass
    
    @abstractmethod
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
        pass
    
    def fit_predict(self, Z, t=None, c=None):
        """
        Fit the clustering algorithm and predict cluster labels.
        
        This method incorporates temporal and contextual constraints if provided.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
        t : numpy.ndarray, default=None
            Timestamps.
        c : numpy.ndarray, default=None
            Contextual attributes.
            
        Returns
        -------
        numpy.ndarray
            Cluster labels.
        """
        self.logger.info(f"Fitting {self.__class__.__name__} with {self.n_clusters} clusters...")
        
        # Basic clustering without temporal or contextual constraints
        if t is None and c is None:
            self.fit(Z)
            return self.labels_
        
        # Initialize with basic clustering
        self.fit(Z)
        labels = self.labels_.copy()
        
        # Apply temporal constraints if timestamps are provided
        if t is not None:
            labels = self._apply_temporal_constraints(Z, labels, t)
        
        # Apply contextual constraints if contextual attributes are provided
        if c is not None:
            labels = self._apply_contextual_constraints(Z, labels, c)
        
        self.labels_ = labels
        self.logger.info(f"Clustering completed with {self.n_clusters} clusters")
        
        return labels
    
    def _apply_temporal_constraints(self, Z, labels, t):
        """
        Apply temporal constraints to cluster assignments.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
        labels : numpy.ndarray
            Initial cluster labels.
        t : numpy.ndarray
            Timestamps.
            
        Returns
        -------
        numpy.ndarray
            Updated cluster labels.
        """
        self.logger.info("Applying temporal constraints...")
        
        # Convert timestamps to numeric values if they are not already
        if not np.issubdtype(t.dtype, np.number):
            t_numeric = np.array([ts.timestamp() for ts in t])
        else:
            t_numeric = t
        
        # Sort data by timestamp
        sort_idx = np.argsort(t_numeric)
        Z_sorted = Z[sort_idx]
        labels_sorted = labels[sort_idx]
        
        # Apply temporal smoothing
        alpha = config.TEMPORAL_PARAMS['alpha']
        beta = config.TEMPORAL_PARAMS['beta']
        
        # Initialize new labels with original assignments
        new_labels = labels_sorted.copy()
        
        # Iterate until convergence or max iterations
        max_iter = 10
        for iteration in range(max_iter):
            changes = 0
            
            for i in range(1, len(Z_sorted) - 1):
                # Get current cluster and neighbors
                curr_cluster = new_labels[i]
                prev_cluster = new_labels[i-1]
                next_cluster = new_labels[i+1]
                
                # Calculate distances to cluster centers
                distances = np.zeros(self.n_clusters)
                for k in range(self.n_clusters):
                    # Distance to cluster center
                    center_dist = np.linalg.norm(Z_sorted[i] - self.cluster_centers_[k])
                    
                    # Temporal penalty (higher if different from neighbors)
                    temporal_penalty = beta * (int(k != prev_cluster) + int(k != next_cluster))
                    
                    # Combined distance
                    distances[k] = alpha * center_dist + temporal_penalty
                
                # Assign to cluster with minimum distance
                best_cluster = np.argmin(distances)
                if best_cluster != curr_cluster:
                    new_labels[i] = best_cluster
                    changes += 1
            
            self.logger.info(f"Temporal iteration {iteration+1}: {changes} changes")
            
            # Check for convergence
            if changes == 0:
                break
        
        # Reorder labels to original order
        reorder_idx = np.argsort(sort_idx)
        updated_labels = new_labels[reorder_idx]
        
        return updated_labels
    
    def _apply_contextual_constraints(self, Z, labels, c):
        """
        Apply contextual constraints to cluster assignments.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
        labels : numpy.ndarray
            Initial cluster labels.
        c : numpy.ndarray
            Contextual attributes.
            
        Returns
        -------
        numpy.ndarray
            Updated cluster labels.
        """
        self.logger.info("Applying contextual constraints...")
        
        # Group data by context
        unique_contexts = np.unique(c, axis=0)
        
        # Initialize new labels with original assignments
        new_labels = labels.copy()
        
        # Apply contextual coherence
        gamma = config.TEMPORAL_PARAMS['gamma']
        
        # Process each context separately
        for context in unique_contexts:
            # Find instances with this context
            context_mask = np.all(c == context, axis=1)
            context_indices = np.where(context_mask)[0]
            
            if len(context_indices) <= 1:
                continue
            
            # Get influence vectors for this context
            Z_context = Z[context_indices]
            
            # Get current labels for this context
            labels_context = labels[context_indices]
            
            # Calculate cluster distribution in this context
            cluster_counts = np.bincount(labels_context, minlength=self.n_clusters)
            dominant_cluster = np.argmax(cluster_counts)
            
            # Apply contextual coherence: instances close to cluster center
            # but assigned to different clusters may be reassigned
            for i, idx in enumerate(context_indices):
                curr_cluster = labels[idx]
                
                if curr_cluster != dominant_cluster:
                    # Distance to current cluster center
                    curr_dist = np.linalg.norm(Z[idx] - self.cluster_centers_[curr_cluster])
                    
                    # Distance to dominant cluster center
                    dom_dist = np.linalg.norm(Z[idx] - self.cluster_centers_[dominant_cluster])
                    
                    # If close enough to dominant cluster, reassign
                    if dom_dist < curr_dist * (1 + gamma):
                        new_labels[idx] = dominant_cluster
        
        return new_labels
