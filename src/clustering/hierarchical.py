"""
Hierarchical clustering implementation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.clustering.base_clustering import BaseClustering


class HierarchicalClustering(BaseClustering):
    """
    Hierarchical clustering implementation.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    linkage : str, default='ward'
        Linkage criterion. Options: 'ward', 'complete', 'average', 'single'.
    affinity : str, default='euclidean'
        Metric used to compute the linkage. Options: 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'.
    """

    def __init__(self, n_clusters=3, linkage='ward', affinity='euclidean', **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.linkage = linkage
        self.affinity = affinity
        self.logger = logging.getLogger(__name__)
        self.model = None

    def fit(self, Z):
        """
        Fit the hierarchical clustering algorithm to the influence space.

        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.

        Returns
        -------
        self
            Fitted clustering instance.
        """
        self.logger.info(f"Fitting hierarchical clustering with {self.n_clusters} clusters...")

        # Create and fit hierarchical clustering model
        # Note: 'ward' linkage only supports 'euclidean' affinity
        if self.linkage == 'ward':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage
            )
        else:
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                affinity=self.affinity
            )

        self.model.fit(Z)

        # Store labels
        self.labels_ = self.model.labels_

        # Compute cluster centers (not provided by AgglomerativeClustering)
        self._compute_cluster_centers(Z)

        self.is_fitted = True

        self.logger.info("Hierarchical clustering fitted successfully")
        return self

    def predict(self, Z):
        """
        Predict cluster labels for new data.

        Note: AgglomerativeClustering does not implement predict(). This method
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

    def get_linkage_matrix(self, Z):
        """
        Compute the linkage matrix for visualization.

        Note: This requires refitting the model with scipy's hierarchical clustering.

        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.

        Returns
        -------
        numpy.ndarray
            Linkage matrix.
        """
        from scipy.cluster.hierarchy import linkage

        # Compute linkage matrix
        if self.linkage == 'ward':
            method = 'ward'
        elif self.linkage == 'complete':
            method = 'complete'
        elif self.linkage == 'average':
            method = 'average'
        elif self.linkage == 'single':
            method = 'single'
        else:
            method = 'ward'

        return linkage(Z, method=method)
