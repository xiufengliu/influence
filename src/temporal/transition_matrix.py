"""
Transition matrix computation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
import pandas as pd


class TransitionMatrix:
    """
    Class for computing transition matrices between clusters over time.
    
    This class implements methods to analyze how cluster assignments evolve over time,
    providing insights into pattern transitions and stability.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.transition_matrix = None
        self.stationary_distribution = None
    
    def compute(self, clusters, timestamps, time_window=None):
        """
        Compute the transition matrix between clusters.
        
        Parameters
        ----------
        clusters : numpy.ndarray
            Cluster assignments.
        timestamps : numpy.ndarray or pandas.Series
            Timestamps for each instance.
        time_window : str, default=None
            Time window for grouping timestamps (e.g., 'D' for daily, 'H' for hourly).
            If None, uses the natural ordering of timestamps.
            
        Returns
        -------
        numpy.ndarray
            Transition matrix P where P[i,j] is the probability of transitioning
            from cluster i to cluster j.
        """
        self.logger.info("Computing transition matrix...")
        
        # Convert timestamps to pandas Series if not already
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        # Convert clusters to numpy array if not already
        clusters = np.array(clusters)
        
        # Get number of clusters
        n_clusters = len(np.unique(clusters))
        
        # Group by time periods if time_window is specified
        if time_window is not None:
            # Create time period labels
            time_periods = timestamps.dt.to_period(time_window)
            
            # Group clusters by time period
            df = pd.DataFrame({'cluster': clusters, 'time_period': time_periods})
            grouped = df.groupby('time_period')['cluster'].apply(lambda x: x.mode()[0]).reset_index()
            
            # Get ordered clusters
            ordered_clusters = grouped['cluster'].values
        else:
            # Sort clusters by timestamp
            sorted_indices = np.argsort(timestamps)
            ordered_clusters = clusters[sorted_indices]
        
        # Initialize transition count matrix
        transition_counts = np.zeros((n_clusters, n_clusters))
        
        # Count transitions
        for i in range(len(ordered_clusters) - 1):
            from_cluster = ordered_clusters[i]
            to_cluster = ordered_clusters[i + 1]
            transition_counts[from_cluster, to_cluster] += 1
        
        # Compute transition probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        
        # Handle zero rows (no outgoing transitions)
        zero_rows = row_sums == 0
        row_sums[zero_rows] = 1  # Avoid division by zero
        
        transition_matrix = transition_counts / row_sums
        
        # For rows with no outgoing transitions, set equal probability to all states
        transition_matrix[zero_rows.flatten()] = 1.0 / n_clusters
        
        self.transition_matrix = transition_matrix
        
        # Compute stationary distribution
        self._compute_stationary_distribution()
        
        self.logger.info("Transition matrix computed successfully")
        return transition_matrix
    
    def _compute_stationary_distribution(self):
        """
        Compute the stationary distribution of the transition matrix.
        
        The stationary distribution π satisfies π = π * P, where P is the transition matrix.
        """
        if self.transition_matrix is None:
            self.logger.error("Transition matrix not computed. Call compute() first.")
            raise ValueError("Transition matrix not computed. Call compute() first.")
        
        # Initialize with uniform distribution
        n_clusters = self.transition_matrix.shape[0]
        pi = np.ones(n_clusters) / n_clusters
        
        # Power iteration to find stationary distribution
        max_iter = 1000
        tol = 1e-8
        
        for _ in range(max_iter):
            pi_new = pi @ self.transition_matrix
            
            # Check convergence
            if np.max(np.abs(pi_new - pi)) < tol:
                break
            
            pi = pi_new
        
        self.stationary_distribution = pi
        return pi
    
    def get_n_step_transition(self, n_steps):
        """
        Compute the n-step transition matrix.
        
        Parameters
        ----------
        n_steps : int
            Number of steps.
            
        Returns
        -------
        numpy.ndarray
            n-step transition matrix.
        """
        if self.transition_matrix is None:
            self.logger.error("Transition matrix not computed. Call compute() first.")
            raise ValueError("Transition matrix not computed. Call compute() first.")
        
        # Compute matrix power
        return np.linalg.matrix_power(self.transition_matrix, n_steps)
    
    def get_expected_time_to_cluster(self, target_cluster):
        """
        Compute the expected number of steps to reach a target cluster from each cluster.
        
        Parameters
        ----------
        target_cluster : int
            Target cluster index.
            
        Returns
        -------
        numpy.ndarray
            Expected number of steps to reach the target cluster from each cluster.
        """
        if self.transition_matrix is None:
            self.logger.error("Transition matrix not computed. Call compute() first.")
            raise ValueError("Transition matrix not computed. Call compute() first.")
        
        n_clusters = self.transition_matrix.shape[0]
        
        # Check if target cluster is valid
        if target_cluster < 0 or target_cluster >= n_clusters:
            self.logger.error(f"Invalid target cluster: {target_cluster}. Must be between 0 and {n_clusters-1}.")
            raise ValueError(f"Invalid target cluster: {target_cluster}. Must be between 0 and {n_clusters-1}.")
        
        # Initialize expected times
        expected_times = np.zeros(n_clusters)
        
        # For each starting cluster
        for i in range(n_clusters):
            if i == target_cluster:
                expected_times[i] = 0
                continue
            
            # Create modified transition matrix where target cluster is absorbing
            P_modified = self.transition_matrix.copy()
            P_modified[target_cluster, :] = 0
            P_modified[target_cluster, target_cluster] = 1
            
            # Compute fundamental matrix
            Q = P_modified.copy()
            Q = np.delete(Q, target_cluster, axis=0)
            Q = np.delete(Q, target_cluster, axis=1)
            
            # Compute expected time
            N = np.linalg.inv(np.eye(n_clusters - 1) - Q)
            
            # Adjust index if needed
            idx = i if i < target_cluster else i - 1
            expected_times[i] = np.sum(N[idx, :])
        
        return expected_times
    
    def get_cluster_stability(self):
        """
        Compute the stability of each cluster.
        
        Stability is defined as the probability of remaining in the same cluster
        in the next time step.
        
        Returns
        -------
        numpy.ndarray
            Stability scores for each cluster.
        """
        if self.transition_matrix is None:
            self.logger.error("Transition matrix not computed. Call compute() first.")
            raise ValueError("Transition matrix not computed. Call compute() first.")
        
        # Diagonal elements represent self-transition probabilities
        return np.diag(self.transition_matrix)
