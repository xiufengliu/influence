"""
Dynamic K-means Clustering Implementation for Influence-Based Time Series Clustering.

This module implements Algorithm 1 from the paper: "Learning Interpretable Dynamics: 
Influence-Based Clustering of Energy Consumption Time Series"

The algorithm optimizes a composite objective function that balances:
1. Cluster cohesion in influence space
2. Temporal continuity constraints  
3. Contextual alignment constraints
"""

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from src.clustering.base_clustering import BaseClustering


class DynamicKMeansClustering(BaseClustering):
    """
    Dynamic K-means clustering implementation based on Algorithm 1 from the paper.
    
    This algorithm incorporates temporal and contextual constraints into the 
    clustering objective function to maintain stable cluster assignments over time
    while ensuring contextual coherence.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form
    alpha : float, default=1.0
        Weight for cluster cohesion term
    beta : float, default=1.0
        Weight for temporal consistency term
    gamma : float, default=1.0
        Weight for contextual alignment term
    J_penalty : float, default=1.0
        Penalty value for temporal transitions
    max_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=42
        Random seed for reproducibility
    """

    def __init__(self, n_clusters=3, alpha=1.0, beta=1.0, gamma=1.0, J_penalty=1.0,
                 max_iter=300, tol=1e-4, random_state=42, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kwargs)
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        self.J_penalty = J_penalty
        self.max_iter = max_iter
        self.tol = tol
        self.logger = logging.getLogger(__name__)
        
        # Algorithm state
        self.cluster_centers_ = None
        self.context_centers_ = {}  # Centers for each context
        self.labels_ = None
        self.is_fitted = False
        self.objective_values_ = []
        
    def _compute_cohesion_cost(self, z_i, centroid):
        """Compute cohesion cost (alpha term in objective)"""
        return self.alpha * np.linalg.norm(z_i - centroid) ** 2
        
    def _compute_temporal_penalty(self, k, entity_prev_cluster):
        """Compute temporal penalty (beta term in objective)"""
        if entity_prev_cluster is None:
            return 0.0
        return self.beta * self.J_penalty * (k != entity_prev_cluster)
        
    def _compute_contextual_penalty(self, z_i, k, context_i):
        """Compute contextual penalty (gamma term in objective)"""
        if (k, context_i) not in self.context_centers_:
            return 0.0
        context_centroid = self.context_centers_[(k, context_i)]
        return self.gamma * np.linalg.norm(z_i - context_centroid) ** 2
        
    def _update_context_centers(self, Z, labels, contexts):
        """Update context-specific centroids"""
        self.context_centers_ = {}
        for k in range(self.n_clusters):
            for context in np.unique(contexts):
                mask = (labels == k) & (contexts == context)
                if np.sum(mask) > 0:
                    self.context_centers_[(k, context)] = np.mean(Z[mask], axis=0)
                    
    def _check_convergence(self, old_labels, new_labels):
        """Check convergence based on label changes"""
        if old_labels is None:
            return False
        return np.mean(old_labels == new_labels) > (1 - self.tol)
        
    def fit(self, Z, timestamps=None, contexts=None, entity_ids=None):
        """
        Fit the Dynamic K-means clustering algorithm.
        
        Parameters
        ----------
        Z : np.ndarray of shape (n_samples, n_features)
            Influence vectors in the influence space
        timestamps : np.ndarray of shape (n_samples,), optional
            Timestamps for temporal analysis
        contexts : np.ndarray of shape (n_samples,), optional
            Context attributes for each sample
        entity_ids : np.ndarray of shape (n_samples,), optional
            Entity identifiers for tracking temporal transitions
            
        Returns
        -------
        self : DynamicKMeansClustering
            Fitted clustering instance
        """
        self.logger.info("Starting Dynamic K-means clustering...")
        
        n_samples, n_features = Z.shape
        
        # Initialize with standard K-means++
        kmeans_init = KMeans(n_clusters=self.n_clusters, 
                           random_state=self.random_state,
                           n_init=1).fit(Z)
        
        # Initialize cluster assignments and centroids
        labels = kmeans_init.labels_.copy()
        self.cluster_centers_ = kmeans_init.cluster_centers_.copy()
        
        # Initialize context centers
        if contexts is not None:
            self._update_context_centers(Z, labels, contexts)
            
        # Set up entity tracking for temporal constraints
        entity_prev_clusters = {}
        if timestamps is not None and entity_ids is not None:
            # Sort by timestamp and build previous cluster mapping
            sorted_indices = np.argsort(timestamps)
            for idx in sorted_indices:
                entity_id = entity_ids[idx]
                if entity_id in entity_prev_clusters:
                    # Entity has previous cluster assignment
                    continue
                entity_prev_clusters[entity_id] = None
        
        # Main optimization loop (Algorithm 1)
        converged = False
        iteration = 0
        
        while iteration < self.max_iter and not converged:
            old_labels = labels.copy()
            
            # Assignment step: Optimize Eq. (7) from the paper
            for i in range(n_samples):
                z_i = Z[i]
                context_i = contexts[i] if contexts is not None else None
                entity_i = entity_ids[i] if entity_ids is not None else None
                
                # Get previous cluster for this entity
                entity_prev_cluster = entity_prev_clusters.get(entity_i, None)
                
                # Compute cost for each cluster
                min_cost = float('inf')
                best_cluster = labels[i]
                
                for k in range(self.n_clusters):
                    # Compute total cost (Eq. 7)
                    cohesion_cost = self._compute_cohesion_cost(z_i, self.cluster_centers_[k])
                    temporal_cost = self._compute_temporal_penalty(k, entity_prev_cluster)
                    contextual_cost = self._compute_contextual_penalty(z_i, k, context_i)
                    
                    total_cost = cohesion_cost + temporal_cost + contextual_cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_cluster = k
                        
                labels[i] = best_cluster
                
                # Update entity tracking
                if entity_i is not None:
                    entity_prev_clusters[entity_i] = best_cluster
            
            # Update step: Recalculate centroids
            for k in range(self.n_clusters):
                cluster_mask = labels == k
                if np.sum(cluster_mask) > 0:
                    self.cluster_centers_[k] = np.mean(Z[cluster_mask], axis=0)
                else:
                    # Reinitialize empty cluster
                    self.cluster_centers_[k] = Z[np.random.choice(n_samples)]
                    
            # Update context-specific centroids
            if contexts is not None:
                self._update_context_centers(Z, labels, contexts)
                
            # Check convergence
            converged = self._check_convergence(old_labels, labels)
            iteration += 1
            
            # Compute and store objective value for monitoring
            objective_value = self._compute_objective_value(Z, labels, contexts, entity_prev_clusters)
            self.objective_values_.append(objective_value)
            
            if iteration % 10 == 0:
                self.logger.debug(f"Iteration {iteration}, Objective: {objective_value:.4f}")
        
        self.labels_ = labels
        self.is_fitted = True
        
        self.logger.info(f"Converged in {iteration} iterations")
        return self
        
    def _compute_objective_value(self, Z, labels, contexts, entity_prev_clusters):
        """Compute the full objective function value (Eq. 1)"""
        # Cohesion term
        cohesion_loss = 0.0
        for i, label in enumerate(labels):
            cohesion_loss += np.linalg.norm(Z[i] - self.cluster_centers_[label]) ** 2
        cohesion_loss *= self.alpha
        
        # Temporal term (simplified for monitoring)
        temporal_loss = 0.0
        for entity_id, prev_cluster in entity_prev_clusters.items():
            if prev_cluster is not None:
                # Count transitions
                current_clusters = [labels[i] for i, eid in enumerate(entity_prev_clusters.keys()) 
                                  if eid == entity_id]
                if current_clusters and current_clusters[0] != prev_cluster:
                    temporal_loss += 1.0
        temporal_loss *= self.beta * self.J_penalty
        
        # Contextual term
        contextual_loss = 0.0
        if contexts is not None:
            for i, (label, context) in enumerate(zip(labels, contexts)):
                if (label, context) in self.context_centers_:
                    contextual_loss += np.linalg.norm(Z[i] - self.context_centers_[(label, context)]) ** 2
        contextual_loss *= self.gamma
        
        return cohesion_loss + temporal_loss + contextual_loss
        
    def predict(self, Z):
        """
        Predict cluster labels for new data.
        
        Parameters
        ----------
        Z : np.ndarray of shape (n_samples, n_features)
            Influence vectors to predict
            
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Simple assignment to nearest centroid for new data
        distances = np.linalg.norm(Z[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
        
    def get_transition_matrix(self, entity_ids, timestamps, labels=None):
        """
        Compute transition matrix for Markov analysis.
        
        Parameters
        ----------
        entity_ids : np.ndarray
            Entity identifiers
        timestamps : np.ndarray
            Timestamps for ordering
        labels : np.ndarray, optional
            Cluster labels (uses self.labels_ if None)
            
        Returns
        -------
        transition_matrix : np.ndarray of shape (n_clusters, n_clusters)
            Transition probability matrix
        """
        if labels is None:
            labels = self.labels_
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing transitions")
            
        # Build transition counts
        transition_counts = np.zeros((self.n_clusters, self.n_clusters))
        
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        entity_states = {}
        
        for idx in sorted_indices:
            entity_id = entity_ids[idx]
            current_state = labels[idx]
            
            if entity_id in entity_states:
                prev_state = entity_states[entity_id]
                transition_counts[prev_state, current_state] += 1
                
            entity_states[entity_id] = current_state
            
        # Convert to probabilities
        transition_matrix = np.zeros_like(transition_counts)
        for i in range(self.n_clusters):
            row_sum = np.sum(transition_counts[i])
            if row_sum > 0:
                transition_matrix[i] = transition_counts[i] / row_sum
                
        return transition_matrix
        current_context = contexts_window[i:i+1, :] # Ensure current_context is (1, num_features)

        # Get all instances in the target cluster k that share the same context
        cluster_context_indices = np.where(np.all(contexts_window == current_context, axis=1) & (current_window_labels == k))[0]
        
        if len(cluster_context_indices) > 0:
            # Calculate the centroid of the context-specific sub-cluster
            context_specific_centroid = np.mean(Z_window[cluster_context_indices], axis=0)
            # The penalty is the distance to this context-specific centroid
            total_penalty = np.linalg.norm(Z_window[i] - context_specific_centroid)**2
        
        return total_penalty


    def fit(self, Z, timestamps, contexts, entity_ids):
        """
        Fit the Dynamic K-means clustering algorithm to the influence space.
        Processes data in time windows.
        """
        self.logger.info(f"Fitting Dynamic K-means with alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")

        if self.beta > 0 and (timestamps is None or entity_ids is None):
            raise ValueError("Timestamps and entity_ids are required for temporal penalty (beta > 0).")
        if self.gamma > 0 and contexts is None:
            raise ValueError("Contexts are required for contextual penalty (gamma > 0).")

        timestamps_np = np.array(timestamps)
        if not np.all(timestamps_np[:-1] <= timestamps_np[1:]):
            raise ValueError("Timestamps must be sorted.")

        unique_timestamps = np.unique(timestamps)
        num_windows = (len(unique_timestamps) + self.window_size - 1) // self.window_size

        prev_window_entity_assignments = {}
        all_labels = np.full(Z.shape[0], -1, dtype=int)

        for window_idx in range(num_windows):
            self.logger.info(f"Processing window {window_idx + 1}/{num_windows}")
            start_time_idx = window_idx * self.window_size
            end_time_idx = min((window_idx + 1) * self.window_size, len(unique_timestamps))
            
            current_window_timestamps = unique_timestamps[start_time_idx:end_time_idx]
            
            if len(current_window_timestamps) == 0:
                continue

            window_data_indices = np.where(
                (timestamps_np >= current_window_timestamps[0]) & 
                (timestamps_np <= current_window_timestamps[-1])
            )[0]

            if len(window_data_indices) == 0:
                self.logger.warning(f"Window {window_idx + 1} has no data points. Skipping.")
                continue

            Z_window = Z[window_data_indices]
            contexts_window = contexts[window_data_indices] if self.gamma > 0 else None
            entity_ids_window = entity_ids[window_data_indices] if self.beta > 0 else None

            kmeans_init = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
            kmeans_init.fit(Z_window)
            current_window_labels = kmeans_init.labels_
            current_window_centers = kmeans_init.cluster_centers_

            for iter_count in range(self.max_iter):
                new_labels_iter = np.copy(current_window_labels)
                
                # Assignment step
                for i in range(Z_window.shape[0]):
                    min_cost = float('inf')
                    best_k = -1
                    
                    for k in range(self.n_clusters):
                        cohesion_cost = self.alpha * np.linalg.norm(Z_window[i] - current_window_centers[k])**2
                        
                        temporal_cost = 0.0
                        if self.beta > 0:
                            temporal_cost = self.beta * self._temporal_penalty(entity_ids_window[i], k, prev_window_entity_assignments)
                        
                        context_cost = 0.0
                        if self.gamma > 0:
                            context_cost = self.gamma * self._contextual_penalty(i, k, contexts_window, Z_window, current_window_labels)
                        
                        total_cost = cohesion_cost + temporal_cost + context_cost
                        
                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_k = k
                    new_labels_iter[i] = best_k
            
                # Update step
                for k in range(self.n_clusters):
                    points_in_cluster = Z_window[new_labels_iter == k]
                    if len(points_in_cluster) > 0:
                        current_window_centers[k] = np.mean(points_in_cluster, axis=0)
                    else:
                        self.logger.warning(f"Window {window_idx + 1}, Cluster {k} became empty. Reinitializing centroid.")
                        distances_to_all_centroids = np.linalg.norm(Z_window[:, np.newaxis] - current_window_centers, axis=2)
                        min_distances = np.min(distances_to_all_centroids, axis=1)
                        furthest_point_idx = np.argmax(min_distances)
                        current_window_centers[k] = Z_window[furthest_point_idx]

                if np.all(new_labels_iter == current_window_labels):
                    self.logger.debug(f"Window {window_idx + 1} converged at iteration {iter_count + 1}.")
                    break
                current_window_labels = new_labels_iter

            all_labels[window_data_indices] = current_window_labels
            self.cluster_centers_ = current_window_centers

            if self.beta > 0:
                prev_window_entity_assignments.update({entity_ids_window[i]: current_window_labels[i] for i in range(len(entity_ids_window))})

        self.labels_ = all_labels
        self.is_fitted = True
        self.logger.info("Dynamic K-means fitted successfully across all windows.")
        return self

    def predict(self, Z):
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() before predict().")
        
        distances = np.linalg.norm(Z[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

    def get_cluster_centers(self):
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() before get_cluster_centers().")
        return self.cluster_centers_