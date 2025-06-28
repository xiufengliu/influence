"""
Comprehensive evaluation metrics for Dynamic Influence-Based Clustering.

This module implements all evaluation metrics described in the paper:
"Learning Interpretable Dynamics: Influence-Based Clustering of Energy Consumption Time Series"

Metrics include:
- Silhouette Score
- Davies-Bouldin Index  
- Temporal Consistency (TC)
- Normalized Information Gain (NIG)
- Cluster Stability
- Anomaly Rate
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
import logging


class ClusteringEvaluator:
    """
    Comprehensive evaluator for clustering performance with temporal and contextual metrics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def evaluate_comprehensive(self, 
                             Z: np.ndarray,
                             labels: np.ndarray, 
                             timestamps: Optional[np.ndarray] = None,
                             contexts: Optional[np.ndarray] = None,
                             entity_ids: Optional[np.ndarray] = None,
                             transition_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Parameters
        ----------
        Z : np.ndarray
            Influence vectors or feature matrix
        labels : np.ndarray  
            Cluster assignments
        timestamps : np.ndarray, optional
            Timestamps for temporal analysis
        contexts : np.ndarray, optional
            Context variables for NIG calculation
        entity_ids : np.ndarray, optional
            Entity identifiers for temporal consistency
        transition_matrix : np.ndarray, optional
            Precomputed transition matrix
            
        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Basic cluster quality metrics
        results.update(self._compute_intrinsic_metrics(Z, labels))
        
        # Temporal metrics
        if timestamps is not None and entity_ids is not None:
            results.update(self._compute_temporal_metrics(labels, timestamps, entity_ids))
            
        # Contextual metrics
        if contexts is not None:
            results.update(self._compute_contextual_metrics(labels, contexts))
            
        # Stability metrics
        if transition_matrix is not None:
            results.update(self._compute_stability_metrics(transition_matrix))
            
        return results
        
    def _compute_intrinsic_metrics(self, Z: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute intrinsic cluster quality metrics."""
        metrics = {}
        
        # Silhouette Score (higher is better)
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(Z, labels)
        else:
            metrics['silhouette_score'] = -1.0
            
        # Davies-Bouldin Index (lower is better)  
        if len(np.unique(labels)) > 1:
            metrics['davies_bouldin_score'] = davies_bouldin_score(Z, labels)
        else:
            metrics['davies_bouldin_score'] = float('inf')
            
        # Calinski-Harabasz Index (higher is better)
        if len(np.unique(labels)) > 1:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(Z, labels)
        else:
            metrics['calinski_harabasz_score'] = 0.0
            
        return metrics
        
    def _compute_temporal_metrics(self, 
                                labels: np.ndarray, 
                                timestamps: np.ndarray, 
                                entity_ids: np.ndarray) -> Dict[str, float]:
        """Compute temporal consistency metrics."""
        metrics = {}
        
        # Temporal Consistency (TC) using Jaccard Index
        tc_scores = []
        
        # Group by entity and compute consistency
        df = pd.DataFrame({
            'entity_id': entity_ids,
            'timestamp': timestamps, 
            'label': labels
        })
        
        for entity_id in df['entity_id'].unique():
            entity_data = df[df['entity_id'] == entity_id].sort_values('timestamp')
            if len(entity_data) < 2:
                continue
                
            # Compute pairwise Jaccard similarity for consecutive time windows
            entity_labels = entity_data['label'].values
            consistency_scores = []
            
            # Use sliding window approach
            window_size = max(1, len(entity_labels) // 10)  # Adaptive window size
            
            for i in range(len(entity_labels) - window_size):
                window1 = set(entity_labels[i:i+window_size])
                window2 = set(entity_labels[i+1:i+1+window_size])
                
                if len(window1) == 0 and len(window2) == 0:
                    jaccard = 1.0
                elif len(window1.union(window2)) == 0:
                    jaccard = 0.0
                else:
                    jaccard = len(window1.intersection(window2)) / len(window1.union(window2))
                    
                consistency_scores.append(jaccard)
                
            if consistency_scores:
                tc_scores.extend(consistency_scores)
                
        metrics['temporal_consistency'] = np.mean(tc_scores) if tc_scores else 0.0
        
        return metrics
        
    def _compute_contextual_metrics(self, labels: np.ndarray, contexts: np.ndarray) -> Dict[str, float]:
        """Compute contextual alignment metrics."""
        metrics = {}
        
        # Handle both single and multi-dimensional contexts
        if contexts.ndim == 1:
            contexts = contexts.reshape(-1, 1)
            
        # Normalized Information Gain for each context dimension
        nig_scores = []
        
        for dim in range(contexts.shape[1]):
            context_dim = contexts[:, dim]
            
            # Remove any NaN values
            valid_mask = ~(pd.isna(context_dim) | pd.isna(labels))
            if np.sum(valid_mask) < 2:
                continue
                
            context_clean = context_dim[valid_mask]
            labels_clean = labels[valid_mask]
            
            # Compute normalized mutual information
            if len(np.unique(context_clean)) > 1 and len(np.unique(labels_clean)) > 1:
                nmi = normalized_mutual_info_score(labels_clean, context_clean)
                nig_scores.append(nmi)
                
        metrics['normalized_information_gain'] = np.mean(nig_scores) if nig_scores else 0.0
        
        # Contextual Entropy (lower is better)
        contextual_entropies = []
        
        for k in np.unique(labels):
            cluster_mask = labels == k
            if np.sum(cluster_mask) < 2:
                continue
                
            cluster_contexts = contexts[cluster_mask]
            
            # Compute entropy for this cluster across context dimensions
            for dim in range(contexts.shape[1]):
                context_values = cluster_contexts[:, dim]
                valid_values = context_values[~pd.isna(context_values)]
                
                if len(valid_values) < 2:
                    continue
                    
                # Compute entropy
                unique_values, counts = np.unique(valid_values, return_counts=True)
                probabilities = counts / len(valid_values)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                contextual_entropies.append(entropy)
                
        metrics['contextual_entropy'] = np.mean(contextual_entropies) if contextual_entropies else 0.0
        
        return metrics
        
    def _compute_stability_metrics(self, transition_matrix: np.ndarray) -> Dict[str, float]:
        """Compute cluster stability metrics from transition matrix."""
        metrics = {}
        
        # Cluster Stability (average diagonal of transition matrix)
        diagonal_values = np.diag(transition_matrix)
        metrics['cluster_stability'] = np.mean(diagonal_values)
        
        # Anomaly Rate (percentage of low-probability transitions)
        anomaly_threshold = 0.05
        anomaly_count = np.sum(transition_matrix < anomaly_threshold) - np.sum(transition_matrix == 0)
        total_transitions = np.sum(transition_matrix > 0)
        
        if total_transitions > 0:
            metrics['anomaly_rate'] = anomaly_count / total_transitions
        else:
            metrics['anomaly_rate'] = 0.0
            
        # Stationary Distribution Analysis
        try:
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
            stationary_idx = np.argmax(np.real(eigenvalues))
            stationary_dist = np.real(eigenvectors[:, stationary_idx])
            stationary_dist = stationary_dist / np.sum(stationary_dist)
            
            # Entropy of stationary distribution
            stationary_entropy = -np.sum(stationary_dist * np.log2(stationary_dist + 1e-10))
            metrics['stationary_entropy'] = stationary_entropy
            
        except Exception as e:
            self.logger.warning(f"Could not compute stationary distribution: {e}")
            metrics['stationary_entropy'] = 0.0
            
        return metrics


# Legacy functions for backward compatibility
def evaluate_clustering(Z, labels):
    """Legacy function for basic clustering evaluation."""
    evaluator = ClusteringEvaluator()
    return evaluator._compute_intrinsic_metrics(Z, labels)


def calculate_temporal_consistency(labels, timestamps, entity_ids, window_size=24):
    """Legacy function for temporal consistency calculation.""" 
    evaluator = ClusteringEvaluator()
    return evaluator._compute_temporal_metrics(labels, timestamps, entity_ids)


def calculate_conditional_entropy(labels, contexts):
    """Legacy function for conditional entropy calculation."""
    evaluator = ClusteringEvaluator()
    result = evaluator._compute_contextual_metrics(labels, contexts)
    return result.get('contextual_entropy', 0.0)


def calculate_entropy(labels):
    """Calculate entropy of cluster labels."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def calculate_spearman_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate Spearman rank correlations between features and target.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray  
        Target variable
        
    Returns
    -------
    np.ndarray
        Array of absolute Spearman correlation coefficients
    """
    correlations = []
    
    for i in range(X.shape[1]):
        corr, _ = spearmanr(X[:, i], y)
        correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
    return np.array(correlations)


def evaluate_clustering_stability(labels_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Evaluate clustering stability across multiple runs.
    
    Parameters
    ----------
    labels_list : List[np.ndarray]
        List of label arrays from multiple clustering runs
        
    Returns
    -------
    Dict[str, float]
        Stability metrics
    """
    if len(labels_list) < 2:
        return {'stability_mean': 1.0, 'stability_std': 0.0}
        
    # Compute pairwise adjusted mutual information
    ami_scores = []
    
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            ami = adjusted_mutual_info_score(labels_list[i], labels_list[j])
            ami_scores.append(ami)
            
    return {
        'stability_mean': np.mean(ami_scores),
        'stability_std': np.std(ami_scores)
    }
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Check if we have at least 2 clusters (required for most metrics)
    if n_clusters < 2:
        logger.warning(f"Only {n_clusters} cluster found. At least 2 clusters are required for evaluation metrics.")
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
            'entropy': calculate_entropy(labels) if n_clusters > 0 else np.nan
        }

    if n_samples < n_clusters + 1:
        logger.warning(f"Not enough samples ({n_samples}) for {n_clusters} clusters. Skipping evaluation.")
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
            'entropy': calculate_entropy(labels)
        }

    # Calculate silhouette score
    try:
        silhouette = silhouette_score(Z, labels)
    except Exception as e:
        logger.warning(f"Error calculating silhouette score: {e}")
        silhouette = np.nan

    # Calculate Davies-Bouldin index
    try:
        davies_bouldin = davies_bouldin_score(Z, labels)
    except Exception as e:
        logger.warning(f"Error calculating Davies-Bouldin index: {e}")
        davies_bouldin = np.nan

    # Calculate Calinski-Harabasz index
    try:
        calinski_harabasz = calinski_harabasz_score(Z, labels)
    except Exception as e:
        logger.warning(f"Error calculating Calinski-Harabasz index: {e}")
        calinski_harabasz = np.nan

    # Calculate entropy
    try:
        entropy = calculate_entropy(labels)
    except Exception as e:
        logger.warning(f"Error calculating entropy: {e}")
        entropy = np.nan

    metrics = {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'entropy': entropy
    }

    logger.info(f"Clustering evaluation metrics: {metrics}")
    return metrics


def calculate_entropy(labels):
    """
    Calculate entropy of cluster assignments.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster labels.

    Returns
    -------
    float
        Entropy value.
    """
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(labels)

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def calculate_conditional_entropy(labels, context):
    """
    Calculate conditional entropy of cluster assignments given context.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster labels.
    context : numpy.ndarray
        Contextual attributes.

    Returns
    -------
    float
        Conditional entropy value.
    """
    # Handle multi-dimensional context by treating each row as a unique context combination
    if context.ndim > 1:
        # Get unique rows (context combinations) and their inverse indices
        unique_contexts_rows, inverse_indices = np.unique(context, axis=0, return_inverse=True)
        unique_contexts = [tuple(row) for row in unique_contexts_rows]
    else:
        unique_contexts = np.unique(context)

    conditional_entropy = 0.0

    for i, ctx_val in enumerate(unique_contexts):
        if context.ndim > 1:
            # Create mask for current unique context row
            ctx_mask = (inverse_indices == i)
        else:
            ctx_mask = (context == ctx_val)

        ctx_labels = labels[ctx_mask]

        if len(ctx_labels) == 0:
            continue

        ctx_entropy = calculate_entropy(ctx_labels)
        ctx_prob = np.sum(ctx_mask) / len(labels)
        conditional_entropy += ctx_prob * ctx_entropy

    return conditional_entropy


def calculate_temporal_consistency(labels, timestamps, time_window=None):
    """
    Calculate temporal consistency of cluster assignments.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster labels.
    timestamps : numpy.ndarray or pandas.Series
        Timestamps for each instance.
    time_window : str, default=None
        Time window for grouping timestamps (e.g., 'D' for daily, 'H' for hourly).
        If None, uses consecutive instances.

    Returns
    -------
    float
        Temporal consistency score between 0 and 1.
    """
    import pandas as pd

    # Convert timestamps to pandas Series if not already
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)

    # Group by time periods if time_window is specified
    if time_window is not None:
        # Create time period labels
        time_periods = timestamps.dt.to_period(time_window)

        # Group labels by time period
        df = pd.DataFrame({'label': labels, 'time_period': time_periods})
        grouped = df.groupby('time_period')['label'].apply(lambda x: x.mode()[0]).reset_index()

        # Get ordered labels
        ordered_labels = grouped['label'].values
    else:
        # Sort labels by timestamp
        sorted_indices = np.argsort(timestamps)
        ordered_labels = labels[sorted_indices]

    # Count consistent transitions (same cluster in consecutive time steps)
    consistent_transitions = np.sum(ordered_labels[:-1] == ordered_labels[1:])

    # Calculate consistency score
    consistency_score = consistent_transitions / (len(ordered_labels) - 1)

    return consistency_score
