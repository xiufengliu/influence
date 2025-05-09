"""
Evaluation metrics for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_clustering(Z, labels):
    """
    Evaluate clustering quality using various metrics.

    Parameters
    ----------
    Z : numpy.ndarray
        Influence space matrix.
    labels : numpy.ndarray
        Cluster labels.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating clustering quality...")

    # Check if there are enough samples and clusters for evaluation
    n_samples = len(Z)
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
    # Convert context to tuple of values for each instance
    if context.ndim > 1:
        context_tuples = [tuple(c) for c in context]
    else:
        context_tuples = context

    # Get unique contexts
    unique_contexts = np.unique(context_tuples)

    # Calculate conditional entropy
    conditional_entropy = 0

    for ctx in unique_contexts:
        # Get labels for this context
        ctx_mask = np.array(context_tuples) == ctx
        ctx_labels = labels[ctx_mask]

        # Calculate entropy for this context
        ctx_entropy = calculate_entropy(ctx_labels)

        # Weight by context probability
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
