"""
Anomaly detection for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
import pandas as pd


class AnomalyDetection:
    """
    Class for detecting anomalies in cluster transitions.
    
    This class implements methods to identify rare or unexpected transitions
    between clusters, which may indicate anomalous energy consumption patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.anomaly_scores = None
        self.anomaly_threshold = None
        self.anomalies = None
    
    def detect(self, clusters, transition_matrix, timestamps, threshold=0.05, time_window=None):
        """
        Detect anomalies in cluster transitions.
        
        Parameters
        ----------
        clusters : numpy.ndarray
            Cluster assignments.
        transition_matrix : numpy.ndarray
            Transition matrix between clusters.
        timestamps : numpy.ndarray or pandas.Series
            Timestamps for each instance.
        threshold : float, default=0.05
            Probability threshold for anomaly detection. Transitions with
            probability below this threshold are considered anomalous.
        time_window : str, default=None
            Time window for grouping timestamps (e.g., 'D' for daily, 'H' for hourly).
            If None, uses the natural ordering of timestamps.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing anomalies with timestamps, from_cluster, to_cluster,
            transition_probability, and anomaly_score.
        """
        self.logger.info("Detecting anomalies in cluster transitions...")
        
        # Convert timestamps to pandas Series if not already
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        # Convert clusters to numpy array if not already
        clusters = np.array(clusters)
        
        # Group by time periods if time_window is specified
        if time_window is not None:
            # Create time period labels
            time_periods = timestamps.dt.to_period(time_window)
            
            # Group clusters by time period
            df = pd.DataFrame({'cluster': clusters, 'time_period': time_periods, 'timestamp': timestamps})
            grouped = df.groupby('time_period')['cluster'].apply(lambda x: x.mode()[0]).reset_index()
            
            # Get ordered clusters and corresponding timestamps
            ordered_clusters = grouped['cluster'].values
            ordered_timestamps = grouped['time_period'].dt.to_timestamp()
        else:
            # Sort clusters by timestamp
            sorted_indices = np.argsort(timestamps)
            ordered_clusters = clusters[sorted_indices]
            ordered_timestamps = timestamps.iloc[sorted_indices].values
        
        # Initialize anomaly detection
        anomalies = []
        
        # Detect anomalies
        for i in range(len(ordered_clusters) - 1):
            from_cluster = ordered_clusters[i]
            to_cluster = ordered_clusters[i + 1]
            
            # Get transition probability
            transition_prob = transition_matrix[from_cluster, to_cluster]
            
            # Compute anomaly score (inverse of probability)
            anomaly_score = 1.0 - transition_prob
            
            # Check if anomalous
            if transition_prob < threshold:
                anomalies.append({
                    'timestamp': ordered_timestamps[i],
                    'from_cluster': from_cluster,
                    'to_cluster': to_cluster,
                    'transition_probability': transition_prob,
                    'anomaly_score': anomaly_score
                })
        
        # Create DataFrame of anomalies
        anomalies_df = pd.DataFrame(anomalies)
        
        # Store results
        self.anomaly_scores = anomaly_score
        self.anomaly_threshold = threshold
        self.anomalies = anomalies_df
        
        self.logger.info(f"Detected {len(anomalies_df)} anomalies")
        return anomalies_df
    
    def detect_contextual_anomalies(self, clusters, contexts, transition_matrix, timestamps, 
                                   threshold=0.05, time_window=None):
        """
        Detect contextual anomalies in cluster transitions.
        
        This method identifies transitions that are anomalous within specific contexts.
        
        Parameters
        ----------
        clusters : numpy.ndarray
            Cluster assignments.
        contexts : numpy.ndarray
            Contextual attributes for each instance.
        transition_matrix : numpy.ndarray
            Transition matrix between clusters.
        timestamps : numpy.ndarray or pandas.Series
            Timestamps for each instance.
        threshold : float, default=0.05
            Probability threshold for anomaly detection. Transitions with
            probability below this threshold are considered anomalous.
        time_window : str, default=None
            Time window for grouping timestamps (e.g., 'D' for daily, 'H' for hourly).
            If None, uses the natural ordering of timestamps.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing contextual anomalies with timestamps, from_cluster,
            to_cluster, context, transition_probability, and anomaly_score.
        """
        self.logger.info("Detecting contextual anomalies in cluster transitions...")
        
        # Convert timestamps to pandas Series if not already
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        # Convert clusters and contexts to numpy arrays if not already
        clusters = np.array(clusters)
        contexts = np.array(contexts)
        
        # Create DataFrame with clusters, contexts, and timestamps
        df = pd.DataFrame({
            'cluster': clusters,
            'timestamp': timestamps
        })
        
        # Add context columns
        for i in range(contexts.shape[1]):
            df[f'context_{i}'] = contexts[:, i]
        
        # Group by time periods if time_window is specified
        if time_window is not None:
            # Create time period labels
            df['time_period'] = timestamps.dt.to_period(time_window)
            
            # Group by time period
            grouped = df.groupby('time_period').agg({
                'cluster': lambda x: x.mode()[0],
                'timestamp': 'first'
            })
            
            # Add context columns to grouped data
            for i in range(contexts.shape[1]):
                grouped[f'context_{i}'] = df.groupby('time_period')[f'context_{i}'].first()
            
            # Reset index
            grouped = grouped.reset_index()
            
            # Get ordered data
            ordered_df = grouped
        else:
            # Sort by timestamp
            ordered_df = df.sort_values('timestamp')
        
        # Initialize anomaly detection
        anomalies = []
        
        # Compute context-specific transition matrices
        context_matrices = {}
        
        # Detect anomalies
        for i in range(len(ordered_df) - 1):
            from_cluster = ordered_df['cluster'].iloc[i]
            to_cluster = ordered_df['cluster'].iloc[i + 1]
            
            # Get context
            context_cols = [col for col in ordered_df.columns if col.startswith('context_')]
            context = tuple(ordered_df[context_cols].iloc[i].values)
            
            # Get or compute context-specific transition matrix
            if context not in context_matrices:
                # Filter data for this context
                context_mask = np.all(contexts == context, axis=1)
                context_clusters = clusters[context_mask]
                
                # Compute transition counts
                n_clusters = len(np.unique(clusters))
                context_counts = np.zeros((n_clusters, n_clusters))
                
                for j in range(len(context_clusters) - 1):
                    from_c = context_clusters[j]
                    to_c = context_clusters[j + 1]
                    context_counts[from_c, to_c] += 1
                
                # Compute transition probabilities
                row_sums = context_counts.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                context_matrix = context_counts / row_sums
                
                # Store matrix
                context_matrices[context] = context_matrix
            
            # Get transition probability from context-specific matrix
            context_matrix = context_matrices[context]
            transition_prob = context_matrix[from_cluster, to_cluster]
            
            # If no transitions observed in this context, use global matrix
            if transition_prob == 0:
                transition_prob = transition_matrix[from_cluster, to_cluster]
            
            # Compute anomaly score (inverse of probability)
            anomaly_score = 1.0 - transition_prob
            
            # Check if anomalous
            if transition_prob < threshold:
                anomalies.append({
                    'timestamp': ordered_df['timestamp'].iloc[i],
                    'from_cluster': from_cluster,
                    'to_cluster': to_cluster,
                    'context': context,
                    'transition_probability': transition_prob,
                    'anomaly_score': anomaly_score
                })
        
        # Create DataFrame of anomalies
        anomalies_df = pd.DataFrame(anomalies)
        
        self.logger.info(f"Detected {len(anomalies_df)} contextual anomalies")
        return anomalies_df
