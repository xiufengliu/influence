"""
Spearman-based influence generation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from scipy.stats import spearmanr


class SpearmanInfluence:
    """
    Spearman-based influence generation.
    
    This class uses Spearman rank correlation to generate feature importance scores
    that represent the influence of each feature on the target variable.
    
    Parameters
    ----------
    method : str, default='spearman'
        Correlation method. Options: 'spearman', 'pearson'.
    """
    
    def __init__(self, method='spearman'):
        self.method = method
        self.logger = logging.getLogger(__name__)
    
    def generate_influence(self, model, X):
        """
        Generate influence scores using Spearman correlation.
        
        Parameters
        ----------
        model : BaseModel
            Fitted model instance.
        X : numpy.ndarray
            Feature matrix.
            
        Returns
        -------
        numpy.ndarray
            Influence scores matrix with the same shape as X.
        """
        self.logger.info(f"Generating {self.method}-based influence scores...")
        
        if not model.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before generate_influence().")
            raise ValueError("Model is not fitted. Call fit() before generate_influence().")
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate influence scores
        influence_scores = np.zeros_like(X, dtype=float)
        
        for i in range(X.shape[1]):
            if self.method == 'spearman':
                corr, _ = spearmanr(X[:, i], y_pred)
            else:  # pearson
                corr = np.corrcoef(X[:, i], y_pred)[0, 1]
            
            # Handle NaN values
            if np.isnan(corr):
                corr = 0
            
            # Assign correlation as influence score for all instances
            influence_scores[:, i] = corr
        
        self.logger.info(f"{self.method.capitalize()} influence scores generated with shape {influence_scores.shape}")
        
        return influence_scores
    
    def generate_influence_local(self, model, X, window_size=100):
        """
        Generate local influence scores using Spearman correlation within sliding windows.
        
        Parameters
        ----------
        model : BaseModel
            Fitted model instance.
        X : numpy.ndarray
            Feature matrix.
        window_size : int, default=100
            Size of the sliding window for local correlation.
            
        Returns
        -------
        numpy.ndarray
            Influence scores matrix with the same shape as X.
        """
        self.logger.info(f"Generating local {self.method}-based influence scores with window size {window_size}...")
        
        if not model.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before generate_influence_local().")
            raise ValueError("Model is not fitted. Call fit() before generate_influence_local().")
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate local influence scores
        influence_scores = np.zeros_like(X, dtype=float)
        n_samples = len(X)
        
        for i in range(n_samples):
            # Define window boundaries
            start = max(0, i - window_size // 2)
            end = min(n_samples, i + window_size // 2)
            
            # Extract window data
            X_window = X[start:end]
            y_window = y_pred[start:end]
            
            # Calculate correlations for each feature
            for j in range(X.shape[1]):
                if self.method == 'spearman':
                    corr, _ = spearmanr(X_window[:, j], y_window)
                else:  # pearson
                    corr = np.corrcoef(X_window[:, j], y_window)[0, 1]
                
                # Handle NaN values
                if np.isnan(corr):
                    corr = 0
                
                # Assign correlation as influence score
                influence_scores[i, j] = corr
        
        self.logger.info(f"Local {self.method.capitalize()} influence scores generated with shape {influence_scores.shape}")
        
        return influence_scores
    
    def generate_influence_coalitional(self, model, X, n_coalitions=10, random_state=42):
        """
        Generate coalitional influence scores using Spearman correlation.
        
        This method evaluates feature importance by measuring the correlation
        between features and predictions across different feature subsets (coalitions).
        
        Parameters
        ----------
        model : BaseModel
            Fitted model instance.
        X : numpy.ndarray
            Feature matrix.
        n_coalitions : int, default=10
            Number of random feature coalitions to evaluate.
        random_state : int, default=42
            Random seed for reproducibility.
            
        Returns
        -------
        numpy.ndarray
            Influence scores matrix with the same shape as X.
        """
        self.logger.info(f"Generating coalitional {self.method}-based influence scores...")
        
        if not model.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before generate_influence_coalitional().")
            raise ValueError("Model is not fitted. Call fit() before generate_influence_coalitional().")
        
        np.random.seed(random_state)
        n_features = X.shape[1]
        influence_scores = np.zeros_like(X, dtype=float)
        
        # Generate random coalitions
        for _ in range(n_coalitions):
            # Randomly select features to include
            coalition_size = np.random.randint(1, n_features + 1)
            coalition_features = np.random.choice(n_features, size=coalition_size, replace=False)
            
            # Create masked data with only coalition features
            X_masked = np.zeros_like(X)
            X_masked[:, coalition_features] = X[:, coalition_features]
            
            # Get predictions
            y_pred = model.predict(X_masked)
            
            # Calculate correlations for coalition features
            for feature in coalition_features:
                if self.method == 'spearman':
                    corr, _ = spearmanr(X[:, feature], y_pred)
                else:  # pearson
                    corr = np.corrcoef(X[:, feature], y_pred)[0, 1]
                
                # Handle NaN values
                if np.isnan(corr):
                    corr = 0
                
                # Accumulate correlation as influence score
                influence_scores[:, feature] += corr / n_coalitions
        
        self.logger.info(f"Coalitional {self.method.capitalize()} influence scores generated with shape {influence_scores.shape}")
        
        return influence_scores
