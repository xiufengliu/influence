"""
Base model class for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

import config


class BaseModel(ABC):
    """
    Abstract base class for predictive models.
    
    This class defines the interface for all predictive models used in the framework.
    """
    
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Target variable.
            
        Returns
        -------
        self
            Fitted model instance.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
            
        Returns
        -------
        numpy.ndarray
            Predictions.
        """
        pass
    
    def evaluate(self, X, y, metric='mse'):
        """
        Evaluate the model performance.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Target variable.
        metric : str, default='mse'
            Evaluation metric. Options: 'mse', 'rmse', 'r2', 'accuracy', 'f1'.
            
        Returns
        -------
        float
            Evaluation metric value.
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before evaluate().")
            raise ValueError("Model is not fitted. Call fit() before evaluate().")
        
        y_pred = self.predict(X)
        
        if metric == 'mse':
            return mean_squared_error(y, y_pred)
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(y, y_pred))
        elif metric == 'r2':
            return r2_score(y, y_pred)
        elif metric == 'accuracy':
            return accuracy_score(y, np.round(y_pred))
        elif metric == 'f1':
            return f1_score(y, np.round(y_pred), average='weighted')
        else:
            self.logger.warning(f"Unknown metric: {metric}. Using MSE instead.")
            return mean_squared_error(y, y_pred)
    
    def cross_validate(self, X, y, cv=None, metric='mse'):
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Target variable.
        cv : int, default=None
            Number of cross-validation folds. If None, uses the value from config.
        metric : str, default='mse'
            Evaluation metric. Options: 'mse', 'rmse', 'r2', 'accuracy', 'f1'.
            
        Returns
        -------
        list
            Cross-validation scores.
        """
        if cv is None:
            cv = config.EVALUATION_PARAMS['cv_folds']
        
        if metric == 'mse':
            scoring = 'neg_mean_squared_error'
        elif metric == 'rmse':
            scoring = 'neg_root_mean_squared_error'
        elif metric == 'r2':
            scoring = 'r2'
        elif metric == 'accuracy':
            scoring = 'accuracy'
        elif metric == 'f1':
            scoring = 'f1_weighted'
        else:
            self.logger.warning(f"Unknown metric: {metric}. Using MSE instead.")
            scoring = 'neg_mean_squared_error'
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        # Convert negative scores to positive for MSE and RMSE
        if scoring.startswith('neg_'):
            scores = -scores
        
        return scores
    
    def train_test_evaluate(self, X, y, test_size=None, random_state=None, metrics=None):
        """
        Split data into train and test sets, fit the model, and evaluate performance.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Target variable.
        test_size : float, default=None
            Proportion of data to use for testing. If None, uses the value from config.
        random_state : int, default=None
            Random seed for reproducibility. If None, uses the value from config.
        metrics : list, default=None
            List of metrics to evaluate. If None, uses ['mse', 'r2'].
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics.
        """
        if test_size is None:
            test_size = config.EVALUATION_PARAMS['test_size']
        
        if random_state is None:
            random_state = config.EVALUATION_PARAMS['random_state']
        
        if metrics is None:
            metrics = ['mse', 'r2']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit model
        self.fit(X_train, y_train)
        
        # Evaluate
        results = {}
        for metric in metrics:
            train_score = self.evaluate(X_train, y_train, metric=metric)
            test_score = self.evaluate(X_test, y_test, metric=metric)
            results[f'train_{metric}'] = train_score
            results[f'test_{metric}'] = test_score
        
        return results
