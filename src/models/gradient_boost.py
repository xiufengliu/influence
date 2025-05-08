"""
Gradient Boosting model implementation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from xgboost import XGBRegressor, XGBClassifier

from src.models.base_model import BaseModel


class GradientBoostModel(BaseModel):
    """
    Gradient Boosting model implementation using XGBoost.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth.
    learning_rate : float, default=0.1
        Learning rate.
    objective : str, default=None
        Objective function. If None, automatically determined based on target variable.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional parameters to pass to XGBoost.
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 objective=None, random_state=42, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.random_state = random_state
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y):
        """
        Fit the Gradient Boosting model to the data.
        
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
        self.logger.info("Fitting Gradient Boosting model...")
        
        # Determine if classification or regression based on target values
        unique_values = np.unique(y)
        is_classification = len(unique_values) <= 10 and np.all(np.mod(unique_values, 1) == 0)
        
        # Set objective if not specified
        if self.objective is None:
            if is_classification:
                if len(unique_values) == 2:
                    self.objective = 'binary:logistic'
                else:
                    self.objective = 'multi:softprob'
            else:
                self.objective = 'reg:squarederror'
        
        # Create model
        if is_classification:
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=self.objective,
                random_state=self.random_state,
                **self.kwargs
            )
        else:
            self.model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=self.objective,
                random_state=self.random_state,
                **self.kwargs
            )
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.logger.info("Gradient Boosting model fitted successfully")
        return self
    
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
        if not self.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before predict().")
            raise ValueError("Model is not fitted. Call fit() before predict().")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification models.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
            
        Returns
        -------
        numpy.ndarray
            Class probabilities.
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before predict_proba().")
            raise ValueError("Model is not fitted. Call fit() before predict_proba().")
        
        if not hasattr(self.model, 'predict_proba'):
            self.logger.error("Model does not support probability predictions.")
            raise ValueError("Model does not support probability predictions.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance from the fitted model.
        
        Parameters
        ----------
        importance_type : str, default='gain'
            Type of feature importance. Options: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'.
            
        Returns
        -------
        numpy.ndarray
            Feature importance scores.
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before get_feature_importance().")
            raise ValueError("Model is not fitted. Call fit() before get_feature_importance().")
        
        return self.model.get_booster().get_score(importance_type=importance_type)
