"""
SHAP-based influence generation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
import shap


class ShapInfluence:
    """
    SHAP-based influence generation.

    This class uses SHAP (SHapley Additive exPlanations) to generate feature importance
    scores that represent the influence of each feature on the model's predictions.

    Parameters
    ----------
    n_samples : int, default=100
        Number of background samples for SHAP explainer.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, n_samples=100, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def generate_influence(self, model, X):
        """
        Generate influence scores using SHAP.

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
        self.logger.info("Generating SHAP-based influence scores...")

        if not model.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before generate_influence().")
            raise ValueError("Model is not fitted. Call fit() before generate_influence().")

        # Select background samples
        np.random.seed(self.random_state)
        if len(X) > self.n_samples:
            background_indices = np.random.choice(len(X), self.n_samples, replace=False)
            background = X[background_indices]
        else:
            background = X

        # For very large datasets, use batching to reduce memory usage
        batch_size = 1000  # Process 1000 samples at a time
        n_samples = len(X)

        if n_samples > batch_size:
            self.logger.info(f"Processing {n_samples} samples in batches of {batch_size} to reduce memory usage")

            # Initialize array to store results
            shap_values_list = []

            # Process in batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: samples {i} to {end_idx-1}")

                X_batch = X[i:end_idx]

                # Create explainer based on model type
                if hasattr(model.model, 'predict_proba'):
                    # For classification models
                    explainer = shap.TreeExplainer(model.model, background)
                    batch_shap_values = explainer.shap_values(X_batch)

                    # For multi-class, take the mean across classes
                    if isinstance(batch_shap_values, list):
                        batch_shap_values = np.mean(np.array(batch_shap_values), axis=0)
                else:
                    # For regression models
                    explainer = shap.TreeExplainer(model.model, background)
                    batch_shap_values = explainer.shap_values(X_batch)

                shap_values_list.append(batch_shap_values)

            # Combine batches
            shap_values = np.vstack(shap_values_list)
        else:
            # For smaller datasets, process all at once
            # Create explainer based on model type
            if hasattr(model.model, 'predict_proba'):
                # For classification models
                explainer = shap.TreeExplainer(model.model, background)
                shap_values = explainer.shap_values(X)

                # For multi-class, take the mean across classes
                if isinstance(shap_values, list):
                    shap_values = np.mean(np.array(shap_values), axis=0)
            else:
                # For regression models
                explainer = shap.TreeExplainer(model.model, background)
                shap_values = explainer.shap_values(X)

        self.logger.info(f"SHAP influence scores generated with shape {shap_values.shape}")

        return shap_values

    def generate_influence_kernel(self, model, X, feature_names=None):
        """
        Generate influence scores using KernelExplainer for non-tree models.

        Parameters
        ----------
        model : BaseModel
            Fitted model instance.
        X : numpy.ndarray
            Feature matrix.
        feature_names : list, default=None
            List of feature names.

        Returns
        -------
        numpy.ndarray
            Influence scores matrix with the same shape as X.
        """
        self.logger.info("Generating SHAP-based influence scores using KernelExplainer...")

        if not model.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before generate_influence_kernel().")
            raise ValueError("Model is not fitted. Call fit() before generate_influence_kernel().")

        # Select background samples
        np.random.seed(self.random_state)
        if len(X) > self.n_samples:
            background_indices = np.random.choice(len(X), self.n_samples, replace=False)
            background = X[background_indices]
        else:
            background = X

        # Create a prediction function
        def predict_fn(x):
            return model.predict(x)

        # Create explainer
        explainer = shap.KernelExplainer(predict_fn, background, feature_names=feature_names)

        # For very large datasets, use batching to reduce memory usage
        batch_size = 500  # Smaller batch size for KernelExplainer as it's more memory intensive
        n_samples = len(X)

        if n_samples > batch_size:
            self.logger.info(f"Processing {n_samples} samples in batches of {batch_size} to reduce memory usage")

            # Initialize array to store results
            shap_values_list = []

            # Process in batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: samples {i} to {end_idx-1}")

                X_batch = X[i:end_idx]

                # Generate SHAP values for this batch
                batch_shap_values = explainer.shap_values(X_batch)

                # For multi-class, take the mean across classes
                if isinstance(batch_shap_values, list):
                    batch_shap_values = np.mean(np.array(batch_shap_values), axis=0)

                shap_values_list.append(batch_shap_values)

            # Combine batches
            shap_values = np.vstack(shap_values_list)
        else:
            # For smaller datasets, process all at once
            # Generate SHAP values
            shap_values = explainer.shap_values(X)

            # For multi-class, take the mean across classes
            if isinstance(shap_values, list):
                shap_values = np.mean(np.array(shap_values), axis=0)

        self.logger.info(f"SHAP influence scores generated with shape {shap_values.shape}")

        return shap_values
