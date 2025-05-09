"""
LIME-based influence generation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
from lime.lime_tabular import LimeTabularExplainer


class LimeInfluence:
    """
    LIME-based influence generation.

    This class uses LIME (Local Interpretable Model-agnostic Explanations) to generate
    feature importance scores that represent the influence of each feature on the model's predictions.

    Parameters
    ----------
    n_samples : int, default=5000
        Number of samples for LIME explainer.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, n_samples=5000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def generate_influence(self, model, X, feature_names=None, categorical_features=None):
        """
        Generate influence scores using LIME.

        Parameters
        ----------
        model : BaseModel
            Fitted model instance.
        X : numpy.ndarray
            Feature matrix.
        feature_names : list, default=None
            List of feature names.
        categorical_features : list, default=None
            List of indices of categorical features.

        Returns
        -------
        numpy.ndarray
            Influence scores matrix with the same shape as X.
        """
        self.logger.info("Generating LIME-based influence scores...")

        if not model.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before generate_influence().")
            raise ValueError("Model is not fitted. Call fit() before generate_influence().")

        # For large datasets, use the batch method automatically
        if len(X) > 500:
            self.logger.info(f"Large dataset detected ({len(X)} samples). Using batch processing.")
            # Use a smaller batch size for better memory management
            batch_size = 100
            return self.generate_influence_batch(model, X, feature_names, categorical_features, batch_size)

        # Set feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Set categorical features if not provided
        if categorical_features is None:
            categorical_features = []

        # Determine mode based on model type
        if hasattr(model.model, 'predict_proba'):
            mode = 'classification'
        else:
            mode = 'regression'

        # Create explainer with a sample of the data to reduce memory usage
        sample_size = min(1000, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]

        self.logger.info(f"Creating LIME explainer with {sample_size} samples to reduce memory usage")
        explainer = LimeTabularExplainer(
            X_sample,  # Use a sample of the data for the explainer
            feature_names=feature_names,
            categorical_features=categorical_features,
            mode=mode,
            random_state=self.random_state
        )

        # Generate explanations for each instance
        influence_scores = np.zeros_like(X, dtype=float)

        for i in range(len(X)):
            if i % 100 == 0:
                self.logger.info(f"Generating LIME explanations: {i}/{len(X)}")

            # Use fewer samples for each explanation to reduce memory usage
            num_samples = min(self.n_samples, 1000)

            try:
                if mode == 'classification':
                    explanation = explainer.explain_instance(
                        X[i], model.predict_proba, num_features=X.shape[1], num_samples=num_samples
                    )
                else:
                    explanation = explainer.explain_instance(
                        X[i], model.predict, num_features=X.shape[1], num_samples=num_samples
                    )

                # Extract feature importance scores
                for feature, importance in explanation.local_exp[1]:
                    influence_scores[i, feature] = importance
            except Exception as e:
                self.logger.warning(f"Error generating LIME explanation for instance {i}: {e}")
                # Use zeros for this instance
                influence_scores[i, :] = 0

        self.logger.info(f"LIME influence scores generated with shape {influence_scores.shape}")

        return influence_scores

    def generate_influence_batch(self, model, X, feature_names=None, categorical_features=None, batch_size=100):
        """
        Generate influence scores using LIME in batches for better performance.

        Parameters
        ----------
        model : BaseModel
            Fitted model instance.
        X : numpy.ndarray
            Feature matrix.
        feature_names : list, default=None
            List of feature names.
        categorical_features : list, default=None
            List of indices of categorical features.
        batch_size : int, default=100
            Batch size for processing.

        Returns
        -------
        numpy.ndarray
            Influence scores matrix with the same shape as X.
        """
        self.logger.info(f"Generating LIME-based influence scores in batches of {batch_size}...")

        if not model.is_fitted:
            self.logger.error("Model is not fitted. Call fit() before generate_influence_batch().")
            raise ValueError("Model is not fitted. Call fit() before generate_influence_batch().")

        # Set feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Set categorical features if not provided
        if categorical_features is None:
            categorical_features = []

        # Determine mode based on model type
        if hasattr(model.model, 'predict_proba'):
            mode = 'classification'
        else:
            mode = 'regression'

        # Create explainer with a sample of the data to reduce memory usage
        sample_size = min(1000, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]

        self.logger.info(f"Creating LIME explainer with {sample_size} samples to reduce memory usage")
        explainer = LimeTabularExplainer(
            X_sample,  # Use a sample of the data for the explainer
            feature_names=feature_names,
            categorical_features=categorical_features,
            mode=mode,
            random_state=self.random_state
        )

        # Generate explanations in batches
        influence_scores = np.zeros_like(X, dtype=float)

        # Use fewer samples for each explanation to reduce memory usage
        num_samples = min(self.n_samples, 500)
        self.logger.info(f"Using {num_samples} samples per explanation to reduce memory usage")

        for batch_start in range(0, len(X), batch_size):
            batch_end = min(batch_start + batch_size, len(X))
            self.logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(X)-1)//batch_size + 1}")

            for i in range(batch_start, batch_end):
                try:
                    if mode == 'classification':
                        explanation = explainer.explain_instance(
                            X[i], model.predict_proba, num_features=X.shape[1], num_samples=num_samples
                        )
                    else:
                        explanation = explainer.explain_instance(
                            X[i], model.predict, num_features=X.shape[1], num_samples=num_samples
                        )

                    # Extract feature importance scores
                    for feature, importance in explanation.local_exp[1]:
                        influence_scores[i, feature] = importance
                except Exception as e:
                    self.logger.warning(f"Error generating LIME explanation for instance {i}: {e}")
                    # Use zeros for this instance
                    influence_scores[i, :] = 0

            # Force garbage collection after each batch to free memory
            import gc
            gc.collect()

        self.logger.info(f"LIME influence scores generated with shape {influence_scores.shape}")

        return influence_scores
