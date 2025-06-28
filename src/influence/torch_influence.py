"""
PyTorch-based influence generation for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
import torch
from captum.attr import IntegratedGradients
import shap


class IntegratedGradientsInfluence:
    """
    Integrated Gradients influence generation.
    """

    def __init__(self, n_steps=50, random_state=42):
        self.n_steps = n_steps
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def generate_influence(self, model, X):
        """
        Generate influence scores using Integrated Gradients.
        """
        self.logger.info("Generating Integrated Gradients influence scores...")
        torch.manual_seed(self.random_state)

        if not model.is_fitted:
            raise ValueError("Model is not fitted.")

        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)

        X_tensor = torch.from_numpy(X).float().to(model.device)

        ig = IntegratedGradients(model.model)
        baselines = torch.zeros_like(X_tensor)
        attributions, delta = ig.attribute(
            X_tensor, baselines, target=0, return_convergence_delta=True
        )

        self.logger.info("Integrated Gradients influence scores generated.")
        return attributions.cpu().numpy().squeeze(axis=1)


class DeepShapInfluence:
    """
    DeepSHAP influence generation.
    """

    def __init__(self, n_samples=100, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def generate_influence(self, model, X):
        """
        Generate influence scores using DeepSHAP.
        """
        self.logger.info("Generating DeepSHAP influence scores...")
        torch.manual_seed(self.random_state)

        if not model.is_fitted:
            raise ValueError("Model is not fitted.")

        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)

        X_tensor = torch.from_numpy(X).float().to(model.device)

        # Select background samples
        np.random.seed(self.random_state)
        if len(X) > self.n_samples:
            background_indices = np.random.choice(len(X), self.n_samples, replace=False)
            background = X_tensor[background_indices]
        else:
            background = X_tensor

        explainer = shap.DeepExplainer(model.model, background)
        shap_values = explainer.shap_values(X_tensor)

        self.logger.info("DeepSHAP influence scores generated.")
        # The output of shap_values is a numpy array, so we don't need to move it to cpu
        return shap_values.squeeze(axis=1)
