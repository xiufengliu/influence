"""
Hessian-based influence function implementation.
"""

import logging
import numpy as np
import torch
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector

from src.influence.base_influence import BaseInfluence

class HessianInfluence(BaseInfluence):
    """
    Computes the influence of training points on the loss of the model using
    the Hessian matrix. This implementation calculates self-influence scores.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_influence(self, model, X):
        """
        Generate influence scores for the given model and data using the Hessian.

        The self-influence of a training point is calculated as:
        I(z_i) = grad_L(z_i)^T H^{-1} grad_L(z_i)
        where H is the Hessian of the loss function.

        Parameters
        ----------
        model : object
            The trained predictive model (must be a PyTorch model).
        X : numpy.ndarray
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Influence scores for each sample in X.
        """
        self.logger.info("Generating Hessian-based influence scores...")

        if not hasattr(model, 'model') or not isinstance(model.model, torch.nn.Module):
            raise ValueError("The model must be a PyTorch model.")

        if not model.is_fitted:
            raise ValueError("Model is not fitted.")

        device = model.device
        torch_model = model.model.to(device)
        torch_model.eval()

        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.from_numpy(X).float().to(device)
        else:
            X_tensor = X.to(device)
            
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1)

        # 1. Calculate the Hessian of the total loss
        self.logger.info("Calculating Hessian of the loss...")
        loss_fn = torch.nn.MSELoss()

        def get_total_loss(params):
            # Temporarily load the new parameters into the model
            original_params = parameters_to_vector(torch_model.parameters())
            torch.nn.utils.vector_to_parameters(params, torch_model.parameters())
            
            # Assuming a regression task where target is the next value in the sequence
            # This part might need adjustment based on the specific task
            inputs = X_tensor[:, :-1]
            targets = X_tensor[:, 1:]
            
            outputs = torch_model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Restore original parameters
            torch.nn.utils.vector_to_parameters(original_params, torch_model.parameters())
            return loss

        params_vector = parameters_to_vector(torch_model.parameters())
        
        try:
            hessian = torch.autograd.functional.hessian(get_total_loss, params_vector)
            self.logger.info("Hessian calculated. Inverting Hessian...")
            hessian_inv = torch.inverse(hessian)
            self.logger.info("Hessian inverted.")
        except torch.linalg.LinAlgError:
            self.logger.warning("Hessian is singular. Using pseudo-inverse.")
            hessian_inv = torch.linalg.pinv(hessian)
            self.logger.info("Pseudo-inverse calculated.")


        # 2. Calculate per-sample gradients and influence scores
        influences = []
        self.logger.info("Calculating per-sample gradients and influence scores...")
        for i in range(len(X_tensor)):
            # This assumes a time-series prediction task where we predict the next step
            if X_tensor.shape[1] <= 1:
                # Cannot form input/target pair
                influences.append(0)
                continue

            sample_input = X_tensor[i:i+1, :-1]
            sample_target = X_tensor[i:i+1, 1:]

            if sample_input.nelement() == 0 or sample_target.nelement() == 0:
                influences.append(0)
                continue

            torch_model.zero_grad()
            output = torch_model(sample_input)
            loss = loss_fn(output, sample_target)
            
            # Get gradient of the loss for this sample w.r.t. model parameters
            sample_grad = grad(loss, torch_model.parameters())
            sample_grad_vec = parameters_to_vector(sample_grad)

            # 3. Calculate influence: grad^T * H_inv * grad
            influence = sample_grad_vec.T @ hessian_inv @ sample_grad_vec
            influences.append(influence.item())

        self.logger.info("Hessian-based influence scores generated.")
        return np.array(influences)
