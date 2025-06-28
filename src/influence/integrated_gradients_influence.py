"""
Integrated Gradients-based influence generation for deep learning models.

This module implements Integrated Gradients attribution method as described in:
Sundararajan et al. "Axiomatic Attribution for Deep Networks" (2017)

Used with LSTM and Transformer models in the Dynamic Influence-Based Clustering framework.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union

from src.influence.base_influence import BaseInfluence


class IntegratedGradientsInfluence(BaseInfluence):
    """
    Integrated Gradients-based influence generation for PyTorch models.
    
    This class computes feature attributions using the Integrated Gradients method,
    which satisfies important axioms like sensitivity and implementation invariance.
    
    Parameters
    ----------
    n_steps : int, default=50
        Number of steps for the Riemann approximation of the integral
    baseline : str or np.ndarray, default='zero'
        Baseline input for computing gradients. Can be 'zero', 'mean', or custom array
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, n_steps: int = 50, baseline: Union[str, np.ndarray] = 'zero', 
                 random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.n_steps = n_steps
        self.baseline = baseline
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
    def generate_influence(self, model, X: np.ndarray) -> np.ndarray:
        """
        Generate influence scores using Integrated Gradients.
        
        Parameters
        ----------
        model : BaseModel
            Fitted PyTorch model wrapper (LSTMModel, TransformerModel, etc.)
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Integrated gradients attribution scores
        """
        self.logger.info("Generating Integrated Gradients influence scores...")
        
        if not model.is_fitted:
            raise ValueError("Model must be fitted before generating influence scores")
            
        # Try using Captum first (more robust)
        try:
            return self._generate_influence_captum(model, X)
        except ImportError:
            self.logger.warning("Captum not available, using manual implementation")
            return self._generate_influence_manual(model, X)
        except Exception as e:
            self.logger.warning(f"Captum implementation failed: {e}, falling back to manual")
            return self._generate_influence_manual(model, X)
            
    def _generate_influence_captum(self, model, X: np.ndarray) -> np.ndarray:
        """Generate influence scores using Captum library."""
        from captum.attr import IntegratedGradients
        
        # Set model to evaluation mode
        model.model.eval()
        
        # Initialize Integrated Gradients
        ig = IntegratedGradients(model.model)
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X).requires_grad_(True)
        
        # Handle different input shapes for different models
        if hasattr(model, 'model_type'):
            if model.model_type in ['lstm', 'transformer']:
                # For sequence models, add sequence dimension if needed
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)  # (batch, seq_len=1, features)
        
        # Generate baseline
        baseline_tensor = self._get_baseline(X_tensor)
        
        # Compute attributions
        attributions = ig.attribute(
            X_tensor,
            baseline_tensor,
            n_steps=self.n_steps,
            return_convergence_delta=False
        )
        
        # Convert back to numpy and reshape to match input
        attributions_np = attributions.detach().cpu().numpy()
        
        # Reshape to match original input shape
        if attributions_np.shape != X.shape:
            if attributions_np.ndim == 3 and X.ndim == 2:
                attributions_np = attributions_np.squeeze(1)  # Remove sequence dimension
                
        self.logger.info(f"Generated influence scores with shape: {attributions_np.shape}")
        return attributions_np
        
    def _generate_influence_manual(self, model, X: np.ndarray) -> np.ndarray:
        """Manual implementation of Integrated Gradients."""
        # Set model to evaluation mode
        model.model.eval()
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X)
        
        # Handle model-specific input shapes
        if hasattr(model, 'model_type'):
            if model.model_type in ['lstm', 'transformer']:
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension
                    
        # Generate baseline
        baseline_tensor = self._get_baseline(X_tensor)
        
        # Compute integrated gradients for each sample
        all_attributions = []
        
        for i in range(X.shape[0]):
            sample_input = X_tensor[i:i+1]  # Keep batch dimension
            sample_baseline = baseline_tensor if baseline_tensor.shape[0] == 1 else baseline_tensor[i:i+1]
            
            # Compute integrated gradients for this sample
            sample_attributions = self._compute_integrated_gradients(
                model.model, sample_input, sample_baseline
            )
            
            all_attributions.append(sample_attributions)
            
        # Convert back to numpy and reshape
        attributions = np.array(all_attributions)
        if attributions.ndim == 3 and attributions.shape[0] == X.shape[0]:
            if attributions.shape[1] == 1:  # Remove sequence dimension
                attributions = attributions.squeeze(1)
                
        return attributions
        
    def _get_baseline(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """Generate baseline input based on self.baseline setting."""
        if isinstance(self.baseline, str):
            if self.baseline == 'zero':
                return torch.zeros_like(X_tensor[0:1])  # Single baseline for all samples
            elif self.baseline == 'mean':
                return torch.mean(X_tensor, dim=0, keepdim=True)
            elif self.baseline == 'random':
                torch.manual_seed(self.random_state)
                return torch.randn_like(X_tensor[0:1]) * 0.1
            else:
                raise ValueError(f"Unknown baseline type: {self.baseline}")
        else:
            # Custom baseline provided
            baseline_array = np.array(self.baseline)
            baseline_tensor = torch.FloatTensor(baseline_array)
            
            # Reshape to match X_tensor
            if baseline_tensor.dim() != X_tensor.dim():
                if X_tensor.dim() == 3:  # (batch, seq, features)
                    baseline_tensor = baseline_tensor.unsqueeze(0).unsqueeze(0)
                else:
                    baseline_tensor = baseline_tensor.unsqueeze(0)
                    
            return baseline_tensor
            
    def _compute_integrated_gradients(self, model: torch.nn.Module, 
                                    inputs: torch.Tensor, 
                                    baseline: torch.Tensor) -> np.ndarray:
        """
        Compute integrated gradients between baseline and input.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model
        inputs : torch.Tensor
            Input tensor
        baseline : torch.Tensor
            Baseline tensor
            
        Returns
        -------
        np.ndarray
            Integrated gradients attribution
        """
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps + 1)
        
        # Initialize gradients accumulator
        gradients = []
        
        for alpha in alphas:
            # Create interpolated input
            interpolated_input = baseline + alpha * (inputs - baseline)
            interpolated_input.requires_grad_(True)
            
            # Forward pass
            try:
                output = model(interpolated_input)
                
                # Handle different output types
                if isinstance(output, tuple):
                    output = output[0]  # Take first element if tuple
                    
                # Ensure scalar output for gradient computation
                if output.dim() > 1:
                    output = output.sum()  # Sum if multiple outputs
                    
                # Compute gradients
                model.zero_grad()
                output.backward(retain_graph=True)
                
                if interpolated_input.grad is not None:
                    gradients.append(interpolated_input.grad.detach().clone())
                else:
                    # If no gradients, append zeros
                    gradients.append(torch.zeros_like(interpolated_input))
                    
            except Exception as e:
                self.logger.warning(f"Error in gradient computation at alpha={alpha}: {e}")
                gradients.append(torch.zeros_like(interpolated_input))
                
        # Average gradients (Riemann sum approximation)
        if gradients:
            avg_gradients = torch.stack(gradients).mean(dim=0)
        else:
            avg_gradients = torch.zeros_like(inputs)
            
        # Compute integrated gradients
        integrated_gradients = (inputs - baseline) * avg_gradients
        
        return integrated_gradients.detach().cpu().numpy()


class DeepLiftInfluence(BaseInfluence):
    """
    DeepLift-based influence generation as an alternative to Integrated Gradients.
    """
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
    def generate_influence(self, model, X: np.ndarray) -> np.ndarray:
        """
        Generate influence scores using DeepLift.
        
        Parameters
        ----------
        model : BaseModel
            Fitted PyTorch model
        X : np.ndarray
            Input feature matrix
            
        Returns
        -------
        np.ndarray
            DeepLift attribution scores
        """
        try:
            from captum.attr import DeepLift
        except ImportError:
            self.logger.error("Captum library required for DeepLift")
            raise ImportError("Please install captum: pip install captum")
            
        self.logger.info("Generating DeepLift influence scores...")
        
        if not model.is_fitted:
            raise ValueError("Model must be fitted before generating influence scores")
        
        # Set model to evaluation mode
        model.model.eval()
        
        # Initialize DeepLift
        dl = DeepLift(model.model)
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Handle model-specific input shapes
        if hasattr(model, 'model_type'):
            if model.model_type in ['lstm', 'transformer']:
                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension
        
        # Generate baseline (zero baseline for DeepLift)
        baseline_tensor = torch.zeros_like(X_tensor)
        
        # Compute attributions
        attributions = dl.attribute(X_tensor, baseline_tensor)
        
        # Convert to numpy and reshape to match input
        attributions_np = attributions.detach().cpu().numpy()
        
        # Reshape to match original input shape
        if attributions_np.shape != X.shape:
            if attributions_np.ndim == 3 and X.ndim == 2:
                attributions_np = attributions_np.squeeze(1)  # Remove sequence dimension
                
        return attributions_np
        if len(X.shape) == 2:
            X_tensor = torch.from_numpy(X).float().unsqueeze(1) # Add sequence_length dimension
        else:
            X_tensor = torch.from_numpy(X).float()

        # Determine device from model
        device = next(torch_model.parameters()).device
        X_tensor = X_tensor.to(device)

        # IntegratedGradients requires a baseline. Using zeros as a common choice.
        # Baseline should have the same shape as input.
        baseline = torch.zeros_like(X_tensor).to(device)

        ig = IntegratedGradients(torch_model)

        # Compute attributions. The target is the index of the output to explain.
        # For regression, it's usually 0 as there's one output.
        attributions = ig.attribute(X_tensor, baseline, n_steps=self.n_steps, target=0)

        # If input was (batch, 1, features), squeeze back to (batch, features)
        if len(X.shape) == 2:
            return attributions.squeeze(1).cpu().numpy()
        else:
            # For actual sequence data, we might want to sum/average attributions over the sequence length
            # For now, let's sum them to get a single attribution per feature per sample
            return attributions.sum(dim=1).cpu().numpy() # Sum attributions across the sequence length
