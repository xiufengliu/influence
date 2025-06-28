"""
PyTorch-based models for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel


class LSTMNetwork(nn.Module):
    """A complete LSTM network including a fully connected output layer."""
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)
        # out shape: (batch_size, sequence_length, hidden_dim)
        # We take the output of the last time step for the final prediction
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(BaseModel):
    """
    LSTM model implementation using PyTorch.
    """

    def __init__(self, input_dim, hidden_dim=50, n_layers=2, output_dim=1,
                 epochs=100, batch_size=32, learning_rate=1e-3,
                 random_state=42, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """Build the LSTM model."""
        return LSTMNetwork(self.input_dim, self.hidden_dim, self.n_layers, self.output_dim)

    def fit(self, X, y):
        """Fit the LSTM model.
        X: numpy.ndarray, shape (n_samples, n_features)
        y: numpy.ndarray, shape (n_samples,)
        """
        self.logger.info("Fitting LSTM model...")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state) # For numpy operations

        # Reshape X to (n_samples, sequence_length, n_features)
        # Assuming each sample is a sequence of length 1 for now, as per data_loader output
        if len(X.shape) == 2:
            X_reshaped = np.expand_dims(X, axis=1) # (n_samples, 1, n_features)
        else: # Assume X is already (n_samples, sequence_length, n_features)
            X_reshaped = X

        X_tensor = torch.from_numpy(X_reshaped).float()
        y_tensor = torch.from_numpy(y).float().view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        self.is_fitted = True
        self.logger.info("LSTM model fitted successfully.")
        return self

    def predict(self, X):
        """Make predictions with the LSTM model.
        X: numpy.ndarray, shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")

        # Reshape X to (n_samples, sequence_length, n_features)
        if len(X.shape) == 2:
            X_reshaped = np.expand_dims(X, axis=1) # (n_samples, 1, n_features)
        else:
            X_reshaped = X

        X_tensor = torch.from_numpy(X_reshaped).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()

    def get_model(self):
        """Returns the underlying PyTorch nn.Module."""
        return self.model


class TransformerNetwork(nn.Module):
    """A complete Transformer network including a fully connected output layer."""
    def __init__(self, input_dim, n_heads, n_layers, output_dim, dropout=0.1):
        super(TransformerNetwork, self).__init__()
        # TransformerEncoderLayer expects d_model to be the feature dimension
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(input_dim, output_dim) # FC layer operates on the last time step's output

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        out = self.transformer_encoder(x)
        # out shape: (batch_size, sequence_length, input_dim)
        # We take the output of the last time step for the final prediction
        out = self.fc(out[:, -1, :])
        return out


class TransformerModel(BaseModel):
    """
    Transformer model implementation using PyTorch.
    """

    def __init__(self, input_dim, n_heads=8, n_layers=2, output_dim=1,
                 epochs=100, batch_size=32, learning_rate=1e-3,
                 random_state=42, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """Build the Transformer model."""
        return TransformerNetwork(self.input_dim, self.n_heads, self.n_layers, self.output_dim)

    def fit(self, X, y):
        """Fit the Transformer model.
        X: numpy.ndarray, shape (n_samples, n_features)
        y: numpy.ndarray, shape (n_samples,)
        """
        self.logger.info("Fitting Transformer model...")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state) # For numpy operations

        # Reshape X to (n_samples, sequence_length, n_features)
        if len(X.shape) == 2:
            X_reshaped = np.expand_dims(X, axis=1) # (n_samples, 1, n_features)
        else:
            X_reshaped = X

        X_tensor = torch.from_numpy(X_reshaped).float()
        y_tensor = torch.from_numpy(y).float().view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        self.is_fitted = True
        self.logger.info("Transformer model fitted successfully.")
        return self

    def predict(self, X):
        """Make predictions with the Transformer model.
        X: numpy.ndarray, shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")

        # Reshape X to (n_samples, sequence_length, n_features)
        if len(X.shape) == 2:
            X_reshaped = np.expand_dims(X, axis=1) # (n_samples, 1, n_features)
        else:
            X_reshaped = X

        X_tensor = torch.from_numpy(X_reshaped).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()

    def get_model(self):
        """Returns the underlying PyTorch nn.Module."""
        return self.model