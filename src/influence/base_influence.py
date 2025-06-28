"""
Base class for influence methods.
"""

from abc import ABC, abstractmethod
import logging

class BaseInfluence(ABC):
    """
    Abstract base class for influence methods.

    This class defines the interface for all influence methods used in the framework.
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.kwargs = kwargs

    @abstractmethod
    def generate_influence(self, model, X):
        """
        Generate influence scores for the given model and data.

        Parameters
        ----------
        model : object
            The trained predictive model.
        X : numpy.ndarray
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Influence scores.
        """
        pass
