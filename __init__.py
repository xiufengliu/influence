"""
Dynamic Influence-Based Clustering Framework

A comprehensive framework for interpretable clustering of time series data using 
explainable machine learning techniques.

This implementation corresponds to the methods described in:
"Learning Interpretable Dynamics: Influence-Based Clustering of Energy Consumption Time Series"

Authors: Binbin Li, Xiufeng Liu, Rongfei Ma, Yuhao Ma
"""

__version__ = "1.0.0"
__author__ = "Binbin Li, Xiufeng Liu, Rongfei Ma, Yuhao Ma"
__email__ = "xiuli@dtu.dk"

from .src.clustering.dynamic_kmeans import DynamicKMeansClustering
from .src.influence.shap_influence import ShapInfluence
from .src.influence.lime_influence import LimeInfluence
from .src.influence.spearman_influence import SpearmanInfluence
from .src.influence.integrated_gradients_influence import IntegratedGradientsInfluence
from .src.models.gradient_boost import GradientBoostModel
from .src.models.torch_models import LSTMModel, TransformerModel
from .src.preprocessing.data_loader import DataLoader
from .src.preprocessing.preprocessor import Preprocessor
from .src.utils.metrics import ClusteringEvaluator
from .src.utils.visualization import ClusteringVisualizer

__all__ = [
    'DynamicKMeansClustering',
    'ShapInfluence',
    'LimeInfluence', 
    'SpearmanInfluence',
    'IntegratedGradientsInfluence',
    'GradientBoostModel',
    'LSTMModel',
    'TransformerModel',
    'DataLoader',
    'Preprocessor',
    'ClusteringEvaluator',
    'ClusteringVisualizer'
]
