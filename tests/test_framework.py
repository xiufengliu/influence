"""
Basic tests for the Dynamic Influence-Based Clustering Framework.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.gradient_boost import GradientBoostModel
from src.influence.shap_influence import ShapInfluence
from src.clustering.kmeans import KMeansClustering
from src.temporal.transition_matrix import TransitionMatrix
from src.temporal.anomaly_detection import AnomalyDetection


class TestFramework(unittest.TestCase):
    """
    Test case for the Dynamic Influence-Based Clustering Framework.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.X = np.random.rand(n_samples, n_features)
        self.y = np.random.rand(n_samples)
        self.t = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
        self.c = np.random.randint(0, 3, size=(n_samples, 2))
    
    def test_gradient_boost_model(self):
        """Test gradient boosting model."""
        model = GradientBoostModel(n_estimators=10, max_depth=3)
        model.fit(self.X, self.y)
        
        # Test prediction
        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertGreater(len(importance), 0)
    
    def test_shap_influence(self):
        """Test SHAP influence generation."""
        model = GradientBoostModel(n_estimators=10, max_depth=3)
        model.fit(self.X, self.y)
        
        influence_generator = ShapInfluence(n_samples=10)
        Z = influence_generator.generate_influence(model, self.X)
        
        self.assertEqual(Z.shape, self.X.shape)
    
    def test_kmeans_clustering(self):
        """Test K-means clustering."""
        n_clusters = 3
        clustering = KMeansClustering(n_clusters=n_clusters)
        
        # Create random influence space
        Z = np.random.rand(len(self.X), self.X.shape[1])
        
        # Fit and predict
        clustering.fit(Z)
        labels = clustering.predict(Z)
        
        self.assertEqual(len(labels), len(Z))
        self.assertEqual(len(np.unique(labels)), n_clusters)
        
        # Test cluster centers
        centers = clustering.get_cluster_centers()
        self.assertEqual(centers.shape, (n_clusters, Z.shape[1]))
    
    def test_transition_matrix(self):
        """Test transition matrix computation."""
        # Create random cluster assignments
        n_clusters = 3
        clusters = np.random.randint(0, n_clusters, size=len(self.X))
        
        # Compute transition matrix
        transition_matrix = TransitionMatrix()
        P = transition_matrix.compute(clusters, self.t)
        
        self.assertEqual(P.shape, (n_clusters, n_clusters))
        
        # Check stochastic property
        for i in range(n_clusters):
            self.assertAlmostEqual(np.sum(P[i]), 1.0, places=5)
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Create random cluster assignments
        n_clusters = 3
        clusters = np.random.randint(0, n_clusters, size=len(self.X))
        
        # Compute transition matrix
        transition_matrix = TransitionMatrix()
        P = transition_matrix.compute(clusters, self.t)
        
        # Detect anomalies
        anomaly_detector = AnomalyDetection()
        anomalies = anomaly_detector.detect(clusters, P, self.t, threshold=0.1)
        
        self.assertIsInstance(anomalies, pd.DataFrame)
    
    def test_end_to_end(self):
        """Test end-to-end workflow."""
        # Train model
        model = GradientBoostModel(n_estimators=10, max_depth=3)
        model.fit(self.X, self.y)
        
        # Generate influence space
        influence_generator = ShapInfluence(n_samples=10)
        Z = influence_generator.generate_influence(model, self.X)
        
        # Perform clustering
        clustering = KMeansClustering(n_clusters=3)
        clusters = clustering.fit_predict(Z)
        
        # Compute transition matrix
        transition_matrix = TransitionMatrix()
        P = transition_matrix.compute(clusters, self.t)
        
        # Detect anomalies
        anomaly_detector = AnomalyDetection()
        anomalies = anomaly_detector.detect(clusters, P, self.t, threshold=0.1)
        
        self.assertIsInstance(anomalies, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
