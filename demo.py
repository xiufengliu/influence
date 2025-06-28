#!/usr/bin/env python3
"""
Quick demo script for the Dynamic Influence-Based Clustering Framework.

This script demonstrates the core functionality of the framework with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_energy_data(n_samples=500, n_features=6, random_state=42):
    """Generate synthetic energy consumption data."""
    np.random.seed(random_state)
    
    # Features: temperature, hour, day_of_week, occupancy, equipment_state, season
    feature_names = ['temperature', 'hour', 'day_of_week', 'occupancy', 'equipment_state', 'season']
    
    # Generate features
    temperature = np.random.normal(20, 5, n_samples)  # Temperature in Celsius
    hour = np.random.randint(0, 24, n_samples)        # Hour of day
    day_of_week = np.random.randint(0, 7, n_samples)  # Day of week
    occupancy = np.random.beta(2, 5, n_samples)       # Occupancy rate [0,1]
    equipment_state = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Equipment on/off
    season = np.random.randint(0, 4, n_samples)       # Season (0-3)
    
    X = np.column_stack([temperature, hour, day_of_week, occupancy, equipment_state, season])
    
    # Generate energy consumption based on realistic patterns
    base_consumption = 50  # Base consumption
    temp_effect = (temperature - 20) * 2  # Temperature effect
    hour_effect = 20 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
    occupancy_effect = occupancy * 30  # Occupancy effect
    equipment_effect = equipment_state * 25  # Equipment effect
    
    y = (base_consumption + temp_effect + hour_effect + 
         occupancy_effect + equipment_effect + 
         np.random.normal(0, 5, n_samples))  # Add noise
    
    # Generate timestamps and contexts
    timestamps = np.arange(n_samples)
    contexts = np.where(day_of_week < 5, 'weekday', 'weekend')
    
    return X, y, timestamps, contexts, feature_names

def demo_influence_clustering():
    """Demonstrate the Dynamic Influence-Based Clustering framework."""
    
    print("Dynamic Influence-Based Clustering - Demo")
    print("=" * 50)
    
    # Generate synthetic data
    print("1. Generating synthetic energy consumption data...")
    X, y, timestamps, contexts, feature_names = generate_synthetic_energy_data()
    print(f"   Generated {len(X)} samples with {X.shape[1]} features")
    
    # Import framework components
    try:
        from src.models.gradient_boost import GradientBoostModel
        from src.influence.spearman_influence import SpearmanInfluence
        from src.clustering.dynamic_kmeans import DynamicKMeansClustering
        from src.utils.metrics import ClusteringEvaluator
    except ImportError as e:
        print(f"Error importing framework components: {e}")
        return
    
    # Train predictive model
    print("\n2. Training predictive model...")
    model = GradientBoostModel()
    model.fit(X, y)
    
    # Evaluate model performance
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    print(f"   Model performance: MAE = {mae:.2f}, R² = {r2:.3f}")
    
    # Generate influence vectors
    print("\n3. Computing influence vectors...")
    influence_extractor = SpearmanInfluence()
    influence_vectors = influence_extractor.compute_influence(X, y)
    print(f"   Generated influence vectors of shape: {influence_vectors.shape}")
    
    # Display top influential features
    feature_importance = np.abs(np.mean(influence_vectors, axis=0))
    top_features = np.argsort(feature_importance)[::-1]
    print("   Top influential features:")
    for i, idx in enumerate(top_features[:3]):
        print(f"      {i+1}. {feature_names[idx]}: {feature_importance[idx]:.3f}")
    
    # Perform dynamic clustering
    print("\n4. Performing dynamic clustering...")
    clustering = DynamicKMeansClustering(
        n_clusters=3,
        alpha=1.0,   # Cohesion weight
        beta=1.0,    # Temporal weight
        gamma=1.0    # Contextual weight
    )
    
    cluster_labels = clustering.fit_predict(influence_vectors, timestamps, contexts)
    print(f"   Created {len(np.unique(cluster_labels))} clusters")
    
    # Cluster distribution
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print("   Cluster distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"      Cluster {label}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    # Evaluate clustering quality
    print("\n5. Evaluating clustering quality...")
    evaluator = ClusteringEvaluator()
    
    # Basic clustering metrics
    silhouette = evaluator.silhouette_score(influence_vectors, cluster_labels)
    davies_bouldin = evaluator.davies_bouldin_score(influence_vectors, cluster_labels)
    
    print(f"   Silhouette Score: {silhouette:.3f}")
    print(f"   Davies-Bouldin Index: {davies_bouldin:.3f}")
    
    # Temporal consistency
    temporal_consistency = evaluator.temporal_consistency(cluster_labels, timestamps)
    print(f"   Temporal Consistency: {temporal_consistency:.3f}")
    
    # Analyze cluster characteristics
    print("\n6. Analyzing cluster characteristics...")
    cluster_centers = clustering.get_cluster_centers()
    
    for cluster_id in range(len(cluster_centers)):
        center = cluster_centers[cluster_id]
        print(f"\n   Cluster {cluster_id} characteristics:")
        
        # Find most influential features for this cluster
        top_influences = np.argsort(np.abs(center))[::-1]
        for i, feature_idx in enumerate(top_influences[:3]):
            influence_val = center[feature_idx]
            feature_name = feature_names[feature_idx]
            direction = "increases" if influence_val > 0 else "decreases"
            print(f"      {feature_name} {direction} consumption (influence: {influence_val:.3f})")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("\nKey findings:")
    print(f"• Identified {len(unique_labels)} distinct consumption patterns")
    print(f"• Achieved silhouette score of {silhouette:.3f} (higher is better)")
    print(f"• Temporal consistency of {temporal_consistency:.3f} (higher is better)")
    print("\nThis demonstrates how the framework:")
    print("• Transforms raw features into interpretable influence space")
    print("• Performs clustering with temporal and contextual constraints")
    print("• Provides interpretable cluster characteristics")

if __name__ == "__main__":
    demo_influence_clustering()
