"""
Experiment script for temporal pattern analysis.

This script analyzes the temporal evolution of clusters and transitions between them.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.gradient_boost import GradientBoostModel
from src.influence.shap_influence import ShapInfluence
from src.clustering.kmeans import KMeansClustering
from src.temporal.transition_matrix import TransitionMatrix
from src.temporal.anomaly_detection import AnomalyDetection
from src.utils.logger import setup_logger
from src.utils.visualization import visualize_transitions, visualize_temporal_evolution


def run_experiment(dataset_name, influence_method="shap", clustering_algorithm="kmeans", 
                  n_clusters=3, time_window=None, output_dir=None):
    """
    Run temporal pattern analysis experiment.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    influence_method : str, default="shap"
        Influence method to use.
    clustering_algorithm : str, default="kmeans"
        Clustering algorithm to use.
    n_clusters : int, default=3
        Number of clusters.
    time_window : str, default=None
        Time window for temporal analysis (e.g., 'D' for daily, 'H' for hourly).
    output_dir : str, default=None
        Directory to save results.
        
    Returns
    -------
    dict
        Dictionary containing experiment results.
    """
    # Setup logging
    logger = setup_logger("temporal_analysis", "INFO")
    logger.info(f"Starting temporal pattern analysis with {dataset_name} dataset")
    
    # Set output directory
    if output_dir is None:
        output_dir = config.RESULTS_DIR / dataset_name / "temporal_analysis"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data_loader = DataLoader(dataset_name=dataset_name)
    data = data_loader.load_data()
    
    preprocessor = Preprocessor()
    X, y, t, c = preprocessor.preprocess(data)
    
    # Train predictive model
    logger.info("Training predictive model...")
    model = GradientBoostModel(**config.MODEL_PARAMS["gradient_boost"])
    model.fit(X, y)
    
    # Generate influence space
    logger.info(f"Generating {influence_method} influence space...")
    if influence_method == "shap":
        influence_generator = ShapInfluence(**config.INFLUENCE_PARAMS["shap"])
    else:
        logger.warning(f"Unsupported influence method: {influence_method}. Using SHAP instead.")
        influence_generator = ShapInfluence(**config.INFLUENCE_PARAMS["shap"])
    
    Z = influence_generator.generate_influence(model, X)
    
    # Perform clustering
    logger.info(f"Performing {clustering_algorithm} clustering with {n_clusters} clusters...")
    if clustering_algorithm == "kmeans":
        clustering_params = config.CLUSTERING_PARAMS["kmeans"].copy()
    else:
        logger.warning(f"Unsupported clustering algorithm: {clustering_algorithm}. Using K-means instead.")
        clustering_params = config.CLUSTERING_PARAMS["kmeans"].copy()
    
    clustering_params["n_clusters"] = n_clusters
    clustering = KMeansClustering(**clustering_params)
    
    clusters = clustering.fit_predict(Z)
    
    # Compute transition matrix
    logger.info("Computing transition matrix...")
    transition_matrix = TransitionMatrix()
    P = transition_matrix.compute(clusters, t, time_window)
    
    # Visualize transition matrix
    visualize_transitions(P, output_path=output_dir / "transition_matrix.png")
    
    # Visualize temporal evolution
    visualize_temporal_evolution(clusters, t, output_path=output_dir / "temporal_evolution.png")
    
    # Detect anomalies
    logger.info("Detecting anomalies...")
    anomaly_detector = AnomalyDetection()
    anomalies = anomaly_detector.detect(clusters, P, t, threshold=0.1, time_window=time_window)
    
    # Save anomalies
    if len(anomalies) > 0:
        anomalies.to_csv(output_dir / "anomalies.csv", index=False)
        
        # Visualize anomalies
        plt.figure(figsize=(12, 6))
        
        # Plot cluster evolution
        plt.scatter(
            pd.to_datetime(t),
            clusters,
            c=clusters,
            cmap='tab10',
            alpha=0.7,
            s=50,
            label='Clusters'
        )
        
        # Highlight anomalies
        if len(anomalies) > 0:
            anomaly_times = pd.to_datetime(anomalies['timestamp'])
            anomaly_clusters = anomalies['from_cluster']
            plt.scatter(
                anomaly_times,
                anomaly_clusters,
                c='red',
                marker='x',
                s=100,
                label='Anomalies'
            )
        
        plt.title('Temporal Evolution with Anomalies')
        plt.xlabel('Time')
        plt.ylabel('Cluster')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "anomalies.png", dpi=300, bbox_inches="tight")
    
    # Compute cluster stability
    stability = transition_matrix.get_cluster_stability()
    
    # Visualize cluster stability
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_clusters), stability)
    plt.title('Cluster Stability')
    plt.xlabel('Cluster')
    plt.ylabel('Stability (Self-transition Probability)')
    plt.xticks(range(n_clusters))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_stability.png", dpi=300, bbox_inches="tight")
    
    # Compute n-step transitions
    n_steps = [2, 3, 5, 10]
    n_step_transitions = {}
    
    for n in n_steps:
        n_step_P = transition_matrix.get_n_step_transition(n)
        n_step_transitions[n] = n_step_P
        
        # Visualize n-step transition matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            n_step_P,
            annot=True,
            cmap='YlGnBu',
            fmt='.2f',
            linewidths=0.5
        )
        plt.title(f'{n}-Step Transition Matrix')
        plt.xlabel('To Cluster')
        plt.ylabel('From Cluster')
        plt.tight_layout()
        plt.savefig(output_dir / f"{n}_step_transition_matrix.png", dpi=300, bbox_inches="tight")
    
    # Compute stationary distribution
    stationary_distribution = transition_matrix.stationary_distribution
    
    # Visualize stationary distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_clusters), stationary_distribution)
    plt.title('Stationary Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Probability')
    plt.xticks(range(n_clusters))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "stationary_distribution.png", dpi=300, bbox_inches="tight")
    
    # Compute expected time to each cluster
    expected_times = {}
    
    for i in range(n_clusters):
        expected_time = transition_matrix.get_expected_time_to_cluster(i)
        expected_times[i] = expected_time
        
        # Visualize expected time
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_clusters), expected_time)
        plt.title(f'Expected Time to Cluster {i}')
        plt.xlabel('From Cluster')
        plt.ylabel('Expected Steps')
        plt.xticks(range(n_clusters))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"expected_time_to_cluster_{i}.png", dpi=300, bbox_inches="tight")
    
    # Save results
    results = {
        "dataset": dataset_name,
        "influence_method": influence_method,
        "clustering_algorithm": clustering_algorithm,
        "n_clusters": n_clusters,
        "transition_matrix": P,
        "stationary_distribution": stationary_distribution,
        "cluster_stability": stability,
        "n_step_transitions": n_step_transitions,
        "expected_times": expected_times,
        "anomalies": anomalies
    }
    
    logger.info(f"Experiment results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Temporal pattern analysis experiment")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["building_genome", "industrial_site1", "industrial_site2", "industrial_site3"],
                        help="Dataset to use for analysis")
    parser.add_argument("--influence", type=str, default="shap",
                        choices=["shap", "lime", "spearman"],
                        help="Influence method to use")
    parser.add_argument("--clustering", type=str, default="kmeans",
                        choices=["kmeans", "hierarchical", "spectral"],
                        help="Clustering algorithm to use")
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of clusters")
    parser.add_argument("--time_window", type=str, default=None,
                        help="Time window for temporal analysis (e.g., 'D' for daily, 'H' for hourly)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = None
    
    run_experiment(
        args.dataset,
        influence_method=args.influence,
        clustering_algorithm=args.clustering,
        n_clusters=args.n_clusters,
        time_window=args.time_window,
        output_dir=output_dir
    )
