"""
Experiment script for evaluating clustering quality.

This script compares the clustering quality of different influence methods and clustering algorithms.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.gradient_boost import GradientBoostModel
from src.influence.shap_influence import ShapInfluence
from src.influence.lime_influence import LimeInfluence
from src.influence.spearman_influence import SpearmanInfluence
from src.clustering.kmeans import KMeansClustering
from src.clustering.hierarchical import HierarchicalClustering
from src.clustering.spectral import SpectralClustering
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_clustering
from src.utils.visualization import visualize_clusters


def run_experiment(dataset_name, output_dir=None):
    """
    Run clustering quality experiment.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    output_dir : str, default=None
        Directory to save results.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing experiment results.
    """
    # Setup logging
    logger = setup_logger("clustering_quality", "INFO")
    logger.info(f"Starting clustering quality experiment with {dataset_name} dataset")
    
    # Set output directory
    if output_dir is None:
        output_dir = config.RESULTS_DIR / dataset_name / "clustering_quality"
    
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
    
    # Define influence methods
    influence_methods = {
        "shap": ShapInfluence(**config.INFLUENCE_PARAMS["shap"]),
        "lime": LimeInfluence(**config.INFLUENCE_PARAMS["lime"]),
        "spearman": SpearmanInfluence(**config.INFLUENCE_PARAMS["spearman"])
    }
    
    # Define clustering algorithms
    clustering_algorithms = {
        "kmeans": KMeansClustering,
        "hierarchical": HierarchicalClustering,
        "spectral": SpectralClustering
    }
    
    # Initialize results
    results = []
    
    # Run experiments
    for influence_name, influence_generator in influence_methods.items():
        logger.info(f"Generating {influence_name} influence space...")
        Z = influence_generator.generate_influence(model, X)
        
        # Save influence space visualization
        visualize_clusters(
            Z, np.zeros(len(Z)),  # Dummy labels for visualization
            output_path=output_dir / f"{influence_name}_influence_space.png",
            method='pca'
        )
        
        for clustering_name, ClusteringAlgorithm in clustering_algorithms.items():
            logger.info(f"Running {clustering_name} clustering with {influence_name} influence...")
            
            # Create clustering algorithm
            clustering_params = config.CLUSTERING_PARAMS[clustering_name].copy()
            clustering = ClusteringAlgorithm(**clustering_params)
            
            # Fit and predict
            clusters = clustering.fit_predict(Z)
            
            # Evaluate clustering
            metrics = evaluate_clustering(Z, clusters)
            
            # Save cluster visualization
            visualize_clusters(
                Z, clusters,
                output_path=output_dir / f"{influence_name}_{clustering_name}_clusters.png",
                method='pca'
            )
            
            # Store results
            result = {
                "dataset": dataset_name,
                "influence_method": influence_name,
                "clustering_algorithm": clustering_name,
                "n_clusters": clustering.n_clusters,
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "entropy": metrics["entropy"]
            }
            
            results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_dir / "clustering_quality_results.csv", index=False)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Silhouette score (higher is better)
    ax = axes[0, 0]
    for influence in influence_methods.keys():
        subset = results_df[results_df["influence_method"] == influence]
        ax.bar(
            np.arange(len(clustering_algorithms)) + 0.2 * list(influence_methods.keys()).index(influence),
            subset["silhouette"],
            width=0.2,
            label=influence
        )
    ax.set_title("Silhouette Score (higher is better)")
    ax.set_xticks(np.arange(len(clustering_algorithms)) + 0.2)
    ax.set_xticklabels(clustering_algorithms.keys())
    ax.legend()
    
    # Davies-Bouldin index (lower is better)
    ax = axes[0, 1]
    for influence in influence_methods.keys():
        subset = results_df[results_df["influence_method"] == influence]
        ax.bar(
            np.arange(len(clustering_algorithms)) + 0.2 * list(influence_methods.keys()).index(influence),
            subset["davies_bouldin"],
            width=0.2,
            label=influence
        )
    ax.set_title("Davies-Bouldin Index (lower is better)")
    ax.set_xticks(np.arange(len(clustering_algorithms)) + 0.2)
    ax.set_xticklabels(clustering_algorithms.keys())
    ax.legend()
    
    # Calinski-Harabasz index (higher is better)
    ax = axes[1, 0]
    for influence in influence_methods.keys():
        subset = results_df[results_df["influence_method"] == influence]
        ax.bar(
            np.arange(len(clustering_algorithms)) + 0.2 * list(influence_methods.keys()).index(influence),
            subset["calinski_harabasz"],
            width=0.2,
            label=influence
        )
    ax.set_title("Calinski-Harabasz Index (higher is better)")
    ax.set_xticks(np.arange(len(clustering_algorithms)) + 0.2)
    ax.set_xticklabels(clustering_algorithms.keys())
    ax.legend()
    
    # Entropy (lower is better)
    ax = axes[1, 1]
    for influence in influence_methods.keys():
        subset = results_df[results_df["influence_method"] == influence]
        ax.bar(
            np.arange(len(clustering_algorithms)) + 0.2 * list(influence_methods.keys()).index(influence),
            subset["entropy"],
            width=0.2,
            label=influence
        )
    ax.set_title("Entropy (lower is better)")
    ax.set_xticks(np.arange(len(clustering_algorithms)) + 0.2)
    ax.set_xticklabels(clustering_algorithms.keys())
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "clustering_quality_summary.png", dpi=300, bbox_inches="tight")
    
    logger.info(f"Experiment results saved to {output_dir}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clustering quality experiment")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["building_genome", "industrial_site1", "industrial_site2", "industrial_site3"],
                        help="Dataset to use for analysis")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = None
    
    run_experiment(args.dataset, output_dir)
