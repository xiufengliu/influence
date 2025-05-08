"""
Experiment script for contextual coherence assessment.

This script evaluates the coherence of clusters within specific contexts.
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
from src.influence.lime_influence import LimeInfluence
from src.influence.spearman_influence import SpearmanInfluence
from src.clustering.kmeans import KMeansClustering
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_entropy, calculate_conditional_entropy


def run_experiment(dataset_name, output_dir=None):
    """
    Run contextual coherence assessment experiment.

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
    logger = setup_logger("contextual_coherence", "INFO")
    logger.info(f"Starting contextual coherence assessment with {dataset_name} dataset")

    # Set output directory
    if output_dir is None:
        output_dir = config.RESULTS_DIR / dataset_name / "contextual_coherence"

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

    # Define number of clusters to try
    # Ensure n_clusters is less than the number of samples
    n_samples = len(X)
    n_clusters_list = [min(3, n_samples-1)]
    if n_samples >= 5:
        n_clusters_list.append(min(5, n_samples-1))
    if n_samples >= 7:
        n_clusters_list.append(min(7, n_samples-1))

    # Initialize results
    results = []

    # Run experiments
    for influence_name, influence_generator in influence_methods.items():
        logger.info(f"Generating {influence_name} influence space...")
        Z = influence_generator.generate_influence(model, X)

        for n_clusters in n_clusters_list:
            logger.info(f"Running K-means clustering with {n_clusters} clusters...")

            # Create clustering algorithm
            clustering = KMeansClustering(n_clusters=n_clusters, random_state=42)

            # Fit and predict
            clusters = clustering.fit_predict(Z)

            # Calculate entropy
            entropy = calculate_entropy(clusters)

            # Calculate conditional entropy for each context dimension
            for i in range(c.shape[1]):
                context_dim = c[:, i]
                cond_entropy = calculate_conditional_entropy(clusters, context_dim)

                # Calculate information gain
                info_gain = entropy - cond_entropy

                # Store results
                result = {
                    "dataset": dataset_name,
                    "influence_method": influence_name,
                    "n_clusters": n_clusters,
                    "context_dimension": i,
                    "entropy": entropy,
                    "conditional_entropy": cond_entropy,
                    "information_gain": info_gain,
                    "normalized_info_gain": info_gain / entropy if entropy > 0 else 0
                }

                results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(output_dir / "contextual_coherence_results.csv", index=False)

    # Create summary plots

    # Information gain by context dimension
    plt.figure(figsize=(12, 8))

    for influence in influence_methods.keys():
        subset = results_df[results_df["influence_method"] == influence]

        # Group by context dimension and take mean across n_clusters
        grouped = subset.groupby("context_dimension")["normalized_info_gain"].mean()

        plt.plot(grouped.index, grouped.values, marker='o', label=influence)

    plt.title("Normalized Information Gain by Context Dimension")
    plt.xlabel("Context Dimension")
    plt.ylabel("Normalized Information Gain")
    plt.xticks(range(c.shape[1]))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "info_gain_by_context.png", dpi=300, bbox_inches="tight")

    # Information gain by number of clusters
    plt.figure(figsize=(12, 8))

    for influence in influence_methods.keys():
        subset = results_df[results_df["influence_method"] == influence]

        # Group by n_clusters and take mean across context dimensions
        grouped = subset.groupby("n_clusters")["normalized_info_gain"].mean()

        plt.plot(grouped.index, grouped.values, marker='o', label=influence)

    plt.title("Normalized Information Gain by Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Normalized Information Gain")
    plt.xticks(n_clusters_list)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "info_gain_by_n_clusters.png", dpi=300, bbox_inches="tight")

    # Heatmap of information gain
    plt.figure(figsize=(15, 10))

    # Pivot table for heatmap
    pivot = results_df.pivot_table(
        index=["influence_method", "n_clusters"],
        columns="context_dimension",
        values="normalized_info_gain"
    )

    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
    plt.title("Normalized Information Gain by Influence Method, Number of Clusters, and Context Dimension")
    plt.xlabel("Context Dimension")
    plt.ylabel("Influence Method / Number of Clusters")
    plt.tight_layout()
    plt.savefig(output_dir / "info_gain_heatmap.png", dpi=300, bbox_inches="tight")

    # Analyze cluster distribution within contexts

    # Select best configuration
    best_config = results_df.loc[results_df["normalized_info_gain"].idxmax()]
    best_influence = best_config["influence_method"]
    best_n_clusters = int(best_config["n_clusters"])
    best_context_dim = int(best_config["context_dimension"])

    logger.info(f"Best configuration: {best_influence} influence, {best_n_clusters} clusters, context dimension {best_context_dim}")

    # Re-run clustering with best configuration
    influence_generator = influence_methods[best_influence]
    Z = influence_generator.generate_influence(model, X)

    clustering = KMeansClustering(n_clusters=best_n_clusters, random_state=42)
    clusters = clustering.fit_predict(Z)

    # Get unique context values
    context_values = np.unique(c[:, best_context_dim])

    # Create cluster distribution plot
    plt.figure(figsize=(15, 10))

    # For each context value, plot cluster distribution
    for i, ctx_val in enumerate(context_values):
        # Get clusters for this context
        ctx_mask = c[:, best_context_dim] == ctx_val
        ctx_clusters = clusters[ctx_mask]

        # Count occurrences of each cluster
        cluster_counts = np.bincount(ctx_clusters, minlength=best_n_clusters)
        cluster_probs = cluster_counts / len(ctx_clusters)

        # Plot
        plt.subplot(len(context_values), 1, i + 1)
        plt.bar(range(best_n_clusters), cluster_probs)
        plt.title(f"Context Value: {ctx_val}")
        plt.xlabel("Cluster")
        plt.ylabel("Probability")
        plt.xticks(range(best_n_clusters))
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / "cluster_distribution_by_context.png", dpi=300, bbox_inches="tight")

    logger.info(f"Experiment results saved to {output_dir}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Contextual coherence assessment experiment")
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
