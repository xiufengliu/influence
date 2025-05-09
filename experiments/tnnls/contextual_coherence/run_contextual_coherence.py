"""
Contextual coherence experiments for TNNLS submission.

This module implements comprehensive contextual coherence experiments to evaluate
the framework's ability to capture meaningful patterns within specific contexts.
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import entropy
from joblib import Parallel, delayed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

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


def run_single_experiment(dataset_name, influence_method, clustering_algorithm,
                         n_clusters, random_seed, output_dir):
    """
    Run a single contextual coherence experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    influence_method : str
        Influence method to use.
    clustering_algorithm : str
        Clustering algorithm to use.
    n_clusters : int
        Number of clusters.
    random_seed : int
        Random seed for reproducibility.
    output_dir : str or Path
        Directory to save results.

    Returns
    -------
    dict
        Dictionary containing experiment results.
    """
    # Set up logging
    logger = logging.getLogger(f"contextual_coherence_{dataset_name}_{influence_method}_{clustering_algorithm}")

    # Set random seed
    np.random.seed(random_seed)

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        # Get preprocessed data directly from the data loader
        X, y, t, c = data_loader.load_data(preprocess=True)

        # Train predictive model
        model_params = config.MODEL_PARAMS["gradient_boost"].copy()
        model_params["random_state"] = random_seed
        model = GradientBoostModel(**model_params)
        model.fit(X, y)

        # Generate influence space
        if influence_method == "shap":
            influence_params = config.INFLUENCE_PARAMS["shap"].copy()
            influence_params["random_state"] = random_seed
            influence_generator = ShapInfluence(**influence_params)
        elif influence_method == "lime":
            influence_params = config.INFLUENCE_PARAMS["lime"].copy()
            influence_params["random_state"] = random_seed
            influence_generator = LimeInfluence(**influence_params)
        else:  # spearman
            influence_params = config.INFLUENCE_PARAMS["spearman"].copy()
            influence_generator = SpearmanInfluence(**influence_params)

        Z = influence_generator.generate_influence(model, X)

        # Perform clustering
        if clustering_algorithm == "kmeans":
            clustering_params = config.CLUSTERING_PARAMS["kmeans"].copy()
            clustering_params["n_clusters"] = n_clusters
            clustering_params["random_state"] = random_seed
            clustering = KMeansClustering(**clustering_params)
        elif clustering_algorithm == "hierarchical":
            clustering_params = config.CLUSTERING_PARAMS["hierarchical"].copy()
            clustering_params["n_clusters"] = n_clusters
            clustering = HierarchicalClustering(**clustering_params)
        else:  # spectral
            clustering_params = config.CLUSTERING_PARAMS["spectral"].copy()
            clustering_params["n_clusters"] = n_clusters
            clustering_params["random_state"] = random_seed
            clustering = SpectralClustering(**clustering_params)

        # Fit and predict
        clusters = clustering.fit_predict(Z)

        # Calculate entropy
        cluster_entropy = calculate_entropy(clusters)

        # Calculate conditional entropy for each context dimension
        context_results = []
        for i in range(c.shape[1]):
            context_dim = c[:, i]

            # Calculate conditional entropy
            cond_entropy = calculate_conditional_entropy(clusters, context_dim)

            # Calculate information gain
            info_gain = cluster_entropy - cond_entropy

            # Calculate normalized information gain
            norm_info_gain = info_gain / cluster_entropy if cluster_entropy > 0 else 0

            # Store results
            context_results.append({
                "context_dimension": i,
                "entropy": cluster_entropy,
                "conditional_entropy": cond_entropy,
                "information_gain": info_gain,
                "normalized_info_gain": norm_info_gain
            })

        # Calculate cluster distribution within contexts
        context_distributions = []

        # For each context dimension
        for i in range(c.shape[1]):
            context_dim = c[:, i]
            unique_contexts = np.unique(context_dim)

            # For each unique context value
            for ctx_val in unique_contexts:
                # Get clusters for this context
                ctx_mask = context_dim == ctx_val
                ctx_clusters = clusters[ctx_mask]

                # Count occurrences of each cluster
                cluster_counts = np.bincount(ctx_clusters, minlength=n_clusters)
                cluster_probs = cluster_counts / len(ctx_clusters)

                # Calculate entropy within this context
                ctx_entropy = entropy(cluster_probs)

                # Store results
                context_distributions.append({
                    "context_dimension": i,
                    "context_value": ctx_val,
                    "cluster_distribution": cluster_probs.tolist(),
                    "context_entropy": ctx_entropy
                })

        # Save visualizations if output_dir is provided
        if output_dir:
            exp_name = f"{dataset_name}_{influence_method}_{clustering_algorithm}_{n_clusters}_{random_seed}"
            vis_dir = Path(output_dir) / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            # Create cluster distribution plot for each context dimension
            for i in range(c.shape[1]):
                context_dim = c[:, i]
                unique_contexts = np.unique(context_dim)

                plt.figure(figsize=(15, 10))

                # For each unique context value
                for j, ctx_val in enumerate(unique_contexts):
                    # Get clusters for this context
                    ctx_mask = context_dim == ctx_val
                    ctx_clusters = clusters[ctx_mask]

                    # Count occurrences of each cluster
                    cluster_counts = np.bincount(ctx_clusters, minlength=n_clusters)
                    cluster_probs = cluster_counts / len(ctx_clusters)

                    # Plot
                    plt.subplot(len(unique_contexts), 1, j + 1)
                    plt.bar(range(n_clusters), cluster_probs)
                    plt.title(f"Context Dimension {i}, Value {ctx_val}")
                    plt.xlabel("Cluster")
                    plt.ylabel("Probability")
                    plt.ylim(0, 1)

                plt.tight_layout()
                plt.savefig(vis_dir / f"{exp_name}_context_dim_{i}_distribution.pdf", format='pdf')
                plt.close()

        # Return results
        result = {
            "dataset": dataset_name,
            "influence_method": influence_method,
            "clustering_algorithm": clustering_algorithm,
            "n_clusters": n_clusters,
            "random_seed": random_seed,
            "entropy": cluster_entropy,
            "context_results": context_results,
            "context_distributions": context_distributions
        }

        return result

    except Exception as e:
        logger.error(f"Error in experiment: {e}", exc_info=True)

        # Return error result
        result = {
            "dataset": dataset_name,
            "influence_method": influence_method,
            "clustering_algorithm": clustering_algorithm,
            "n_clusters": n_clusters,
            "random_seed": random_seed,
            "error": str(e)
        }

        return result


def run_contextual_coherence(datasets, influence_methods, clustering_algorithms,
                            n_clusters_list, random_seeds, output_dir, n_jobs=-1, verbose=False):
    """
    Run comprehensive contextual coherence experiments.

    Parameters
    ----------
    datasets : list
        List of datasets to use.
    influence_methods : list
        List of influence methods to evaluate.
    clustering_algorithms : list
        List of clustering algorithms to evaluate.
    n_clusters_list : list
        List of number of clusters to evaluate.
    random_seeds : list
        List of random seeds for reproducibility and statistical analysis.
    output_dir : str or Path
        Directory to save results.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all available cores).
    verbose : bool, default=False
        Enable verbose output.

    Returns
    -------
    dict
        Dictionary containing experiment results.
    """
    # Set up logging
    logger = setup_logger("contextual_coherence", "INFO")
    logger.info("Starting contextual coherence experiments...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import clustering algorithms here to avoid circular imports
    from src.clustering.hierarchical import HierarchicalClustering
    from src.clustering.spectral import SpectralClustering

    # Add to global namespace for run_single_experiment
    globals()["HierarchicalClustering"] = HierarchicalClustering
    globals()["SpectralClustering"] = SpectralClustering

    # Generate experiment configurations
    experiments = []
    for dataset in datasets:
        for influence_method in influence_methods:
            for clustering_algorithm in clustering_algorithms:
                for n_clusters in n_clusters_list:
                    for random_seed in random_seeds:
                        experiments.append({
                            "dataset_name": dataset,
                            "influence_method": influence_method,
                            "clustering_algorithm": clustering_algorithm,
                            "n_clusters": n_clusters,
                            "random_seed": random_seed,
                            "output_dir": output_dir
                        })

    logger.info(f"Running {len(experiments)} experiments...")

    # Run experiments in parallel with optimized settings
    start_time = time.time()
    results = Parallel(
        n_jobs=n_jobs,
        verbose=10 if verbose else 0,
        batch_size="auto",
        pre_dispatch="2*n_jobs",
        max_nbytes="100M"  # Increase memory limit for better performance
    )(
        delayed(run_single_experiment)(**exp) for exp in experiments
    )

    logger.info(f"Experiments completed in {time.time() - start_time:.2f} seconds")

    # Process context results
    context_results = []
    for result in results:
        # Skip error results
        if "error" in result and result["error"] is not None:
            continue

        # Extract basic info
        dataset = result["dataset"]
        influence_method = result["influence_method"]
        clustering_algorithm = result["clustering_algorithm"]
        n_clusters = result["n_clusters"]
        random_seed = result["random_seed"]
        entropy = result["entropy"]

        # Process context results
        for ctx_result in result["context_results"]:
            context_results.append({
                "dataset": dataset,
                "influence_method": influence_method,
                "clustering_algorithm": clustering_algorithm,
                "n_clusters": n_clusters,
                "random_seed": random_seed,
                "entropy": entropy,
                "context_dimension": ctx_result["context_dimension"],
                "conditional_entropy": ctx_result["conditional_entropy"],
                "information_gain": ctx_result["information_gain"],
                "normalized_info_gain": ctx_result["normalized_info_gain"]
            })

    # Create results DataFrame
    results_df = pd.DataFrame(context_results)

    # Save results
    results_file = output_dir / "contextual_coherence_results.csv"
    results_df.to_csv(results_file, index=False)

    # Create summary visualizations
    logger.info("Creating summary visualizations...")
    create_summary_visualizations(results_df, output_dir)

    logger.info(f"Contextual coherence experiments completed. Results saved to {output_dir}")

    return {
        "results": results_df.to_dict(orient="records")
    }


def create_summary_visualizations(results_df, output_dir):
    """
    Create summary visualizations of contextual coherence results.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing experiment results.
    output_dir : str or Path
        Directory to save visualizations.
    """
    # Create visualizations directory
    vis_dir = Path(output_dir) / "summary_visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Calculate mean metrics across random seeds
    mean_results = results_df.groupby(
        ["dataset", "influence_method", "clustering_algorithm", "n_clusters", "context_dimension"]
    ).mean().reset_index()

    # Create heatmap of normalized information gain
    plt.figure(figsize=(15, 10))
    pivot = mean_results.pivot_table(
        index=["dataset", "context_dimension"],
        columns=["influence_method", "clustering_algorithm"],
        values="normalized_info_gain"
    )
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Normalized Information Gain by Dataset, Context, Method, and Algorithm")
    plt.tight_layout()
    plt.savefig(vis_dir / "normalized_info_gain_heatmap.pdf", format='pdf')
    plt.close()

    # Create bar plot of normalized information gain by influence method
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="normalized_info_gain",
        hue="influence_method"
    )
    plt.title("Normalized Information Gain by Dataset and Influence Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "normalized_info_gain_by_influence.pdf", format='pdf')
    plt.close()

    # Create bar plot of normalized information gain by context dimension
    for dataset in mean_results["dataset"].unique():
        plt.figure(figsize=(15, 8))
        dataset_results = mean_results[mean_results["dataset"] == dataset]

        sns.barplot(
            data=dataset_results,
            x="context_dimension",
            y="normalized_info_gain",
            hue="influence_method"
        )
        plt.title(f"Normalized Information Gain by Context Dimension for {dataset}")
        plt.tight_layout()
        plt.savefig(vis_dir / f"{dataset}_normalized_info_gain_by_context.pdf", format='pdf')
        plt.close()

    # Create line plot of normalized information gain by number of clusters
    plt.figure(figsize=(15, 8))
    sns.lineplot(
        data=mean_results,
        x="n_clusters",
        y="normalized_info_gain",
        hue="influence_method",
        style="clustering_algorithm",
        markers=True
    )
    plt.title("Normalized Information Gain by Number of Clusters")
    plt.tight_layout()
    plt.savefig(vis_dir / "normalized_info_gain_by_n_clusters.pdf", format='pdf')
    plt.close()
