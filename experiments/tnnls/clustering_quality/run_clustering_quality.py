"""
Clustering quality experiments for TNNLS submission.

This module implements comprehensive clustering quality experiments comparing
different influence methods and clustering algorithms across multiple datasets.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wilcoxon
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
from src.clustering.hierarchical import HierarchicalClustering
from src.clustering.spectral import SpectralClustering
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_clustering
from src.utils.visualization import visualize_clusters


def run_single_experiment(dataset_name, influence_method, clustering_algorithm,
                         n_clusters, random_seed, output_dir):
    """
    Run a single clustering quality experiment.

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
    logger = logging.getLogger(f"clustering_quality_{dataset_name}_{influence_method}_{clustering_algorithm}")

    # Set random seed
    np.random.seed(random_seed)

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
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

        # Check if we have only one cluster
        unique_clusters = np.unique(clusters)
        n_unique_clusters = len(unique_clusters)

        if n_unique_clusters <= 1:
            logger.warning(f"Only {n_unique_clusters} cluster found for {dataset_name}, {influence_method}, {clustering_algorithm}, n_clusters={n_clusters}")

        # Evaluate clustering
        metrics = evaluate_clustering(Z, clusters)

        # Save cluster visualization if output_dir is provided
        if output_dir:
            exp_name = f"{dataset_name}_{influence_method}_{clustering_algorithm}_{n_clusters}_{random_seed}"
            vis_dir = Path(output_dir) / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            # Visualize clusters with PCA
            visualize_clusters(
                Z, clusters,
                output_path=vis_dir / f"{exp_name}_pca.pdf",
                method='pca'
            )

            # Visualize clusters with t-SNE only if we have more than one cluster
            # This avoids segmentation faults with t-SNE
            if n_unique_clusters > 1:
                visualize_clusters(
                    Z, clusters,
                    output_path=vis_dir / f"{exp_name}_tsne.pdf",
                    method='tsne'
                )

        # Return results
        result = {
            "dataset": dataset_name,
            "influence_method": influence_method,
            "clustering_algorithm": clustering_algorithm,
            "n_clusters": n_clusters,
            "random_seed": random_seed,
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "entropy": metrics["entropy"]
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


def run_clustering_quality(datasets, influence_methods, clustering_algorithms,
                          n_clusters_list, random_seeds, output_dir, n_jobs=-1, verbose=False):
    """
    Run comprehensive clustering quality experiments.

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
    logger = setup_logger("clustering_quality", "INFO")
    logger.info("Starting clustering quality experiments...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_file = output_dir / "clustering_quality_results.csv"
    results_df.to_csv(results_file, index=False)

    # Perform statistical analysis if we have valid results
    logger.info("Performing statistical analysis...")
    try:
        # Check if we have any valid results (without errors)
        if "error" in results_df.columns:
            valid_results = results_df[results_df["error"].isna()]
        else:
            valid_results = results_df

        if len(valid_results) > 0 and all(col in valid_results.columns for col in ["silhouette", "davies_bouldin", "calinski_harabasz", "entropy"]):
            statistical_analysis = perform_statistical_analysis(results_df)

            # Save statistical analysis
            stats_file = output_dir / "clustering_quality_statistics.csv"
            statistical_analysis.to_csv(stats_file, index=False)
        else:
            logger.warning("Not enough valid results for statistical analysis")
            statistical_analysis = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error in statistical analysis: {e}")
        statistical_analysis = pd.DataFrame()

    # Create visualizations if we have valid results
    logger.info("Creating summary visualizations...")
    try:
        # Check if we have any valid results (without errors)
        if "error" in results_df.columns:
            valid_results = results_df[results_df["error"].isna()]
        else:
            valid_results = results_df

        if len(valid_results) > 0 and all(col in valid_results.columns for col in ["silhouette", "davies_bouldin", "calinski_harabasz", "entropy"]):
            create_summary_visualizations(results_df, output_dir)
        else:
            logger.warning("Not enough valid results for visualizations")
    except Exception as e:
        logger.error(f"Error in creating visualizations: {e}")

    logger.info(f"Clustering quality experiments completed. Results saved to {output_dir}")

    return {
        "results": results_df.to_dict(orient="records"),
        "statistics": statistical_analysis.to_dict(orient="records")
    }


def perform_statistical_analysis(results_df):
    """
    Perform statistical analysis on experiment results.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing experiment results.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing statistical analysis results.
    """
    # Filter out error results
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    # Group by dataset, influence method, clustering algorithm, and n_clusters
    grouped = results_df.groupby(["dataset", "influence_method", "clustering_algorithm", "n_clusters"])

    # Calculate mean and standard deviation
    stats = grouped.agg({
        "silhouette": ["mean", "std"],
        "davies_bouldin": ["mean", "std"],
        "calinski_harabasz": ["mean", "std"],
        "entropy": ["mean", "std"]
    }).reset_index()

    # Flatten multi-level columns
    stats.columns = ["_".join(col).strip("_") for col in stats.columns.values]

    # Perform statistical tests
    statistical_tests = []

    # For each dataset and n_clusters
    for dataset in results_df["dataset"].unique():
        for n_clusters in results_df["n_clusters"].unique():
            # Get baseline (raw features with k-means)
            baseline = results_df[
                (results_df["dataset"] == dataset) &
                (results_df["influence_method"] == "shap") &
                (results_df["clustering_algorithm"] == "kmeans") &
                (results_df["n_clusters"] == n_clusters)
            ]

            # Compare each method against baseline
            for influence_method in results_df["influence_method"].unique():
                for clustering_algorithm in results_df["clustering_algorithm"].unique():
                    # Skip baseline
                    if influence_method == "shap" and clustering_algorithm == "kmeans":
                        continue

                    # Get comparison method
                    comparison = results_df[
                        (results_df["dataset"] == dataset) &
                        (results_df["influence_method"] == influence_method) &
                        (results_df["clustering_algorithm"] == clustering_algorithm) &
                        (results_df["n_clusters"] == n_clusters)
                    ]

                    # Skip if not enough samples
                    if len(baseline) < 5 or len(comparison) < 5:
                        continue

                    # Perform Wilcoxon signed-rank test for each metric
                    for metric in ["silhouette", "davies_bouldin", "calinski_harabasz", "entropy"]:
                        try:
                            stat, p_value = wilcoxon(baseline[metric], comparison[metric])

                            statistical_tests.append({
                                "dataset": dataset,
                                "baseline_influence": "shap",
                                "baseline_clustering": "kmeans",
                                "comparison_influence": influence_method,
                                "comparison_clustering": clustering_algorithm,
                                "n_clusters": n_clusters,
                                "metric": metric,
                                "statistic": stat,
                                "p_value": p_value,
                                "significant": p_value < 0.05
                            })
                        except Exception as e:
                            # Skip if test fails
                            pass

    # Create statistical tests DataFrame
    tests_df = pd.DataFrame(statistical_tests)

    return tests_df


def create_summary_visualizations(results_df, output_dir):
    """
    Create summary visualizations of experiment results.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing experiment results.
    output_dir : str or Path
        Directory to save visualizations.
    """
    # Filter out error results
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    # Create visualizations directory
    vis_dir = Path(output_dir) / "summary_visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Calculate mean metrics across random seeds
    mean_results = results_df.groupby(
        ["dataset", "influence_method", "clustering_algorithm", "n_clusters"]
    ).mean().reset_index()

    # Create heatmap of silhouette scores
    plt.figure(figsize=(15, 10))
    pivot = mean_results.pivot_table(
        index=["dataset", "n_clusters"],
        columns=["influence_method", "clustering_algorithm"],
        values="silhouette"
    )
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Silhouette Score by Dataset, Method, and Algorithm")
    plt.tight_layout()
    plt.savefig(vis_dir / "silhouette_heatmap.pdf", format='pdf')
    plt.close()

    # Create heatmap of Davies-Bouldin index
    plt.figure(figsize=(15, 10))
    pivot = mean_results.pivot_table(
        index=["dataset", "n_clusters"],
        columns=["influence_method", "clustering_algorithm"],
        values="davies_bouldin"
    )
    sns.heatmap(pivot, annot=True, cmap="YlGnBu_r", fmt=".3f")
    plt.title("Davies-Bouldin Index by Dataset, Method, and Algorithm")
    plt.tight_layout()
    plt.savefig(vis_dir / "davies_bouldin_heatmap.png", dpi=300)
    plt.close()

    # Create bar plot of silhouette scores by influence method
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="silhouette",
        hue="influence_method"
    )
    plt.title("Silhouette Score by Dataset and Influence Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "silhouette_by_influence.pdf", format='pdf')
    plt.close()

    # Create bar plot of silhouette scores by clustering algorithm
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="silhouette",
        hue="clustering_algorithm"
    )
    plt.title("Silhouette Score by Dataset and Clustering Algorithm")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "silhouette_by_algorithm.pdf", format='pdf')
    plt.close()

    # Create line plot of silhouette scores by number of clusters
    plt.figure(figsize=(15, 8))
    sns.lineplot(
        data=mean_results,
        x="n_clusters",
        y="silhouette",
        hue="influence_method",
        style="clustering_algorithm",
        markers=True
    )
    plt.title("Silhouette Score by Number of Clusters")
    plt.tight_layout()
    plt.savefig(vis_dir / "silhouette_by_n_clusters.pdf", format='pdf')
    plt.close()
