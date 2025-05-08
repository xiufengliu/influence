"""
Ablation studies for TNNLS submission.

This module implements comprehensive ablation studies to evaluate the contribution
of different components of the Dynamic Influence-Based Clustering Framework.
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
from src.temporal.transition_matrix import TransitionMatrix
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_clustering, calculate_temporal_consistency
from src.utils.visualization import visualize_clusters


def run_influence_space_ablation(dataset_name, clustering_algorithm, n_clusters,
                                random_seed, output_dir):
    """
    Run ablation study on the impact of influence space.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
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
    # Import clustering algorithms here to avoid circular imports
    from src.clustering.hierarchical import HierarchicalClustering
    from src.clustering.spectral import SpectralClustering

    # Add to global namespace
    globals()["HierarchicalClustering"] = HierarchicalClustering
    globals()["SpectralClustering"] = SpectralClustering
    # Set up logging
    logger = logging.getLogger(f"ablation_influence_{dataset_name}_{clustering_algorithm}")

    # Set random seed
    np.random.seed(random_seed)

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        data = data_loader.load_data()

        preprocessor = Preprocessor()
        X, y, t, c = preprocessor.preprocess(data)

        # Train predictive model
        model_params = config.MODEL_PARAMS["gradient_boost"].copy()
        model_params["random_state"] = random_seed
        model = GradientBoostModel(**model_params)
        model.fit(X, y)

        # Generate influence spaces
        influence_spaces = {}

        # Raw feature space
        influence_spaces["raw"] = X

        # SHAP influence space
        influence_params = config.INFLUENCE_PARAMS["shap"].copy()
        influence_params["random_state"] = random_seed
        shap_generator = ShapInfluence(**influence_params)
        influence_spaces["shap"] = shap_generator.generate_influence(model, X)

        # LIME influence space
        influence_params = config.INFLUENCE_PARAMS["lime"].copy()
        influence_params["random_state"] = random_seed
        lime_generator = LimeInfluence(**influence_params)
        influence_spaces["lime"] = lime_generator.generate_influence(model, X)

        # Spearman influence space
        influence_params = config.INFLUENCE_PARAMS["spearman"].copy()
        spearman_generator = SpearmanInfluence(**influence_params)
        influence_spaces["spearman"] = spearman_generator.generate_influence(model, X)

        # Perform clustering and evaluation for each influence space
        results = []

        for influence_name, Z in influence_spaces.items():
            # Create clustering algorithm
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

            # Evaluate clustering
            metrics = evaluate_clustering(Z, clusters)

            # Compute temporal consistency
            temporal_consistency = calculate_temporal_consistency(clusters, t)

            # Compute transition matrix
            transition_matrix = TransitionMatrix()
            P = transition_matrix.compute(clusters, t)

            # Compute cluster stability
            cluster_stability = transition_matrix.get_cluster_stability()

            # Save visualization if output_dir is provided
            if output_dir:
                exp_name = f"{dataset_name}_{influence_name}_{clustering_algorithm}_{n_clusters}_{random_seed}"
                vis_dir = Path(output_dir) / "visualizations"
                vis_dir.mkdir(exist_ok=True)

                # Visualize clusters
                visualize_clusters(
                    Z, clusters,
                    output_path=vis_dir / f"{exp_name}_clusters.png",
                    method='pca'
                )

            # Store results
            result = {
                "dataset": dataset_name,
                "influence_method": influence_name,
                "clustering_algorithm": clustering_algorithm,
                "n_clusters": n_clusters,
                "random_seed": random_seed,
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "entropy": metrics["entropy"],
                "temporal_consistency": temporal_consistency,
                "mean_cluster_stability": np.mean(cluster_stability)
            }

            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Error in experiment: {e}", exc_info=True)

        # Return error result
        result = {
            "dataset": dataset_name,
            "clustering_algorithm": clustering_algorithm,
            "n_clusters": n_clusters,
            "random_seed": random_seed,
            "error": str(e)
        }

        return [result]


def run_temporal_integration_ablation(dataset_name, influence_method, clustering_algorithm,
                                     n_clusters, random_seed, output_dir):
    """
    Run ablation study on the effect of temporal integration.

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
    # Import clustering algorithms here to avoid circular imports
    from src.clustering.hierarchical import HierarchicalClustering
    from src.clustering.spectral import SpectralClustering

    # Add to global namespace
    globals()["HierarchicalClustering"] = HierarchicalClustering
    globals()["SpectralClustering"] = SpectralClustering
    # Set up logging
    logger = logging.getLogger(f"ablation_temporal_{dataset_name}_{influence_method}_{clustering_algorithm}")

    # Set random seed
    np.random.seed(random_seed)

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        data = data_loader.load_data()

        preprocessor = Preprocessor()
        X, y, t, c = preprocessor.preprocess(data)

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

        # Create clustering algorithm
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

        # Run with and without temporal constraints
        results = []

        # Without temporal constraints
        clusters_no_temporal = clustering.fit_predict(Z)

        # With temporal constraints
        clusters_with_temporal = clustering.fit_predict(Z, t)

        # Evaluate both
        for temporal_mode, clusters in [("no_temporal", clusters_no_temporal),
                                       ("with_temporal", clusters_with_temporal)]:
            # Evaluate clustering
            metrics = evaluate_clustering(Z, clusters)

            # Compute temporal consistency
            temporal_consistency = calculate_temporal_consistency(clusters, t)

            # Compute transition matrix
            transition_matrix = TransitionMatrix()
            P = transition_matrix.compute(clusters, t)

            # Compute cluster stability
            cluster_stability = transition_matrix.get_cluster_stability()

            # Save visualization if output_dir is provided
            if output_dir:
                exp_name = f"{dataset_name}_{influence_method}_{clustering_algorithm}_{n_clusters}_{temporal_mode}_{random_seed}"
                vis_dir = Path(output_dir) / "visualizations"
                vis_dir.mkdir(exist_ok=True)

                # Visualize clusters
                visualize_clusters(
                    Z, clusters,
                    output_path=vis_dir / f"{exp_name}_clusters.png",
                    method='pca'
                )

            # Store results
            result = {
                "dataset": dataset_name,
                "influence_method": influence_method,
                "clustering_algorithm": clustering_algorithm,
                "n_clusters": n_clusters,
                "temporal_mode": temporal_mode,
                "random_seed": random_seed,
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "entropy": metrics["entropy"],
                "temporal_consistency": temporal_consistency,
                "mean_cluster_stability": np.mean(cluster_stability)
            }

            results.append(result)

        return results

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

        return [result]


def run_contextual_alignment_ablation(dataset_name, influence_method, clustering_algorithm,
                                     n_clusters, random_seed, output_dir):
    """
    Run ablation study on the role of contextual alignment.

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
    # Import clustering algorithms here to avoid circular imports
    from src.clustering.hierarchical import HierarchicalClustering
    from src.clustering.spectral import SpectralClustering

    # Add to global namespace
    globals()["HierarchicalClustering"] = HierarchicalClustering
    globals()["SpectralClustering"] = SpectralClustering
    # Set up logging
    logger = logging.getLogger(f"ablation_contextual_{dataset_name}_{influence_method}_{clustering_algorithm}")

    # Set random seed
    np.random.seed(random_seed)

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        data = data_loader.load_data()

        preprocessor = Preprocessor()
        X, y, t, c = preprocessor.preprocess(data)

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

        # Create clustering algorithm
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

        # Run with and without contextual constraints
        results = []

        # Without contextual constraints
        clusters_no_contextual = clustering.fit_predict(Z)

        # With contextual constraints
        clusters_with_contextual = clustering.fit_predict(Z, t=None, c=c)

        # Evaluate both
        for contextual_mode, clusters in [("no_contextual", clusters_no_contextual),
                                         ("with_contextual", clusters_with_contextual)]:
            # Evaluate clustering
            metrics = evaluate_clustering(Z, clusters)

            # Compute temporal consistency
            temporal_consistency = calculate_temporal_consistency(clusters, t)

            # Compute transition matrix
            transition_matrix = TransitionMatrix()
            P = transition_matrix.compute(clusters, t)

            # Compute cluster stability
            cluster_stability = transition_matrix.get_cluster_stability()

            # Save visualization if output_dir is provided
            if output_dir:
                exp_name = f"{dataset_name}_{influence_method}_{clustering_algorithm}_{n_clusters}_{contextual_mode}_{random_seed}"
                vis_dir = Path(output_dir) / "visualizations"
                vis_dir.mkdir(exist_ok=True)

                # Visualize clusters
                visualize_clusters(
                    Z, clusters,
                    output_path=vis_dir / f"{exp_name}_clusters.png",
                    method='pca'
                )

            # Store results
            result = {
                "dataset": dataset_name,
                "influence_method": influence_method,
                "clustering_algorithm": clustering_algorithm,
                "n_clusters": n_clusters,
                "contextual_mode": contextual_mode,
                "random_seed": random_seed,
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "entropy": metrics["entropy"],
                "temporal_consistency": temporal_consistency,
                "mean_cluster_stability": np.mean(cluster_stability)
            }

            results.append(result)

        return results

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

        return [result]


def run_ablation_studies(datasets, influence_methods, clustering_algorithms,
                        n_clusters_list, random_seeds, output_dir, n_jobs=-1, verbose=False):
    """
    Run comprehensive ablation studies.

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
    logger = setup_logger("ablation_studies", "INFO")
    logger.info("Starting ablation studies...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each ablation study
    influence_dir = output_dir / "influence_space"
    influence_dir.mkdir(exist_ok=True)

    temporal_dir = output_dir / "temporal_integration"
    temporal_dir.mkdir(exist_ok=True)

    contextual_dir = output_dir / "contextual_alignment"
    contextual_dir.mkdir(exist_ok=True)

    # Run influence space ablation
    logger.info("Running influence space ablation studies...")
    influence_experiments = []

    for dataset in datasets:
        for clustering_algorithm in clustering_algorithms:
            for n_clusters in n_clusters_list:
                for random_seed in random_seeds:
                    influence_experiments.append({
                        "dataset": dataset,
                        "clustering_algorithm": clustering_algorithm,
                        "n_clusters": n_clusters,
                        "random_seed": random_seed,
                        "output_dir": influence_dir
                    })

    logger.info(f"Running {len(influence_experiments)} influence space experiments...")

    influence_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(run_influence_space_ablation)(**exp) for exp in influence_experiments
    )

    # Flatten results
    influence_results_flat = []
    for result_list in influence_results:
        influence_results_flat.extend(result_list)

    # Create DataFrame
    influence_df = pd.DataFrame(influence_results_flat)

    # Save results
    influence_file = influence_dir / "influence_space_results.csv"
    influence_df.to_csv(influence_file, index=False)

    # Run temporal integration ablation
    logger.info("Running temporal integration ablation studies...")
    temporal_experiments = []

    for dataset in datasets:
        for influence_method in influence_methods:
            for clustering_algorithm in clustering_algorithms:
                for n_clusters in n_clusters_list:
                    for random_seed in random_seeds:
                        temporal_experiments.append({
                            "dataset": dataset,
                            "influence_method": influence_method,
                            "clustering_algorithm": clustering_algorithm,
                            "n_clusters": n_clusters,
                            "random_seed": random_seed,
                            "output_dir": temporal_dir
                        })

    logger.info(f"Running {len(temporal_experiments)} temporal integration experiments...")

    temporal_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(run_temporal_integration_ablation)(**exp) for exp in temporal_experiments
    )

    # Flatten results
    temporal_results_flat = []
    for result_list in temporal_results:
        temporal_results_flat.extend(result_list)

    # Create DataFrame
    temporal_df = pd.DataFrame(temporal_results_flat)

    # Save results
    temporal_file = temporal_dir / "temporal_integration_results.csv"
    temporal_df.to_csv(temporal_file, index=False)

    # Run contextual alignment ablation
    logger.info("Running contextual alignment ablation studies...")
    contextual_experiments = []

    for dataset in datasets:
        for influence_method in influence_methods:
            for clustering_algorithm in clustering_algorithms:
                for n_clusters in n_clusters_list:
                    for random_seed in random_seeds:
                        contextual_experiments.append({
                            "dataset": dataset,
                            "influence_method": influence_method,
                            "clustering_algorithm": clustering_algorithm,
                            "n_clusters": n_clusters,
                            "random_seed": random_seed,
                            "output_dir": contextual_dir
                        })

    logger.info(f"Running {len(contextual_experiments)} contextual alignment experiments...")

    contextual_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(run_contextual_alignment_ablation)(**exp) for exp in contextual_experiments
    )

    # Flatten results
    contextual_results_flat = []
    for result_list in contextual_results:
        contextual_results_flat.extend(result_list)

    # Create DataFrame
    contextual_df = pd.DataFrame(contextual_results_flat)

    # Save results
    contextual_file = contextual_dir / "contextual_alignment_results.csv"
    contextual_df.to_csv(contextual_file, index=False)

    # Create summary visualizations
    logger.info("Creating summary visualizations...")

    # Influence space visualizations
    create_influence_space_visualizations(influence_df, output_dir)

    # Temporal integration visualizations
    create_temporal_integration_visualizations(temporal_df, output_dir)

    # Contextual alignment visualizations
    create_contextual_alignment_visualizations(contextual_df, output_dir)

    logger.info(f"Ablation studies completed. Results saved to {output_dir}")

    return {
        "influence_space": influence_df.to_dict(orient="records"),
        "temporal_integration": temporal_df.to_dict(orient="records"),
        "contextual_alignment": contextual_df.to_dict(orient="records")
    }


def create_influence_space_visualizations(results_df, output_dir):
    """
    Create summary visualizations for influence space ablation.

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

    # Filter out error results
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    # Calculate mean metrics across random seeds
    mean_results = results_df.groupby(
        ["dataset", "influence_method", "clustering_algorithm", "n_clusters"]
    ).mean().reset_index()

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
    plt.savefig(vis_dir / "influence_silhouette_by_method.png", dpi=300)
    plt.close()

    # Create bar plot of temporal consistency by influence method
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="temporal_consistency",
        hue="influence_method"
    )
    plt.title("Temporal Consistency by Dataset and Influence Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "influence_temporal_consistency_by_method.png", dpi=300)
    plt.close()


def create_temporal_integration_visualizations(results_df, output_dir):
    """
    Create summary visualizations for temporal integration ablation.

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

    # Filter out error results
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    # Calculate mean metrics across random seeds
    mean_results = results_df.groupby(
        ["dataset", "influence_method", "clustering_algorithm", "n_clusters", "temporal_mode"]
    ).mean().reset_index()

    # Create bar plot of temporal consistency by temporal mode
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="temporal_consistency",
        hue="temporal_mode"
    )
    plt.title("Temporal Consistency by Dataset and Temporal Mode")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "temporal_consistency_by_mode.png", dpi=300)
    plt.close()

    # Create bar plot of silhouette scores by temporal mode
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="silhouette",
        hue="temporal_mode"
    )
    plt.title("Silhouette Score by Dataset and Temporal Mode")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "temporal_silhouette_by_mode.png", dpi=300)
    plt.close()


def create_contextual_alignment_visualizations(results_df, output_dir):
    """
    Create summary visualizations for contextual alignment ablation.

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

    # Filter out error results
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    # Calculate mean metrics across random seeds
    mean_results = results_df.groupby(
        ["dataset", "influence_method", "clustering_algorithm", "n_clusters", "contextual_mode"]
    ).mean().reset_index()

    # Create bar plot of entropy by contextual mode
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="entropy",
        hue="contextual_mode"
    )
    plt.title("Entropy by Dataset and Contextual Mode")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "contextual_entropy_by_mode.png", dpi=300)
    plt.close()

    # Create bar plot of silhouette scores by contextual mode
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="silhouette",
        hue="contextual_mode"
    )
    plt.title("Silhouette Score by Dataset and Contextual Mode")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "contextual_silhouette_by_mode.png", dpi=300)
    plt.close()