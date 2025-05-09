"""
Temporal analysis experiments for TNNLS submission.

This module implements comprehensive temporal analysis experiments to evaluate
the framework's ability to capture temporal patterns and transitions.
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
from src.temporal.transition_matrix import TransitionMatrix
from src.temporal.anomaly_detection import AnomalyDetection
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_temporal_consistency
from src.utils.visualization import visualize_transitions, visualize_temporal_evolution


def run_single_experiment(dataset_name, influence_method, clustering_algorithm,
                         n_clusters, random_seed, output_dir):
    """
    Run a single temporal analysis experiment.

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

    # Set up logging
    logger = logging.getLogger(f"temporal_analysis_{dataset_name}_{influence_method}_{clustering_algorithm}")

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

        # Check if we have only one cluster
        unique_clusters = np.unique(clusters)
        n_unique_clusters = len(unique_clusters)

        if n_unique_clusters <= 1:
            logger.warning(f"Only {n_unique_clusters} cluster found for {dataset_name}, {influence_method}, {clustering_algorithm}, n_clusters={n_clusters}")
            # For a single cluster, set default values
            P = np.array([[1.0]])
            stationary_distribution = np.array([1.0])
            cluster_stability = np.array([1.0])
            temporal_consistency = 1.0
            mean_transition_entropy = 0.0
        else:
            # Compute transition matrix
            transition_matrix = TransitionMatrix()
            P = transition_matrix.compute(clusters, t)

            # Compute stationary distribution
            stationary_distribution = transition_matrix.stationary_distribution

            # Compute cluster stability
            cluster_stability = transition_matrix.get_cluster_stability()

            # Compute temporal consistency
            temporal_consistency = calculate_temporal_consistency(clusters, t)

            # Compute transition entropy
            transition_entropy = np.array([entropy(P[i]) for i in range(n_unique_clusters)])
            mean_transition_entropy = np.mean(transition_entropy)

        # Detect anomalies only if we have more than one cluster
        if n_unique_clusters > 1:
            try:
                anomaly_detector = AnomalyDetection()
                anomalies = anomaly_detector.detect(clusters, P, t, threshold=0.1)
                anomaly_rate = len(anomalies) / len(clusters) if len(anomalies) > 0 else 0
            except Exception as e:
                logger.warning(f"Error detecting anomalies: {e}")
                anomalies = pd.DataFrame()
                anomaly_rate = 0.0
        else:
            # No anomalies with only one cluster
            anomalies = pd.DataFrame()
            anomaly_rate = 0.0

        # Save visualizations if output_dir is provided
        if output_dir:
            exp_name = f"{dataset_name}_{influence_method}_{clustering_algorithm}_{n_clusters}_{random_seed}"
            vis_dir = Path(output_dir) / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            # Only create visualizations if we have more than one cluster
            if n_unique_clusters > 1:
                # Visualize transition matrix
                visualize_transitions(
                    P,
                    output_path=vis_dir / f"{exp_name}_transition_matrix.pdf"
                )

                # Use a sample of the data if it's large to reduce memory usage
                if len(clusters) > 1000:
                    # Take a random sample of 1000 points
                    sample_indices = np.random.choice(len(clusters), size=1000, replace=False)
                    # Sort indices to maintain temporal order
                    sample_indices = np.sort(sample_indices)
                    clusters_sample = clusters[sample_indices]
                    t_sample = t.iloc[sample_indices] if hasattr(t, 'iloc') else t[sample_indices]
                    logger.info(f"Using a sample of 1000 points for visualization (from {len(clusters)} total)")

                    # Visualize temporal evolution with sample
                    visualize_temporal_evolution(
                        clusters_sample, t_sample,
                        output_path=vis_dir / f"{exp_name}_temporal_evolution.pdf"
                    )
                else:
                    # Visualize temporal evolution with all data
                    visualize_temporal_evolution(
                        clusters, t,
                        output_path=vis_dir / f"{exp_name}_temporal_evolution.pdf"
                    )
            else:
                logger.warning(f"Skipping visualizations for {exp_name} due to only having {n_unique_clusters} cluster")

            # Save transition matrix as CSV
            pd.DataFrame(P).to_csv(vis_dir / f"{exp_name}_transition_matrix.csv")

            # Save anomalies if any
            if len(anomalies) > 0:
                anomalies.to_csv(vis_dir / f"{exp_name}_anomalies.csv", index=False)

        # Return results
        result = {
            "dataset": dataset_name,  # Use "dataset" as the key for consistency
            "influence_method": influence_method,
            "clustering_algorithm": clustering_algorithm,
            "n_clusters": n_clusters,
            "random_seed": random_seed,
            "temporal_consistency": temporal_consistency,
            "mean_cluster_stability": np.mean(cluster_stability),
            "min_cluster_stability": np.min(cluster_stability),
            "max_cluster_stability": np.max(cluster_stability),
            "mean_transition_entropy": mean_transition_entropy,
            "anomaly_rate": anomaly_rate,
            "stationary_distribution": stationary_distribution.tolist(),
            "transition_matrix": P.tolist()
        }

        return result

    except Exception as e:
        logger.error(f"Error in experiment: {e}", exc_info=True)

        # Return error result
        result = {
            "dataset": dataset_name,  # Use "dataset" as the key for consistency
            "influence_method": influence_method,
            "clustering_algorithm": clustering_algorithm,
            "n_clusters": n_clusters,
            "random_seed": random_seed,
            "error": str(e)
        }

        return result


def run_temporal_analysis(datasets, influence_methods, clustering_algorithms,
                         n_clusters_list, random_seeds, output_dir, n_jobs=-1, verbose=False):
    """
    Run comprehensive temporal analysis experiments.

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
    logger = setup_logger("temporal_analysis", "INFO")
    logger.info("Starting temporal analysis experiments...")

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

    # Run experiments in parallel with memory-optimized settings
    start_time = time.time()

    # Use a more conservative approach to parallel processing
    # Limit the number of jobs to avoid memory issues
    safe_n_jobs = min(4, os.cpu_count() or 1) if n_jobs == -1 else min(n_jobs, 4)
    logger.info(f"Using {safe_n_jobs} parallel jobs to avoid memory issues")

    results = Parallel(
        n_jobs=safe_n_jobs,
        verbose=10 if verbose else 0,
        batch_size=1,  # Process one task at a time
        pre_dispatch="1*n_jobs",  # Limit pre-dispatched tasks
        max_nbytes="50M",  # Reduce memory limit
        timeout=None  # No timeout
    )(
        delayed(run_single_experiment)(**exp) for exp in experiments
    )

    logger.info(f"Experiments completed in {time.time() - start_time:.2f} seconds")

    # Process results
    processed_results = []
    for result in results:
        # Skip error results
        if "error" in result and result["error"] is not None:
            continue

        # Extract basic info
        processed_result = {
            "dataset": result["dataset"],  # Keep as "dataset" for consistency with other modules
            "influence_method": result["influence_method"],
            "clustering_algorithm": result["clustering_algorithm"],
            "n_clusters": result["n_clusters"],
            "random_seed": result["random_seed"],
            "temporal_consistency": result["temporal_consistency"],
            "mean_cluster_stability": result["mean_cluster_stability"],
            "min_cluster_stability": result["min_cluster_stability"],
            "max_cluster_stability": result["max_cluster_stability"],
            "mean_transition_entropy": result["mean_transition_entropy"],
            "anomaly_rate": result["anomaly_rate"]
        }

        processed_results.append(processed_result)

    # Create results DataFrame
    results_df = pd.DataFrame(processed_results)

    # Save results
    results_file = output_dir / "temporal_analysis_results.csv"
    results_df.to_csv(results_file, index=False)

    # Create summary visualizations
    logger.info("Creating summary visualizations...")
    create_summary_visualizations(results_df, output_dir)

    # Create transition matrix visualizations
    logger.info("Creating transition matrix visualizations...")
    create_transition_visualizations(results, output_dir)

    logger.info(f"Temporal analysis experiments completed. Results saved to {output_dir}")

    return {
        "results": results_df.to_dict(orient="records")
    }


def create_summary_visualizations(results_df, output_dir):
    """
    Create summary visualizations of temporal analysis results.

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
        ["dataset", "influence_method", "clustering_algorithm", "n_clusters"]
    ).mean().reset_index()

    # Create heatmap of temporal consistency
    plt.figure(figsize=(15, 10))
    pivot = mean_results.pivot_table(
        index=["dataset", "n_clusters"],
        columns=["influence_method", "clustering_algorithm"],
        values="temporal_consistency"
    )
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Temporal Consistency by Dataset, Method, and Algorithm")
    plt.tight_layout()
    plt.savefig(vis_dir / "temporal_consistency_heatmap.pdf", format='pdf')
    plt.close()

    # Create heatmap of cluster stability
    plt.figure(figsize=(15, 10))
    pivot = mean_results.pivot_table(
        index=["dataset", "n_clusters"],
        columns=["influence_method", "clustering_algorithm"],
        values="mean_cluster_stability"
    )
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Cluster Stability by Dataset, Method, and Algorithm")
    plt.tight_layout()
    plt.savefig(vis_dir / "cluster_stability_heatmap.pdf", format='pdf')
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
    plt.savefig(vis_dir / "temporal_consistency_by_influence.pdf", format='pdf')
    plt.close()

    # Create bar plot of anomaly rate by influence method
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=mean_results,
        x="dataset",
        y="anomaly_rate",
        hue="influence_method"
    )
    plt.title("Anomaly Rate by Dataset and Influence Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "anomaly_rate_by_influence.pdf", format='pdf')
    plt.close()


def create_transition_visualizations(results, output_dir):
    """
    Create visualizations of transition matrices.

    Parameters
    ----------
    results : list
        List of experiment results.
    output_dir : str or Path
        Directory to save visualizations.
    """
    # Set up logging
    logger = logging.getLogger("transition_visualizations")

    # Create visualizations directory
    vis_dir = Path(output_dir) / "transition_visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Process each result
    for result in results:
        try:
            # Skip error results
            if "error" in result and result["error"] is not None:
                continue

            # Skip if no transition matrix
            if "transition_matrix" not in result:
                continue

            # Extract info
            dataset = result["dataset"]
            influence_method = result["influence_method"]
            clustering_algorithm = result["clustering_algorithm"]
            n_clusters = result["n_clusters"]
            random_seed = result["random_seed"]

            # Create experiment name
            exp_name = f"{dataset}_{influence_method}_{clustering_algorithm}_{n_clusters}_{random_seed}"

            # Extract transition matrix
            P = np.array(result["transition_matrix"])

            # Check if we have a single-cluster case (1x1 matrix)
            if P.shape[0] <= 1 or P.shape[1] <= 1:
                logger.warning(f"Skipping transition visualization for {exp_name} - only one cluster found")
                continue

            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(P, annot=True, cmap="YlGnBu", fmt=".3f")
            plt.title(f"Transition Matrix: {dataset}, {influence_method}, {clustering_algorithm}")
            plt.xlabel("To Cluster")
            plt.ylabel("From Cluster")
            plt.tight_layout()
            plt.savefig(vis_dir / f"{exp_name}_transition_heatmap.pdf", format='pdf')
            plt.close()

            # Create network diagram
            plt.figure(figsize=(10, 8))

            # Get actual number of clusters from the transition matrix
            actual_n_clusters = P.shape[0]

            # Create positions for nodes in a circle
            pos = {}
            for i in range(actual_n_clusters):
                angle = 2 * np.pi * i / actual_n_clusters
                pos[i] = (np.cos(angle), np.sin(angle))

            # Draw nodes
            for i in range(actual_n_clusters):
                plt.plot(pos[i][0], pos[i][1], 'o', markersize=20,
                        color=plt.cm.tab10(i % 10))
                plt.text(pos[i][0], pos[i][1], str(i),
                        horizontalalignment='center', verticalalignment='center')

            # Draw edges
            for i in range(actual_n_clusters):
                for j in range(actual_n_clusters):
                    if P[i, j] > 0.1:  # Only draw significant transitions
                        plt.arrow(pos[i][0], pos[i][1],
                                0.8 * (pos[j][0] - pos[i][0]),
                                0.8 * (pos[j][1] - pos[i][1]),
                                head_width=0.05, head_length=0.1,
                                fc=plt.cm.Blues(P[i, j]), ec=plt.cm.Blues(P[i, j]),
                                alpha=P[i, j])

            plt.title(f"Transition Network: {dataset}, {influence_method}, {clustering_algorithm}")
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(vis_dir / f"{exp_name}_transition_network.pdf", format='pdf')
            plt.close()

        except Exception as e:
            logger.error(f"Error creating transition visualization: {e}")
            plt.close()  # Make sure to close any open figures
