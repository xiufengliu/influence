"""
Computational analysis for TNNLS submission.

This module implements comprehensive computational analysis to evaluate the
efficiency and scalability of the Dynamic Influence-Based Clustering Framework.
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
from memory_profiler import memory_usage
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


def measure_runtime(func, *args, **kwargs):
    """
    Measure the runtime of a function.

    Parameters
    ----------
    func : callable
        Function to measure.
    *args : tuple
        Positional arguments to pass to the function.
    **kwargs : dict
        Keyword arguments to pass to the function.

    Returns
    -------
    tuple
        (result, runtime_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    return result, end_time - start_time


def measure_memory_usage(func, *args, **kwargs):
    """
    Measure the memory usage of a function.

    Parameters
    ----------
    func : callable
        Function to measure.
    *args : tuple
        Positional arguments to pass to the function.
    **kwargs : dict
        Keyword arguments to pass to the function.

    Returns
    -------
    tuple
        (result, memory_usage_mb)
    """
    def wrapper():
        return func(*args, **kwargs)

    mem_usage = memory_usage(wrapper, interval=0.1, timeout=None, max_usage=True)

    return wrapper(), mem_usage


def run_influence_method_benchmark(dataset_name, influence_method, n_samples_list, output_dir):
    """
    Benchmark influence method performance.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    influence_method : str
        Influence method to benchmark.
    n_samples_list : list
        List of sample sizes to benchmark.
    output_dir : str or Path
        Directory to save results.

    Returns
    -------
    dict
        Dictionary containing benchmark results.
    """
    # Set up logging
    logger = logging.getLogger(f"benchmark_influence_{dataset_name}_{influence_method}")

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        data = data_loader.load_data()

        preprocessor = Preprocessor()
        X, y, t, c = preprocessor.preprocess(data)

        # Create influence method
        if influence_method == "shap":
            influence_params = config.INFLUENCE_PARAMS["shap"].copy()
            influence_generator = ShapInfluence(**influence_params)
        elif influence_method == "lime":
            influence_params = config.INFLUENCE_PARAMS["lime"].copy()
            influence_generator = LimeInfluence(**influence_params)
        else:  # spearman
            influence_params = config.INFLUENCE_PARAMS["spearman"].copy()
            influence_generator = SpearmanInfluence(**influence_params)

        # Benchmark for different sample sizes
        results = []

        for n_samples in n_samples_list:
            if n_samples > len(X):
                logger.warning(f"Sample size {n_samples} exceeds dataset size {len(X)}. Skipping.")
                continue

            # Subsample data
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]

            # Train model
            model = GradientBoostModel(**config.MODEL_PARAMS["gradient_boost"])
            model.fit(X_sample, y_sample)

            # Measure influence generation time
            _, runtime = measure_runtime(
                influence_generator.generate_influence,
                model, X_sample
            )

            # Measure memory usage
            _, memory_usage = measure_memory_usage(
                influence_generator.generate_influence,
                model, X_sample
            )

            # Store results
            results.append({
                "dataset": dataset_name,
                "influence_method": influence_method,
                "n_samples": n_samples,
                "runtime_seconds": runtime,
                "memory_usage_mb": memory_usage
            })

        return results

    except Exception as e:
        logger.error(f"Error in benchmark: {e}", exc_info=True)

        # Return error result
        return [{
            "dataset": dataset_name,
            "influence_method": influence_method,
            "error": str(e)
        }]


def run_clustering_algorithm_benchmark(dataset_name, clustering_algorithm, n_samples_list, n_clusters_list, output_dir):
    """
    Benchmark clustering algorithm performance.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    clustering_algorithm : str
        Clustering algorithm to benchmark.
    n_samples_list : list
        List of sample sizes to benchmark.
    n_clusters_list : list
        List of number of clusters to benchmark.
    output_dir : str or Path
        Directory to save results.

    Returns
    -------
    dict
        Dictionary containing benchmark results.
    """
    # Set up logging
    logger = logging.getLogger(f"benchmark_clustering_{dataset_name}_{clustering_algorithm}")

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        data = data_loader.load_data()

        preprocessor = Preprocessor()
        X, y, t, c = preprocessor.preprocess(data)

        # Train model
        model = GradientBoostModel(**config.MODEL_PARAMS["gradient_boost"])
        model.fit(X, y)

        # Generate influence space
        influence_generator = ShapInfluence(**config.INFLUENCE_PARAMS["shap"])
        Z = influence_generator.generate_influence(model, X)

        # Benchmark for different sample sizes and number of clusters
        results = []

        for n_samples in n_samples_list:
            if n_samples > len(Z):
                logger.warning(f"Sample size {n_samples} exceeds dataset size {len(Z)}. Skipping.")
                continue

            # Subsample data
            indices = np.random.choice(len(Z), n_samples, replace=False)
            Z_sample = Z[indices]

            for n_clusters in n_clusters_list:
                # Create clustering algorithm
                if clustering_algorithm == "kmeans":
                    clustering_params = config.CLUSTERING_PARAMS["kmeans"].copy()
                    clustering_params["n_clusters"] = n_clusters
                    clustering = KMeansClustering(**clustering_params)
                elif clustering_algorithm == "hierarchical":
                    clustering_params = config.CLUSTERING_PARAMS["hierarchical"].copy()
                    clustering_params["n_clusters"] = n_clusters
                    clustering = HierarchicalClustering(**clustering_params)
                else:  # spectral
                    clustering_params = config.CLUSTERING_PARAMS["spectral"].copy()
                    clustering_params["n_clusters"] = n_clusters
                    clustering = SpectralClustering(**clustering_params)

                # Measure clustering time
                _, runtime = measure_runtime(
                    clustering.fit_predict,
                    Z_sample
                )

                # Measure memory usage
                _, memory_usage = measure_memory_usage(
                    clustering.fit_predict,
                    Z_sample
                )

                # Store results
                results.append({
                    "dataset": dataset_name,
                    "clustering_algorithm": clustering_algorithm,
                    "n_samples": n_samples,
                    "n_clusters": n_clusters,
                    "runtime_seconds": runtime,
                    "memory_usage_mb": memory_usage
                })

        return results

    except Exception as e:
        logger.error(f"Error in benchmark: {e}", exc_info=True)

        # Return error result
        return [{
            "dataset": dataset_name,
            "clustering_algorithm": clustering_algorithm,
            "error": str(e)
        }]


def run_end_to_end_benchmark(dataset_name, influence_method, clustering_algorithm, n_clusters, output_dir):
    """
    Benchmark end-to-end framework performance.

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
    output_dir : str or Path
        Directory to save results.

    Returns
    -------
    dict
        Dictionary containing benchmark results.
    """
    # Set up logging
    logger = logging.getLogger(f"benchmark_end_to_end_{dataset_name}_{influence_method}_{clustering_algorithm}")

    try:
        # Load and preprocess data
        start_time = time.time()
        data_loader = DataLoader(dataset_name=dataset_name)
        data = data_loader.load_data()

        preprocessor = Preprocessor()
        X, y, t, c = preprocessor.preprocess(data)
        preprocessing_time = time.time() - start_time

        # Train model
        start_time = time.time()
        model = GradientBoostModel(**config.MODEL_PARAMS["gradient_boost"])
        model.fit(X, y)
        model_training_time = time.time() - start_time

        # Generate influence space
        start_time = time.time()
        if influence_method == "shap":
            influence_generator = ShapInfluence(**config.INFLUENCE_PARAMS["shap"])
        elif influence_method == "lime":
            influence_generator = LimeInfluence(**config.INFLUENCE_PARAMS["lime"])
        else:  # spearman
            influence_generator = SpearmanInfluence(**config.INFLUENCE_PARAMS["spearman"])

        Z = influence_generator.generate_influence(model, X)
        influence_generation_time = time.time() - start_time

        # Perform clustering
        start_time = time.time()
        if clustering_algorithm == "kmeans":
            clustering_params = config.CLUSTERING_PARAMS["kmeans"].copy()
            clustering_params["n_clusters"] = n_clusters
            clustering = KMeansClustering(**clustering_params)
        elif clustering_algorithm == "hierarchical":
            clustering_params = config.CLUSTERING_PARAMS["hierarchical"].copy()
            clustering_params["n_clusters"] = n_clusters
            clustering = HierarchicalClustering(**clustering_params)
        else:  # spectral
            clustering_params = config.CLUSTERING_PARAMS["spectral"].copy()
            clustering_params["n_clusters"] = n_clusters
            clustering = SpectralClustering(**clustering_params)

        clusters = clustering.fit_predict(Z)
        clustering_time = time.time() - start_time

        # Total time
        total_time = preprocessing_time + model_training_time + influence_generation_time + clustering_time

        # Return results
        return {
            "dataset": dataset_name,
            "influence_method": influence_method,
            "clustering_algorithm": clustering_algorithm,
            "n_clusters": n_clusters,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "preprocessing_time": preprocessing_time,
            "model_training_time": model_training_time,
            "influence_generation_time": influence_generation_time,
            "clustering_time": clustering_time,
            "total_time": total_time
        }

    except Exception as e:
        logger.error(f"Error in benchmark: {e}", exc_info=True)

        # Return error result
        return {
            "dataset": dataset_name,
            "influence_method": influence_method,
            "clustering_algorithm": clustering_algorithm,
            "n_clusters": n_clusters,
            "error": str(e)
        }


def run_computational_analysis(datasets, influence_methods, clustering_algorithms,
                              n_clusters_list, random_seeds, output_dir, n_jobs=-1, verbose=False):
    """
    Run comprehensive computational analysis.

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
        Dictionary containing computational analysis results.
    """
    # Set up logging
    logger = setup_logger("computational_analysis", "INFO")
    logger.info("Starting computational analysis...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    influence_dir = output_dir / "influence_benchmarks"
    influence_dir.mkdir(exist_ok=True)

    clustering_dir = output_dir / "clustering_benchmarks"
    clustering_dir.mkdir(exist_ok=True)

    end_to_end_dir = output_dir / "end_to_end_benchmarks"
    end_to_end_dir.mkdir(exist_ok=True)

    # Define sample sizes for benchmarks
    n_samples_list = [100, 500, 1000, 5000, 10000]

    # Run influence method benchmarks
    logger.info("Running influence method benchmarks...")
    influence_results = []

    for dataset in datasets:
        for influence_method in influence_methods:
            logger.info(f"Benchmarking {influence_method} on {dataset}...")
            results = run_influence_method_benchmark(
                dataset, influence_method, n_samples_list, influence_dir
            )
            influence_results.extend(results)

    # Create DataFrame and save results
    influence_df = pd.DataFrame(influence_results)
    influence_file = influence_dir / "influence_benchmarks.csv"
    influence_df.to_csv(influence_file, index=False)

    # Run clustering algorithm benchmarks
    logger.info("Running clustering algorithm benchmarks...")
    clustering_results = []

    for dataset in datasets:
        for clustering_algorithm in clustering_algorithms:
            logger.info(f"Benchmarking {clustering_algorithm} on {dataset}...")
            results = run_clustering_algorithm_benchmark(
                dataset, clustering_algorithm, n_samples_list, n_clusters_list, clustering_dir
            )
            clustering_results.extend(results)

    # Create DataFrame and save results
    clustering_df = pd.DataFrame(clustering_results)
    clustering_file = clustering_dir / "clustering_benchmarks.csv"
    clustering_df.to_csv(clustering_file, index=False)

    # Run end-to-end benchmarks
    logger.info("Running end-to-end benchmarks...")
    end_to_end_results = []

    for dataset in datasets:
        for influence_method in influence_methods:
            for clustering_algorithm in clustering_algorithms:
                for n_clusters in n_clusters_list:
                    logger.info(f"Benchmarking {influence_method} + {clustering_algorithm} with {n_clusters} clusters on {dataset}...")
                    result = run_end_to_end_benchmark(
                        dataset, influence_method, clustering_algorithm, n_clusters, end_to_end_dir
                    )
                    end_to_end_results.append(result)

    # Create DataFrame and save results
    end_to_end_df = pd.DataFrame(end_to_end_results)
    end_to_end_file = end_to_end_dir / "end_to_end_benchmarks.csv"
    end_to_end_df.to_csv(end_to_end_file, index=False)

    # Create visualizations
    logger.info("Creating visualizations...")
    create_influence_visualizations(influence_df, output_dir)
    create_clustering_visualizations(clustering_df, output_dir)
    create_end_to_end_visualizations(end_to_end_df, output_dir)

    logger.info(f"Computational analysis completed. Results saved to {output_dir}")

    return {
        "influence_benchmarks": influence_df.to_dict(orient="records"),
        "clustering_benchmarks": clustering_df.to_dict(orient="records"),
        "end_to_end_benchmarks": end_to_end_df.to_dict(orient="records")
    }


def create_influence_visualizations(results_df, output_dir):
    """
    Create visualizations for influence method benchmarks.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing benchmark results.
    output_dir : str or Path
        Directory to save visualizations.
    """
    # Create visualizations directory
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Filter out error results
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    # Create runtime plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df,
        x="n_samples",
        y="runtime_seconds",
        hue="influence_method",
        marker="o"
    )
    plt.title("Influence Method Runtime by Sample Size")
    plt.xlabel("Number of Samples")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "influence_runtime.png", dpi=300)
    plt.close()

    # Create memory usage plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df,
        x="n_samples",
        y="memory_usage_mb",
        hue="influence_method",
        marker="o"
    )
    plt.title("Influence Method Memory Usage by Sample Size")
    plt.xlabel("Number of Samples")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "influence_memory.png", dpi=300)
    plt.close()

    # Create runtime comparison bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=results_df[results_df["n_samples"] == 1000],
        x="influence_method",
        y="runtime_seconds",
        hue="dataset"
    )
    plt.title("Influence Method Runtime Comparison (1000 samples)")
    plt.xlabel("Influence Method")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "influence_runtime_comparison.png", dpi=300)
    plt.close()


def create_clustering_visualizations(results_df, output_dir):
    """
    Create visualizations for clustering algorithm benchmarks.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing benchmark results.
    output_dir : str or Path
        Directory to save visualizations.
    """
    # Create visualizations directory
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Filter out error results
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    # Create runtime plot by sample size
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df[results_df["n_clusters"] == 5],
        x="n_samples",
        y="runtime_seconds",
        hue="clustering_algorithm",
        marker="o"
    )
    plt.title("Clustering Algorithm Runtime by Sample Size (5 clusters)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "clustering_runtime_by_samples.png", dpi=300)
    plt.close()

    # Create runtime plot by number of clusters
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df[results_df["n_samples"] == 1000],
        x="n_clusters",
        y="runtime_seconds",
        hue="clustering_algorithm",
        marker="o"
    )
    plt.title("Clustering Algorithm Runtime by Number of Clusters (1000 samples)")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "clustering_runtime_by_clusters.png", dpi=300)
    plt.close()

    # Create memory usage plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df[results_df["n_clusters"] == 5],
        x="n_samples",
        y="memory_usage_mb",
        hue="clustering_algorithm",
        marker="o"
    )
    plt.title("Clustering Algorithm Memory Usage by Sample Size (5 clusters)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "clustering_memory.png", dpi=300)
    plt.close()


def create_end_to_end_visualizations(results_df, output_dir):
    """
    Create visualizations for end-to-end benchmarks.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing benchmark results.
    output_dir : str or Path
        Directory to save visualizations.
    """
    # Create visualizations directory
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Filter out error results
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isna()]

    # Create stacked bar plot of component times
    plt.figure(figsize=(15, 10))

    # Melt the DataFrame to get it in the right format for stacked bars
    melted_df = pd.melt(
        results_df,
        id_vars=["dataset", "influence_method", "clustering_algorithm", "n_clusters"],
        value_vars=["preprocessing_time", "model_training_time", "influence_generation_time", "clustering_time"],
        var_name="component",
        value_name="time"
    )

    # Create the stacked bar plot
    sns.barplot(
        data=melted_df,
        x="influence_method",
        y="time",
        hue="component"
    )

    plt.title("End-to-End Runtime Breakdown by Component")
    plt.xlabel("Influence Method")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "end_to_end_component_times.png", dpi=300)
    plt.close()

    # Create total time comparison
    plt.figure(figsize=(15, 10))
    sns.barplot(
        data=results_df,
        x="influence_method",
        y="total_time",
        hue="clustering_algorithm"
    )
    plt.title("Total Runtime by Method and Algorithm")
    plt.xlabel("Influence Method")
    plt.ylabel("Total Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "end_to_end_total_time.png", dpi=300)
    plt.close()

    # Create influence generation time comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=results_df,
        x="influence_method",
        y="influence_generation_time",
        hue="dataset"
    )
    plt.title("Influence Generation Time by Method and Dataset")
    plt.xlabel("Influence Method")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "influence_generation_time.png", dpi=300)
    plt.close()

    # Create clustering time comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=results_df,
        x="clustering_algorithm",
        y="clustering_time",
        hue="n_clusters"
    )
    plt.title("Clustering Time by Algorithm and Number of Clusters")
    plt.xlabel("Clustering Algorithm")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(vis_dir / "clustering_time.png", dpi=300)
    plt.close()
