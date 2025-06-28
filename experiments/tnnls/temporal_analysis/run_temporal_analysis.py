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
from src.models.gradient_boost import GradientBoostModel
from src.models.torch_models import LSTMModel, TransformerModel
from src.influence.shap_influence import ShapInfluence
from src.influence.lime_influence import LimeInfluence
from src.influence.spearman_influence import SpearmanInfluence
from src.influence.torch_influence import IntegratedGradientsInfluence, DeepShapInfluence
from src.influence.hessian_influence import HessianInfluence
from src.clustering.kmeans import KMeansClustering
from src.clustering.hierarchical import HierarchicalClustering
from src.clustering.spectral import SpectralClustering
from src.temporal.transition_matrix import TransitionMatrix
from src.temporal.anomaly_detection import AnomalyDetection
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_temporal_consistency
from src.utils.visualization import visualize_transitions, visualize_temporal_evolution


def get_model(model_name, random_seed, input_dim=None):
    """Get model instance from name."""
    if model_name == "gradient_boost":
        model_params = config.MODEL_PARAMS["gradient_boost"].copy()
        model_params["random_state"] = random_seed
        return GradientBoostModel(**model_params)
    elif model_name == "lstm":
        model_params = config.MODEL_PARAMS["lstm"].copy()
        model_params["random_state"] = random_seed
        model_params["input_dim"] = input_dim
        return LSTMModel(**model_params)
    elif model_name == "transformer":
        model_params = config.MODEL_PARAMS["transformer"].copy()
        model_params["random_state"] = random_seed
        model_params["input_dim"] = input_dim
        return TransformerModel(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_influence_generator(influence_method, random_seed):
    """Get influence generator instance from name."""
    if influence_method == "shap":
        influence_params = config.INFLUENCE_PARAMS["shap"].copy()
        influence_params["random_state"] = random_seed
        return ShapInfluence(**influence_params)
    elif influence_method == "lime":
        influence_params = config.INFLUENCE_PARAMS["lime"].copy()
        influence_params["random_state"] = random_seed
        return LimeInfluence(**influence_params)
    elif influence_method == "spearman":
        influence_params = config.INFLUENCE_PARAMS["spearman"].copy()
        return SpearmanInfluence(**influence_params)
    elif influence_method == "integrated_gradients":
        influence_params = config.INFLUENCE_PARAMS["integrated_gradients"].copy()
        influence_params["random_state"] = random_seed
        return IntegratedGradientsInfluence(**influence_params)
    elif influence_method == "deepshap":
        influence_params = config.INFLUENCE_PARAMS["deepshap"].copy()
        influence_params["random_state"] = random_seed
        return DeepShapInfluence(**influence_params)
    elif influence_method == "hessian":
        influence_params = config.INFLUENCE_PARAMS["hessian"].copy()
        return HessianInfluence(**influence_params)
    else:
        raise ValueError(f"Unknown influence method: {influence_method}")


def run_single_experiment(dataset_name, model_name, influence_method, clustering_algorithm,
                         n_clusters, random_seed, output_dir):
    """
    Run a single temporal analysis experiment.
    """
    # ... (setup is the same)

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        X, y, t, c, _ = data_loader.load_data(preprocess=True)

        # Train predictive model
        model = get_model(model_name, random_seed, input_dim=X.shape[1])
        model.fit(X, y)

        # Generate influence space
        influence_generator = get_influence_generator(influence_method, random_seed)
        Z = influence_generator.generate_influence(model, X)

        # ... (rest of the function is the same)

    except Exception as e:
        logger.error(f"Error in experiment: {e}", exc_info=True)
        return None


def run_temporal_analysis(datasets, models, influence_methods, clustering_algorithms,
                         n_clusters_list, random_seeds, output_dir, logger, n_jobs=-1, verbose=False):
    """
    Run comprehensive temporal analysis experiments.
    """
    # ... (setup is the same)

    # Generate experiment configurations
    experiment_configs = []
    for dataset in datasets:
        for model in models:
            for influence_method in influence_methods:
                for clustering_algorithm in clustering_algorithms:
                    for n_clusters in n_clusters_list:
                        for random_seed in random_seeds:
                            experiment_configs.append({
                                "dataset_name": dataset,
                                "model_name": model,
                                "influence_method": influence_method,
                                "clustering_algorithm": clustering_algorithm,
                                "n_clusters": n_clusters,
                                "random_seed": random_seed,
                                "output_dir": output_dir
                            })

    # Run experiments in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_experiment)(**config) for config in experiment_configs
    )

    # Filter out None results (from failed experiments)
    results = [r for r in results if r is not None]

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_file = output_dir / "temporal_analysis_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Temporal analysis results saved to {results_file}")

    # Perform statistical analysis (e.g., Wilcoxon signed-rank test)
    # This part can be expanded based on specific analysis needs
    logger.info("Performing statistical analysis...")

    # Visualization (example: box plots of metrics)
    logger.info("Generating visualizations...")
    if not results_df.empty:
        # Example: Visualize temporal consistency
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x="influence_method", y="temporal_consistency")
        plt.title("Temporal Consistency by Influence Method")
        plt.savefig(output_dir / "temporal_consistency_boxplot.png")
        plt.close()

    return results_df.to_dict(orient="records")
