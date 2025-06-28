"""
Main entry point for the Dynamic Influence-Based Clustering Framework.
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.gradient_boost import GradientBoostModel
from src.models.torch_models import LSTMModel, TransformerModel # New imports
from src.influence.shap_influence import ShapInfluence
from src.influence.lime_influence import LimeInfluence
from src.influence.spearman_influence import SpearmanInfluence
from src.influence.integrated_gradients_influence import IntegratedGradientsInfluence # New import
from src.clustering.kmeans import KMeansClustering
from src.clustering.hierarchical import HierarchicalClustering
from src.clustering.spectral import SpectralClustering
from src.clustering.dynamic_kmeans import DynamicKMeansClustering # New import
from src.temporal.transition_matrix import TransitionMatrix
from src.temporal.anomaly_detection import AnomalyDetection
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_clustering, calculate_temporal_consistency, calculate_conditional_entropy, calculate_entropy # Updated imports
from src.utils.visualization import visualize_clusters, visualize_transitions
from src.utils.hyperparameter_tuning import HyperparameterTuner # New import


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dynamic Influence-Based Clustering Framework")
    
    parser.add_argument("--dataset", type=str, required=True, \
                        choices=["building_genome", "industrial_site1", "industrial_site2", "industrial_site3", "energy_data", "steel_industry", "household_power_consumption", "air_quality"],\
                        help="Dataset to use for analysis")

    parser.add_argument("--model", type=str, default="gradient_boost",\
                        choices=["gradient_boost", "lstm", "transformer"],\
                        help="Predictive model to use")
    
    parser.add_argument("--influence", type=str, default="shap",\
                        choices=["shap", "lime", "spearman", "integrated_gradients"],\
                        help="Influence method to use")
    
    parser.add_argument("--clustering", type=str, default="kmeans",\
                        choices=["kmeans", "hierarchical", "spectral", "dynamic_kmeans"],\
                        help="Clustering algorithm to use")
    
    parser.add_argument("--n_clusters", type=int, default=3,\
                        help="Number of clusters")
    
    parser.add_argument("--output_dir", type=str, default=None,\
                        help="Directory to save results")

    parser.add_argument("--tune_hyperparameters", action="store_true",\
                        help="Enable hyperparameter tuning for dynamic_kmeans")
    parser.add_argument("--alpha_range", type=float, nargs=2, default=[0.1, 10.0],\
                        help="Range for alpha in hyperparameter tuning (min max)")
    parser.add_argument("--beta_range", type=float, nargs=2, default=[0.1, 10.0],                        help="Range for beta in hyperparameter tuning (min max)")
    parser.add_argument("--window_size", type=int, default=24,\
                        help="Window size for dynamic K-means")

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("main", config.LOGGING_PARAMS["level"])
    logger.info(f"Starting Dynamic Influence-Based Clustering with {args.dataset} dataset")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else config.RESULTS_DIR / args.dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_loader = DataLoader(dataset_name=args.dataset)
        X, y, t, c, entity_ids = data_loader.load_data()
        logger.info(f"Data shapes: X={X.shape}, y={y.shape}, t={t.shape}, c={c.shape}, entity_ids={entity_ids.shape}")
        
        # Train predictive model
        logger.info(f"Training {args.model} predictive model...")
        input_dim = X.shape[1] # Number of features

        if args.model == "gradient_boost":
            model = GradientBoostModel(**config.MODEL_PARAMS["gradient_boost"])
        elif args.model == "lstm":
            model_params = config.MODEL_PARAMS["lstm"].copy()
            model_params["input_dim"] = input_dim
            model = LSTMModel(**model_params)
        elif args.model == "transformer":
            model_params = config.MODEL_PARAMS["transformer"].copy()
            model_params["input_dim"] = input_dim
            model = TransformerModel(**model_params)
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        model.fit(X, y)
        
        # Generate influence space
        logger.info(f"Generating influence space using {args.influence}...")
        if args.influence == "shap":
            influence_generator = ShapInfluence(**config.INFLUENCE_PARAMS["shap"])
        elif args.influence == "lime":
            influence_generator = LimeInfluence(**config.INFLUENCE_PARAMS["lime"])
        elif args.influence == "spearman":
            influence_generator = SpearmanInfluence(**config.INFLUENCE_PARAMS["spearman"])
        elif args.influence == "integrated_gradients":
            influence_generator = IntegratedGradientsInfluence(**config.INFLUENCE_PARAMS["integrated_gradients"])
        else:
            raise ValueError(f"Unknown influence method: {args.influence}")
        
        Z = influence_generator.generate_influence(model, X)
        
        # Perform clustering
        logger.info(f"Performing {args.clustering} clustering with {args.n_clusters} clusters...")
        clustering_params = config.CLUSTERING_PARAMS.get(args.clustering, {}).copy()
        clustering_params["n_clusters"] = args.n_clusters

        if args.clustering == "dynamic_kmeans":
            clustering_params["window_size"] = args.window_size
            
            if args.tune_hyperparameters:
                logger.info("Starting hyperparameter tuning for Dynamic K-means...")
                param_grid = {
                    'alpha': np.linspace(args.alpha_range[0], args.alpha_range[1], 3).tolist(),
                    'beta': np.linspace(args.beta_range[0], args.beta_range[1], 3).tolist(),
                    'n_clusters': [args.n_clusters],
                    'window_size': [args.window_size]
                }
                # Define evaluation metrics and weights for tuning
                # Assuming higher is better for all these metrics
                evaluation_metrics = ['silhouette', 'temporal_consistency', 'normalized_information_gain']
                metric_weights = {
                    'silhouette': 0.4, \
                    'temporal_consistency': 0.3, \
                    'normalized_information_gain': 0.3
                } # Example weights

                tuner = HyperparameterTuner(
                    model_class=DynamicKMeansClustering,\
                    param_grid=param_grid,\
                    evaluation_metrics=evaluation_metrics,\
                    metric_weights=metric_weights
                )
                best_params, best_score = tuner.tune(Z, t, c, entity_ids)
                logger.info(f"Best hyperparameters found: {best_params} with score: {best_score}")
                clustering_params.update(best_params)
            
            clustering = DynamicKMeansClustering(**clustering_params)
            clusters = clustering.fit(Z, t, c, entity_ids).labels_

        elif args.clustering == "kmeans":
            clustering = KMeansClustering(**clustering_params)
            clusters = clustering.fit_predict(Z)
        elif args.clustering == "hierarchical":
            clustering = HierarchicalClustering(**clustering_params)
            clusters = clustering.fit_predict(Z)
        else:
            clustering = SpectralClustering(**clustering_params)
            clusters = clustering.fit_predict(Z)
        
        # Compute transition matrix
        logger.info("Computing transition matrix...")
        transition_matrix = TransitionMatrix()
        P = transition_matrix.compute(clusters, t)
        
        # Detect anomalies
        logger.info("Detecting anomalies...")
        anomaly_detector = AnomalyDetection()
        anomalies = anomaly_detector.detect(clusters, P, t)
        
        # Evaluate clustering
        logger.info("Evaluating clustering results...")
        metrics = evaluate_clustering(Z, clusters)

        # Add temporal consistency and normalized information gain to metrics
        tc_score = calculate_temporal_consistency(clusters, t)
        metrics['temporal_consistency'] = tc_score

        entropy_labels = calculate_entropy(clusters)
        conditional_entropy_labels_context = calculate_conditional_entropy(clusters, c)
        nig_score = (entropy_labels - conditional_entropy_labels_context) / (entropy_labels + 1e-9)
        metrics['normalized_information_gain'] = nig_score
        
        # Visualize results
        logger.info("Visualizing results...")
        visualize_clusters(Z, clusters, output_dir / "clusters.png")
        visualize_transitions(P, output_dir / "transitions.png")
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Clustering metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error in execution: {e}")
        raise
    
    logger.info("Execution completed successfully")


if __name__ == "__main__":
    main()