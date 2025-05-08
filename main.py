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
from src.influence.shap_influence import ShapInfluence
from src.influence.lime_influence import LimeInfluence
from src.influence.spearman_influence import SpearmanInfluence
from src.clustering.kmeans import KMeansClustering
from src.clustering.hierarchical import HierarchicalClustering
from src.clustering.spectral import SpectralClustering
from src.temporal.transition_matrix import TransitionMatrix
from src.temporal.anomaly_detection import AnomalyDetection
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_clustering
from src.utils.visualization import visualize_clusters, visualize_transitions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dynamic Influence-Based Clustering Framework")
    
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
    
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")
    
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
        data = data_loader.load_data()
        
        preprocessor = Preprocessor()
        X, y, t, c = preprocessor.preprocess(data)
        
        # Train predictive model
        logger.info("Training predictive model...")
        model = GradientBoostModel(**config.MODEL_PARAMS["gradient_boost"])
        model.fit(X, y)
        
        # Generate influence space
        logger.info(f"Generating influence space using {args.influence}...")
        if args.influence == "shap":
            influence_generator = ShapInfluence(**config.INFLUENCE_PARAMS["shap"])
        elif args.influence == "lime":
            influence_generator = LimeInfluence(**config.INFLUENCE_PARAMS["lime"])
        else:
            influence_generator = SpearmanInfluence(**config.INFLUENCE_PARAMS["spearman"])
        
        Z = influence_generator.generate_influence(model, X)
        
        # Perform clustering
        logger.info(f"Performing {args.clustering} clustering with {args.n_clusters} clusters...")
        clustering_params = config.CLUSTERING_PARAMS[args.clustering].copy()
        clustering_params["n_clusters"] = args.n_clusters
        
        if args.clustering == "kmeans":
            clustering = KMeansClustering(**clustering_params)
        elif args.clustering == "hierarchical":
            clustering = HierarchicalClustering(**clustering_params)
        else:
            clustering = SpectralClustering(**clustering_params)
        
        clusters = clustering.fit_predict(Z, t, c)
        
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
