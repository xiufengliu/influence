#!/usr/bin/env python3
"""
Simple experiment runner for Dynamic Influence-Based Clustering.

This script provides basic functionality to run clustering experiments
with different influence methods and compare results.

Usage:
    python run_experiments.py --dataset your_data.csv
    python run_experiments.py --help
"""

import argparse
import logging
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import config
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.gradient_boost import GradientBoostModel
from src.influence.shap_influence import ShapInfluence
from src.influence.lime_influence import LimeInfluence
from src.influence.spearman_influence import SpearmanInfluence
from src.clustering.kmeans import KMeansClustering
from src.clustering.dynamic_kmeans import DynamicKMeansClustering
from src.utils.metrics import ClusteringEvaluator
from src.utils.logger import setup_logger


def run_clustering_experiment(dataset_name, n_clusters=3, influence_method='spearman', output_dir='results'):
    """Run a single clustering experiment."""
    
    # Setup logging
    logger = setup_logger("experiment", level="INFO")
    logger.info(f"Starting experiment: {dataset_name} with {influence_method} influence")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        data_loader = DataLoader(dataset_name=dataset_name)
        X, y, timestamps, contexts, entity_ids = data_loader.load_data()
        
        preprocessor = Preprocessor()
        X_processed = preprocessor.fit_transform(X)
        
        logger.info(f"Data loaded: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        
        # Train predictive model
        logger.info("Training predictive model...")
        model = GradientBoostModel()
        model.fit(X_processed, y)
        
        # Generate influence vectors
        logger.info(f"Computing {influence_method} influence vectors...")
        if influence_method == 'spearman':
            influence_extractor = SpearmanInfluence()
            influence_vectors = influence_extractor.compute_influence(X_processed, y)
        elif influence_method == 'shap':
            influence_extractor = ShapInfluence(model)
            influence_vectors = influence_extractor.compute_influence(X_processed)
        elif influence_method == 'lime':
            influence_extractor = LimeInfluence(model)
            influence_vectors = influence_extractor.compute_influence(X_processed)
        else:
            raise ValueError(f"Unknown influence method: {influence_method}")
        
        logger.info(f"Generated influence vectors: {influence_vectors.shape}")
        
        # Run clustering experiments
        results = {}
        
        # Standard K-means on raw features
        logger.info("Running standard K-means...")
        kmeans = KMeansClustering(n_clusters=n_clusters, random_state=42)
        labels_raw = kmeans.fit_predict(X_processed)
        
        # Dynamic K-means on influence vectors
        logger.info("Running dynamic K-means on influence space...")
        dynamic_kmeans = DynamicKMeansClustering(
            n_clusters=n_clusters,
            alpha=1.0, beta=1.0, gamma=1.0,
            random_state=42
        )
        labels_influence = dynamic_kmeans.fit_predict(influence_vectors, timestamps, contexts)
        
        # Evaluate results
        logger.info("Evaluating clustering results...")
        evaluator = ClusteringEvaluator()
        
        # Evaluate raw features clustering
        results['raw_features'] = {
            'silhouette': evaluator.silhouette_score(X_processed, labels_raw),
            'davies_bouldin': evaluator.davies_bouldin_score(X_processed, labels_raw),
            'temporal_consistency': evaluator.temporal_consistency(labels_raw, timestamps),
        }
        
        # Evaluate influence-based clustering
        results['influence_based'] = {
            'silhouette': evaluator.silhouette_score(influence_vectors, labels_influence),
            'davies_bouldin': evaluator.davies_bouldin_score(influence_vectors, labels_influence),
            'temporal_consistency': evaluator.temporal_consistency(labels_influence, timestamps),
        }
        
        # Save results
        results_file = output_path / f"experiment_{dataset_name}_{influence_method}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        logger.info("Experiment completed!")
        logger.info("Results Summary:")
        logger.info(f"Raw Features - Silhouette: {results['raw_features']['silhouette']:.3f}")
        logger.info(f"Influence-Based - Silhouette: {results['influence_based']['silhouette']:.3f}")
        logger.info(f"Improvement: {results['influence_based']['silhouette'] - results['raw_features']['silhouette']:+.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Dynamic Influence-Based Clustering Experiments")
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name or path to CSV file")
    parser.add_argument("--n_clusters", type=int, default=3,
                       help="Number of clusters (default: 3)")
    parser.add_argument("--influence", type=str, default="spearman",
                       choices=["spearman", "shap", "lime"],
                       help="Influence method to use (default: spearman)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all influence methods")
    
    args = parser.parse_args()
    
    if args.compare_all:
        print("Comparing all influence methods...")
        methods = ["spearman", "shap", "lime"]
        all_results = {}
        
        for method in methods:
            print(f"\n--- Running with {method} influence ---")
            try:
                results = run_clustering_experiment(
                    dataset_name=args.dataset,
                    n_clusters=args.n_clusters,
                    influence_method=method,
                    output_dir=args.output_dir
                )
                all_results[method] = results
            except Exception as e:
                print(f"Failed with {method}: {e}")
                continue
        
        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Method':<15} {'Raw Sil':<10} {'Inf Sil':<10} {'Improvement':<12}")
        print("-"*60)
        
        for method, results in all_results.items():
            raw_sil = results['raw_features']['silhouette']
            inf_sil = results['influence_based']['silhouette']
            improvement = inf_sil - raw_sil
            print(f"{method:<15} {raw_sil:<10.3f} {inf_sil:<10.3f} {improvement:<+12.3f}")
    
    else:
        run_clustering_experiment(
            dataset_name=args.dataset,
            n_clusters=args.n_clusters,
            influence_method=args.influence,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
