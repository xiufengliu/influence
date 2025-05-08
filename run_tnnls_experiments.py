"""
Main script to run TNNLS-quality experiments for the Dynamic Influence-Based Clustering Framework.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import config
from src.utils.logger import setup_logger
from experiments.tnnls.run_experiments import run_experiments


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TNNLS-quality experiments for Dynamic Influence-Based Clustering Framework"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["energy_data", "steel_industry"],
        help="Datasets to use for experiments"
    )

    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["clustering_quality", "temporal_analysis", "contextual_coherence",
                 "ablation_studies", "case_studies", "computational_analysis"],
        help="Experiments to run"
    )

    parser.add_argument(
        "--influence_methods",
        type=str,
        nargs="+",
        default=["shap", "lime", "spearman"],
        help="Influence methods to evaluate"
    )

    parser.add_argument(
        "--clustering_algorithms",
        type=str,
        nargs="+",
        default=["kmeans", "hierarchical", "spectral"],
        help="Clustering algorithms to evaluate"
    )

    parser.add_argument(
        "--n_clusters_list",
        type=int,
        nargs="+",
        default=config.TNNLS_EXPERIMENT_PARAMS["n_clusters_list"],
        help="Number of clusters to evaluate"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.RESULTS_DIR / "tnnls_experiments"),
        help="Directory to save results"
    )

    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="+",
        default=config.TNNLS_EXPERIMENT_PARAMS["random_seeds"],
        help="Random seeds for reproducibility and statistical analysis"
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=config.TNNLS_EXPERIMENT_PARAMS["n_jobs"],
        help="Number of parallel jobs (-1 for all available cores)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=config.TNNLS_EXPERIMENT_PARAMS["verbose"],
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Set up logging
    logger = setup_logger("tnnls_main", "INFO")
    logger.info("Starting TNNLS experiments...")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    try:
        results = run_experiments(
            datasets=args.datasets,
            experiments=args.experiments,
            influence_methods=args.influence_methods,
            clustering_algorithms=args.clustering_algorithms,
            n_clusters_list=args.n_clusters_list,
            random_seeds=args.random_seeds,
            output_dir=output_dir,
            n_jobs=args.n_jobs,
            verbose=args.verbose
        )

        logger.info("TNNLS experiments completed successfully.")

    except Exception as e:
        logger.error(f"Error in TNNLS experiments: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
