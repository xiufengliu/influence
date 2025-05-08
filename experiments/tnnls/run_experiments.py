"""
Main experiment runner for TNNLS-quality experiments.

This script coordinates all experiments for the Dynamic Influence-Based Clustering Framework
to produce results suitable for submission to IEEE Transactions on Neural Networks and Learning Systems.
"""

import os
import sys
import argparse
import logging
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.utils.logger import setup_logger
from experiments.tnnls.clustering_quality.run_clustering_quality import run_clustering_quality
from experiments.tnnls.temporal_analysis.run_temporal_analysis import run_temporal_analysis
from experiments.tnnls.contextual_coherence.run_contextual_coherence import run_contextual_coherence
from experiments.tnnls.ablation_studies.run_ablation_studies import run_ablation_studies
from experiments.tnnls.case_studies.run_case_studies import run_case_studies
from experiments.tnnls.computational_analysis.run_computational_analysis import run_computational_analysis


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
        default=[3, 5, 7],
        help="Number of clusters to evaluate"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results"
    )

    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 101112],
        help="Random seeds for reproducibility and statistical analysis"
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all available cores)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def setup_experiment(args):
    """Set up experiment environment."""
    # Create timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set output directory
    if args.output_dir is None:
        output_dir = config.RESULTS_DIR / f"tnnls_experiments_{timestamp}"
    else:
        output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = output_dir / "experiment.log"
    logger = setup_logger("tnnls_experiments", "INFO", log_file)

    # Log experiment configuration
    logger.info(f"Starting TNNLS experiments at {timestamp}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Experiments: {args.experiments}")
    logger.info(f"Influence methods: {args.influence_methods}")
    logger.info(f"Clustering algorithms: {args.clustering_algorithms}")
    logger.info(f"Number of clusters: {args.n_clusters_list}")
    logger.info(f"Random seeds: {args.random_seeds}")
    logger.info(f"Number of parallel jobs: {args.n_jobs}")
    logger.info(f"Output directory: {output_dir}")

    # Save experiment configuration
    config_file = output_dir / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=4)

    return output_dir, logger


def run_all_experiments(args, output_dir, logger):
    """Run all specified experiments."""
    # Create experiment parameters
    experiment_params = {
        "datasets": args.datasets,
        "influence_methods": args.influence_methods,
        "clustering_algorithms": args.clustering_algorithms,
        "n_clusters_list": args.n_clusters_list,
        "random_seeds": args.random_seeds,
        "n_jobs": args.n_jobs,
        "verbose": args.verbose
    }

    # Track experiment results
    results = {}

    # Run clustering quality experiments
    if "clustering_quality" in args.experiments:
        logger.info("Running clustering quality experiments...")
        start_time = time.time()
        clustering_quality_dir = output_dir / "clustering_quality"
        clustering_quality_dir.mkdir(exist_ok=True)

        clustering_results = run_clustering_quality(
            output_dir=clustering_quality_dir,
            **experiment_params
        )

        results["clustering_quality"] = clustering_results
        logger.info(f"Clustering quality experiments completed in {time.time() - start_time:.2f} seconds")

    # Run temporal analysis experiments
    if "temporal_analysis" in args.experiments:
        logger.info("Running temporal analysis experiments...")
        start_time = time.time()
        temporal_analysis_dir = output_dir / "temporal_analysis"
        temporal_analysis_dir.mkdir(exist_ok=True)

        temporal_results = run_temporal_analysis(
            output_dir=temporal_analysis_dir,
            **experiment_params
        )

        results["temporal_analysis"] = temporal_results
        logger.info(f"Temporal analysis experiments completed in {time.time() - start_time:.2f} seconds")

    # Run contextual coherence experiments
    if "contextual_coherence" in args.experiments:
        logger.info("Running contextual coherence experiments...")
        start_time = time.time()
        contextual_coherence_dir = output_dir / "contextual_coherence"
        contextual_coherence_dir.mkdir(exist_ok=True)

        contextual_results = run_contextual_coherence(
            output_dir=contextual_coherence_dir,
            **experiment_params
        )

        results["contextual_coherence"] = contextual_results
        logger.info(f"Contextual coherence experiments completed in {time.time() - start_time:.2f} seconds")

    # Run ablation studies
    if "ablation_studies" in args.experiments:
        logger.info("Running ablation studies...")
        start_time = time.time()
        ablation_studies_dir = output_dir / "ablation_studies"
        ablation_studies_dir.mkdir(exist_ok=True)

        ablation_results = run_ablation_studies(
            output_dir=ablation_studies_dir,
            **experiment_params
        )

        results["ablation_studies"] = ablation_results
        logger.info(f"Ablation studies completed in {time.time() - start_time:.2f} seconds")

    # Run case studies
    if "case_studies" in args.experiments:
        logger.info("Running case studies...")
        start_time = time.time()
        case_studies_dir = output_dir / "case_studies"
        case_studies_dir.mkdir(exist_ok=True)

        case_study_results = run_case_studies(
            output_dir=case_studies_dir,
            **experiment_params
        )

        results["case_studies"] = case_study_results
        logger.info(f"Case studies completed in {time.time() - start_time:.2f} seconds")

    # Run computational analysis
    if "computational_analysis" in args.experiments:
        logger.info("Running computational analysis...")
        start_time = time.time()
        computational_analysis_dir = output_dir / "computational_analysis"
        computational_analysis_dir.mkdir(exist_ok=True)

        computational_results = run_computational_analysis(
            output_dir=computational_analysis_dir,
            **experiment_params
        )

        results["computational_analysis"] = computational_results
        logger.info(f"Computational analysis completed in {time.time() - start_time:.2f} seconds")

    return results


def run_experiments(datasets, experiments, influence_methods, clustering_algorithms,
                   n_clusters_list, random_seeds, output_dir, n_jobs=-1, verbose=False):
    """
    Run TNNLS-quality experiments for Dynamic Influence-Based Clustering Framework.

    Parameters
    ----------
    datasets : list
        List of datasets to use for experiments.
    experiments : list
        List of experiments to run.
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
    # Create timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set output directory
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = output_dir / "experiment.log"
    logger = setup_logger("tnnls_experiments", "INFO", log_file)

    # Log experiment configuration
    logger.info(f"Starting TNNLS experiments at {timestamp}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Experiments: {experiments}")
    logger.info(f"Influence methods: {influence_methods}")
    logger.info(f"Clustering algorithms: {clustering_algorithms}")
    logger.info(f"Number of clusters: {n_clusters_list}")
    logger.info(f"Random seeds: {random_seeds}")
    logger.info(f"Number of parallel jobs: {n_jobs}")
    logger.info(f"Output directory: {output_dir}")

    # Save experiment configuration
    config_file = output_dir / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump({
            "datasets": datasets,
            "experiments": experiments,
            "influence_methods": influence_methods,
            "clustering_algorithms": clustering_algorithms,
            "n_clusters_list": n_clusters_list,
            "random_seeds": random_seeds,
            "n_jobs": n_jobs,
            "verbose": verbose
        }, f, indent=4)

    # Create experiment parameters
    experiment_params = {
        "datasets": datasets,
        "influence_methods": influence_methods,
        "clustering_algorithms": clustering_algorithms,
        "n_clusters_list": n_clusters_list,
        "random_seeds": random_seeds,
        "n_jobs": n_jobs,
        "verbose": verbose
    }

    # Track experiment results
    results = {}

    try:
        # Run clustering quality experiments
        if "clustering_quality" in experiments:
            logger.info("Running clustering quality experiments...")
            start_time = time.time()
            clustering_quality_dir = output_dir / "clustering_quality"
            clustering_quality_dir.mkdir(exist_ok=True)

            clustering_results = run_clustering_quality(
                output_dir=clustering_quality_dir,
                **experiment_params
            )

            results["clustering_quality"] = clustering_results
            logger.info(f"Clustering quality experiments completed in {time.time() - start_time:.2f} seconds")

        # Run temporal analysis experiments
        if "temporal_analysis" in experiments:
            logger.info("Running temporal analysis experiments...")
            start_time = time.time()
            temporal_analysis_dir = output_dir / "temporal_analysis"
            temporal_analysis_dir.mkdir(exist_ok=True)

            temporal_results = run_temporal_analysis(
                output_dir=temporal_analysis_dir,
                **experiment_params
            )

            results["temporal_analysis"] = temporal_results
            logger.info(f"Temporal analysis experiments completed in {time.time() - start_time:.2f} seconds")

        # Run contextual coherence experiments
        if "contextual_coherence" in experiments:
            logger.info("Running contextual coherence experiments...")
            start_time = time.time()
            contextual_coherence_dir = output_dir / "contextual_coherence"
            contextual_coherence_dir.mkdir(exist_ok=True)

            contextual_results = run_contextual_coherence(
                output_dir=contextual_coherence_dir,
                **experiment_params
            )

            results["contextual_coherence"] = contextual_results
            logger.info(f"Contextual coherence experiments completed in {time.time() - start_time:.2f} seconds")

        # Run ablation studies
        if "ablation_studies" in experiments:
            logger.info("Running ablation studies...")
            start_time = time.time()
            ablation_studies_dir = output_dir / "ablation_studies"
            ablation_studies_dir.mkdir(exist_ok=True)

            ablation_results = run_ablation_studies(
                output_dir=ablation_studies_dir,
                **experiment_params
            )

            results["ablation_studies"] = ablation_results
            logger.info(f"Ablation studies completed in {time.time() - start_time:.2f} seconds")

        # Run case studies
        if "case_studies" in experiments:
            logger.info("Running case studies...")
            start_time = time.time()
            case_studies_dir = output_dir / "case_studies"
            case_studies_dir.mkdir(exist_ok=True)

            case_study_results = run_case_studies(
                output_dir=case_studies_dir,
                **experiment_params
            )

            results["case_studies"] = case_study_results
            logger.info(f"Case studies completed in {time.time() - start_time:.2f} seconds")

        # Run computational analysis
        if "computational_analysis" in experiments:
            logger.info("Running computational analysis...")
            start_time = time.time()
            computational_analysis_dir = output_dir / "computational_analysis"
            computational_analysis_dir.mkdir(exist_ok=True)

            computational_results = run_computational_analysis(
                output_dir=computational_analysis_dir,
                **experiment_params
            )

            results["computational_analysis"] = computational_results
            logger.info(f"Computational analysis completed in {time.time() - start_time:.2f} seconds")

        # Save overall results
        results_file = output_dir / "all_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"All experiments completed. Results saved to {output_dir}")

        return results

    except Exception as e:
        logger.error(f"Error in experiment execution: {e}", exc_info=True)
        raise


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Set up experiment
    output_dir, logger = setup_experiment(args)

    try:
        # Run all experiments
        start_time = time.time()
        results = run_all_experiments(args, output_dir, logger)

        # Save overall results
        results_file = output_dir / "all_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"All experiments completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in experiment execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
