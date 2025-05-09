"""
Test script to run all TNNLS experiments with real datasets.
"""

import os
from pathlib import Path
from experiments.tnnls.run_experiments import run_experiments

# Define a comprehensive experiment to test all methods and algorithms
datasets = ["energy_data", "steel_industry"]
# Run all experiment types
experiments = [
    "clustering_quality",
    "temporal_analysis",
    "contextual_coherence",
    "ablation_studies",
    "case_studies",
    "computational_analysis"
]
influence_methods = ["shap", "lime", "spearman"]
clustering_algorithms = ["kmeans", "hierarchical"]  # Removed "spectral" as it's causing hanging issues
n_clusters_list = [3, 5]
random_seeds = [42]
output_dir = Path("data/results/test_tnnls_real_datasets")
# Set to -1 to use all available cores for parallel processing
n_jobs = 1  # Using 1 for sequential processing, change to -1 for parallel
verbose = True

# Run all experiments
try:
    print("Starting all TNNLS experiments with real datasets...")
    results = run_experiments(
        datasets=datasets,
        experiments=experiments,
        influence_methods=influence_methods,
        clustering_algorithms=clustering_algorithms,
        n_clusters_list=n_clusters_list,
        random_seeds=random_seeds,
        output_dir=output_dir,
        n_jobs=n_jobs,
        verbose=verbose
    )
    print("All TNNLS experiments completed successfully!")
    print(f"Results saved to {output_dir}")
except Exception as e:
    import traceback
    print(f"TNNLS experiments failed: {e}")
    traceback.print_exc()
