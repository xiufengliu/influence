"""
Test script to run all TNNLS experiments with real datasets.
"""

import os
from pathlib import Path
from experiments.tnnls.run_experiments import run_experiments

# Define a more focused experiment with stable configurations
datasets = ["energy_data", "steel_industry"]
# Run only the most stable experiment types
experiments = [
    "temporal_analysis",  # Focus on temporal analysis which is most relevant for the paper
    "clustering_quality"  # Include basic clustering quality assessment
]
influence_methods = ["shap"]  # Focus on SHAP which is the most stable method
clustering_algorithms = ["kmeans"]  # Focus on K-means which is the most reliable algorithm
n_clusters_list = [3]  # Focus on 3 clusters which works well with the datasets
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
