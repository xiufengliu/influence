"""
Test script to run a small TNNLS experiment.
"""

import os
from pathlib import Path
from experiments.tnnls.run_experiments import run_experiments

# Define a minimal experiment
datasets = ["building_genome"]
experiments = ["contextual_coherence"]
influence_methods = ["shap"]
clustering_algorithms = ["kmeans"]
n_clusters_list = [3]
random_seeds = [42]
output_dir = Path("data/results/test_tnnls_experiment")
n_jobs = 1
verbose = True

# Run the experiment
try:
    print("Starting test experiment...")
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
    print("Test experiment completed successfully!")
    print(f"Results saved to {output_dir}")
except Exception as e:
    print(f"Test experiment failed: {e}")
