"""
Test script for contextual coherence experiments.
"""

import os
from pathlib import Path
from experiments.tnnls.contextual_coherence.run_contextual_coherence import run_single_experiment

# Define test parameters
dataset_name = "building_genome"
influence_method = "shap"
clustering_algorithm = "kmeans"
n_clusters = 3
random_seed = 42
output_dir = Path("data/results/test_contextual_coherence")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Run the experiment
try:
    print("Starting test experiment...")
    result = run_single_experiment(
        dataset_name=dataset_name,
        influence_method=influence_method,
        clustering_algorithm=clustering_algorithm,
        n_clusters=n_clusters,
        random_seed=random_seed,
        output_dir=output_dir
    )
    print("Test experiment completed successfully!")
    print(f"Results saved to {output_dir}")
except Exception as e:
    print(f"Test experiment failed: {e}")
