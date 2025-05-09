"""
Test script to run a minimal TNNLS experiment with the smallest possible settings.
This is designed to run successfully even on machines with limited memory.
"""

import os
import sys
from pathlib import Path
from experiments.tnnls.run_experiments import run_experiments

# Define minimal experiment parameters
datasets = ["energy_data"]  # Use only one dataset
experiments = ["clustering_quality"]  # Run only one experiment type
influence_methods = ["spearman"]  # Use only Spearman (least memory-intensive)
clustering_algorithms = ["kmeans"]  # Use only K-means
n_clusters_list = [3]  # Use only one cluster size
random_seeds = [42]  # Use only one random seed
output_dir = Path("data/results/test_tnnls_minimal")
# Use minimal parallel processing
n_jobs = 1  # Use a single job
verbose = True

# Allow command-line override of experiment type
if len(sys.argv) > 1:
    requested_experiment = sys.argv[1].lower()
    valid_experiments = [
        "clustering_quality", 
        "temporal_analysis", 
        "contextual_coherence", 
        "ablation_studies", 
        "case_studies", 
        "computational_analysis"
    ]
    if requested_experiment in valid_experiments:
        experiments = [requested_experiment]
        print(f"Running only {requested_experiment} experiment as requested")
    else:
        print(f"Warning: Unknown experiment '{requested_experiment}'. Using default.")

# Run the minimal experiment
print(f"Starting minimal TNNLS experiment: {experiments[0]}...")
print(f"Using dataset: {datasets[0]}, influence: {influence_methods[0]}, clustering: {clustering_algorithms[0]}")
print(f"Using {n_jobs} parallel job to minimize memory usage")

try:
    # Run the experiment with minimal settings
    result = run_experiments(
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
    
    print(f"\n{'='*80}")
    print(f"Minimal TNNLS experiment completed successfully!")
    print(f"{'='*80}")
    print(f"Results saved to {output_dir}")
    
except Exception as e:
    import traceback
    print(f"\n{'='*80}")
    print(f"Minimal TNNLS experiment failed: {e}")
    print(f"{'='*80}")
    traceback.print_exc()
