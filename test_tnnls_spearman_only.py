"""
Test script to run TNNLS experiments with only Spearman influence method.
This is a simplified version that avoids memory-intensive SHAP and LIME calculations.
"""

import os
from pathlib import Path
from experiments.tnnls.run_experiments import run_experiments

# Define a simplified experiment to avoid memory issues
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
# Only use Spearman influence method which is less memory-intensive
influence_methods = ["spearman"]
clustering_algorithms = ["kmeans", "hierarchical"]  # Spectral clustering removed due to stability issues
n_clusters_list = [3, 5]
random_seeds = [42]
output_dir = Path("data/results/test_tnnls_spearman_only")
# Use conservative parallel processing
safe_n_jobs = min(4, os.cpu_count() or 1)
verbose = True

# Run experiments one by one to avoid memory issues
print("Starting TNNLS experiments with Spearman influence only...")
print(f"Using {safe_n_jobs} parallel jobs to avoid memory issues")

# Run each experiment type separately
all_results = {}
for experiment in experiments:
    try:
        print(f"\n\n{'='*80}\nRunning {experiment} experiment...\n{'='*80}\n")
        
        # Run just this experiment
        result = run_experiments(
            datasets=datasets,
            experiments=[experiment],  # Run only one experiment type at a time
            influence_methods=influence_methods,  # Only use Spearman
            clustering_algorithms=clustering_algorithms,
            n_clusters_list=n_clusters_list,
            random_seeds=random_seeds,
            output_dir=output_dir,
            n_jobs=safe_n_jobs,  # Use conservative parallel settings
            verbose=verbose
        )
        
        all_results[experiment] = result
        print(f"{experiment} experiment completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"{experiment} experiment failed: {e}")
        traceback.print_exc()
        print(f"Continuing with next experiment...")

# Report overall results
print("\n\n" + "="*80)
print("TNNLS experiments summary:")
print("="*80)
for experiment in experiments:
    if experiment in all_results:
        print(f"✅ {experiment}: Completed successfully")
    else:
        print(f"❌ {experiment}: Failed")
print(f"\nResults saved to {output_dir}")
